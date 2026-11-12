#!/usr/bin/env python3
"""
controller_node -- lifecycle-managed trajectory controller for the planar arm.

Interfaces
----------
Publishes
    /joint_states        sensor_msgs/JointState            measured (publish_rate_hz)
    /joint_reference     sensor_msgs/JointState            trajectory reference, same stamp
    /controller_status   planar_arm_interfaces/ControllerStatus   (status_rate_hz)
Subscribes
    /target_pose         geometry_msgs/PointStamped        fire-and-forget target
Services
    /move_to_target      planar_arm_interfaces/srv/MoveToTarget
    /move_joints         planar_arm_interfaces/srv/MoveJoints  (joint-space goal)
    /home                std_srvs/srv/Trigger
    /stop                std_srvs/srv/Trigger
    /gripper             std_srvs/srv/SetBool              (true = close)
Action
    /pick_place          planar_arm_interfaces/action/PickPlace  (feedback + cancel)

Threading / timing model
------------------------
Three loops at different rates:

    control  (control_rate_hz, default 200 Hz): read backend -> sample
             trajectory -> control law -> write backend. Never does IK.
             Runs as an rclpy timer in its own MutuallyExclusive group
             (control_thread:=ros_timer, default) or in a dedicated thread on
             absolute deadlines (control_thread:=dedicated).
    publish  (publish_rate_hz, default 50 Hz):  JointState + reference
    status   (status_rate_hz,  default 10 Hz):  ControllerStatus incl. jitter

Services/topic/action callbacks live in a Reentrant group of a
MultiThreadedExecutor, so a 10 s pick-and-place action never blocks anything.
The IK (~15 ms pure Python with ik_solver:=library, <1 ms with the default
corrected solver) runs in a worker *process* (ik_in_subprocess) so it
cannot hold the GIL against the control loop -- measured in docs/PROFILING.md.
Shared state is protected by a single lock held only for short sections.
"""

import multiprocessing as mp
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                   ReentrantCallbackGroup)
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger

from planar_arm_interfaces.action import PickPlace
from planar_arm_interfaces.msg import ControllerStatus
from planar_arm_interfaces.srv import MoveJoints, MoveToTarget

# The provided kinematics library -- used unmodified.
from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.backends import CONTROL_MODES, SimBackend, SimJointParams
from planar_arm_control.control_modes import make_control_mode
from planar_arm_control.pick_place import MotionInterface, Payload, PickPlaceSequencer
from planar_arm_control.planning import HOME_Q, MotionPlanner
from planar_arm_control.timing import LoopTimer
from planar_arm_control.trajectory import JointTrajectory

LINK_LENGTHS = [3.0, 2.0, 1.5]
JOINT_NAMES = ["joint1", "joint2", "joint3"]


def _pt(xy):
    return Point(x=float(xy[0]), y=float(xy[1]), z=0.0)


class ControllerNode(LifecycleNode, MotionInterface):

    def __init__(self, **kwargs):
        super().__init__("controller_node", **kwargs)
        d = self.declare_parameter
        d("backend", "sim", ParameterDescriptor(
            description="sim | gazebo", read_only=True))
        d("control_mode", "position", ParameterDescriptor(
            description="position | velocity | current (switchable when idle)"))
        d("control_rate_hz", 200.0, ParameterDescriptor(read_only=True))
        d("publish_rate_hz", 50.0, ParameterDescriptor(read_only=True))
        d("status_rate_hz", 10.0, ParameterDescriptor(read_only=True))
        d("max_joint_velocity", 1.0, ParameterDescriptor(
            description="rad/s, used to time-scale trajectories"))
        d("max_joint_acceleration", 2.0, ParameterDescriptor(description="rad/s^2"))
        d("min_trajectory_duration", 0.5)
        d("trajectory_duration", 0.0, ParameterDescriptor(
            description="fixed duration in s; <= 0 = derive from limits"))
        d("settle_tolerance_deg", 0.5)
        d("settle_timeout", 2.0)
        d("initial_joint_positions_deg", [90.0, 0.0, 0.0])
        d("home_on_activate", False)
        d("grip_time", 0.5)
        d("autostart", True, ParameterDescriptor(
            description="configure + activate on startup", read_only=True))
        d("vel_kp", 8.0)
        d("vel_ki", 4.0)
        d("cur_kp", 400.0)
        d("cur_ki", 800.0)
        d("cur_kd", 40.0)
        d("encoder_ticks_per_rev", 4096)
        d("sim_gravity", True)
        d("control_thread", "ros_timer", ParameterDescriptor(
            description="dedicated: control loop in its own thread sleeping to "
                        "absolute deadlines | ros_timer: rclpy timer in the "
                        "executor", read_only=True))
        d("ik_solver", "corrected", ParameterDescriptor(
            description="corrected: provided IK with its elbow (beta) formula "
                        "fixed, so q3 is used | library: provided "
                        "PlanarArm.inverse_kinematics as is (q3 always 0)",
            read_only=True))
        d("ik_in_subprocess", True, ParameterDescriptor(
            description="run the pure-Python IK in a worker process so it "
                        "cannot hold the GIL against the control loop",
            read_only=True))
        d("python_switch_interval_ms", 1.0, ParameterDescriptor(
            description="sys.setswitchinterval: max time another Python thread "
                        "(e.g. an IK solve) can hold the GIL before the control "
                        "thread gets a turn", read_only=True))

        self.arm = PlanarArm(LINK_LENGTHS)
        self._lock = threading.RLock()
        self._cb_control = MutuallyExclusiveCallbackGroup()
        self._cb_publish = MutuallyExclusiveCallbackGroup()
        self._cb_status = MutuallyExclusiveCallbackGroup()
        self._cb_cmd = ReentrantCallbackGroup()
        self._active = False
        self._loop_timers = []
        self._entities = []
        self.backend = None
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.get_logger().info("controller_node created (unconfigured)")

    # ================================================================ lifecycle
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        p = self.get_parameter
        try:
            self.sim_params = SimJointParams(
                encoder_ticks_per_rev=int(p("encoder_ticks_per_rev").value),
                gravity=bool(p("sim_gravity").value))
            self.ik_pool = None
            if p("ik_in_subprocess").value:
                self.ik_pool = ProcessPoolExecutor(
                    max_workers=1, mp_context=mp.get_context("forkserver"))
            self.planner = MotionPlanner(
                self.arm, p("max_joint_velocity").value,
                p("max_joint_acceleration").value,
                p("min_trajectory_duration").value, ik_executor=self.ik_pool,
                ik_solver=p("ik_solver").value)
            if self.ik_pool is not None:      # warm up: first spawn is slow
                self.planner.solve_ik((4.0, 2.0), HOME_Q)
            q0 = np.radians(p("initial_joint_positions_deg").value)
            backend = p("backend").value
            if backend == "sim":
                self.backend = SimBackend(self.arm, q0, self.sim_params)
            elif backend == "gazebo":
                from planar_arm_control.gazebo_backend import GazeboBackend
                self.backend = GazeboBackend(self, JOINT_NAMES, q0)
            else:
                raise ValueError(f"unknown backend '{backend}'")
            mode = p("control_mode").value
            if not self.backend.supports(mode):
                self.get_logger().warning(
                    f"backend '{backend}' does not support '{mode}' mode; "
                    f"falling back to 'position'")
                mode = "position"
            self.mode_name = mode
            self.control = make_control_mode(mode, self.sim_params, self._gains())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"configure failed: {exc}")
            return TransitionCallbackReturn.FAILURE

        # motion state
        self._traj = JointTrajectory.hold(q0)
        self._t0 = time.monotonic()
        self._motion_state = "idle"
        self._motion_id = 0
        self._busy_action = False
        self._settled_since = None
        self._ref = self._traj.sample(0.0)
        self._meas = self.backend.read()
        self._last_cmd = np.zeros(3)
        self._requested = np.array(self.arm.end_effector(q0))
        self._reachable = self._requested.copy()
        self._projected = False
        self._message = "configured"
        self._gripper_closed = False
        self.payload = Payload(position=(4.0, 2.0), grasp_tolerance=0.1)
        self._anchor_pending = True
        self._home_pending = False

        self.control_timer_stats = LoopTimer(1.0 / p("control_rate_hz").value)

        # publishers (lifecycle-managed: silent unless active)
        self.pub_js = self.create_lifecycle_publisher(JointState, "joint_states", 10)
        self.pub_ref = self.create_lifecycle_publisher(JointState, "joint_reference", 10)
        self.pub_status = self.create_lifecycle_publisher(
            ControllerStatus, "controller_status", 10)

        cg = self._cb_cmd
        self._entities = [
            self.create_service(MoveToTarget, "move_to_target",
                                self._srv_move, callback_group=cg),
            self.create_service(MoveJoints, "move_joints",
                                self._srv_move_joints, callback_group=cg),
            self.create_service(Trigger, "home", self._srv_home, callback_group=cg),
            self.create_service(Trigger, "stop", self._srv_stop, callback_group=cg),
            self.create_service(SetBool, "gripper", self._srv_gripper,
                                callback_group=cg),
            self.create_subscription(PointStamped, "target_pose",
                                     self._on_target_pose, 10, callback_group=cg),
        ]
        self.action_server = ActionServer(
            self, PickPlace, "pick_place",
            execute_callback=self._execute_pick_place,
            goal_callback=self._on_goal, cancel_callback=self._on_cancel,
            callback_group=cg)
        self.sequencer = PickPlaceSequencer(self, grip_time=p("grip_time").value)
        self.get_logger().info(
            f"configured: backend={backend} mode={self.mode_name} "
            f"control={p('control_rate_hz').value:.0f} Hz "
            f"publish={p('publish_rate_hz').value:.0f} Hz")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        p = self.get_parameter
        self.backend.enable()
        # Always start from the *measured* position: no jump on (re)activation.
        # A remote backend (Gazebo, hardware) may not have reported yet; the
        # control loop then waits for the first measurement before commanding.
        self._anchor_pending = True
        self._home_pending = bool(p("home_on_activate").value)
        self._anchor_to_measurement()
        self.control.reset()
        self.control_timer_stats.reset()
        self._control_stop = threading.Event()
        self._control_thread = None
        if p("control_thread").value == "dedicated":
            self._control_thread = threading.Thread(
                target=self._control_thread_main, name="control-loop", daemon=True)
            self._control_thread.start()
        else:
            self._loop_timers.append(self.create_timer(
                1.0 / p("control_rate_hz").value, self._control_step,
                callback_group=self._cb_control))
        self._loop_timers += [
            self.create_timer(1.0 / p("publish_rate_hz").value, self._publish_state,
                              callback_group=self._cb_publish),
            self.create_timer(1.0 / p("status_rate_hz").value, self._publish_status,
                              callback_group=self._cb_status),
        ]
        self._active = True
        ret = super().on_activate(state)
        self.get_logger().info("activated")
        return ret

    def _anchor_to_measurement(self):
        """Hold at the measured position once the backend is ready."""
        if not self._anchor_pending or not self.backend.ready:
            return
        meas = self.backend.read()
        with self._lock:
            self._meas = meas
            self._set_trajectory(JointTrajectory.hold(meas.position), "idle")
            self.control.reset()
            self._anchor_pending = False
        self.get_logger().info(
            f"anchored at measured q = "
            f"{np.round(np.degrees(meas.position), 1).tolist()} deg")
        if self._home_pending:
            self._home_pending = False
            ok, msg = self.start_home()
            self.get_logger().info(f"home on activate: {msg}")

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._active = False
        self._destroy_timers()
        with self._lock:
            self._set_trajectory(JointTrajectory.hold(self._ref.position), "idle")
        self.get_logger().info("deactivated (holding last reference)")
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._active = False
        self._teardown()
        return TransitionCallbackReturn.SUCCESS

    def _control_thread_main(self):
        """
        Fixed-rate loop on absolute deadlines (no drift). If a cycle overruns,
        missed ticks are skipped rather than run back-to-back.
        """
        period = 1.0 / self.get_parameter("control_rate_hz").value
        next_t = time.perf_counter()
        while not self._control_stop.is_set():
            self._control_step()
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t += period * np.ceil(-delay / period)

    def _destroy_timers(self):
        if getattr(self, "_control_thread", None) is not None:
            self._control_stop.set()
            self._control_thread.join(timeout=1.0)
            self._control_thread = None
        for t in self._loop_timers:
            self.destroy_timer(t)
        self._loop_timers = []

    def _teardown(self):
        self._destroy_timers()
        for e in self._entities:
            if hasattr(e, "srv_name"):
                self.destroy_service(e)
            else:
                self.destroy_subscription(e)
        self._entities = []
        if getattr(self, "action_server", None) is not None:
            self.action_server.destroy()
            self.action_server = None
        for pub in ("pub_js", "pub_ref", "pub_status"):
            if getattr(self, pub, None) is not None:
                self.destroy_lifecycle_publisher(getattr(self, pub))
                setattr(self, pub, None)
        if getattr(self, "ik_pool", None) is not None:
            self.ik_pool.shutdown(wait=False, cancel_futures=True)
            self.ik_pool = None
        if self.backend is not None:
            self.backend.disable()
            self.backend.close()
            self.backend = None

    # ============================================================ parameters
    def _gains(self):
        return {k: self.get_parameter(k).value
                for k in ("vel_kp", "vel_ki", "cur_kp", "cur_ki", "cur_kd")}

    def _on_set_parameters(self, params):
        for prm in params:
            if prm.name == "control_mode":
                if prm.value not in CONTROL_MODES:
                    return SetParametersResult(
                        successful=False,
                        reason=f"control_mode must be one of {CONTROL_MODES}")
                if self.backend is not None:
                    if not self.backend.supports(prm.value):
                        return SetParametersResult(
                            successful=False,
                            reason=f"backend '{self.backend.name}' does not "
                                   f"support '{prm.value}'")
                    with self._lock:
                        if self._motion_state == "moving" or self._busy_action:
                            return SetParametersResult(
                                successful=False,
                                reason="cannot switch control mode while moving")
                        gains = self._gains()
                        self.control = make_control_mode(
                            prm.value, self.sim_params, gains)
                        # re-anchor the reference to the measured position
                        self._set_trajectory(
                            JointTrajectory.hold(self._meas.position), "idle")
                        self.mode_name = prm.value
                        self._message = f"control mode -> {prm.value}"
                    self.get_logger().info(f"control mode switched to {prm.value}")
            elif prm.name == "max_joint_velocity" and hasattr(self, "planner"):
                self.planner.max_velocity[:] = float(prm.value)
            elif prm.name == "max_joint_acceleration" and hasattr(self, "planner"):
                self.planner.max_acceleration[:] = float(prm.value)
        return SetParametersResult(successful=True)

    # ============================================================ control loop
    def _set_trajectory(self, traj, state):
        """Caller holds the lock."""
        self._traj = traj
        self._t0 = time.monotonic()
        self._ref = traj.sample(0.0)     # keep the reference consistent at once
        self._motion_state = state
        self._motion_id += 1
        self._settled_since = None

    def _control_step(self):
        dt = self.control_timer_stats.start()
        dt = min(dt, 0.05)   # don't integrate across a long stall
        backend = self.backend
        if backend is None or not backend.ready:
            return                               # no measurement yet: don't command
        if self._anchor_pending:
            self._anchor_to_measurement()
        backend.update(dt)                       # sim physics (no-op on hw)
        meas = backend.read()
        with self._lock:
            t = time.monotonic() - self._t0
            ref = self._traj.sample(t)
            cmd = self.control.compute(ref, meas, dt)
            backend.write(self.mode_name, cmd)
            self._ref, self._meas, self._last_cmd = ref, meas, cmd
            ee = self.arm.end_effector(meas.position)
            self.payload.follow(ee)
            if self._motion_state == "moving" and t >= self._traj.duration:
                err = np.max(np.abs(ref.position - meas.position))
                tol = np.radians(self.get_parameter("settle_tolerance_deg").value)
                timeout = self.get_parameter("settle_timeout").value
                if err < tol or t > self._traj.duration + timeout:
                    self._motion_state = "holding"
                    if err >= tol:
                        self._message = (f"settle timeout, residual error "
                                         f"{np.degrees(err):.2f} deg")
        self.control_timer_stats.stop()

    def _publish_state(self):
        with self._lock:
            ref, meas = self._ref, self._meas
        stamp = self.get_clock().now().to_msg()
        js = JointState(name=JOINT_NAMES, position=meas.position.tolist(),
                        velocity=meas.velocity.tolist(),
                        effort=meas.effort.tolist())
        js.header.stamp = stamp
        rf = JointState(name=JOINT_NAMES, position=ref.position.tolist(),
                        velocity=ref.velocity.tolist(),
                        effort=ref.acceleration.tolist())
        rf.header.stamp = stamp
        self.pub_js.publish(js)
        self.pub_ref.publish(rf)

    def _publish_status(self):
        st = self.control_timer_stats.stats()
        with self._lock:
            msg = ControllerStatus(
                state=self._motion_state, control_mode=self.mode_name,
                backend=self.backend.name if self.backend else "",
                requested_target=_pt(self._requested),
                reachable_target=_pt(self._reachable),
                target_projected=self._projected,
                progress=self._traj.progress(time.monotonic() - self._t0),
                tracking_error=(self._ref.position - self._meas.position).tolist(),
                gripper_closed=self._gripper_closed,
                object_attached=self.payload.attached,
                object_position=_pt(self.payload.position),
                control_period_target_ms=st["target_ms"],
                control_period_mean_ms=st["mean_ms"],
                control_period_std_ms=st["std_ms"],
                control_period_max_ms=st["max_ms"],
                control_jitter_p99_ms=st["p99_jitter_ms"],
                control_loop_exec_mean_ms=st["exec_mean_ms"],
                message=self._message)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_status.publish(msg)

    # ======================================== MotionInterface (used by sequencer)
    def current_q(self):
        with self._lock:
            return self._ref.position.copy()

    def end_effector(self):
        with self._lock:
            return tuple(self.arm.end_effector(self._meas.position))

    def check_target(self, xy, q_from):
        ik = self.planner.solve_ik(xy, q_from)
        if ik.ok and ik.projected:
            return False, None, ik.message
        return ik.ok, ik.q_goal, ik.message

    def start_move(self, xy, duration=None):
        ok, info = self._plan_and_start(xy, duration)
        return ok, info["message"]

    def start_home(self):
        with self._lock:
            q_from = self._ref.position.copy()
        plan = self.planner.plan(q_from, HOME_Q)
        if not plan.ok:
            return False, plan.message
        with self._lock:
            self._set_trajectory(plan.trajectory, "moving")
            self._requested = np.array(self.arm.end_effector(HOME_Q))
            self._reachable, self._projected = self._requested.copy(), False
            self._message = f"homing ({plan.message})"
        return True, plan.message

    def motion_done(self):
        with self._lock:
            return self._motion_state != "moving"

    def motion_progress(self):
        with self._lock:
            return self._traj.progress(time.monotonic() - self._t0)

    def stop(self):
        with self._lock:
            self._set_trajectory(JointTrajectory.hold(self._ref.position), "holding")
            self._message = "stopped"

    def set_gripper(self, closed):
        with self._lock:
            self._gripper_closed = bool(closed)
            ee = self.arm.end_effector(self._meas.position)
            if closed:
                attached = self.payload.try_attach(ee)
                self._message = ("gripper closed, object grasped" if attached
                                 else "gripper closed, nothing to grasp")
            else:
                self.payload.release(ee)
                attached = False
                self._message = "gripper opened, object released"
        self.get_logger().info(self._message)
        return attached

    # ======================================================= planning helper
    def _plan_and_start(self, xy, duration=None):
        info = {"accepted": False, "projected": False, "reachable": np.zeros(2),
                "q_goal": [], "duration": 0.0, "message": ""}
        if not self._active or self._anchor_pending:
            info["message"] = "controller is not active / no measurement yet"
            return False, info
        with self._lock:
            q_from = self._ref.position.copy()
        ik = self.planner.solve_ik(xy, q_from)       # ~15 ms, outside the lock
        if ik.library_log:
            lines = dict.fromkeys(ik.library_log.splitlines())  # dedupe, keep order
            self.get_logger().info("[planar_arm] " + " | ".join(lines))
        info.update(projected=ik.projected, reachable=ik.reachable,
                    message=ik.message)
        if not ik.ok:
            self._record_target(ik, ik.message)
            self.get_logger().warning(ik.message)
            return False, info
        if duration is None or duration <= 0:
            duration = self.get_parameter("trajectory_duration").value
        plan = self.planner.plan(q_from, ik.q_goal, duration)
        if not plan.ok:
            msg = f"planning failed: {plan.message}"
            self._record_target(ik, msg)
            info["message"] = msg
            self.get_logger().warning(msg)
            return False, info
        with self._lock:
            if self._motion_state == "moving":
                info["message"] = "busy: a motion is already executing"
                return False, info
            self._set_trajectory(plan.trajectory, "moving")
        msg = f"{ik.message}; {plan.message}"
        self._record_target(ik, msg)
        info.update(accepted=True, q_goal=ik.q_goal.tolist(),
                    duration=plan.trajectory.duration, message=msg)
        # (rclpy forbids one call site logging at two severities)
        if ik.projected:
            self.get_logger().warning(msg)
        else:
            self.get_logger().info(msg)
        return True, info

    def _record_target(self, ik, message):
        with self._lock:
            self._requested = ik.requested
            self._reachable = ik.reachable
            self._projected = ik.projected
            self._message = message

    # ============================================================ services
    def _srv_move(self, req, resp):
        if self._busy_action:
            resp.accepted, resp.message = False, "busy: pick-and-place running"
            return resp
        try:
            ok, info = self._plan_and_start((req.target.x, req.target.y),
                                            req.duration)
        except Exception as exc:  # noqa: BLE001 - never let a request kill the node
            self.get_logger().error(f"move_to_target failed: {exc}")
            resp.accepted, resp.message = False, f"internal error: {exc}"
            return resp
        resp.accepted = ok
        resp.projected = bool(info["projected"])
        resp.reachable_target = _pt(info["reachable"])
        resp.joint_goal = [float(v) for v in info["q_goal"]]
        resp.duration = float(info["duration"])
        resp.message = info["message"]
        return resp

    def _srv_move_joints(self, req, resp):
        """Joint-space move (GUI sliders): plan straight to the given angles."""
        resp.accepted = False
        if self._busy_action:
            resp.message = "busy: pick-and-place running"
            return resp
        if not self._active or self._anchor_pending:
            resp.message = "controller is not active / no measurement yet"
            return resp
        if len(req.positions) != len(JOINT_NAMES):
            resp.message = (f"expected {len(JOINT_NAMES)} joint positions, "
                            f"got {len(req.positions)}")
            return resp
        q_goal = np.asarray(req.positions, dtype=float)
        if not np.all(np.isfinite(q_goal)):
            resp.message = "joint positions must be finite"
            return resp
        with self._lock:
            q_from = self._ref.position.copy()
        duration = req.duration
        if duration <= 0:
            duration = self.get_parameter("trajectory_duration").value
        plan = self.planner.plan(q_from, q_goal, duration)   # limits + ground
        if not plan.ok:
            resp.message = f"joint move rejected: {plan.message}"
            with self._lock:
                self._message = resp.message
            self.get_logger().warning(resp.message)
            return resp
        ee = np.array(self.arm.end_effector(q_goal))
        with self._lock:
            if self._motion_state == "moving":
                resp.message = "busy: a motion is already executing"
                return resp
            self._set_trajectory(plan.trajectory, "moving")
            self._requested, self._reachable = ee, ee.copy()
            self._projected = False
            self._message = (f"joint move to "
                             f"{np.round(np.degrees(q_goal), 1).tolist()} deg "
                             f"({plan.message})")
            resp.message = self._message
        resp.accepted = True
        resp.duration = float(plan.trajectory.duration)
        self.get_logger().info(resp.message)
        return resp

    def _on_target_pose(self, msg):
        if self._busy_action:
            self.get_logger().warning("target_pose ignored: pick-and-place running")
            return
        try:
            self._plan_and_start((msg.point.x, msg.point.y))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"target_pose failed: {exc}")

    def _srv_home(self, req, resp):
        if self._busy_action or not self._active:
            resp.success, resp.message = False, "busy or inactive"
            return resp
        resp.success, resp.message = self.start_home()
        return resp

    def _srv_stop(self, req, resp):
        self.stop()
        resp.success, resp.message = True, "holding current position"
        return resp

    def _srv_gripper(self, req, resp):
        if self._busy_action:       # the sequencer owns the gripper
            resp.success, resp.message = False, "busy: pick-and-place running"
            return resp
        attached = self.set_gripper(req.data)
        resp.success = True
        resp.message = self._message + (" (attached)" if attached else "")
        return resp

    # ============================================================== action
    def _on_goal(self, goal_request):
        if not self._active:
            return GoalResponse.REJECT
        with self._lock:
            if self._busy_action or self._motion_state == "moving":
                self.get_logger().warning("pick_place goal rejected: busy")
                return GoalResponse.REJECT
            self._busy_action = True
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        self.get_logger().info("pick_place cancel requested")
        return CancelResponse.ACCEPT

    def _execute_pick_place(self, goal_handle):
        g = goal_handle.request
        pick, place = (g.pick.x, g.pick.y), (g.place.x, g.place.y)
        self.get_logger().info(f"pick_place: pick={pick} place={place}")
        with self._lock:
            self.payload.reset(pick)
            self._gripper_closed = False

        def publish(fb):
            if goal_handle.is_cancel_requested or not goal_handle.is_active:
                return
            m = PickPlace.Feedback(
                phase=fb.phase, step=fb.step, total_steps=fb.total_steps,
                phase_progress=fb.phase_progress, total_progress=fb.total_progress,
                end_effector=_pt(fb.end_effector))
            goal_handle.publish_feedback(m)

        try:
            out = self.sequencer.run(
                pick, place, g.return_home,
                is_cancel_requested=lambda: goal_handle.is_cancel_requested,
                publish_feedback=publish)
        except Exception as exc:  # noqa: BLE001
            self.stop()
            out_msg = f"pick_place error: {exc}"
            self.get_logger().error(out_msg)
            goal_handle.abort()
            with self._lock:
                self._busy_action = False
            return PickPlace.Result(success=False, message=out_msg)

        with self._lock:
            self._busy_action = False
            self._message = out.message
        result = PickPlace.Result(success=out.success, message=out.message,
                                  elapsed=out.elapsed)
        if out.canceled:
            goal_handle.canceled()
        elif out.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(f"pick_place finished: {out.message}")
        return result


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None):
    # We own SIGINT/SIGTERM instead of rclpy: if rclpy's handler shuts the
    # context down first, the lifecycle node can no longer destroy its
    # entities and destroy_node() spins forever. Here we shut down in order:
    # lifecycle shutdown transition -> executor -> node -> context.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    node = ControllerNode()
    # The control loop shares the GIL with other Python threads; a shorter
    # switch interval bounds how long one can hold it (docs/PROFILING.md).
    sys.setswitchinterval(node.get_parameter("python_switch_interval_ms").value / 1e3)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    if node.get_parameter("autostart").value:
        if node.trigger_configure() == TransitionCallbackReturn.SUCCESS:
            node.trigger_activate()
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            node.trigger_shutdown()       # on_shutdown: stop loop, free backend/pool
        except Exception:  # noqa: BLE001 - already finalized
            pass
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
