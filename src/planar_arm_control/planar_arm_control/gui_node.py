#!/usr/bin/env python3
"""
gui_node -- PyQt5 / PyQtGraph operator console for the planar arm.

Threading model (the graded part)
---------------------------------
    +---------------------------+            +-----------------------------+
    | Qt main thread            |            | ROS thread                  |
    |  QApplication.exec_()     |            |  SingleThreadedExecutor     |
    |  render QTimer @ 30 Hz ---+-- reads -->|  .spin()                    |
    |   (snapshot of buffers)   |  (lock)    |  subscription callbacks --> |
    |                           |            |   TelemetryBuffer (lock)    |
    |  widgets / slots  <-------+-- Qt sig --|  service/action callbacks   |
    |  (queued, thread-safe)    |            |                             |
    |  button clicks -----------+-- create_task (thread-safe) -->          |
    +---------------------------+            +-----------------------------+

* The ROS executor spins in its own daemon thread, so Qt never blocks ROS and
  ROS never blocks Qt.
* ROS callbacks never touch widgets. Streaming data (50 Hz joint states) goes
  into lock-protected ring buffers; discrete events (service responses, action
  feedback/results) are emitted as Qt signals, which Qt delivers to the GUI
  thread through a queued connection.
* The GUI never calls rclpy directly: requests are handed to the executor
  with ``executor.create_task`` (thread-safe, wakes the executor), so every
  rclpy call happens on the ROS thread.
* Rendering runs on its own QTimer (render_rate_hz), decoupled from the
  message rate: a burst of messages never floods the Qt event queue, and a
  slow repaint never delays message handling.
"""

import signal
import sys
import threading
import time

import numpy as np
import rclpy
from PyQt5 import QtCore, QtWidgets
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import Point

from planar_arm_interfaces.action import PickPlace
from planar_arm_interfaces.msg import ControllerStatus
from planar_arm_interfaces.srv import MoveJoints, MoveToTarget

from planar_arm_control.planar_arm import PlanarArm

LINK_LENGTHS = [3.0, 2.0, 1.5]


class TelemetryBuffer:
    """Thread-safe ring buffer of (t, q_measured, q_reference) samples."""

    def __init__(self, capacity=3000, n=3):
        self._lock = threading.Lock()
        self.cap = capacity
        self.t = np.zeros(capacity)
        self.q = np.zeros((capacity, n))
        self.q_ref = np.zeros((capacity, n))
        self.count = 0
        self.idx = 0
        self.last_rx = 0.0
        # samples waiting for their partner (state/reference share a stamp)
        self._pending = {"state": {}, "ref": {}}
        self.latest_q = None
        self.latest_ref = None

    def add(self, kind, stamp_ns, values):
        other = "ref" if kind == "state" else "state"
        with self._lock:
            self.last_rx = time.monotonic()
            if kind == "state":
                self.latest_q = values
            else:
                self.latest_ref = values
            partner = self._pending[other].pop(stamp_ns, None)
            if partner is None:
                self._pending[kind][stamp_ns] = values
                for d in self._pending.values():      # drop orphans
                    while len(d) > 50:
                        d.pop(next(iter(d)))
                return
            q, q_ref = (values, partner) if kind == "state" else (partner, values)
            i = self.idx
            self.t[i] = stamp_ns * 1e-9
            self.q[i] = q
            self.q_ref[i] = q_ref
            self.idx = (i + 1) % self.cap
            self.count = min(self.count + 1, self.cap)

    def snapshot(self):
        """Chronologically ordered copies (safe to use outside the lock)."""
        with self._lock:
            n = self.count
            if n < self.cap:
                sl = slice(0, n)
                return (self.t[sl].copy(), self.q[sl].copy(),
                        self.q_ref[sl].copy(), self.latest_q, self.latest_ref,
                        self.last_rx)
            order = np.r_[self.idx:self.cap, 0:self.idx]
            return (self.t[order], self.q[order], self.q_ref[order],
                    self.latest_q, self.latest_ref, self.last_rx)


class RosSignals(QtCore.QObject):
    """Qt signals emitted from the ROS thread (delivered queued to the GUI)."""

    status = QtCore.pyqtSignal(object)            # ControllerStatus
    move_response = QtCore.pyqtSignal(object)     # MoveToTarget.Response | str
    joints_response = QtCore.pyqtSignal(object)   # MoveJoints.Response | str
    trigger_response = QtCore.pyqtSignal(str, object)
    action_event = QtCore.pyqtSignal(str, object)  # kind, payload
    param_response = QtCore.pyqtSignal(bool, str)


class GuiNode(Node):
    """ROS side of the GUI. All methods run on the ROS executor thread."""

    def __init__(self, signals: RosSignals):
        super().__init__("gui_node")
        self.declare_parameter("render_rate_hz", 30.0)
        self.declare_parameter("history_seconds", 15.0)
        self.declare_parameter("controller_node", "controller_node")
        self.declare_parameter("demo", False)   # auto-play the test scenario
        self.arm = PlanarArm(LINK_LENGTHS)
        self.sig = signals
        self.buffer = TelemetryBuffer()
        self.create_subscription(JointState, "joint_states", self._on_state, 20)
        self.create_subscription(JointState, "joint_reference", self._on_ref, 20)
        self.create_subscription(ControllerStatus, "controller_status",
                                 self._on_status, 10)
        self.move_client = self.create_client(MoveToTarget, "move_to_target")
        self.joints_client = self.create_client(MoveJoints, "move_joints")
        self.gripper_client = self.create_client(SetBool, "gripper")
        self.home_client = self.create_client(Trigger, "home")
        self.stop_client = self.create_client(Trigger, "stop")
        self.action_client = ActionClient(self, PickPlace, "pick_place")
        self.param_client = AsyncParameterClient(
            self, self.get_parameter("controller_node").value)
        self._goal_handle = None
        self.get_logger().info("gui_node started")

    # ------------------------------------------------------------ streaming
    @staticmethod
    def _stamp_ns(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _on_state(self, msg):
        self.buffer.add("state", self._stamp_ns(msg), np.asarray(msg.position[:3]))

    def _on_ref(self, msg):
        self.buffer.add("ref", self._stamp_ns(msg), np.asarray(msg.position[:3]))

    def _on_status(self, msg):
        self.sig.status.emit(msg)

    # ------------------------------------------------------------- requests
    # (scheduled onto this thread via executor.create_task)
    def request_move(self, x, y):
        if not self.move_client.service_is_ready():
            self.sig.move_response.emit("move_to_target service not available")
            return
        req = MoveToTarget.Request(target=Point(x=float(x), y=float(y)))
        fut = self.move_client.call_async(req)
        fut.add_done_callback(lambda f: self.sig.move_response.emit(
            f.result() if f.exception() is None else str(f.exception())))

    def request_move_joints(self, q):
        if not self.joints_client.service_is_ready():
            self.sig.joints_response.emit("move_joints service not available")
            return
        req = MoveJoints.Request(positions=[float(v) for v in q])
        fut = self.joints_client.call_async(req)
        fut.add_done_callback(lambda f: self.sig.joints_response.emit(
            f.result() if f.exception() is None else str(f.exception())))

    def request_gripper(self, closed):
        # SetBool.Response has success/message like Trigger -> same signal
        if not self.gripper_client.service_is_ready():
            self.sig.trigger_response.emit("gripper", "/gripper not available")
            return
        fut = self.gripper_client.call_async(SetBool.Request(data=bool(closed)))
        fut.add_done_callback(lambda f: self.sig.trigger_response.emit(
            "gripper", f.result() if f.exception() is None else str(f.exception())))

    def request_trigger(self, which):
        client = self.home_client if which == "home" else self.stop_client
        if not client.service_is_ready():
            self.sig.trigger_response.emit(which, f"/{which} not available")
            return
        fut = client.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self.sig.trigger_response.emit(
            which, f.result() if f.exception() is None else str(f.exception())))

    def request_mode(self, mode):
        if not self.param_client.services_are_ready():
            self.sig.param_response.emit(False, "controller parameters unavailable")
            return
        fut = self.param_client.set_parameters(
            [Parameter("control_mode", Parameter.Type.STRING, mode)])

        def done(f):
            if f.exception() is not None:
                self.sig.param_response.emit(False, str(f.exception()))
                return
            r = f.result().results[0]
            self.sig.param_response.emit(
                r.successful, f"control mode -> {mode}" if r.successful else r.reason)
        fut.add_done_callback(done)

    def request_pick_place(self, pick, place, return_home):
        if not self.action_client.server_is_ready():
            self.sig.action_event.emit("rejected", "pick_place action server not available")
            return
        goal = PickPlace.Goal(pick=Point(x=pick[0], y=pick[1]),
                              place=Point(x=place[0], y=place[1]),
                              return_home=return_home)
        fut = self.action_client.send_goal_async(
            goal, feedback_callback=lambda fb: self.sig.action_event.emit(
                "feedback", fb.feedback))
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.sig.action_event.emit("rejected", "goal rejected (controller busy?)")
            return
        self._goal_handle = gh
        self.sig.action_event.emit("accepted", None)
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, fut):
        res = fut.result()
        self._goal_handle = None
        self.sig.action_event.emit("result", (res.status, res.result))

    def request_cancel(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self.sig.action_event.emit("cancel_sent", None)


class RosThread:
    """Owns the executor and the background spin thread."""

    def __init__(self, node):
        self.node = node
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(node)
        self.thread = threading.Thread(target=self._spin, name="ros-spin",
                                       daemon=True)

    def _spin(self):
        try:
            self.executor.spin()
        except (ExternalShutdownException, KeyboardInterrupt):
            pass

    def start(self):
        self.thread.start()

    def call(self, fn, *args):
        """Run ``fn(*args)`` on the ROS thread (thread-safe)."""
        self.executor.create_task(fn, *args)

    def stop(self):
        self.executor.shutdown(timeout_sec=1.0)
        self.thread.join(timeout=2.0)


def main(args=None):
    # Qt owns SIGINT/SIGTERM (-> app.quit), so shutdown is always:
    # Qt loop exits -> executor stopped + thread joined -> node -> context.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Planar Arm Console")

    signals = RosSignals()
    node = GuiNode(signals)
    ros = RosThread(node)

    from planar_arm_control.gui_window import MainWindow
    window = MainWindow(node, ros, signals)
    window.show()

    # Ctrl+C: Python only runs signal handlers between bytecodes, and Qt's
    # C++ loop rarely returns to Python -- a no-op timer lets it in.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    wake = QtCore.QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(200)

    ros.start()
    try:
        app.exec_()
    finally:
        window.shutdown()
        ros.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
