import numpy as np
import pytest

from planar_arm_control.backends import SimBackend, SimJointParams
from planar_arm_control.control_modes import JointPID, make_control_mode
from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.planning import HOME_Q, MotionPlanner

ARM = PlanarArm([3.0, 2.0, 1.5])


def run_scenario(mode, dt=0.005, targets=((4.0, 2.0), (-3.0, 3.0))):
    params = SimJointParams()
    backend = SimBackend(ARM, HOME_Q, params)
    ctl = make_control_mode(mode, params)
    planner = MotionPlanner(ARM)
    q = HOME_Q.copy()
    worst, finals = 0.0, []
    for tgt in targets:
        ik = planner.solve_ik(tgt, q)
        traj = planner.plan(q, ik.q_goal).trajectory
        for k in range(int((traj.duration + 1.0) / dt)):
            backend.update(dt)
            meas = backend.read()
            ref = traj.sample(k * dt)
            backend.write(mode, ctl.compute(ref, meas, dt))
            worst = max(worst, np.max(np.abs(ref.position - backend.q)))
            assert ARM.arm_above_base(backend.q, tolerance=1e-3)
        finals.append(np.linalg.norm(np.asarray(ARM.end_effector(backend.q)) - tgt))
        q = ik.q_goal
    return np.degrees(worst), finals


@pytest.mark.parametrize("mode,max_err_deg", [
    ("position", 4.0), ("velocity", 1.0), ("current", 1.5)])
def test_modes_track_scenario(mode, max_err_deg):
    worst, finals = run_scenario(mode)
    assert worst < max_err_deg
    assert max(finals) < 0.01        # within 1 cm of every target


def test_current_mode_integral_rejects_gravity():
    """Hold a horizontal arm (max gravity torque) - steady-state error -> 0."""
    params = SimJointParams()
    q_hold = np.radians([10.0, 0.0, 0.0])
    backend = SimBackend(ARM, q_hold, params)
    ctl = make_control_mode("current", params)
    from planar_arm_control.trajectory import JointTrajectory
    traj = JointTrajectory.hold(q_hold)
    dt = 0.005
    for k in range(int(4.0 / dt)):
        backend.update(dt)
        backend.write("current", ctl.compute(traj.sample(k * dt), backend.read(), dt))
    assert np.max(np.abs(backend.q - q_hold)) < np.radians(0.2)


def test_pid_integral_clamp():
    pid = JointPID(0.0, 1.0, 0.0, integral_limit=0.5)
    for _ in range(1000):
        out = pid.update(np.ones(3), 0.0, 0.01)
    np.testing.assert_allclose(out, 0.5)
    pid.reset()
    np.testing.assert_allclose(pid.integral, 0.0)


def test_sim_hard_stops():
    backend = SimBackend(ARM, HOME_Q, SimJointParams(gravity=False))
    backend.write("velocity", [-3.0, 0.0, 0.0])
    backend.update(3.0)
    assert backend.q[0] >= ARM.joint_limits[0, 0] - 1e-12


def test_unknown_mode():
    with pytest.raises(ValueError):
        make_control_mode("torque_magic")
