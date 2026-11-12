import hashlib
import os

import numpy as np
import pytest

from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.planning import HOME_Q, MotionPlanner, corrected_ik
from planar_arm_control.trajectory import QuinticSegment

ARM = PlanarArm([3.0, 2.0, 1.5])


@pytest.fixture
def planner():
    return MotionPlanner(ARM, max_velocity=1.0, max_acceleration=2.0)


def test_provided_library_is_unmodified():
    path = os.path.join(os.path.dirname(__file__), "..", "planar_arm_control",
                        "planar_arm.py")
    with open(path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    assert digest == "4aa74cb74144bc9d4b3e2dfd32327f3d"


@pytest.mark.parametrize("target", [(4.0, 2.0), (-3.0, 3.0)])
def test_scenario_targets_are_reached(planner, target):
    ik = planner.solve_ik(target, HOME_Q)
    assert ik.ok and not ik.projected, ik.message
    np.testing.assert_allclose(ARM.end_effector(ik.q_goal), target, atol=1e-4)
    assert planner.is_valid(ik.q_goal)


def test_unreachable_target_is_projected_to_boundary(planner):
    ik = planner.solve_ik((7.0, 3.0), HOME_Q)
    assert ik.ok and ik.projected
    assert np.linalg.norm(ik.reachable) == pytest.approx(6.5)
    # projection is along the ray from the base
    np.testing.assert_allclose(ik.reachable / np.linalg.norm(ik.reachable),
                               np.array([7.0, 3.0]) / np.hypot(7, 3))
    np.testing.assert_allclose(ARM.end_effector(ik.q_goal), ik.reachable, atol=1e-3)
    assert "outside the workspace" in ik.message


def test_target_below_ground_rejected(planner):
    ik = planner.solve_ik((2.0, -1.0), HOME_Q)
    assert not ik.ok and "below the ground" in ik.message


@pytest.fixture
def library_planner():
    return MotionPlanner(ARM, max_velocity=1.0, max_acceleration=2.0,
                         ik_solver="library")


@pytest.mark.parametrize("target", [(-6.5, 0.0), (0.5, 0.5)])
def test_invalid_jacobian_fallback_is_rejected(library_planner, target):
    """The library's fallback IK ignores limits/ground; we must catch it."""
    ik = library_planner.solve_ik(target, HOME_Q)
    assert ik.used_fallback
    assert not ik.ok


def test_unreachable_within_limits_still_falls_back_and_is_rejected(planner):
    # (0.5, 0.5) needs joint 2 beyond -120 deg: no valid pose exists, the
    # corrected IK finds nothing, the library fallback result is rejected.
    ik = planner.solve_ik((0.5, 0.5), HOME_Q)
    assert ik.used_fallback and not ik.ok


def test_library_ik_never_bends_joint_3(library_planner):
    """Documents the bug in the provided IK (docs/DESIGN_NOTE.md)."""
    start = np.radians([45.0, -60.0, -60.0])
    ik = library_planner.solve_ik((4.0, 2.0), start)
    assert ik.ok and abs(ik.q_goal[2]) < 1e-9


def test_corrected_ik_uses_joint_3():
    start = np.radians([45.0, -60.0, -60.0])
    q = corrected_ik(ARM, (4.0, 2.0), start)
    np.testing.assert_allclose(ARM.end_effector(q), (4.0, 2.0), atol=1e-5)
    assert abs(np.degrees(q[2])) > 30.0
    # least movement: much closer to the start than the library's q3 = 0 pose
    lib = np.asarray(ARM.inverse_kinematics((4.0, 2.0), initial_guess=start))
    assert np.abs(q - start).sum() < np.abs(lib - start).sum()


@pytest.mark.parametrize("target", [(1.0, 1.0), (-6.5, 0.0)])
def test_corrected_ik_solves_what_the_library_could_not(planner, target):
    # (1, 1): needs a bent wrist. (-6.5, 0): q1 = 180 deg exactly on its limit.
    ik = planner.solve_ik(target, HOME_Q)
    assert ik.ok and not ik.used_fallback, ik.message
    np.testing.assert_allclose(ARM.end_effector(ik.q_goal), target, atol=1e-4)
    assert planner.is_valid(ik.q_goal)


def test_corrected_ik_solutions_are_always_valid():
    rng = np.random.default_rng(0)
    solved = 0
    for _ in range(500):
        r, a = rng.uniform(0.3, 6.5), rng.uniform(0.0, np.pi)
        target = np.array([r * np.cos(a), r * np.sin(a)])
        start = rng.uniform(ARM.joint_limits[:, 0], ARM.joint_limits[:, 1])
        q = corrected_ik(ARM, target, start)
        if q is None:
            continue
        solved += 1
        np.testing.assert_allclose(ARM.end_effector(q), target, atol=1e-5)
        assert ARM.within_joint_limits(q) and ARM.arm_above_base(q)
    assert solved > 400


def test_scenario_plan_is_safe(planner):
    q = HOME_Q
    for target in [(4.0, 2.0), (-3.0, 3.0), (7.0, 3.0)]:
        ik = planner.solve_ik(target, q)
        plan = planner.plan(q, ik.q_goal)
        assert plan.ok, plan.message
        traj = plan.trajectory
        for t in np.linspace(0, traj.duration, 400):
            assert planner.is_valid(traj.sample(t).position)
        q = ik.q_goal


def test_via_point_when_direct_path_hits_ground(planner):
    a = np.radians([46.1, -102.6, -115.7])
    b = np.radians([104.4, -74.1, 114.1])
    assert planner.is_valid(a) and planner.is_valid(b)
    assert not planner.segment_is_safe(QuinticSegment(a, b, 1.0))
    plan = planner.plan(a, b)
    assert plan.ok and len(plan.via_points) == 1
    for t in np.linspace(0, plan.trajectory.duration, 600):
        assert planner.is_valid(plan.trajectory.sample(t).position)


def test_random_plans_never_violate_constraints(planner):
    rng = np.random.default_rng(1)
    lo, hi = ARM.joint_limits[:, 0], ARM.joint_limits[:, 1]
    qs = [q for q in rng.uniform(lo, hi, (600, 3)) if planner.is_valid(q)]
    for a, b in zip(qs[::2], qs[1::2]):
        plan = planner.plan(a, b)
        if not plan.ok:
            continue          # rejecting is allowed, violating is not
        for t in np.linspace(0, plan.trajectory.duration, 60):
            assert planner.is_valid(plan.trajectory.sample(t).position)
