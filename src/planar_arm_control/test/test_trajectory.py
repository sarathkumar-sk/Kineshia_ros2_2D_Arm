import numpy as np
import pytest

from planar_arm_control.trajectory import (
    JointTrajectory, QuinticSegment, quintic_min_duration)


def test_quintic_boundary_conditions():
    q0, q1 = np.array([0.1, -0.5, 1.0]), np.array([1.2, 0.7, -0.3])
    seg = QuinticSegment(q0, q1, 2.0)
    a, b = seg.sample(0.0), seg.sample(2.0)
    np.testing.assert_allclose(a.position, q0)
    np.testing.assert_allclose(b.position, q1)
    for p in (a, b):
        np.testing.assert_allclose(p.velocity, 0.0, atol=1e-12)
        np.testing.assert_allclose(p.acceleration, 0.0, atol=1e-12)
    # midpoint is exactly halfway and at peak velocity 15/8 * dq / T
    mid = seg.sample(1.0)
    np.testing.assert_allclose(mid.position, 0.5 * (q0 + q1))
    np.testing.assert_allclose(mid.velocity, 15 / 8 * (q1 - q0) / 2.0)


def test_derivatives_are_consistent():
    seg = QuinticSegment([0, 0, 0], [1, -2, 0.5], 1.5)
    h = 1e-6
    for t in np.linspace(0.05, 1.45, 15):
        num_v = (seg.sample(t + h).position - seg.sample(t - h).position) / (2 * h)
        num_a = (seg.sample(t + h).velocity - seg.sample(t - h).velocity) / (2 * h)
        np.testing.assert_allclose(seg.sample(t).velocity, num_v, atol=1e-5)
        np.testing.assert_allclose(seg.sample(t).acceleration, num_a, atol=1e-4)


@pytest.mark.parametrize("vmax,amax", [(1.0, 2.0), (0.5, 10.0), (3.0, 0.5)])
def test_min_duration_respects_limits(vmax, amax):
    dq = np.array([1.5, -3.0, 0.2])
    T = quintic_min_duration(dq, vmax, amax)
    seg = QuinticSegment(np.zeros(3), dq, T)
    ts = np.linspace(0, T, 2001)
    v = max(np.max(np.abs(seg.sample(t).velocity)) for t in ts)
    a = max(np.max(np.abs(seg.sample(t).acceleration)) for t in ts)
    assert v <= vmax * (1 + 1e-6)
    assert a <= amax * (1 + 1e-3)
    # and at least one limit is active (time-optimal for this profile)
    assert v > 0.99 * vmax or a > 0.99 * amax


def test_min_duration_floor():
    assert quintic_min_duration([0.0, 0.0, 0.0], 1.0, 1.0, 0.5) == 0.5


def test_multi_segment_trajectory():
    s1 = QuinticSegment([0, 0, 0], [1, 1, 1], 1.0)
    s2 = QuinticSegment([1, 1, 1], [2, 0, 1], 2.0)
    traj = JointTrajectory([s1, s2])
    assert traj.duration == pytest.approx(3.0)
    np.testing.assert_allclose(traj.sample(1.0).position, [1, 1, 1])
    np.testing.assert_allclose(traj.sample(10.0).position, [2, 0, 1])
    np.testing.assert_allclose(traj.sample(10.0).velocity, 0.0)
    assert traj.progress(1.5) == pytest.approx(0.5)


def test_hold():
    traj = JointTrajectory.hold([0.3, 0.2, 0.1])
    np.testing.assert_allclose(traj.sample(5.0).position, [0.3, 0.2, 0.1])
    assert traj.progress(0.0) == 1.0
