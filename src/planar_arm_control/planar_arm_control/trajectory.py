"""
Joint-space trajectory generation.

We use a *quintic* (5th-order, minimum-jerk) time scaling for every joint:

    q(t) = q0 + (q1 - q0) * s(tau),   tau = t / T,
    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5

Why quintic rather than trapezoidal:
  * position, velocity AND acceleration are continuous, with zero velocity and
    zero acceleration at both ends -> no torque steps at start/stop, which is
    kinder to real servos (Dynamixel) and to a PID tracking loop;
  * it is a single closed-form expression, trivially sampled at any rate, so
    the planner is completely decoupled from the control/publish rate;
  * all joints share the same s(tau), so the motion is synchronised (every joint
    starts and finishes together) and the joint-space path is a straight line.
    Because the path is a straight line between two in-limit configurations,
    it can never leave the (box-shaped) joint limits.

The price is a ~1.88x peak velocity versus the average velocity (trapezoidal
is ~1.0-1.5x), i.e. slightly slower motions for the same velocity limit.
"""

from dataclasses import dataclass

import numpy as np

# Peak |s'(tau)| and |s''(tau)| of the quintic time scaling.
QUINTIC_PEAK_VEL = 15.0 / 8.0               # at tau = 0.5
QUINTIC_PEAK_ACC = 10.0 / np.sqrt(3.0)      # at tau = 0.5 -/+ sqrt(3)/6


@dataclass
class TrajectoryPoint:
    """Reference state at one instant."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class QuinticSegment:
    """Rest-to-rest quintic segment from ``q_start`` to ``q_goal``."""

    def __init__(self, q_start, q_goal, duration):
        self.q_start = np.asarray(q_start, dtype=float).copy()
        self.q_goal = np.asarray(q_goal, dtype=float).copy()
        self.delta = self.q_goal - self.q_start
        self.duration = max(float(duration), 1e-6)

    def sample(self, t):
        T = self.duration
        tau = float(np.clip(t / T, 0.0, 1.0))
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
        dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
        if t >= T:
            ds = dds = 0.0
        return TrajectoryPoint(
            position=self.q_start + self.delta * s,
            velocity=self.delta * ds,
            acceleration=self.delta * dds,
        )

    def path_samples(self, n=100):
        """Configurations along the (straight, joint-space) path."""
        s = np.linspace(0.0, 1.0, n)
        return self.q_start[None, :] + s[:, None] * self.delta[None, :]


def quintic_min_duration(delta_q, max_velocity, max_acceleration,
                         min_duration=0.0):
    """Shortest duration that respects per-joint |qd| and |qdd| limits."""
    dq = np.abs(np.asarray(delta_q, dtype=float))
    v_max = np.broadcast_to(np.asarray(max_velocity, dtype=float), dq.shape)
    a_max = np.broadcast_to(np.asarray(max_acceleration, dtype=float), dq.shape)
    t_vel = np.max(QUINTIC_PEAK_VEL * dq / v_max) if dq.size else 0.0
    t_acc = np.max(np.sqrt(QUINTIC_PEAK_ACC * dq / a_max)) if dq.size else 0.0
    return float(max(t_vel, t_acc, min_duration))


class JointTrajectory:
    """A chain of rest-to-rest quintic segments (via-points stop briefly)."""

    def __init__(self, segments):
        if not segments:
            raise ValueError("JointTrajectory needs at least one segment")
        self.segments = list(segments)
        self._starts = np.cumsum([0.0] + [s.duration for s in self.segments])
        self.duration = float(self._starts[-1])

    @classmethod
    def hold(cls, q):
        """Zero-length trajectory that just holds ``q``."""
        return cls([QuinticSegment(q, q, 1e-6)])

    @property
    def q_start(self):
        return self.segments[0].q_start

    @property
    def q_goal(self):
        return self.segments[-1].q_goal

    @property
    def waypoints(self):
        return [self.q_start] + [s.q_goal for s in self.segments]

    def sample(self, t):
        if t >= self.duration:
            return self.segments[-1].sample(self.segments[-1].duration + 1.0)
        idx = int(np.searchsorted(self._starts, t, side="right") - 1)
        idx = min(max(idx, 0), len(self.segments) - 1)
        return self.segments[idx].sample(t - self._starts[idx])

    def progress(self, t):
        if self.duration <= 1e-6:
            return 1.0
        return float(np.clip(t / self.duration, 0.0, 1.0))
