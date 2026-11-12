"""
Motion planning on top of the provided (unmodified) PlanarArm library.

Responsibilities:
  1. Cartesian target -> joint goal. By default with ``corrected_ik`` (below):
     the provided ``PlanarArm.inverse_kinematics`` drops the beta offset of
     the link-2+3 virtual link, so it only ever finds q3 = 0 (see
     docs/DESIGN_NOTE.md). The library is still used for the workspace
     projection and as the fallback, and ``ik_solver="library"`` restores its
     original behaviour.
  2. Make the IK's silent behaviour explicit: detect when the target was
     projected onto the workspace boundary, and when the damped-Jacobian
     fallback produced a solution that violates the joint limits or the ground
     constraint (it does not check them -- see docs/DESIGN_NOTE.md).
  3. Build a quintic joint-space trajectory and verify that *every* sampled
     configuration along it respects the joint limits and the ground (y >= 0).
     If the straight joint-space path dips below the ground, retry through a
     small set of safe via-points.

No ROS imports here, so this is unit-testable with plain pytest.
"""

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np

from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.trajectory import (
    JointTrajectory, QuinticSegment, quintic_min_duration)

HOME_Q = np.radians([90.0, 0.0, 0.0])   # arm pointing straight up

IK_SOLVERS = ("corrected", "library")

_WORKER_ARM = {}


def _wrap(a):
    """Wrap to (-pi, pi]. Unlike ``PlanarArm.wrap_angle``, +pi stays +pi, so a
    joint sitting exactly on a 180 deg limit is not thrown away."""
    return np.arctan2(np.sin(a), np.cos(a))


def corrected_ik(arm, target_xy, q_current, q3_samples=361, tolerance=1e-5):
    """
    The provided multiple-solution IK with its elbow formula fixed.

    Same method as ``PlanarArm.inverse_kinematics``: sample q3, treat links
    2+3 as one virtual link of length R at angle beta from link 2, solve the
    resulting 2-link problem for both elbow signs, keep candidates inside the
    joint limits, above the ground and on the target, and return the one
    with the least joint movement from ``q_current``.

    The fix: the virtual link is at q1 + q2 + beta, so q2 = delta - beta. The
    library uses q2 = delta, so every candidate with q3 != 0 misses the target
    and only q3 = 0 survives. Vectorised over the q3 samples (well under 1 ms
    versus ~15 ms for the library's Python loop).

    ``target_xy`` must already be reachable (use ``arm.reachable_target``).
    Returns the joint goal (rad) or None if no candidate passes, in which
    case the caller falls back to the library (and its Jacobian IK).
    """
    L1, L2, L3 = arm.link_lengths
    target = np.asarray(target_xy, dtype=float)
    q_current = np.asarray(q_current, dtype=float)
    dist = float(np.linalg.norm(target))
    if dist < tolerance:
        return None

    q3 = np.linspace(-np.pi, np.pi, q3_samples)
    A = L2 + L3 * np.cos(q3)
    B = L3 * np.sin(q3)
    R = np.hypot(A, B)
    beta = np.arctan2(B, A)
    cos_delta = (dist ** 2 - L1 ** 2 - R ** 2) / (2 * L1 * R)
    keep = np.abs(cos_delta) <= 1.000001        # same slack as the library
    if not np.any(keep):
        return None
    q3, R, beta = q3[keep], R[keep], beta[keep]
    delta = np.arccos(np.clip(cos_delta[keep], -1.0, 1.0))
    phi = np.arctan2(target[1], target[0])

    cands = []
    for sign in (1.0, -1.0):
        d = sign * delta
        q1 = phi - np.arctan2(R * np.sin(d), L1 + R * np.cos(d))
        q2 = d - beta                           # library: q2 = d  (the bug)
        cands.append(np.column_stack([q1, q2, q3]))
    q = _wrap(np.vstack(cands))

    lo, hi = arm.joint_limits[:, 0], arm.joint_limits[:, 1]
    ok = np.all((q >= lo) & (q <= hi), axis=1)
    # forward kinematics of every candidate (absolute link angles)
    a1 = q[:, 0]
    a2 = a1 + q[:, 1]
    a3 = a2 + q[:, 2]
    y1 = L1 * np.sin(a1)
    y2 = y1 + L2 * np.sin(a2)
    y3 = y2 + L3 * np.sin(a3)
    x3 = L1 * np.cos(a1) + L2 * np.cos(a2) + L3 * np.cos(a3)
    ok &= np.minimum(np.minimum(y1, y2), y3) >= arm.min_y - 1e-6
    ok &= np.hypot(x3 - target[0], y3 - target[1]) <= tolerance
    if not np.any(ok):
        return None
    q = q[ok]
    movement = np.sum(np.abs(_wrap(q - q_current)), axis=1)
    return q[int(np.argmin(movement))]


def ik_worker(link_lengths, target_xy, q_current, solver="corrected"):
    """
    Run the IK -- module-level so it can be executed in a separate
    *process*: the library's ~15 ms pure-Python loop in a thread holds the
    GIL and delays the control thread (see docs/PROFILING.md).
    Returns (reachable_target, q_goal, captured stdout).
    """
    key = tuple(link_lengths)
    arm = _WORKER_ARM.get(key)
    if arm is None:
        arm = _WORKER_ARM[key] = PlanarArm(list(link_lengths))
    q_current = np.asarray(q_current, float)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        reachable = np.asarray(arm.reachable_target(target_xy), float)
        q_goal = None
        if solver == "corrected":
            q_goal = corrected_ik(arm, reachable, q_current)
        if q_goal is None:      # library IK (and its Jacobian fallback)
            q_goal = arm.inverse_kinematics(target_xy, initial_guess=q_current)
    return reachable, np.asarray(q_goal, float), log.getvalue()


@dataclass
class IKResult:
    ok: bool
    requested: np.ndarray
    reachable: np.ndarray
    projected: bool = False
    q_goal: np.ndarray = None
    used_fallback: bool = False
    position_error: float = 0.0
    message: str = ""
    library_log: str = ""


@dataclass
class PlanResult:
    ok: bool
    trajectory: JointTrajectory = None
    via_points: list = field(default_factory=list)
    message: str = ""


class MotionPlanner:
    """IK + validation + safe trajectory generation."""

    def __init__(self, arm: PlanarArm, max_velocity=1.0, max_acceleration=2.0,
                 min_duration=0.5, check_samples=200, ground_margin=1e-4,
                 ik_tolerance=1e-3, ik_executor=None, ik_solver="corrected"):
        if ik_solver not in IK_SOLVERS:
            raise ValueError(f"ik_solver must be one of {IK_SOLVERS}")
        self.ik_solver = ik_solver
        self.arm = arm
        self.max_velocity = np.broadcast_to(
            np.asarray(max_velocity, dtype=float), (arm.n_joints,)).copy()
        self.max_acceleration = np.broadcast_to(
            np.asarray(max_acceleration, dtype=float), (arm.n_joints,)).copy()
        self.min_duration = float(min_duration)
        self.check_samples = int(check_samples)
        self.ground_margin = float(ground_margin)
        self.ik_tolerance = float(ik_tolerance)
        # Optional concurrent.futures executor (e.g. a ProcessPoolExecutor)
        self.ik_executor = ik_executor

    # ------------------------------------------------------------------ checks
    @property
    def max_reach(self):
        return float(sum(self.arm.link_lengths))

    def min_link_height(self, q):
        return min(y for _, y in self.arm.forward_kinematics(q))

    def is_valid(self, q, ground_tol=None):
        tol = self.ground_margin if ground_tol is None else ground_tol
        q = np.asarray(q, dtype=float)
        lo, hi = self.arm.joint_limits[:, 0], self.arm.joint_limits[:, 1]
        if np.any(q < lo - 1e-9) or np.any(q > hi + 1e-9):
            return False
        return self.min_link_height(q) >= -tol

    def violation(self, q):
        """Human-readable reason why q is invalid ('' if valid)."""
        q = np.asarray(q, dtype=float)
        lo, hi = self.arm.joint_limits[:, 0], self.arm.joint_limits[:, 1]
        for i in range(len(q)):
            if q[i] < lo[i] - 1e-9 or q[i] > hi[i] + 1e-9:
                return (f"joint {i + 1} = {np.degrees(q[i]):.1f} deg outside "
                        f"[{np.degrees(lo[i]):.0f}, {np.degrees(hi[i]):.0f}]")
        h = self.min_link_height(q)
        if h < -self.ground_margin:
            return f"arm goes below the ground (min y = {h:.3f})"
        return ""

    # ---------------------------------------------------------------------- IK
    def solve_ik(self, target_xy, q_current):
        requested = np.asarray(target_xy, dtype=float)[:2]
        if requested[1] < 0.0:
            return IKResult(False, requested, requested,
                            message=(f"target ({requested[0]:.2f}, "
                                     f"{requested[1]:.2f}) is below the ground "
                                     f"(y < 0) - rejected"))

        # The library reports projection / fallback only via print();
        # capture it so the node can log it through ROS and surface it.
        args = (tuple(self.arm.link_lengths), requested,
                np.asarray(q_current, float), self.ik_solver)
        if self.ik_executor is not None:
            reachable, q_goal, text = self.ik_executor.submit(
                ik_worker, *args).result()
        else:
            reachable, q_goal, text = ik_worker(*args)
        projected = bool(np.linalg.norm(requested) > self.max_reach + 1e-12)
        used_fallback = "Jacobian" in text
        ee = np.asarray(self.arm.end_effector(q_goal), dtype=float)
        err = float(np.linalg.norm(ee - reachable))

        res = IKResult(True, requested, reachable, projected, q_goal,
                       used_fallback, err, "", text.strip())
        reason = self.violation(q_goal)
        if reason:
            res.ok = False
            res.message = (f"IK solution rejected: {reason}"
                           + (" (Jacobian fallback)" if used_fallback else ""))
        elif err > self.ik_tolerance:
            res.ok = False
            res.message = (f"IK did not converge: end-effector error "
                           f"{err:.3f} m")
        elif projected:
            res.message = (f"target ({requested[0]:.2f}, {requested[1]:.2f}) "
                           f"is outside the workspace (reach "
                           f"{self.max_reach:.1f} m); projected to the "
                           f"boundary at ({reachable[0]:.3f}, "
                           f"{reachable[1]:.3f})")
        else:
            res.message = (f"IK ok: q = "
                           f"{np.round(np.degrees(q_goal), 1).tolist()} deg")
        return res

    # --------------------------------------------------------------- planning
    def _segment(self, q_a, q_b, duration=None):
        if duration is None or duration <= 0:
            duration = quintic_min_duration(
                np.asarray(q_b) - np.asarray(q_a), self.max_velocity,
                self.max_acceleration, self.min_duration)
        return QuinticSegment(q_a, q_b, duration)

    def segment_is_safe(self, seg):
        # Skip s=0: the measured start state may sit a hair outside the
        # constraint (sensor noise, quantisation) and we still want to leave it.
        for q in seg.path_samples(self.check_samples)[1:]:
            if not self.is_valid(q):
                return False
        return True

    def plan(self, q_start, q_goal, duration=None):
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)
        reason = self.violation(q_goal)
        if reason:
            return PlanResult(False, message=f"goal invalid: {reason}")

        direct = self._segment(q_start, q_goal, duration)
        if self.segment_is_safe(direct):
            return PlanResult(True, JointTrajectory([direct]),
                              message=f"direct path, {direct.duration:.2f} s")

        # Straight joint-space path dips below the ground: go through a
        # retracted / upright configuration instead.
        candidates = [
            np.array([q_start[0], 0.0, 0.0]),
            np.array([q_goal[0], 0.0, 0.0]),
            HOME_Q.copy(),
        ]
        for via in candidates:
            if not self.is_valid(via):
                continue
            s1, s2 = self._segment(q_start, via), self._segment(via, q_goal)
            if self.segment_is_safe(s1) and self.segment_is_safe(s2):
                traj = JointTrajectory([s1, s2])
                return PlanResult(
                    True, traj, [via],
                    message=(f"direct path hits the ground; via "
                             f"{np.round(np.degrees(via), 1).tolist()} deg, "
                             f"{traj.duration:.2f} s"))
        return PlanResult(False, message="no collision-free path found")
