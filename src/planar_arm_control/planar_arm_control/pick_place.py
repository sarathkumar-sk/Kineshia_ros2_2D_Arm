"""
Pick-and-place orchestration, independent of ROS.

The sequence is  move -> pick (close gripper) -> move -> place (open gripper)
[-> return home]. The sequencer drives the controller through a small
``MotionInterface`` so it can be unit-tested with a fake, and it is executed
by the ROS 2 action server in controller_node.py, which supplies the cancel
check and the feedback publisher.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class MotionInterface(ABC):
    """What the sequencer needs from the controller."""

    @abstractmethod
    def check_target(self, xy, q_from):
        """Return (ok, q_goal, message) without moving."""

    @abstractmethod
    def start_move(self, xy):
        """Plan from the current state and start executing. -> (ok, msg)"""

    @abstractmethod
    def start_home(self):
        """-> (ok, msg)"""

    @abstractmethod
    def motion_done(self):
        """True once the current motion has finished and settled."""

    @abstractmethod
    def motion_progress(self):
        """Progress of the current motion in [0, 1]."""

    @abstractmethod
    def stop(self):
        """Abort the current motion and hold position."""

    @abstractmethod
    def set_gripper(self, closed):
        """Close/open the (virtual) gripper. -> object attached?"""

    @abstractmethod
    def current_q(self):
        """Current joint configuration."""

    @abstractmethod
    def end_effector(self):
        """Current end-effector (x, y)."""


@dataclass
class Feedback:
    phase: str
    step: int
    total_steps: int
    phase_progress: float
    total_progress: float
    end_effector: tuple


@dataclass
class Outcome:
    success: bool
    canceled: bool
    message: str
    elapsed: float


class PickPlaceSequencer:
    def __init__(self, motion: MotionInterface, grip_time=0.5,
                 poll_period=0.02, feedback_period=0.1, motion_timeout=30.0):
        self.motion = motion
        self.grip_time = grip_time
        self.poll_period = poll_period
        self.feedback_period = feedback_period
        self.motion_timeout = motion_timeout

    def run(self, pick, place, return_home=False,
            is_cancel_requested=lambda: False, publish_feedback=lambda fb: None):
        t0 = time.monotonic()
        pick = np.asarray(pick, float)[:2]
        place = np.asarray(place, float)[:2]

        # Validate both targets up front: fail before moving, not halfway.
        ok, q_pick, msg = self.motion.check_target(pick, self.motion.current_q())
        if not ok:
            return Outcome(False, False, f"pick target rejected: {msg}", 0.0)
        ok, _, msg = self.motion.check_target(place, q_pick)
        if not ok:
            return Outcome(False, False, f"place target rejected: {msg}", 0.0)

        steps = [("move_to_pick", "move", pick),
                 ("grasp", "grip", True),
                 ("move_to_place", "move", place),
                 ("release", "grip", False)]
        if return_home:
            steps.append(("return_home", "home", None))
        n = len(steps)

        for k, (phase, kind, arg) in enumerate(steps):
            if kind == "grip":
                ok, msg = self._grip(arg, phase, k, n, is_cancel_requested,
                                     publish_feedback)
            else:
                ok, msg = self._move(kind, arg, phase, k, n,
                                     is_cancel_requested, publish_feedback)
            if not ok:
                canceled = msg == "canceled"
                return Outcome(False, canceled,
                               f"{phase}: {msg}", time.monotonic() - t0)
        publish_feedback(self._fb("done", n, n, 1.0))
        return Outcome(True, False, "pick-and-place complete",
                       time.monotonic() - t0)

    # ------------------------------------------------------------------
    def _fb(self, phase, k, n, phase_progress):
        total = min(1.0, (k + float(np.clip(phase_progress, 0, 1))) / n)
        return Feedback(phase, min(k + 1, n), n, float(phase_progress), total,
                        tuple(self.motion.end_effector()))

    def _move(self, kind, xy, phase, k, n, is_cancel, publish):
        ok, msg = (self.motion.start_home() if kind == "home"
                   else self.motion.start_move(xy))
        if not ok:
            return False, msg
        t_start = time.monotonic()
        last_fb = 0.0
        while not self.motion.motion_done():
            if is_cancel():
                self.motion.stop()
                return False, "canceled"
            now = time.monotonic()
            if now - t_start > self.motion_timeout:
                self.motion.stop()
                return False, "motion timed out"
            if now - last_fb >= self.feedback_period:
                publish(self._fb(phase, k, n, self.motion.motion_progress()))
                last_fb = now
            time.sleep(self.poll_period)
        publish(self._fb(phase, k, n, 1.0))
        return True, "ok"

    def _grip(self, close, phase, k, n, is_cancel, publish):
        t_start = time.monotonic()
        while time.monotonic() - t_start < self.grip_time:
            if is_cancel():
                return False, "canceled"
            publish(self._fb(phase, k, n,
                             (time.monotonic() - t_start) / self.grip_time))
            time.sleep(self.feedback_period)
        attached = self.motion.set_gripper(close)
        if close and not attached:
            return False, "no object within reach of the gripper"
        publish(self._fb(phase, k, n, 1.0))
        return True, "ok"


class Payload:
    """A virtual block the gripper can pick up (for visualisation/logic)."""

    def __init__(self, position=(0.0, 0.0), grasp_tolerance=0.05):
        self.position = np.asarray(position, float)
        self.attached = False
        self.grasp_tolerance = grasp_tolerance

    def reset(self, position):
        self.position = np.asarray(position, float)[:2].copy()
        self.attached = False

    def try_attach(self, ee_xy):
        if np.linalg.norm(np.asarray(ee_xy) - self.position) <= self.grasp_tolerance:
            self.attached = True
        return self.attached

    def release(self, ee_xy):
        if self.attached:
            self.position = np.asarray(ee_xy, float).copy()
        self.attached = False

    def follow(self, ee_xy):
        if self.attached:
            self.position = np.asarray(ee_xy, float).copy()
