"""
Hardware abstraction layer.

The controller node only ever talks to an ``ArmBackend``:

    backend.write(mode, command)   # rad | rad/s | A, one value per joint
    backend.update(dt)             # advance physics (sim) / no-op (hardware)
    sample = backend.read()        # measured position / velocity / effort

Swapping the simulated arm for real hardware (e.g. a Dynamixel chain) means
writing one more subclass -- ``DynamixelBackend.write`` would map the three
modes onto Goal Position / Goal Velocity / Goal Current using a GroupSyncWrite,
and ``read`` onto a GroupSyncRead of Present Position / Velocity / Current.
The planner, the control laws, the ROS interfaces and the GUI do not change.
The Gazebo backend (gazebo_backend.py) is a second, working example of this.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

CONTROL_MODES = ("position", "velocity", "current")
GRAVITY = 9.81


@dataclass
class JointSample:
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray


class ArmBackend(ABC):
    """Interface every arm driver (sim, Gazebo, Dynamixel, ...) implements."""

    name = "abstract"
    supported_modes = CONTROL_MODES

    def supports(self, mode):
        return mode in self.supported_modes

    @property
    def ready(self):
        """True once the backend has produced a real measurement."""
        return True

    @abstractmethod
    def read(self) -> JointSample:
        """Latest measured joint state."""

    @abstractmethod
    def write(self, mode: str, command) -> None:
        """Send one command per joint in the given control mode."""

    def update(self, dt: float) -> None:
        """Advance the backend by dt seconds (physics for sims; no-op else)."""

    def enable(self) -> None:
        """Torque on."""

    def disable(self) -> None:
        """Torque off / safe stop."""

    def close(self) -> None:
        """Release resources."""


@dataclass
class SimJointParams:
    """Parameters of the simplified simulated actuator model."""

    # Effective (decoupled) joint inertia incl. reflected rotor inertia [kg m^2]
    inertia: tuple = (12.0, 3.0, 0.4)
    # Viscous friction [N m s / rad] and Coulomb friction [N m]
    damping: tuple = (4.0, 1.5, 0.3)
    coulomb: tuple = (0.5, 0.3, 0.1)
    # Link masses for the gravity disturbance [kg] (centre of mass mid-link)
    link_masses: tuple = (1.0, 0.6, 0.3)
    # Torque constant after the gearbox [N m / A] and current limit [A]
    torque_constant: tuple = (20.0, 10.0, 4.0)
    max_current: tuple = (5.0, 5.0, 5.0)
    # Built-in servo loops for position / velocity mode (first-order lags)
    position_time_constant: float = 0.05
    velocity_time_constant: float = 0.03
    max_velocity: float = 3.0
    # Encoder resolution (ticks / rev); 0 disables quantisation.
    encoder_ticks_per_rev: int = 4096
    # Internal integration step [s]
    substep: float = 0.001
    gravity: bool = True


class SimBackend(ArmBackend):
    """
    Lightweight simulated arm (no external simulator needed).

    * position mode: an ideal servo with a first-order lag towards the goal
      position (like a Dynamixel's internal position loop);
    * velocity mode: first-order lag towards the goal velocity;
    * current mode: torque = Kt * i drives a per-joint rigid-body model with
      viscous + Coulomb friction and a gravity disturbance computed from the
      real arm geometry. This is the mode where the external PID matters.

    Mechanical joint limits act as hard stops.
    """

    name = "sim"

    def __init__(self, arm, initial_q, params: SimJointParams = None):
        self.arm = arm
        self.p = params or SimJointParams()
        n = arm.n_joints
        self.q = np.asarray(initial_q, dtype=float).copy()
        self.qd = np.zeros(n)
        self.effort = np.zeros(n)
        self.mode = "position"
        self.command = self.q.copy()
        self.enabled = True
        self._J = np.asarray(self.p.inertia, float)
        self._b = np.asarray(self.p.damping, float)
        self._fc = np.asarray(self.p.coulomb, float)
        self._kt = np.asarray(self.p.torque_constant, float)
        self._imax = np.asarray(self.p.max_current, float)
        self._lim = arm.joint_limits

    # ------------------------------------------------------------------ API
    def read(self):
        q = self.q.copy()
        ticks = self.p.encoder_ticks_per_rev
        if ticks:
            res = 2 * np.pi / ticks
            q = np.round(q / res) * res
        return JointSample(q, self.qd.copy(), self.effort.copy())

    def write(self, mode, command):
        if mode not in CONTROL_MODES:
            raise ValueError(f"unknown control mode '{mode}'")
        self.mode = mode
        self.command = np.asarray(command, dtype=float).copy()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.qd[:] = 0.0

    def update(self, dt):
        if dt <= 0 or not self.enabled:
            return
        n_sub = max(1, int(np.ceil(dt / self.p.substep)))
        h = dt / n_sub
        for _ in range(n_sub):
            self._step(h)

    # ------------------------------------------------------------- physics
    def gravity_torque(self, q):
        """Torque gravity applies at each joint (y is up)."""
        if not self.p.gravity:
            return np.zeros_like(q)
        pts = np.asarray(self.arm.forward_kinematics(q))
        com_x = 0.5 * (pts[:-1, 0] + pts[1:, 0])
        m = np.asarray(self.p.link_masses, float)
        tau = np.zeros(len(q))
        for i in range(len(q)):
            tau[i] = -GRAVITY * np.sum(m[i:] * (com_x[i:] - pts[i, 0]))
        return tau

    def _step(self, h):
        if self.mode == "position":
            qd = (self.command - self.q) / self.p.position_time_constant
            self.qd = np.clip(qd, -self.p.max_velocity, self.p.max_velocity)
            self.effort = -self.gravity_torque(self.q)
        elif self.mode == "velocity":
            target = np.clip(self.command, -self.p.max_velocity,
                             self.p.max_velocity)
            self.qd += (target - self.qd) * h / self.p.velocity_time_constant
            self.effort = -self.gravity_torque(self.q)
        else:  # current
            i = np.clip(self.command, -self._imax, self._imax)
            tau_m = self._kt * i
            tau = (tau_m + self.gravity_torque(self.q) - self._b * self.qd
                   - self._fc * np.tanh(self.qd / 0.01))
            self.qd = self.qd + h * tau / self._J
            self.effort = tau_m
        self.q = self.q + h * self.qd
        # mechanical hard stops
        lo, hi = self._lim[:, 0], self._lim[:, 1]
        hit = (self.q < lo) | (self.q > hi)
        if np.any(hit):
            self.q = np.clip(self.q, lo, hi)
            self.qd[hit] = 0.0
