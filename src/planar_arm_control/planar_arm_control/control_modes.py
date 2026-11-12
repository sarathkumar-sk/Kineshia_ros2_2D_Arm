"""
Control laws: turn a trajectory reference + measured state into a backend
command. Selected at runtime with the ``control_mode`` parameter.

  position : q_cmd = q_ref                          (servo closes the loop)
  velocity : qd_cmd = qd_ref + PI(q_ref - q)        (feed-forward + PI)
  current  : i_cmd = [J (qdd_ref + PID(e)) + b qd_ref] / Kt
             (model-scaled PID: gains are in acceleration units, so the same
              gains give the same closed-loop bandwidth on every joint)
"""

from abc import ABC, abstractmethod

import numpy as np


class JointPID:
    """Vector PID with derivative-on-error-rate and integral anti-windup."""

    def __init__(self, kp, ki, kd, integral_limit=np.inf, n=3):
        self.kp = np.broadcast_to(np.asarray(kp, float), (n,)).copy()
        self.ki = np.broadcast_to(np.asarray(ki, float), (n,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, float), (n,)).copy()
        self.integral_limit = np.broadcast_to(
            np.asarray(integral_limit, float), (n,)).copy()
        self.integral = np.zeros(n)

    def reset(self):
        self.integral[:] = 0.0

    def update(self, error, error_rate, dt):
        self.integral = np.clip(self.integral + error * dt,
                                -self.integral_limit, self.integral_limit)
        return self.kp * error + self.ki * self.integral + self.kd * error_rate


class ControlMode(ABC):
    name = "abstract"

    def reset(self):
        pass

    @abstractmethod
    def compute(self, ref, meas, dt):
        """ref: TrajectoryPoint, meas: JointSample -> command array."""


class PositionMode(ControlMode):
    name = "position"

    def compute(self, ref, meas, dt):
        return ref.position.copy()


class VelocityMode(ControlMode):
    name = "velocity"

    def __init__(self, kp=8.0, ki=4.0, integral_limit=0.5):
        self.pid = JointPID(kp, ki, 0.0, integral_limit)

    def reset(self):
        self.pid.reset()

    def compute(self, ref, meas, dt):
        e = ref.position - meas.position
        return ref.velocity + self.pid.update(e, 0.0, dt)


class CurrentMode(ControlMode):
    name = "current"

    def __init__(self, inertia, damping, torque_constant, kp=400.0, ki=800.0,
                 kd=40.0, integral_limit=0.2, max_current=5.0):
        self.J = np.asarray(inertia, float)
        self.b = np.asarray(damping, float)
        self.kt = np.asarray(torque_constant, float)
        self.max_current = np.broadcast_to(
            np.asarray(max_current, float), self.J.shape).copy()
        self.pid = JointPID(kp, ki, kd, integral_limit, n=len(self.J))

    def reset(self):
        self.pid.reset()

    def compute(self, ref, meas, dt):
        e = ref.position - meas.position
        e_dot = ref.velocity - meas.velocity
        acc = ref.acceleration + self.pid.update(e, e_dot, dt)
        tau = self.J * acc + self.b * ref.velocity
        i = tau / self.kt
        # anti-windup: undo the last integral step on saturated joints
        sat = np.abs(i) > self.max_current
        if np.any(sat):
            self.pid.integral[sat] -= e[sat] * dt
        return np.clip(i, -self.max_current, self.max_current)


def make_control_mode(name, sim_params=None, gains=None):
    """Factory used by the controller node."""
    gains = gains or {}
    if name == "position":
        return PositionMode()
    if name == "velocity":
        return VelocityMode(gains.get("vel_kp", 8.0), gains.get("vel_ki", 4.0))
    if name == "current":
        if sim_params is None:
            raise ValueError("current mode needs an actuator model")
        return CurrentMode(sim_params.inertia, sim_params.damping,
                           sim_params.torque_constant,
                           gains.get("cur_kp", 400.0),
                           gains.get("cur_ki", 800.0),
                           gains.get("cur_kd", 40.0),
                           max_current=sim_params.max_current)
    raise ValueError(f"unknown control mode '{name}'")
