"""
Gazebo (gz sim) backend -- same controller, bridged command pipeline.

    controller_node --std_msgs/Float64--> /planar_arm/jN/cmd_pos
        --ros_gz_bridge--> gz JointPositionController (one per joint)
    gz JointStatePublisher --> /world/.../joint_state --ros_gz_bridge-->
        /gz/joint_states (sensor_msgs/JointState) --> controller_node

Physics, integration and joint dynamics happen inside Gazebo, so ``update()``
is a no-op -- exactly like a real hardware backend.
"""

import threading

import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from planar_arm_control.backends import ArmBackend, JointSample


class GazeboBackend(ArmBackend):
    name = "gazebo"
    supported_modes = ("position",)

    def __init__(self, node, joint_names, initial_q,
                 cmd_topic_fmt="/planar_arm/{joint}/cmd_pos",
                 state_topic="/gz/joint_states"):
        self.node = node
        self.joint_names = list(joint_names)
        n = len(self.joint_names)
        self._lock = threading.Lock()
        self._q = np.asarray(initial_q, float).copy()
        self._qd = np.zeros(n)
        self._effort = np.zeros(n)
        self.received = False
        self._pubs = [node.create_publisher(
            Float64, cmd_topic_fmt.format(joint=j), 10)
            for j in self.joint_names]
        self._sub = node.create_subscription(
            JointState, state_topic, self._on_state, 10)

    def _on_state(self, msg):
        idx = {name: i for i, name in enumerate(msg.name)}
        with self._lock:
            for k, j in enumerate(self.joint_names):
                i = idx.get(j)
                if i is None:
                    continue
                if i < len(msg.position):
                    self._q[k] = msg.position[i]
                if i < len(msg.velocity):
                    self._qd[k] = msg.velocity[i]
                if i < len(msg.effort):
                    self._effort[k] = msg.effort[i]
            self.received = True

    @property
    def ready(self):
        return self.received

    def read(self):
        with self._lock:
            return JointSample(self._q.copy(), self._qd.copy(),
                               self._effort.copy())

    def write(self, mode, command):
        if mode != "position":
            raise ValueError("GazeboBackend only supports position mode")
        for pub, value in zip(self._pubs, np.asarray(command, float)):
            pub.publish(Float64(data=float(value)))

    def close(self):
        for pub in self._pubs:
            self.node.destroy_publisher(pub)
        self.node.destroy_subscription(self._sub)
