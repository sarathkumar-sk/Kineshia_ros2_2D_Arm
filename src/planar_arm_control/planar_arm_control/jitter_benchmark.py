#!/usr/bin/env python3
"""
jitter_benchmark -- measure the controller's control-loop timing while idle
and while the node is loaded with IK requests.

    ros2 run planar_arm_control jitter_benchmark --ros-args -p seconds:=10.0

Reads the timing fields of /controller_status (rolling 1000-sample window
computed inside the controller) and fires /move_to_target requests at
`ik_rate_hz` during the load phase (each one runs the ~15 ms IK solve).
"""

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node

from planar_arm_interfaces.msg import ControllerStatus
from planar_arm_interfaces.srv import MoveToTarget


class JitterBenchmark(Node):
    def __init__(self):
        super().__init__("jitter_benchmark")
        self.declare_parameter("seconds", 10.0)
        self.declare_parameter("ik_rate_hz", 10.0)
        self.samples = []
        self.create_subscription(ControllerStatus, "controller_status",
                                 lambda m: self.samples.append(m), 10)
        self.cli = self.create_client(MoveToTarget, "move_to_target")

    def phase(self, seconds, load):
        self.samples = []
        period = 1.0 / self.get_parameter("ik_rate_hz").value
        targets = [(4.0, 2.0), (-3.0, 3.0), (2.0, 4.0), (-1.0, 5.0)]
        k, t_next = 0, time.monotonic()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if load and time.monotonic() >= t_next:
                x, y = targets[k % len(targets)]
                self.cli.call_async(MoveToTarget.Request(target=Point(x=x, y=y)))
                k += 1
                t_next += period
            rclpy.spin_once(self, timeout_sec=0.01)
        s = self.samples[len(self.samples) // 2:]  # window filled by the phase
        return {
            "mean": np.mean([m.control_period_mean_ms for m in s]),
            "std": np.max([m.control_period_std_ms for m in s]),
            "p99": np.max([m.control_jitter_p99_ms for m in s]),
            "max": np.max([m.control_period_max_ms for m in s]),
            "exec": np.mean([m.control_loop_exec_mean_ms for m in s]),
            "target": s[-1].control_period_target_ms, "ik_requests": k,
        }


def main(args=None):
    rclpy.init(args=args)
    node = JitterBenchmark()
    if not node.cli.wait_for_service(timeout_sec=10.0):
        node.get_logger().error("controller not running")
        return
    secs = node.get_parameter("seconds").value
    rows = [("idle", node.phase(secs, False)), ("IK load", node.phase(secs, True))]
    print(f"\n{'phase':8s} {'target':>7s} {'mean':>7s} {'std':>6s} {'p99|jit|':>9s} "
          f"{'max':>7s} {'exec':>6s}   [ms]")
    for name, r in rows:
        print(f"{name:8s} {r['target']:7.2f} {r['mean']:7.3f} {r['std']:6.3f} "
              f"{r['p99']:9.3f} {r['max']:7.2f} {r['exec']:6.3f}"
              + (f"   ({r['ik_requests']} IK requests)" if name != "idle" else ""))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
