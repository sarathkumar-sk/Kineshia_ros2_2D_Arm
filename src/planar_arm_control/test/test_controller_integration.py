"""
launch_testing integration test: starts the real controller_node and drives
it through its ROS interfaces (service, action, lifecycle, parameters).

    colcon test --packages-select planar_arm_control
"""

import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import numpy as np
import pytest
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point
from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from sensor_msgs.msg import JointState

from planar_arm_interfaces.action import PickPlace
from planar_arm_interfaces.msg import ControllerStatus
from planar_arm_interfaces.srv import MoveJoints, MoveToTarget
from planar_arm_control.planar_arm import PlanarArm

ARM = PlanarArm([3.0, 2.0, 1.5])


@pytest.mark.launch_test
def generate_test_description():
    controller = launch_ros.actions.Node(
        package="planar_arm_control", executable="controller_node",
        name="controller_node", output="screen",
        parameters=[{"max_joint_velocity": 2.0, "max_joint_acceleration": 4.0,
                     "grip_time": 0.2}],
    )
    return launch.LaunchDescription([
        controller, launch_testing.actions.ReadyToTest()]), {"controller": controller}


class TestController(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("controller_test")
        cls.q = None
        cls.status = None
        cls.node.create_subscription(JointState, "joint_states", cls._on_js, 10)
        cls.node.create_subscription(ControllerStatus, "controller_status",
                                     cls._on_status, 10)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def _on_js(cls, msg):
        cls.q = np.asarray(msg.position)

    @classmethod
    def _on_status(cls, msg):
        cls.status = msg

    # -------------------------------------------------------------- helpers
    def spin_until(self, cond, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if cond():
                return True
        return False

    def call(self, client, req, timeout=10.0):
        self.assertTrue(client.wait_for_service(timeout_sec=10.0))
        fut = client.call_async(req)
        self.assertTrue(self.spin_until(fut.done, timeout))
        return fut.result()

    def ee(self):
        return np.asarray(ARM.end_effector(self.q))

    def wait_idle(self, timeout=20.0):
        # spin (not sleep) so queued joint_states/status are consumed and the
        # caller never reads a stale sample from before the wait
        self.spin_until(lambda: False, 0.3)
        return self.spin_until(lambda: self.status is not None
                               and self.status.state != "moving", timeout)

    # ---------------------------------------------------------------- tests
    def test_1_publishes_joint_states(self):
        self.assertTrue(self.spin_until(lambda: self.q is not None, 10.0))
        self.assertEqual(len(self.q), 3)

    def test_2_move_to_target(self):
        cli = self.node.create_client(MoveToTarget, "move_to_target")
        resp = self.call(cli, MoveToTarget.Request(target=Point(x=4.0, y=2.0)))
        self.assertTrue(resp.accepted, resp.message)
        self.assertFalse(resp.projected)
        self.assertTrue(self.spin_until(
            lambda: np.linalg.norm(self.ee() - [4.0, 2.0]) < 0.01,
            resp.duration + 5.0))
        self.wait_idle()

    def test_3_unreachable_target_is_projected(self):
        cli = self.node.create_client(MoveToTarget, "move_to_target")
        resp = self.call(cli, MoveToTarget.Request(target=Point(x=7.0, y=3.0)))
        self.assertTrue(resp.accepted)
        self.assertTrue(resp.projected)
        r = np.array([resp.reachable_target.x, resp.reachable_target.y])
        self.assertAlmostEqual(np.linalg.norm(r), 6.5, places=3)
        self.assertTrue(self.spin_until(lambda: self.status is not None
                                        and self.status.target_projected, 5.0))
        self.assertTrue(self.spin_until(
            lambda: np.linalg.norm(self.ee() - r) < 0.01, resp.duration + 5.0))
        self.wait_idle()

    def test_4_below_ground_rejected(self):
        cli = self.node.create_client(MoveToTarget, "move_to_target")
        resp = self.call(cli, MoveToTarget.Request(target=Point(x=2.0, y=-1.0)))
        self.assertFalse(resp.accepted)
        self.assertIn("below the ground", resp.message)

    def test_5_pick_place_action_with_feedback(self):
        client = ActionClient(self.node, PickPlace, "pick_place")
        self.assertTrue(client.wait_for_server(timeout_sec=10.0))
        feedback = []
        goal = PickPlace.Goal(pick=Point(x=4.0, y=2.0), place=Point(x=-3.0, y=3.0))
        fut = client.send_goal_async(goal, feedback_callback=lambda f: feedback.append(
            f.feedback))
        self.assertTrue(self.spin_until(fut.done, 10.0))
        gh = fut.result()
        self.assertTrue(gh.accepted)
        res_fut = gh.get_result_async()
        self.assertTrue(self.spin_until(res_fut.done, 60.0))
        res = res_fut.result()
        self.assertEqual(res.status, GoalStatus.STATUS_SUCCEEDED, res.result.message)
        phases = {f.phase for f in feedback}
        self.assertTrue({"move_to_pick", "grasp", "move_to_place", "release"} <= phases)
        np.testing.assert_allclose(self.ee(), [-3.0, 3.0], atol=0.01)
        self.assertTrue(self.spin_until(
            lambda: abs(self.status.object_position.x + 3.0) < 0.05, 3.0))

    def test_6_pick_place_cancel(self):
        client = ActionClient(self.node, PickPlace, "pick_place")
        self.assertTrue(client.wait_for_server(timeout_sec=10.0))
        goal = PickPlace.Goal(pick=Point(x=4.0, y=2.0), place=Point(x=-3.0, y=3.0))
        fut = client.send_goal_async(goal)
        self.assertTrue(self.spin_until(fut.done, 10.0))
        gh = fut.result()
        self.assertTrue(gh.accepted)
        self.spin_until(lambda: False, 0.8)
        cfut = gh.cancel_goal_async()
        self.assertTrue(self.spin_until(cfut.done, 5.0))
        res_fut = gh.get_result_async()
        self.assertTrue(self.spin_until(res_fut.done, 10.0))
        self.assertEqual(res_fut.result().status, GoalStatus.STATUS_CANCELED)
        # arm stops: holds still
        self.wait_idle(5.0)
        q0 = self.q.copy()
        self.spin_until(lambda: False, 0.5)
        np.testing.assert_allclose(self.q, q0, atol=1e-2)

    def test_7_switch_control_mode(self):
        pc = AsyncParameterClient(self.node, "controller_node")
        self.assertTrue(pc.wait_for_services(timeout_sec=10.0))
        fut = pc.set_parameters([Parameter("control_mode", Parameter.Type.STRING,
                                           "current")])
        self.assertTrue(self.spin_until(fut.done, 5.0))
        self.assertTrue(fut.result().results[0].successful)
        self.assertTrue(self.spin_until(
            lambda: self.status.control_mode == "current", 3.0))
        cli = self.node.create_client(MoveToTarget, "move_to_target")
        resp = self.call(cli, MoveToTarget.Request(target=Point(x=4.0, y=2.0)))
        self.assertTrue(resp.accepted, resp.message)
        self.assertTrue(self.spin_until(
            lambda: np.linalg.norm(self.ee() - [4.0, 2.0]) < 0.01,
            resp.duration + 5.0))
        # invalid mode is refused
        fut = pc.set_parameters([Parameter("control_mode", Parameter.Type.STRING,
                                           "bogus")])
        self.assertTrue(self.spin_until(fut.done, 5.0))
        self.assertFalse(fut.result().results[0].successful)

    def test_8_lifecycle_deactivate_activate(self):
        get = self.node.create_client(GetState, "controller_node/get_state")
        change = self.node.create_client(ChangeState, "controller_node/change_state")
        self.assertEqual(self.call(get, GetState.Request()).current_state.label,
                         "active")
        r = self.call(change, ChangeState.Request(
            transition=Transition(id=Transition.TRANSITION_DEACTIVATE)))
        self.assertTrue(r.success)
        cli = self.node.create_client(MoveToTarget, "move_to_target")
        resp = self.call(cli, MoveToTarget.Request(target=Point(x=4.0, y=2.0)))
        self.assertFalse(resp.accepted)       # inactive -> refuses commands
        r = self.call(change, ChangeState.Request(
            transition=Transition(id=Transition.TRANSITION_ACTIVATE)))
        self.assertTrue(r.success)
        type(self).q = None   # class attr is what the callback updates
        self.assertTrue(self.spin_until(lambda: self.q is not None, 5.0))

    def test_9_move_joints(self):
        self.wait_idle()
        cli = self.node.create_client(MoveJoints, "move_joints")
        goal = np.radians([60.0, 30.0, -45.0])
        resp = self.call(cli, MoveJoints.Request(positions=goal.tolist()))
        self.assertTrue(resp.accepted, resp.message)
        self.assertTrue(self.spin_until(
            lambda: np.max(np.abs(self.q - goal)) < np.radians(0.5),
            resp.duration + 5.0))
        self.wait_idle()
        # link 2 would point below the ground
        bad = self.call(cli, MoveJoints.Request(
            positions=np.radians([10.0, -90.0, 0.0]).tolist()))
        self.assertFalse(bad.accepted)
        self.assertIn("below the ground", bad.message)
        short = self.call(cli, MoveJoints.Request(positions=[0.0, 0.0]))
        self.assertFalse(short.accepted)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_code(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, -2, -15, 1])
