"""
Bring up the controller and the operator GUI.

    ros2 launch planar_arm_control bringup.launch.py
    ros2 launch planar_arm_control bringup.launch.py control_mode:=current
    ros2 launch planar_arm_control bringup.launch.py gui:=false   # headless
    ros2 launch planar_arm_control bringup.launch.py demo:=true   # auto-play test scenario
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory("planar_arm_control"),
                          "config", "controller.yaml")
    args = [
        DeclareLaunchArgument("control_mode", default_value="position",
                              description="position | velocity | current"),
        DeclareLaunchArgument("backend", default_value="sim",
                              description="sim | gazebo"),
        DeclareLaunchArgument("control_rate_hz", default_value="200.0"),
        DeclareLaunchArgument("publish_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("max_joint_velocity", default_value="1.0"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("demo", default_value="false",
                              description="GUI auto-plays the test scenario"),
    ]
    lc = LaunchConfiguration
    controller = Node(
        package="planar_arm_control",
        executable="controller_node",
        name="controller_node",
        output="screen",
        emulate_tty=True,
        parameters=[config, {
            "control_mode": lc("control_mode"),
            "backend": lc("backend"),
            "control_rate_hz": ParameterValue(lc("control_rate_hz"), value_type=float),
            "publish_rate_hz": ParameterValue(lc("publish_rate_hz"), value_type=float),
            "max_joint_velocity": ParameterValue(lc("max_joint_velocity"),
                                                 value_type=float),
        }],
    )
    gui = Node(
        package="planar_arm_control",
        executable="gui_node",
        name="gui_node",
        output="screen",
        emulate_tty=True,
        parameters=[config, {"demo": ParameterValue(lc("demo"), value_type=bool)}],
        condition=IfCondition(lc("gui")),
    )
    return LaunchDescription(args + [controller, gui])
