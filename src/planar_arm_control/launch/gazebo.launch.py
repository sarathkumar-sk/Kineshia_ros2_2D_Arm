"""
Same controller + GUI, but the arm lives in Gazebo (gz sim) behind ros_gz_bridge.

    ros2 launch planar_arm_control gazebo.launch.py
    ros2 launch planar_arm_control gazebo.launch.py headless:=true gui:=false

Command pipeline:
    controller_node (backend=gazebo) -> /planar_arm/jointN/cmd_pos (Float64)
      -> ros_gz_bridge -> gz JointPositionController
    gz JointStatePublisher -> ros_gz_bridge -> /gz/joint_states -> controller_node
    controller_node -> /joint_states -> gui_node   (unchanged)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("planar_arm_control")
    world = os.path.join(share, "worlds", "planar_arm.sdf")
    bridge_cfg = os.path.join(share, "config", "gz_bridge.yaml")
    params = os.path.join(share, "config", "controller.yaml")
    lc = LaunchConfiguration

    gz_args = PythonExpression([
        "'-r -s ' if '", lc("headless"), "' == 'true' else '-r '"])
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": [gz_args, world],
                          "on_exit_shutdown": "true"}.items(),
    )
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        name="gz_bridge", output="screen",
        parameters=[{"config_file": bridge_cfg}],
    )
    controller = Node(
        package="planar_arm_control", executable="controller_node",
        name="controller_node", output="screen", emulate_tty=True,
        parameters=[params, {"backend": "gazebo", "control_mode": "position",
                             "home_on_activate": True}],
    )
    gui = Node(
        package="planar_arm_control", executable="gui_node",
        name="gui_node", output="screen",
        parameters=[params, {"demo": ParameterValue(lc("demo"), value_type=bool)}],
        condition=IfCondition(lc("gui")),
    )
    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false",
                              description="run gz sim server only"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="start the PyQt operator GUI"),
        DeclareLaunchArgument("demo", default_value="false",
                              description="GUI auto-plays the test scenario"),
        gz, bridge, controller, gui,
    ])
