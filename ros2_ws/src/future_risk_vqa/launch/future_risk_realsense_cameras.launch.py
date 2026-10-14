"""Launch two RealSense cameras for the future-risk pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _camera_node(name, serial, color_profile, enable_depth):
    return Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="",
        name=name,
        output="screen",
        parameters=[
            {
                "camera_name": name,
                "serial_no": ParameterValue(serial, value_type=str),
                "enable_color": True,
                "enable_depth": ParameterValue(enable_depth, value_type=bool),
                "enable_infra": False,
                "enable_infra1": False,
                "enable_infra2": False,
                "rgb_camera.color_profile": color_profile,
                "initial_reset": False,
                "publish_tf": True,
            }
        ],
    )


def generate_launch_description():
    color_profile = LaunchConfiguration("color_profile")
    enable_depth = LaunchConfiguration("enable_depth")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "left_serial_no",
                default_value="244622072255",
                description="Serial number of the left RealSense camera.",
            ),
            DeclareLaunchArgument(
                "right_serial_no",
                default_value="243522070379",
                description="Serial number of the right RealSense camera.",
            ),
            DeclareLaunchArgument("color_profile", default_value="640x480x30"),
            DeclareLaunchArgument("enable_depth", default_value="false"),
            _camera_node(
                "left_camera",
                LaunchConfiguration("left_serial_no"),
                color_profile,
                enable_depth,
            ),
            _camera_node(
                "right_camera",
                LaunchConfiguration("right_serial_no"),
                color_profile,
                enable_depth,
            ),
        ]
    )
