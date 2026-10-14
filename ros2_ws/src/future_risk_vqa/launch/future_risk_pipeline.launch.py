"""Launch the research pipeline for real or Isaac Sim camera topics."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("future_risk_vqa"), "config", "future_risk_isaac.yaml"
    )
    config = LaunchConfiguration("config")
    left = LaunchConfiguration("left_image_topic")
    right = LaunchConfiguration("right_image_topic")
    compressed = LaunchConfiguration("use_compressed_image")
    sim_time = LaunchConfiguration("use_sim_time")
    common = {
        "left_image_topic": left,
        "right_image_topic": right,
        "use_compressed_image": ParameterValue(compressed, value_type=bool),
        "use_sim_time": ParameterValue(sim_time, value_type=bool),
    }
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("left_image_topic", default_value="/left_camera/rgb"),
            DeclareLaunchArgument("right_image_topic", default_value="/right_camera/rgb"),
            DeclareLaunchArgument("use_compressed_image", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("tracker_backend", default_value="sam3"),
            DeclareLaunchArgument("vqa_backend", default_value="llava_worker"),
            DeclareLaunchArgument("controller_url", default_value="http://127.0.0.1:11000"),
            DeclareLaunchArgument("model_name", default_value=""),
            Node(
                package="future_risk_vqa",
                executable="sam3_person_tracker_node",
                name="sam3_person_tracker",
                output="screen",
                parameters=[config, common, {"backend": LaunchConfiguration("tracker_backend")}],
            ),
            Node(
                package="future_risk_vqa",
                executable="ground_trajectory_node",
                name="ground_trajectory",
                output="screen",
                parameters=[config, {"use_sim_time": ParameterValue(sim_time, value_type=bool)}],
            ),
            Node(
                package="future_risk_vqa",
                executable="future_vqa_node",
                name="future_vqa",
                output="screen",
                parameters=[
                    config,
                    common,
                    {
                        "backend": LaunchConfiguration("vqa_backend"),
                        "controller_url": LaunchConfiguration("controller_url"),
                        "model_name": LaunchConfiguration("model_name"),
                    },
                ],
            ),
            Node(
                package="future_risk_vqa",
                executable="future_policy_node",
                name="future_risk_policy",
                output="screen",
                parameters=[config, {"use_sim_time": ParameterValue(sim_time, value_type=bool)}],
            ),
        ]
    )
