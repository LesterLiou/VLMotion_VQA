"""Launch the future-risk pipeline and UI for selectable live image topics."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("future_risk_vqa")
    config = LaunchConfiguration("config")
    gui = LaunchConfiguration("enable_gui")
    logger = LaunchConfiguration("enable_logger")
    pipeline_enabled = LaunchConfiguration("enable_pipeline")
    compressed = LaunchConfiguration("use_compressed_image")

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "future_risk_pipeline.launch.py")
        ),
        condition=IfCondition(pipeline_enabled),
        launch_arguments={
            "config": config,
            "left_image_topic": LaunchConfiguration("left_image_topic"),
            "right_image_topic": LaunchConfiguration("right_image_topic"),
            "use_compressed_image": compressed,
            "use_sim_time": "false",
            "tracker_backend": LaunchConfiguration("tracker_backend"),
            "vqa_backend": LaunchConfiguration("vqa_backend"),
            "controller_url": LaunchConfiguration("controller_url"),
            "model_name": LaunchConfiguration("model_name"),
        }.items(),
    )
    gui_node = Node(
        package="future_risk_vqa",
        executable="research_gui",
        name="future_risk_research_gui",
        output="screen",
        condition=IfCondition(gui),
        additional_env={"QT_XCB_GL_INTEGRATION": "none"},
    )
    logger_node = Node(
        package="future_risk_vqa",
        executable="experiment_logger_node",
        name="future_risk_experiment_logger",
        output="screen",
        condition=IfCondition(logger),
        parameters=[
            config,
            {
                "use_sim_time": False,
                "run_name": LaunchConfiguration("run_name"),
                "output_root": LaunchConfiguration("output_root"),
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=os.path.join(
                    share, "config", "future_risk_isaac.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "left_image_topic",
                default_value="/left_camera/color/image_raw/compressed",
            ),
            DeclareLaunchArgument(
                "right_image_topic",
                default_value="/right_camera/color/image_raw/compressed",
            ),
            DeclareLaunchArgument("use_compressed_image", default_value="true"),
            DeclareLaunchArgument("enable_pipeline", default_value="true"),
            DeclareLaunchArgument("tracker_backend", default_value="sam3"),
            DeclareLaunchArgument("vqa_backend", default_value="llava_worker"),
            DeclareLaunchArgument(
                "controller_url", default_value="http://127.0.0.1:11000"
            ),
            DeclareLaunchArgument("model_name", default_value=""),
            DeclareLaunchArgument("enable_gui", default_value="true"),
            DeclareLaunchArgument("enable_logger", default_value="true"),
            DeclareLaunchArgument("run_name", default_value="live_camera"),
            DeclareLaunchArgument(
                "output_root", default_value="/workspace/results/future_risk_vqa"
            ),
            pipeline,
            logger_node,
            gui_node,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=gui_node,
                    on_exit=[EmitEvent(event=Shutdown(reason="research GUI closed"))],
                )
            ),
        ]
    )
