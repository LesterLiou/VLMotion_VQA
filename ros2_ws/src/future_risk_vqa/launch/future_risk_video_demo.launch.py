"""Launch prerecorded dual video, research pipeline, and optional GUI."""

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
    tracker = LaunchConfiguration("tracker_backend")
    vqa = LaunchConfiguration("vqa_backend")
    gui = LaunchConfiguration("enable_gui")
    logger = LaunchConfiguration("enable_logger")
    source = Node(
        package="future_risk_vqa",
        executable="dual_video_source_node",
        name="future_risk_video_source",
        output="screen",
        # Video paths and playback settings have one source of truth: the YAML file.
        parameters=[config],
    )
    gui_node = Node(
        package="future_risk_vqa",
        executable="research_gui",
        name="future_risk_research_gui",
        output="screen",
        condition=IfCondition(gui),
        additional_env={"QT_XCB_GL_INTEGRATION": "none"},
    )
    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "future_risk_pipeline.launch.py")
        ),
        launch_arguments={
            "config": config,
            "tracker_backend": tracker,
            "vqa_backend": vqa,
            "controller_url": LaunchConfiguration("controller_url"),
            "model_name": LaunchConfiguration("model_name"),
            "use_sim_time": "false",
        }.items(),
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
                "run_name": LaunchConfiguration("run_name"),
                "output_root": LaunchConfiguration("output_root"),
            },
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=os.path.join(share, "config", "future_risk_video_demo.yaml"),
            ),
            DeclareLaunchArgument("tracker_backend", default_value="sam3"),
            DeclareLaunchArgument("vqa_backend", default_value="llava_worker"),
            DeclareLaunchArgument("controller_url", default_value="http://127.0.0.1:11000"),
            DeclareLaunchArgument("model_name", default_value=""),
            DeclareLaunchArgument("enable_gui", default_value="true"),
            DeclareLaunchArgument("enable_logger", default_value="true"),
            DeclareLaunchArgument("run_name", default_value="video_demo"),
            DeclareLaunchArgument(
                "output_root", default_value="/workspace/results/future_risk_vqa"
            ),
            source,
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
