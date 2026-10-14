"""Launch synchronized video cameras with VQA and deterministic decisions."""

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
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('risk_vqa_sim')
    default_config = os.path.join(
        package_share, 'config', 'risk_vqa_sim.yaml'
    )
    system_launch = os.path.join(
        package_share, 'launch', 'risk_vqa_sim.launch.py'
    )
    config = LaunchConfiguration('config')
    left_video = LaunchConfiguration('left_video_path')
    right_video = LaunchConfiguration('right_video_path')
    left_topic = LaunchConfiguration('left_image_topic')
    right_topic = LaunchConfiguration('right_image_topic')
    compressed = LaunchConfiguration('use_compressed_image')
    loop_videos = LaunchConfiguration('loop_videos')
    playback_rate = LaunchConfiguration('playback_rate')
    output_fps = LaunchConfiguration('output_fps')
    enable_gui = LaunchConfiguration('enable_gui')
    controller_url = LaunchConfiguration('controller_url')
    model_name = LaunchConfiguration('model_name')

    video_source_node = Node(
        package='risk_vqa_sim',
        executable='dual_video_source_node',
        name='dual_video_source',
        output='screen',
        parameters=[
            config,
            {
                'use_sim_time': False,
                'left_video_path': left_video,
                'right_video_path': right_video,
                'left_image_topic': left_topic,
                'right_image_topic': right_topic,
                'use_compressed_image': ParameterValue(
                    compressed, value_type=bool
                ),
                'loop_videos': ParameterValue(
                    loop_videos, value_type=bool
                ),
                'playback_rate': ParameterValue(
                    playback_rate, value_type=float
                ),
                'output_fps': ParameterValue(
                    output_fps, value_type=float
                ),
            },
        ],
    )
    gui_node = Node(
        package='risk_vqa_sim',
        executable='risk_vqa_sim_gui',
        name='risk_vqa_sim_gui',
        output='screen',
        condition=IfCondition(enable_gui),
        parameters=[
            config,
            {
                'use_sim_time': False,
                'left_image_topic': left_topic,
                'right_image_topic': right_topic,
                'use_compressed_image': ParameterValue(
                    compressed, value_type=bool
                ),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config),
        DeclareLaunchArgument(
            'left_video_path',
            default_value='/workspace/example/video/0610_take3n4.mp4',
        ),
        DeclareLaunchArgument(
            'right_video_path',
            default_value='/workspace/example/video/0610_take1n2.mp4',
        ),
        DeclareLaunchArgument(
            'left_image_topic', default_value='/left_camera/rgb'
        ),
        DeclareLaunchArgument(
            'right_image_topic', default_value='/right_camera/rgb'
        ),
        DeclareLaunchArgument('use_compressed_image', default_value='false'),
        DeclareLaunchArgument('loop_videos', default_value='true'),
        DeclareLaunchArgument('playback_rate', default_value='1.0'),
        DeclareLaunchArgument('output_fps', default_value='10.0'),
        DeclareLaunchArgument('enable_gui', default_value='true'),
        DeclareLaunchArgument(
            'controller_url', default_value='http://127.0.0.1:11000'
        ),
        DeclareLaunchArgument('model_name', default_value=''),
        video_source_node,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(system_launch),
            launch_arguments={
                'config': config,
                'left_image_topic': left_topic,
                'right_image_topic': right_topic,
                'use_compressed_image': compressed,
                'controller_url': controller_url,
                'model_name': model_name,
                'use_sim_time': 'false',
            }.items(),
        ),
        gui_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=gui_node,
                on_exit=[
                    EmitEvent(event=Shutdown(reason='risk VQA GUI closed'))
                ],
            )
        ),
    ])
