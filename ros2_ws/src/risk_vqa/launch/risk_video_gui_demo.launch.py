from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    demo_launch = os.path.join(
        get_package_share_directory('risk_vqa'),
        'launch',
        'risk_video_demo.launch.py',
    )
    arguments = {
        'video_path': LaunchConfiguration('video_path'),
        'loop_video': LaunchConfiguration('loop_video'),
        'enable_vlm': LaunchConfiguration('enable_vlm'),
        'enable_gui': 'true',
        'show_window': 'false',
        'controller_url': LaunchConfiguration('controller_url'),
        'model_name': LaunchConfiguration('model_name'),
        'evaluation_period_s': LaunchConfiguration('evaluation_period_s'),
        'semantic_response_mode': LaunchConfiguration('semantic_response_mode'),
        'semantic_question': LaunchConfiguration('semantic_question'),
        'work_context': LaunchConfiguration('work_context'),
        'max_new_tokens': LaunchConfiguration('max_new_tokens'),
        'use_cache': LaunchConfiguration('use_cache'),
        'yolo_model': LaunchConfiguration('yolo_model'),
        'yolo_device': LaunchConfiguration('yolo_device'),
    }
    return LaunchDescription([
        DeclareLaunchArgument('video_path', default_value='/workspace/input.mp4'),
        DeclareLaunchArgument('loop_video', default_value='false'),
        DeclareLaunchArgument('enable_vlm', default_value='true'),
        DeclareLaunchArgument('controller_url', default_value='http://127.0.0.1:11000'),
        DeclareLaunchArgument('model_name', default_value=''),
        DeclareLaunchArgument('evaluation_period_s', default_value='2.0'),
        DeclareLaunchArgument('semantic_response_mode', default_value='compact'),
        DeclareLaunchArgument(
            'semantic_question',
            default_value='How could the visible people affect the current work?',
        ),
        DeclareLaunchArgument(
            'work_context',
            default_value=(
                'Robot manipulation and navigation in a shared workspace'
            ),
        ),
        DeclareLaunchArgument('max_new_tokens', default_value='128'),
        DeclareLaunchArgument('use_cache', default_value='true'),
        DeclareLaunchArgument('yolo_model', default_value='yolov8n.pt'),
        DeclareLaunchArgument('yolo_device', default_value='0'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(demo_launch),
            launch_arguments=arguments.items(),
        ),
    ])
