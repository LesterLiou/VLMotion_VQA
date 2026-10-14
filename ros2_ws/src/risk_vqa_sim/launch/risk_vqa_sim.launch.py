"""Launch the dual-camera VQA observer and deterministic decision node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('risk_vqa_sim'),
        'config',
        'risk_vqa_sim.yaml',
    )
    config = LaunchConfiguration('config')
    left_topic = LaunchConfiguration('left_image_topic')
    right_topic = LaunchConfiguration('right_image_topic')
    compressed = LaunchConfiguration('use_compressed_image')
    controller_url = LaunchConfiguration('controller_url')
    model_name = LaunchConfiguration('model_name')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config),
        DeclareLaunchArgument(
            'left_image_topic', default_value='/left_camera/rgb'
        ),
        DeclareLaunchArgument(
            'right_image_topic', default_value='/right_camera/rgb'
        ),
        DeclareLaunchArgument('use_compressed_image', default_value='false'),
        DeclareLaunchArgument(
            'controller_url', default_value='http://127.0.0.1:11000'
        ),
        DeclareLaunchArgument('model_name', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='risk_vqa_sim',
            executable='dual_camera_vqa_node',
            name='dual_camera_vqa',
            output='screen',
            parameters=[
                config,
                {
                    'left_image_topic': left_topic,
                    'right_image_topic': right_topic,
                    'use_compressed_image': ParameterValue(
                        compressed, value_type=bool
                    ),
                    'controller_url': controller_url,
                    'model_name': model_name,
                    'use_sim_time': ParameterValue(
                        use_sim_time, value_type=bool
                    ),
                },
            ],
        ),
        Node(
            package='risk_vqa_sim',
            executable='risk_decision_node',
            name='risk_decision',
            output='screen',
            parameters=[
                config,
                {
                    'use_sim_time': ParameterValue(
                        use_sim_time, value_type=bool
                    )
                },
            ],
        ),
    ])
