from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('risk_vqa'),
        'config',
        'risk_vqa.yaml',
    )
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config_file),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/d435i/color/image_raw/compressed',
        ),
        Node(
            package='risk_vqa',
            executable='risk_vqa_node',
            name='risk_vqa',
            output='screen',
            parameters=[
                LaunchConfiguration('config'),
                {'image_topic': LaunchConfiguration('image_topic')},
            ],
        ),
    ])
