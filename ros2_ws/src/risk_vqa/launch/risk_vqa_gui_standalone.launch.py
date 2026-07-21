from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'controller_url',
            default_value='http://127.0.0.1:11000',
        ),
        DeclareLaunchArgument('model_name', default_value=''),
        DeclareLaunchArgument(
            'work_context',
            default_value='Robot manipulation and navigation in a shared workspace',
        ),
        DeclareLaunchArgument('horizon_s', default_value='10'),
        Node(
            package='risk_vqa',
            executable='risk_vqa_gui',
            name='risk_vqa_gui',
            output='screen',
            arguments=[
                '--controller-url', LaunchConfiguration('controller_url'),
                '--model-name', LaunchConfiguration('model_name'),
                '--work-context', LaunchConfiguration('work_context'),
                '--horizon-s', LaunchConfiguration('horizon_s'),
            ],
        ),
    ])
