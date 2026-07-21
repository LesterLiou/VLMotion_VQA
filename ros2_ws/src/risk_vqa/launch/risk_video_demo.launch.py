from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('risk_vqa'),
        'config',
        'video_risk.yaml',
    )
    video_path = LaunchConfiguration('video_path')
    show_window = LaunchConfiguration('show_window')
    loop_video = LaunchConfiguration('loop_video')
    enable_vlm = LaunchConfiguration('enable_vlm')
    enable_gui = LaunchConfiguration('enable_gui')
    controller_url = LaunchConfiguration('controller_url')
    model_name = LaunchConfiguration('model_name')
    evaluation_period_s = LaunchConfiguration('evaluation_period_s')
    semantic_response_mode = LaunchConfiguration('semantic_response_mode')
    semantic_question = LaunchConfiguration('semantic_question')
    work_context = LaunchConfiguration('work_context')
    max_new_tokens = LaunchConfiguration('max_new_tokens')
    use_cache = LaunchConfiguration('use_cache')
    yolo_model = LaunchConfiguration('yolo_model')
    yolo_device = LaunchConfiguration('yolo_device')

    video_node = Node(
        package='risk_vqa',
        executable='video_ttc_node',
        name='video_ttc',
        output='screen',
        parameters=[
            LaunchConfiguration('config'),
            {
                'video_path': video_path,
                'show_window': ParameterValue(show_window, value_type=bool),
                'loop_video': ParameterValue(loop_video, value_type=bool),
                'yolo_model': yolo_model,
                'yolo_device': ParameterValue(yolo_device, value_type=str),
            },
        ],
    )
    gui_node = Node(
        package='risk_vqa',
        executable='risk_video_gui',
        name='risk_video_gui',
        output='screen',
        condition=IfCondition(enable_gui),
    )

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config_file),
        DeclareLaunchArgument('video_path', default_value='/workspace/input.mp4'),
        DeclareLaunchArgument('show_window', default_value='true'),
        DeclareLaunchArgument('loop_video', default_value='false'),
        DeclareLaunchArgument('enable_vlm', default_value='true'),
        DeclareLaunchArgument('enable_gui', default_value='false'),
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
        video_node,
        gui_node,
        Node(
            package='risk_vqa',
            executable='risk_vqa_node',
            name='risk_vqa_semantic',
            output='screen',
            condition=IfCondition(enable_vlm),
            parameters=[
                LaunchConfiguration('config'),
                {
                    'image_topic': '/risk_vqa/video/image_raw/compressed',
                    'use_compressed_image': True,
                    'controller_url': controller_url,
                    'model_name': ParameterValue(model_name, value_type=str),
                    'evaluation_period_s': ParameterValue(
                        evaluation_period_s,
                        value_type=float,
                    ),
                    'max_new_tokens': ParameterValue(
                        max_new_tokens,
                        value_type=int,
                    ),
                    'use_cache': ParameterValue(use_cache, value_type=bool),
                    'analysis_mode': 'person_semantic',
                    'semantic_response_mode': ParameterValue(
                        semantic_response_mode,
                        value_type=str,
                    ),
                    'semantic_question': ParameterValue(
                        semantic_question,
                        value_type=str,
                    ),
                    'work_context': ParameterValue(
                        work_context,
                        value_type=str,
                    ),
                },
            ],
        ),
        Node(
            package='risk_vqa',
            executable='risk_fusion_node',
            name='risk_fusion',
            output='screen',
            parameters=[LaunchConfiguration('config')],
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=video_node,
                on_exit=[EmitEvent(event=Shutdown(reason='video TTC node exited'))],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=gui_node,
                on_exit=[EmitEvent(event=Shutdown(reason='video GUI closed'))],
            )
        ),
    ])
