from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration


def _start_worker(context, command, quantization):
    """Append exactly one optional quantization flag after launch expansion."""
    mode = quantization.perform(context).strip().lower()
    flags = {
        '4bit': ['--load-4bit'],
        '8bit': ['--load-8bit'],
        'none': [],
    }
    if mode not in flags:
        raise ValueError(
            f'Unsupported quantization {mode!r}; choose 4bit, 8bit, or none.'
        )
    return [ExecuteProcess(
        cmd=command + flags[mode],
        name='vlpoint_worker',
        output='screen',
    )]


def generate_launch_description():
    worker_host = LaunchConfiguration('worker_host')
    worker_port = LaunchConfiguration('worker_port')
    controller_address = LaunchConfiguration('controller_address')
    worker_address = LaunchConfiguration('worker_address')
    model_path = LaunchConfiguration('model_path')
    quantization = LaunchConfiguration('quantization')
    device = LaunchConfiguration('device')
    limit_model_concurrency = LaunchConfiguration('limit_model_concurrency')
    stream_interval = LaunchConfiguration('stream_interval')
    mm_use_sam3_conditioning = LaunchConfiguration('mm_use_sam3_conditioning')
    mm_sam3_vision_tower = LaunchConfiguration('mm_sam3_vision_tower')
    mm_sam3_blend_alpha = LaunchConfiguration('mm_sam3_blend_alpha')
    mm_sam3_mask_gamma = LaunchConfiguration('mm_sam3_mask_gamma')
    mm_sam3_device = LaunchConfiguration('mm_sam3_device')
    mm_sam3_dtype = LaunchConfiguration('mm_sam3_dtype')
    mm_sam3_unload_after_forward = LaunchConfiguration('mm_sam3_unload_after_forward')
    enable_latency_timing = LaunchConfiguration('enable_latency_timing')
    sam3_detect_device = LaunchConfiguration('sam3_detect_device')
    sam3_detect_dtype = LaunchConfiguration('sam3_detect_dtype')
    sam3_max_candidates = LaunchConfiguration('sam3_max_candidates')
    sam3_detections_out_path = LaunchConfiguration('sam3_detections_out_path')

    set_unbuffered = SetEnvironmentVariable('PYTHONUNBUFFERED', '1')
    set_cuda_alloc = SetEnvironmentVariable(
        'PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    worker_command = [
        'python3', '-m', 'point.serve.model_worker',
        '--host', worker_host,
        '--port', worker_port,
        '--worker-address', worker_address,
        '--controller-address', controller_address,
        '--model-path', model_path,
        '--device', device,
        '--limit-model-concurrency', limit_model_concurrency,
        '--stream-interval', stream_interval,
        '--mm-use-sam3-conditioning', mm_use_sam3_conditioning,
        '--mm-sam3-vision-tower', mm_sam3_vision_tower,
        '--mm-sam3-blend-alpha', mm_sam3_blend_alpha,
        '--mm-sam3-mask-gamma', mm_sam3_mask_gamma,
        '--mm-sam3-device', mm_sam3_device,
        '--mm-sam3-dtype', mm_sam3_dtype,
        '--mm-sam3-unload-after-forward', mm_sam3_unload_after_forward,
        '--enable-latency-timing', enable_latency_timing,
        '--sam3-detect-device', sam3_detect_device,
        '--sam3-detect-dtype', sam3_detect_dtype,
        '--sam3-max-candidates', sam3_max_candidates,
        '--sam3-detections-out-path', sam3_detections_out_path,
    ]

    return LaunchDescription([
        DeclareLaunchArgument('worker_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('worker_port', default_value='22000'),
        DeclareLaunchArgument('controller_address', default_value='http://127.0.0.1:11000'),
        DeclareLaunchArgument('worker_address', default_value='http://127.0.0.1:22000'),
        DeclareLaunchArgument('model_path', default_value='liuhaotian/llava-v1.5-13b'),
        DeclareLaunchArgument(
            'quantization',
            default_value='4bit',
            choices=['4bit', '8bit', 'none'],
            description='Model weight quantization used by the worker.',
        ),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('limit_model_concurrency', default_value='5'),
        DeclareLaunchArgument('stream_interval', default_value='1'),
        DeclareLaunchArgument('mm_use_sam3_conditioning', default_value='true'),
        DeclareLaunchArgument('mm_sam3_vision_tower', default_value='facebook/sam3'),
        DeclareLaunchArgument('mm_sam3_blend_alpha', default_value='0.35'),
        DeclareLaunchArgument('mm_sam3_mask_gamma', default_value='1.0'),
        DeclareLaunchArgument('mm_sam3_device', default_value='cuda'),
        DeclareLaunchArgument('mm_sam3_dtype', default_value='bfloat16'),
        DeclareLaunchArgument('mm_sam3_unload_after_forward', default_value='true'),
        DeclareLaunchArgument('enable_latency_timing', default_value='false'),
        DeclareLaunchArgument('sam3_detect_device', default_value='cuda'),
        DeclareLaunchArgument('sam3_detect_dtype', default_value='bfloat16'),
        DeclareLaunchArgument('sam3_max_candidates', default_value='12'),
        DeclareLaunchArgument('sam3_detections_out_path', default_value='/workspace/sam3_worker_candidates.jpg'),
        set_unbuffered,
        set_cuda_alloc,
        OpaqueFunction(
            function=_start_worker,
            kwargs={
                'command': worker_command,
                'quantization': quantization,
            },
        ),
    ])
