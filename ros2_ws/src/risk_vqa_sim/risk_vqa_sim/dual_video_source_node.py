#!/usr/bin/env python3
"""Publish two prerecorded videos as synchronized ROS 2 camera streams."""

import json
import os

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import SetBool


class _VideoReader:
    """Sequential reader that can skip frames to a shared timeline target."""

    def __init__(self, path, start_offset_s=0.0):
        self.path = path
        self.start_offset_s = max(0.0, float(start_offset_s))
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise RuntimeError(f'OpenCV cannot open video: {path}')
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0.0 else 30.0
        frame_count = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_s = frame_count / self.fps if frame_count > 0.0 else 0.0
        self._next_index = 0
        self.reset()

    @property
    def available_duration_s(self):
        return max(0.0, self.duration_s - self.start_offset_s)

    def reset(self):
        start_index = int(round(self.start_offset_s * self.fps))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        self._next_index = start_index

    def read_at(self, relative_time_s):
        target_s = self.start_offset_s + max(0.0, float(relative_time_s))
        target_index = int(round(target_s * self.fps))
        while self._next_index < target_index:
            if not self.capture.grab():
                return None
            self._next_index += 1
        ok, frame = self.capture.read()
        if not ok:
            return None
        self._next_index += 1
        return frame

    def close(self):
        self.capture.release()


class DualVideoSourceNode(Node):
    """Drive two videos from one clock and publish identical ROS timestamps."""

    def __init__(self):
        super().__init__('dual_video_source')
        self._declare_parameters()
        self._load_parameters()
        self._validate_files()
        self._bridge = CvBridge()
        self._left_reader = _VideoReader(
            self.left_video_path, self.left_start_offset_s
        )
        self._right_reader = _VideoReader(
            self.right_video_path, self.right_start_offset_s
        )
        durations = [
            duration
            for duration in (
                self._left_reader.available_duration_s,
                self._right_reader.available_duration_s,
            )
            if duration > 0.0
        ]
        if len(durations) != 2:
            raise RuntimeError('both video durations must be available')
        self.common_duration_s = min(durations)
        self.timeline_s = 0.0
        self.output_index = 0
        self._ended = False
        self._paused = False

        image_type = CompressedImage if self.use_compressed_image else Image
        self._left_pub = self.create_publisher(
            image_type, self.left_image_topic, qos_profile_sensor_data
        )
        self._right_pub = self.create_publisher(
            image_type, self.right_image_topic, qos_profile_sensor_data
        )
        self._status_pub = self.create_publisher(String, self.status_topic, 10)
        self._pause_service = self.create_service(
            SetBool,
            self.pause_service,
            self._set_paused,
        )
        self._timer = self.create_timer(1.0 / self.output_fps, self._tick)
        self._publish_status('ready')
        self.get_logger().info(
            f'Dual video source ready: left={self.left_video_path}, '
            f'right={self.right_video_path}, output={self.output_fps:.1f} FPS, '
            f'common_duration={self.common_duration_s:.2f}s, loop={self.loop_videos}'
        )

    def _declare_parameters(self):
        self.declare_parameter(
            'left_video_path',
            '/workspace/example/video/0610_take3n4.mp4',
        )
        self.declare_parameter(
            'right_video_path',
            '/workspace/example/video/0610_take1n2.mp4',
        )
        self.declare_parameter('left_image_topic', '/left_camera/rgb')
        self.declare_parameter('right_image_topic', '/right_camera/rgb')
        self.declare_parameter('left_frame_id', 'left_camera_optical_frame')
        self.declare_parameter('right_frame_id', 'right_camera_optical_frame')
        self.declare_parameter('use_compressed_image', False)
        self.declare_parameter('loop_videos', True)
        self.declare_parameter('playback_rate', 1.0)
        self.declare_parameter('output_fps', 10.0)
        self.declare_parameter('output_width', 960)
        self.declare_parameter('output_height', 540)
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('left_start_offset_s', 0.0)
        self.declare_parameter('right_start_offset_s', 0.0)
        self.declare_parameter(
            'status_topic', '/risk_vqa_sim/video_source_status'
        )
        self.declare_parameter(
            'pause_service', '/risk_vqa_sim/set_video_paused'
        )

    def _load_parameters(self):
        def value(name):
            return self.get_parameter(name).value

        self.left_video_path = str(value('left_video_path'))
        self.right_video_path = str(value('right_video_path'))
        self.left_image_topic = str(value('left_image_topic'))
        self.right_image_topic = str(value('right_image_topic'))
        self.left_frame_id = str(value('left_frame_id'))
        self.right_frame_id = str(value('right_frame_id'))
        self.use_compressed_image = bool(value('use_compressed_image'))
        self.loop_videos = bool(value('loop_videos'))
        self.playback_rate = max(0.05, float(value('playback_rate')))
        self.output_fps = min(60.0, max(1.0, float(value('output_fps'))))
        self.output_width = max(0, int(value('output_width')))
        self.output_height = max(0, int(value('output_height')))
        self.jpeg_quality = min(100, max(40, int(value('jpeg_quality'))))
        self.left_start_offset_s = max(
            0.0, float(value('left_start_offset_s'))
        )
        self.right_start_offset_s = max(
            0.0, float(value('right_start_offset_s'))
        )
        self.status_topic = str(value('status_topic'))
        self.pause_service = str(value('pause_service'))

    def _validate_files(self):
        for side, path in (
            ('left', self.left_video_path),
            ('right', self.right_video_path),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f'{side} video does not exist: {path}. '
                    'Use /workspace/... paths inside the project container.'
                )

    def _restart_both(self):
        self._left_reader.reset()
        self._right_reader.reset()
        self.timeline_s = 0.0
        self.output_index = 0
        self._publish_status('looped')

    def _resize(self, frame):
        if self.output_width == 0 or self.output_height == 0:
            return frame
        return cv2.resize(
            frame,
            (self.output_width, self.output_height),
            interpolation=cv2.INTER_AREA,
        )

    def _tick(self):
        if self._ended or self._paused:
            return
        if self.timeline_s >= self.common_duration_s:
            if self.loop_videos:
                self._restart_both()
            else:
                self._ended = True
                self._timer.cancel()
                self._publish_status('ended')
                self.get_logger().info('Dual video playback ended')
                return

        left_frame = self._left_reader.read_at(self.timeline_s)
        right_frame = self._right_reader.read_at(self.timeline_s)
        if left_frame is None or right_frame is None:
            if self.loop_videos:
                self._restart_both()
                left_frame = self._left_reader.read_at(0.0)
                right_frame = self._right_reader.read_at(0.0)
            if left_frame is None or right_frame is None:
                self._ended = True
                self._timer.cancel()
                self._publish_status('error:video_decode')
                self.get_logger().error('Could not decode synchronized video frames')
                return

        left_frame = self._resize(left_frame)
        right_frame = self._resize(right_frame)
        stamp = self.get_clock().now().to_msg()
        self._publish_frame(
            self._left_pub, left_frame, stamp, self.left_frame_id
        )
        self._publish_frame(
            self._right_pub, right_frame, stamp, self.right_frame_id
        )
        self.output_index += 1
        self.timeline_s += self.playback_rate / self.output_fps

    def _set_paused(self, request, response):
        if self._ended:
            response.success = False
            response.message = 'video playback has ended'
            return response
        self._paused = bool(request.data)
        state = 'paused' if self._paused else 'running'
        self._publish_status(state)
        response.success = True
        response.message = state
        self.get_logger().info(f'Dual video playback {state}')
        return response

    def _publish_frame(self, publisher, frame, stamp, frame_id):
        if self.use_compressed_image:
            ok, encoded = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                raise RuntimeError('OpenCV failed to encode a video frame')
            message = CompressedImage()
            message.format = 'jpeg'
            message.data = encoded.tobytes()
        else:
            message = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        publisher.publish(message)

    def _publish_status(self, state):
        message = String()
        message.data = json.dumps({
            'state': state,
            'timeline_s': round(self.timeline_s, 3),
            'output_index': self.output_index,
            'loop': self.loop_videos,
            'paused': self._paused,
        })
        self._status_pub.publish(message)

    def destroy_node(self):
        self._left_reader.close()
        self._right_reader.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DualVideoSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
