#!/usr/bin/env python3
"""Dual-camera temporal storyboard VQA node for Isaac Sim RGB topics."""

from datetime import datetime, timezone
import json
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from risk_vqa_sim.model_client import LlavaWorkerClient
from risk_vqa_sim.prompt import build_observation_prompt, wrap_vicuna_prompt
from risk_vqa_sim.schema import extract_json_object, validate_observation
from risk_vqa_sim.storyboard import FrameBuffer, build_storyboard


class DualCameraVQANode(Node):
    """Sample two image streams, run one temporal VQA, and publish observations."""

    def __init__(self):
        super().__init__('dual_camera_vqa')
        self._declare_parameters()
        self._load_parameters()
        self._bridge = CvBridge()
        self._buffer_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._request_in_flight = False
        self._last_evaluated_key = None
        self._left = FrameBuffer(self.frame_buffer_size)
        self._right = FrameBuffer(self.frame_buffer_size)

        self._observation_pub = self.create_publisher(
            String, self.observation_topic, 10
        )
        self._raw_pub = self.create_publisher(String, self.raw_response_topic, 10)
        self._status_pub = self.create_publisher(String, self.status_topic, 10)
        self._storyboard_pub = self.create_publisher(
            CompressedImage, self.storyboard_topic, 2
        )

        image_type = CompressedImage if self.use_compressed_image else Image
        self._left_sub = self.create_subscription(
            image_type,
            self.left_image_topic,
            lambda message: self._image_callback('left', message),
            qos_profile_sensor_data,
        )
        self._right_sub = self.create_subscription(
            image_type,
            self.right_image_topic,
            lambda message: self._image_callback('right', message),
            qos_profile_sensor_data,
        )
        self._client = LlavaWorkerClient(
            controller_url=self.controller_url,
            model_name=self.model_name,
            request_timeout_s=self.request_timeout_s,
            max_new_tokens=self.max_new_tokens,
            use_cache=self.use_cache,
        )
        user_prompt = build_observation_prompt(
            self.work_context, self.sample_count, self.sample_window_s
        )
        self._prompt = wrap_vicuna_prompt(user_prompt)
        self.create_timer(self.evaluation_period_s, self._start_evaluation)
        self._publish_status('waiting_for_both_cameras')
        self.get_logger().info(
            'Dual-camera VQA ready: '
            f'left={self.left_image_topic}, right={self.right_image_topic}, '
            f'compressed={self.use_compressed_image}, '
            f'{self.sample_count} frames/camera over {self.sample_window_s:.1f}s'
        )

    def _declare_parameters(self):
        self.declare_parameter('left_image_topic', '/left_camera/rgb')
        self.declare_parameter('right_image_topic', '/right_camera/rgb')
        self.declare_parameter('use_compressed_image', False)
        self.declare_parameter('controller_url', 'http://127.0.0.1:11000')
        self.declare_parameter('model_name', '')
        self.declare_parameter('evaluation_period_s', 2.0)
        self.declare_parameter('sample_window_s', 2.0)
        self.declare_parameter('sample_count', 3)
        self.declare_parameter('minimum_temporal_span_s', 0.8)
        self.declare_parameter('frame_buffer_size', 180)
        self.declare_parameter('max_camera_skew_s', 0.20)
        self.declare_parameter('max_frame_age_s', 2.0)
        self.declare_parameter('request_timeout_s', 60.0)
        self.declare_parameter('max_new_tokens', 512)
        self.declare_parameter('use_cache', False)
        self.declare_parameter('jpeg_quality', 88)
        self.declare_parameter('storyboard_cell_width', 448)
        self.declare_parameter('storyboard_cell_height', 336)
        self.declare_parameter(
            'work_context',
            'A machine or robot is about to enter a shared hazard zone. '
            'People may approach from either side.',
        )
        self.declare_parameter('observation_topic', '/risk_vqa_sim/observation')
        self.declare_parameter(
            'raw_response_topic', '/risk_vqa_sim/raw_model_response'
        )
        self.declare_parameter('status_topic', '/risk_vqa_sim/status')
        self.declare_parameter(
            'storyboard_topic', '/risk_vqa_sim/storyboard/compressed'
        )

    def _load_parameters(self):
        def value(name):
            return self.get_parameter(name).value

        self.left_image_topic = str(value('left_image_topic'))
        self.right_image_topic = str(value('right_image_topic'))
        self.use_compressed_image = bool(value('use_compressed_image'))
        self.controller_url = str(value('controller_url')).rstrip('/')
        self.model_name = str(value('model_name')).strip()
        self.evaluation_period_s = max(0.2, float(value('evaluation_period_s')))
        self.sample_window_s = max(0.1, float(value('sample_window_s')))
        self.sample_count = min(6, max(2, int(value('sample_count'))))
        self.minimum_temporal_span_s = max(
            0.0, float(value('minimum_temporal_span_s'))
        )
        self.frame_buffer_size = max(10, int(value('frame_buffer_size')))
        self.max_camera_skew_s = max(0.0, float(value('max_camera_skew_s')))
        self.max_frame_age_s = max(0.2, float(value('max_frame_age_s')))
        self.request_timeout_s = max(1.0, float(value('request_timeout_s')))
        self.max_new_tokens = min(1024, max(128, int(value('max_new_tokens'))))
        self.use_cache = bool(value('use_cache'))
        self.jpeg_quality = min(100, max(40, int(value('jpeg_quality'))))
        self.storyboard_cell_width = max(160, int(value('storyboard_cell_width')))
        self.storyboard_cell_height = max(120, int(value('storyboard_cell_height')))
        self.work_context = str(value('work_context')).strip()
        self.observation_topic = str(value('observation_topic'))
        self.raw_response_topic = str(value('raw_response_topic'))
        self.status_topic = str(value('status_topic'))
        self.storyboard_topic = str(value('storyboard_topic'))

    @staticmethod
    def _message_stamp_s(message, fallback_s):
        stamp = message.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            return fallback_s
        return stamp.sec + stamp.nanosec / 1_000_000_000.0

    def _decode(self, message):
        if self.use_compressed_image:
            array = cv2.imdecode(
                np.frombuffer(message.data, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if array is None:
                raise ValueError('OpenCV could not decode compressed image')
            return array
        return self._bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')

    def _image_callback(self, side, message):
        receipt_s = time.monotonic()
        try:
            frame = self._decode(message)
            stamp_s = self._message_stamp_s(message, receipt_s)
            with self._buffer_lock:
                target = self._left if side == 'left' else self._right
                target.append(stamp_s, receipt_s, frame)
        except Exception as exc:
            self.get_logger().error(f'{side} image conversion failed: {exc}')
            self._publish_status(f'error:{side}_image_conversion')

    def _snapshot(self):
        now_s = time.monotonic()
        with self._buffer_lock:
            left_latest = self._left.latest
            right_latest = self._right.latest
            if left_latest is None or right_latest is None:
                return None, 'waiting_for_both_cameras'
            if (
                now_s - left_latest.receipt_s > self.max_frame_age_s
                or now_s - right_latest.receipt_s > self.max_frame_age_s
            ):
                return None, 'camera_frame_stale'
            if abs(left_latest.stamp_s - right_latest.stamp_s) > self.max_camera_skew_s:
                return None, 'camera_timestamps_not_synchronized'
            key = (left_latest.stamp_s, right_latest.stamp_s)
            if key == self._last_evaluated_key:
                return None, 'waiting_for_new_frames'
            left_samples = self._left.sample(
                self.sample_count, self.sample_window_s
            )
            right_samples = self._right.sample(
                self.sample_count, self.sample_window_s
            )
        left_span = left_samples[-1].stamp_s - left_samples[0].stamp_s
        right_span = right_samples[-1].stamp_s - right_samples[0].stamp_s
        if min(left_span, right_span) < self.minimum_temporal_span_s:
            return None, 'warming_temporal_buffer'
        self._last_evaluated_key = key
        return (left_samples, right_samples), None

    def _start_evaluation(self):
        with self._request_lock:
            if self._request_in_flight:
                self._publish_status('evaluation_already_in_flight')
                return
            snapshot, waiting_reason = self._snapshot()
            if snapshot is None:
                self._publish_status(waiting_reason)
                return
            self._request_in_flight = True
        thread = threading.Thread(
            target=self._evaluate, args=snapshot, daemon=True
        )
        thread.start()

    def _evaluate(self, left_samples, right_samples):
        started = time.perf_counter()
        storyboard = None
        storyboard_metadata = {}
        try:
            self._publish_status('building_storyboard')
            storyboard, storyboard_metadata = build_storyboard(
                left_samples,
                right_samples,
                self.storyboard_cell_width,
                self.storyboard_cell_height,
            )
            ok, encoded = cv2.imencode(
                '.jpg',
                storyboard,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                raise RuntimeError('OpenCV failed to encode storyboard')
            jpeg = encoded.tobytes()
            self._publish_storyboard(jpeg)
            self._publish_status('evaluating')
            completion, model_metadata = self._client.generate(self._prompt, jpeg)
            self._publish_text(self._raw_pub, completion)
            raw_report = extract_json_object(completion)
            report = validate_observation(raw_report)
            latest_receipt_s = min(
                left_samples[-1].receipt_s,
                right_samples[-1].receipt_s,
            )
            report['metadata'] = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'pipeline_ms': round(
                    (time.perf_counter() - started) * 1000.0, 3
                ),
                'latest_frame_age_at_publish_s': round(
                    max(0.0, time.monotonic() - latest_receipt_s), 3
                ),
                'work_context': self.work_context,
                'temporal_window_s': self.sample_window_s,
                **storyboard_metadata,
                **model_metadata,
            }
            self._publish_text(
                self._observation_pub,
                json.dumps(report, ensure_ascii=False),
            )
            if report['validation_errors']:
                self.get_logger().warning(
                    'VLM output normalized: '
                    + '; '.join(report['validation_errors'])
                )
            self._publish_status('ok')
        except Exception as exc:
            self.get_logger().error(f'VQA evaluation failed: {exc}')
            fallback = validate_observation({})
            fallback['validation_errors'].append(f'model_pipeline: {exc}')
            fallback['metadata'] = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'pipeline_ms': round(
                    (time.perf_counter() - started) * 1000.0, 3
                ),
                **storyboard_metadata,
            }
            self._publish_text(
                self._observation_pub,
                json.dumps(fallback, ensure_ascii=False),
            )
            self._publish_status('error:model_pipeline')
        finally:
            with self._request_lock:
                self._request_in_flight = False

    def _publish_storyboard(self, jpeg):
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format = 'jpeg'
        message.data = jpeg
        self._storyboard_pub.publish(message)

    @staticmethod
    def _publish_text(publisher, text):
        message = String()
        message.data = str(text)
        publisher.publish(message)

    def _publish_status(self, text):
        self._publish_text(self._status_pub, text)


def main(args=None):
    rclpy.init(args=args)
    node = DualCameraVQANode()
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
