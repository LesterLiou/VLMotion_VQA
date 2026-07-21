#!/usr/bin/env python3
"""ROS 2 node that performs periodic visual risk assessment."""

import base64
from datetime import datetime, timezone
import json
import threading
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import requests
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from risk_vqa.prompt import build_risk_prompt, wrap_vicuna_prompt
from risk_vqa.person_semantic import (
    build_person_semantic_prompt,
    render_person_semantic_en,
    validate_person_semantic_report,
)
from risk_vqa.schema import (
    extract_json_object,
    render_en,
    select_action,
    validate_report,
)


class RiskVQANode(Node):
    """Capture the latest camera frame and send it to the VLM worker."""

    def __init__(self):
        super().__init__('risk_vqa')
        self._declare_parameters()
        self._load_parameters()

        self._bridge = CvBridge()
        self._image_lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_stamp_s = None
        self._latest_image_seq = 0
        self._last_evaluated_seq = 0
        self._request_lock = threading.Lock()
        self._request_in_flight = False

        self._report_pub = self.create_publisher(String, self.report_topic, 10)
        self._text_pub = self.create_publisher(String, self.text_topic, 10)
        self._action_pub = self.create_publisher(String, self.action_topic, 10)
        self._status_pub = self.create_publisher(String, self.status_topic, 10)

        image_type = CompressedImage if self.use_compressed_image else Image
        self._image_sub = self.create_subscription(
            image_type,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self._timer = self.create_timer(
            self.evaluation_period_s,
            self._start_evaluation,
        )
        self._publish_status('ready')
        self.get_logger().info(
            f'Risk VQA ready: mode={self.analysis_mode}, '
            f'response={self.semantic_response_mode}, image={self.image_topic}, '
            f'period={self.evaluation_period_s:.1f}s, '
            f'horizon={self.prediction_horizon_s:.1f}s'
        )

    def _declare_parameters(self):
        self.declare_parameter('image_topic', '/d435i/color/image_raw/compressed')
        self.declare_parameter('use_compressed_image', True)
        self.declare_parameter('controller_url', 'http://127.0.0.1:11000')
        self.declare_parameter('model_name', '')
        self.declare_parameter('evaluation_period_s', 2.0)
        self.declare_parameter('prediction_horizon_s', 10.0)
        self.declare_parameter('analysis_mode', 'risk_assessment')
        self.declare_parameter('semantic_response_mode', 'compact')
        self.declare_parameter('semantic_question', '')
        self.declare_parameter('request_timeout_s', 60.0)
        self.declare_parameter('max_new_tokens', 384)
        self.declare_parameter('use_cache', False)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter(
            'work_context',
            'Robot manipulation and navigation in a shared workspace',
        )
        self.declare_parameter('allowed_actions', [
            'CONTINUE', 'SLOW_DOWN', 'WAIT', 'STOP',
            'REPLAN', 'ALERT_HUMAN',
        ])
        self.declare_parameter('report_topic', '/risk_vqa/report')
        self.declare_parameter('text_topic', '/risk_vqa/text')
        self.declare_parameter('action_topic', '/risk_vqa/action_suggestion')
        self.declare_parameter('status_topic', '/risk_vqa/status')

    def _load_parameters(self):
        value = lambda name: self.get_parameter(name).value
        self.image_topic = str(value('image_topic'))
        self.use_compressed_image = bool(value('use_compressed_image'))
        self.controller_url = str(value('controller_url')).rstrip('/')
        self.model_name = str(value('model_name')).strip()
        self.evaluation_period_s = max(0.2, float(value('evaluation_period_s')))
        self.prediction_horizon_s = max(0.1, float(value('prediction_horizon_s')))
        self.analysis_mode = str(value('analysis_mode')).strip().lower()
        self.semantic_response_mode = str(
            value('semantic_response_mode')
        ).strip().lower()
        self.semantic_question = str(value('semantic_question')).strip()
        if self.analysis_mode not in {'risk_assessment', 'person_semantic'}:
            raise ValueError('analysis_mode must be risk_assessment or person_semantic')
        if self.semantic_response_mode not in {'compact', 'detailed'}:
            raise ValueError(
                'semantic_response_mode must be compact or detailed'
            )
        self.request_timeout_s = max(1.0, float(value('request_timeout_s')))
        self.max_new_tokens = min(1024, max(64, int(value('max_new_tokens'))))
        self.use_cache = bool(value('use_cache'))
        self.jpeg_quality = min(100, max(1, int(value('jpeg_quality'))))
        self.work_context = str(value('work_context')).strip()
        self.allowed_actions = [
            str(action).upper() for action in value('allowed_actions')
        ]
        self.report_topic = str(value('report_topic'))
        self.text_topic = str(value('text_topic'))
        self.action_topic = str(value('action_topic'))
        self.status_topic = str(value('status_topic'))

    @staticmethod
    def _stamp_to_seconds(header):
        stamp = header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            return None
        return stamp.sec + stamp.nanosec / 1_000_000_000.0

    def _image_callback(self, msg):
        try:
            if self.use_compressed_image:
                jpeg = bytes(msg.data)
            else:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                ok, encoded = cv2.imencode(
                    '.jpg',
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    raise ValueError('OpenCV failed to encode camera frame')
                jpeg = encoded.tobytes()
            with self._image_lock:
                self._latest_jpeg = jpeg
                self._latest_stamp_s = self._stamp_to_seconds(msg.header)
                self._latest_image_seq += 1
        except Exception as exc:
            self.get_logger().error(f'Image conversion failed: {exc}')

    def _start_evaluation(self):
        with self._request_lock:
            if self._request_in_flight:
                return
            with self._image_lock:
                if self._latest_jpeg is None:
                    self._publish_status('waiting_for_image')
                    return
                if self._latest_image_seq == self._last_evaluated_seq:
                    return
                jpeg = self._latest_jpeg
                stamp_s = self._latest_stamp_s
                self._last_evaluated_seq = self._latest_image_seq
            self._request_in_flight = True

        thread = threading.Thread(
            target=self._evaluate,
            args=(jpeg, stamp_s),
            daemon=True,
        )
        thread.start()

    def _resolve_model(self):
        if self.model_name:
            return self.model_name
        response = requests.post(
            self.controller_url + '/list_models',
            timeout=5,
        )
        response.raise_for_status()
        models = response.json().get('models', [])
        if not models:
            raise RuntimeError('controller has no registered models')
        self.model_name = str(models[0])
        self.get_logger().info(f'Using model: {self.model_name}')
        return self.model_name

    def _worker_address(self, model_name):
        response = requests.post(
            self.controller_url + '/get_worker_address',
            json={'model': model_name},
            timeout=5,
        )
        response.raise_for_status()
        address = str(response.json().get('address', '')).rstrip('/')
        if not address:
            raise RuntimeError(f'no worker is available for model {model_name}')
        return address

    def _build_request(self, model_name, jpeg):
        if self.analysis_mode == 'person_semantic':
            user_prompt = build_person_semantic_prompt(
                self.work_context,
                self.allowed_actions,
                self.semantic_question,
                self.semantic_response_mode,
            )
        else:
            user_prompt = build_risk_prompt(
                self.work_context,
                self.prediction_horizon_s,
                self.allowed_actions,
            )
        prompt = wrap_vicuna_prompt(user_prompt)
        return prompt, {
            'model': model_name,
            'prompt': prompt,
            'temperature': 0.0,
            'top_p': 1.0,
            'max_new_tokens': self.max_new_tokens,
            'use_cache': self.use_cache,
            'stop': '###',
            'images': [base64.b64encode(jpeg).decode('ascii')],
            'task_type': 'risk_assessment',
            'use_sam3_candidates': False,
        }

    def _generate(self, worker_address, payload, prompt):
        response = requests.post(
            worker_address + '/worker_generate_stream',
            headers={'User-Agent': 'risk_vqa ROS 2 node'},
            json=payload,
            stream=True,
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()
        output = ''
        worker_latency = {}
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b'\0'):
            if not chunk:
                continue
            data = json.loads(chunk.decode('utf-8'))
            if data.get('error_code', 1) != 0:
                raise RuntimeError(data.get('text', 'model worker error'))
            text = str(data.get('text', ''))
            output = text[len(prompt):] if text.startswith(prompt) else text
            if isinstance(data.get('latency_ms'), dict):
                worker_latency = data['latency_ms']
        if not output.strip():
            raise RuntimeError('model worker returned an empty response')
        return output.strip(), worker_latency

    def _evaluate(self, jpeg, stamp_s):
        total_started = time.perf_counter()
        try:
            self._publish_status('evaluating')
            resolve_started = time.perf_counter()
            model_name = self._resolve_model()
            resolve_ms = (time.perf_counter() - resolve_started) * 1000.0
            worker_started = time.perf_counter()
            worker_address = self._worker_address(model_name)
            worker_lookup_ms = (time.perf_counter() - worker_started) * 1000.0
            prompt_started = time.perf_counter()
            prompt, payload = self._build_request(model_name, jpeg)
            prompt_build_ms = (time.perf_counter() - prompt_started) * 1000.0
            generation_started = time.perf_counter()
            completion, worker_latency = self._generate(
                worker_address, payload, prompt
            )
            worker_generation_ms = (
                time.perf_counter() - generation_started
            ) * 1000.0
            validation_started = time.perf_counter()
            try:
                raw_report = extract_json_object(completion)
            except ValueError as exc:
                preview = ' '.join(completion.split())[:240]
                raise ValueError(
                    f'{exc}; raw model response: {preview!r}'
                ) from exc
            if self.analysis_mode == 'person_semantic':
                report = validate_person_semantic_report(
                    raw_report,
                    self.allowed_actions,
                    self.semantic_response_mode,
                )
                display_text = render_person_semantic_en(report)
            else:
                report = validate_report(
                    raw_report,
                    self.prediction_horizon_s,
                    self.allowed_actions,
                )
                display_text = render_en(report)
            validation_ms = (time.perf_counter() - validation_started) * 1000.0
            total_ms = (time.perf_counter() - total_started) * 1000.0
            metadata = {
                'observed_at_s': stamp_s,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'work_context': self.work_context,
                'analysis_mode': self.analysis_mode,
                'semantic_response_mode': self.semantic_response_mode,
                'model': model_name,
                'max_new_tokens': self.max_new_tokens,
                'use_cache': self.use_cache,
                'latency_ms': {
                    'controller_model_resolve_ms': round(resolve_ms, 3),
                    'controller_worker_lookup_ms': round(worker_lookup_ms, 3),
                    'prompt_build_ms': round(prompt_build_ms, 3),
                    'worker_generation_ms': round(worker_generation_ms, 3),
                    'json_validation_render_ms': round(validation_ms, 3),
                    'vlm_pipeline_ms': round(total_ms, 3),
                },
                'published_monotonic_ns': time.monotonic_ns(),
            }
            for key, value in worker_latency.items():
                try:
                    metadata['latency_ms'][f'worker_{key}'] = round(
                        float(value), 3
                    )
                except (TypeError, ValueError):
                    continue
            if self.analysis_mode == 'risk_assessment':
                metadata['prediction_horizon_s'] = self.prediction_horizon_s
            report.update(metadata)
            self._publish(self._report_pub, json.dumps(report, ensure_ascii=False))
            self._publish(self._text_pub, display_text)
            self._publish(self._action_pub, select_action(report))
            self._publish_status('ok')
            if report['validation_errors']:
                self.get_logger().warning(
                    'Some model factors were rejected: '
                    + '; '.join(report['validation_errors'])
                )
        except Exception as exc:
            message = f'error: {exc}'
            self.get_logger().error(message)
            self._publish_status(message)
        finally:
            with self._request_lock:
                self._request_in_flight = False

    @staticmethod
    def _publish(publisher, text):
        message = String()
        message.data = text
        publisher.publish(message)

    def _publish_status(self, text):
        if hasattr(self, '_status_pub'):
            self._publish(self._status_pub, text)


def main(args=None):
    rclpy.init(args=args)
    node = RiskVQANode()
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
