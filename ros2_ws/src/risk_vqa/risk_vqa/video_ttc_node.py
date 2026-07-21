#!/usr/bin/env python3
"""YOLO person tracking and monocular TTC for prerecorded RGB video."""

import json
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import SetBool

from risk_vqa.ttc_tracker import PersonTTCMonitor


class VideoTTCNode(Node):
    """Detect and track people, estimate optical TTC, and publish safety data."""

    def __init__(self):
        super().__init__('video_ttc')
        self._declare_parameters()
        self._load_parameters()

        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(
                f'Video file does not exist inside this container: {self.video_path}. '
                'If the project is mounted at /workspace, use a /workspace/... path.'
            )
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            raise RuntimeError(
                f'Video exists but OpenCV cannot decode it: {self.video_path}. '
                'Check the file format and container codec support.'
            )
        source_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.source_fps = source_fps if source_fps and source_fps > 0 else 30.0
        self.frame_period_s = 1.0 / (self.source_fps * self.playback_rate)
        self.frame_index = 0
        self.paused = False

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                'ultralytics is required for person detection. Install it with pip.'
            ) from exc
        self.get_logger().info(f'Loading YOLO person detector: {self.yolo_model}')
        self.detector = YOLO(self.yolo_model)
        self.ttc_monitor = PersonTTCMonitor(
            warning_ttc_s=self.warning_ttc_s,
            critical_ttc_s=self.critical_ttc_s,
            min_growth_rate=self.min_growth_rate,
            min_area_ratio=self.min_area_ratio,
            critical_area_ratio=self.critical_area_ratio,
            roi=self.roi,
        )

        self.frame_pub = self.create_publisher(CompressedImage, self.frame_topic, 2)
        self.annotated_pub = self.create_publisher(CompressedImage, self.annotated_topic, 2)
        self.report_pub = self.create_publisher(String, self.report_topic, 10)
        self.action_pub = self.create_publisher(String, self.action_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.pause_service = self.create_service(
            SetBool,
            '/risk_vqa/set_paused',
            self._set_paused,
        )
        self.timer = self.create_timer(self.frame_period_s, self._process_frame)
        self._publish_text(self.status_pub, 'ready')
        self.get_logger().info(
            f'YOLO person TTC started: path={self.video_path}, '
            f'fps={self.source_fps:.2f}, period={self.frame_period_s * 1000:.1f} ms'
        )

    def _declare_parameters(self):
        self.declare_parameter('video_path', '/workspace/input.mp4')
        self.declare_parameter('loop_video', False)
        self.declare_parameter('playback_rate', 1.0)
        self.declare_parameter('show_window', True)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('yolo_device', '0')
        self.declare_parameter('yolo_confidence', 0.35)
        self.declare_parameter('yolo_iou', 0.50)
        self.declare_parameter('yolo_image_size', 640)
        self.declare_parameter('yolo_quantize', 16)
        self.declare_parameter('tracker_config', 'bytetrack.yaml')
        self.declare_parameter('roi', [0.20, 0.15, 0.80, 0.95])
        self.declare_parameter('warning_ttc_s', 3.0)
        self.declare_parameter('critical_ttc_s', 1.0)
        self.declare_parameter('min_growth_rate', 0.04)
        self.declare_parameter('min_area_ratio', 0.002)
        self.declare_parameter('critical_area_ratio', 0.08)
        self.declare_parameter('frame_topic', '/risk_vqa/video/image_raw/compressed')
        self.declare_parameter('annotated_topic', '/risk_vqa/video/annotated/compressed')
        self.declare_parameter('report_topic', '/risk_vqa/realtime_report')
        self.declare_parameter('action_topic', '/risk_vqa/realtime_action')
        self.declare_parameter('status_topic', '/risk_vqa/video_status')

    def _load_parameters(self):
        value = lambda name: self.get_parameter(name).value
        self.video_path = str(value('video_path'))
        self.loop_video = bool(value('loop_video'))
        self.playback_rate = max(0.05, float(value('playback_rate')))
        self.show_window = bool(value('show_window'))
        self.jpeg_quality = min(100, max(1, int(value('jpeg_quality'))))
        self.yolo_model = str(value('yolo_model'))
        raw_device = str(value('yolo_device')).strip()
        self.yolo_device = int(raw_device) if raw_device.isdigit() else raw_device
        self.yolo_confidence = min(1.0, max(0.01, float(value('yolo_confidence'))))
        self.yolo_iou = min(1.0, max(0.01, float(value('yolo_iou'))))
        self.yolo_image_size = max(160, int(value('yolo_image_size')))
        self.yolo_quantize = int(value('yolo_quantize'))
        if self.yolo_quantize not in {16, 32}:
            raise ValueError('yolo_quantize must be 16 or 32 for runtime inference')
        if self.yolo_device == 'cpu':
            self.yolo_quantize = 32
        self.tracker_config = str(value('tracker_config'))
        self.roi = [float(item) for item in value('roi')]
        if len(self.roi) != 4:
            raise ValueError('roi must contain normalized x1, y1, x2, y2')
        self.warning_ttc_s = max(0.1, float(value('warning_ttc_s')))
        self.critical_ttc_s = max(0.05, float(value('critical_ttc_s')))
        self.min_growth_rate = max(0.0, float(value('min_growth_rate')))
        self.min_area_ratio = min(1.0, max(0.0, float(value('min_area_ratio'))))
        self.critical_area_ratio = min(
            1.0, max(self.min_area_ratio, float(value('critical_area_ratio')))
        )
        self.frame_topic = str(value('frame_topic'))
        self.annotated_topic = str(value('annotated_topic'))
        self.report_topic = str(value('report_topic'))
        self.action_topic = str(value('action_topic'))
        self.status_topic = str(value('status_topic'))

    def _reset_tracking(self):
        self.ttc_monitor.reset()
        predictor = getattr(self.detector, 'predictor', None)
        for tracker in getattr(predictor, 'trackers', []) or []:
            reset = getattr(tracker, 'reset', None)
            if callable(reset):
                reset()

    def _person_detections(self, frame):
        """Run YOLO + ByteTrack and return only COCO class-0 person tracks."""
        started = time.perf_counter()
        results = self.detector.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=self.yolo_confidence,
            iou=self.yolo_iou,
            imgsz=self.yolo_image_size,
            device=self.yolo_device,
            quantize=self.yolo_quantize,
            tracker=self.tracker_config,
            verbose=False,
        )
        speed = results[0].speed if results else {}
        if not results or results[0].boxes is None:
            detections = []
        else:
            boxes = results[0].boxes
            if boxes.id is None or len(boxes) == 0:
                detections = []
            else:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                confidences = boxes.conf.detach().cpu().tolist()
                track_ids = boxes.id.detach().cpu().int().tolist()
                detections = [
                    {
                        'track_id': track_id,
                        'bbox': box,
                        'confidence': confidence,
                    }
                    for track_id, box, confidence in zip(
                        track_ids, xyxy, confidences
                    )
                ]
        latency = {
            'yolo_bytetrack_total_ms': (time.perf_counter() - started) * 1000.0,
            'yolo_preprocess_ms': float(speed.get('preprocess', 0.0)),
            'yolo_inference_ms': float(speed.get('inference', 0.0)),
            'yolo_postprocess_ms': float(speed.get('postprocess', 0.0)),
        }
        return detections, latency

    def _publish_image(self, publisher, frame):
        encode_started = time.perf_counter()
        ok, encoded = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        if not ok:
            return {'jpeg_encode_ms': encode_ms, 'ros_publish_ms': 0.0}
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.format = 'jpeg'
        message.data = encoded.tobytes()
        publish_started = time.perf_counter()
        publisher.publish(message)
        return {
            'jpeg_encode_ms': encode_ms,
            'ros_publish_ms': (time.perf_counter() - publish_started) * 1000.0,
        }

    @staticmethod
    def _publish_text(publisher, value):
        message = String()
        message.data = value
        publisher.publish(message)

    def _set_paused(self, request, response):
        self.paused = bool(request.data)
        state = 'paused' if self.paused else 'running'
        self._publish_text(self.status_pub, state)
        response.success = True
        response.message = state
        self.get_logger().info(f'Video playback {state}')
        return response

    def _annotate(self, frame, result, processing_ms):
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        x1, y1, x2, y2 = self.roi
        roi_points = (
            int(x1 * width), int(y1 * height),
            int(x2 * width), int(y2 * height),
        )
        cv2.rectangle(annotated, roi_points[:2], roi_points[2:], (255, 180, 0), 2)
        colors = {
            'CONTINUE': (0, 200, 0),
            'SLOW_DOWN': (0, 200, 255),
            'STOP': (0, 0, 255),
        }
        for track in result['tracks']:
            bx1, by1, bx2, by2 = [int(value) for value in track['bbox']]
            color = colors[track['action']]
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)
            ttc = '--' if track['ttc_s'] is None else f"{track['ttc_s']:.2f}s"
            label = (
                f"Person {track['track_id']} {track['confidence']:.2f} "
                f"TTC {ttc}"
            )
            cv2.putText(
                annotated,
                label,
                (bx1, max(18, by1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        action = result['action']
        cv2.putText(
            annotated,
            f"{action} | people {result['person_count']} | {processing_ms:.1f} ms",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colors[action],
            2,
        )
        return annotated

    def _read_frame(self):
        ok, frame = self.capture.read()
        if ok:
            return frame
        if not self.loop_video:
            return None
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_index = 0
        self._reset_tracking()
        ok, frame = self.capture.read()
        return frame if ok else None

    def _process_frame(self):
        if self.paused:
            return
        pipeline_started = time.perf_counter()
        decode_started = time.perf_counter()
        frame = self._read_frame()
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if frame is None:
            self._publish_text(self.status_pub, 'finished')
            self.timer.cancel()
            self.get_logger().info('Video finished')
            return

        video_time_s = self.frame_index / self.source_fps
        detections, detector_latency = self._person_detections(frame)
        ttc_started = time.perf_counter()
        result = self.ttc_monitor.update(detections, video_time_s, frame.shape)
        ttc_ms = (time.perf_counter() - ttc_started) * 1000.0
        annotate_started = time.perf_counter()
        annotated = self._annotate(
            frame,
            result,
            detector_latency['yolo_bytetrack_total_ms'] + ttc_ms,
        )
        annotate_ms = (time.perf_counter() - annotate_started) * 1000.0
        raw_publish = self._publish_image(self.frame_pub, frame)
        annotated_publish = self._publish_image(self.annotated_pub, annotated)
        processing_ms = (time.perf_counter() - pipeline_started) * 1000.0
        latency_ms = {
            'video_decode_ms': decode_ms,
            **detector_latency,
            'ttc_rules_ms': ttc_ms,
            'overlay_ms': annotate_ms,
            'raw_jpeg_encode_ms': raw_publish['jpeg_encode_ms'],
            'raw_ros_publish_ms': raw_publish['ros_publish_ms'],
            'annotated_jpeg_encode_ms': annotated_publish['jpeg_encode_ms'],
            'annotated_ros_publish_ms': annotated_publish['ros_publish_ms'],
            'realtime_pipeline_ms': processing_ms,
        }
        result.update({
            'frame_index': self.frame_index,
            'video_time_s': round(video_time_s, 3),
            'processing_ms': round(processing_ms, 3),
            'latency_ms': {
                key: round(value, 3) for key, value in latency_ms.items()
            },
            'published_monotonic_ns': time.monotonic_ns(),
            'method': 'YOLO person detection + ByteTrack + optical TTC',
            'assumption': 'RGB-only monocular camera; approximately constant person size',
        })

        self._publish_text(self.report_pub, json.dumps(result))
        self._publish_text(self.action_pub, result['action'])
        self._publish_text(self.status_pub, 'running')

        if self.show_window:
            cv2.imshow('Risk VQA - YOLO Person TTC', annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                self.timer.cancel()
                self._publish_text(self.status_pub, 'stopped_by_user')
        self.frame_index += 1

    def destroy_node(self):
        self.capture.release()
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VideoTTCNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
