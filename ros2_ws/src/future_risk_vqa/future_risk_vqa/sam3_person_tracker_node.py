#!/usr/bin/env python3
"""Shared SAM3 person segmentation and lightweight per-camera tracking node."""

import json
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String

from .geometry import (
    footpoint_from_mask,
    mask_foot_zone_overlap_ratio,
    normalized_polygon_to_image,
    point_in_normalized_polygon,
)
from .qos import pipeline_pause_qos
from .tracking import Detection, PersonTrackManager, suppress_duplicate_detections


class _Sam3Backend:
    """Load one Transformers SAM3 model and reuse it for both views."""

    def __init__(
        self,
        device,
        model_id,
        threshold,
        mask_threshold,
        max_people,
        local_files_only,
    ):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device
        self.threshold = float(threshold)
        self.mask_threshold = float(mask_threshold)
        self.max_people = int(max_people)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Sam3Model.from_pretrained(
            str(model_id),
            dtype=dtype,
            local_files_only=bool(local_files_only),
        ).to(device)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.processor = Sam3Processor.from_pretrained(
            str(model_id), local_files_only=bool(local_files_only)
        )

    def infer(self, bgr, text="person", threshold=None, max_results=None):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = bgr.shape[:2]
        inputs = self.processor(
            images=PilImage.fromarray(rgb), text=str(text), return_tensors="pt"
        ).to(self.device)
        context = (
            self.torch.autocast(
                device_type="cuda",
                dtype=self.torch.bfloat16,
            )
            if self.device.startswith("cuda")
            else _NullContext()
        )
        with self.torch.inference_mode(), context:
            outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold if threshold is None else float(threshold),
            mask_threshold=self.mask_threshold,
            target_sizes=[(height, width)],
        )[0]
        boxes = result["boxes"].detach().float().cpu().numpy()
        scores = result["scores"].detach().float().cpu().numpy()
        masks = result["masks"].detach().cpu().numpy().astype(bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        limit = self.max_people if max_results is None else max(1, int(max_results))
        order = np.argsort(-scores)[:limit]
        return [(boxes[index], float(scores[index]), masks[index]) for index in order]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Sam3PersonTrackerNode(Node):
    def __init__(self):
        super().__init__("sam3_person_tracker")
        defaults = {
            "enable_person_detection": True,
            "enable_tracking": True,
            "left_image_topic": "/left_camera/rgb",
            "right_image_topic": "/right_camera/rgb",
            "use_compressed_image": False,
            "backend": "sam3",
            "device": "cuda",
            "model_id": "facebook/sam3",
            "local_files_only": True,
            "confidence_threshold": 0.35,
            "mask_threshold": 0.5,
            "max_people": 12,
            "suppress_duplicate_detections": True,
            "duplicate_mask_iou_threshold": 0.65,
            "duplicate_bbox_iou_threshold": 0.70,
            "duplicate_bbox_ioa_threshold": 0.85,
            "inference_hz": 2.0,
            "max_missed_frames": 5,
            "minimum_match_score": 0.20,
            "match_mask_iou_weight": 0.15,
            "match_bbox_iou_weight": 0.20,
            "match_foot_distance_weight": 0.50,
            "match_size_similarity_weight": 0.15,
            "match_max_foot_distance_ratio": 0.20,
            "match_velocity_smoothing": 0.60,
            "enable_walkway_detection": True,
            "walkway_prompt": "floor",
            "walkway_confidence_threshold": 0.35,
            "walkway_inference_interval_s": 5.0,
            "show_walkway_overlay": True,
            "enable_approximate_stop_zone": True,
            "approximate_stop_foot_overlap_ratio": 0.05,
            "left_approximate_stop_polygon_normalized": [
                0.20, 1.00, 0.80, 1.00, 0.635, 0.76, 0.365, 0.76,
            ],
            "right_approximate_stop_polygon_normalized": [
                0.20, 1.00, 0.80, 1.00, 0.635, 0.76, 0.365, 0.76,
            ],
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        value = lambda name: self.get_parameter(name).value
        self.bridge = CvBridge()
        self.enable_person_detection = bool(value("enable_person_detection"))
        self.enable_tracking = bool(value("enable_tracking"))
        self.compressed = bool(value("use_compressed_image"))
        self.backend_name = str(value("backend")).lower()
        if not self.enable_person_detection:
            self.backend_name = "disabled"
        self.model_id = str(value("model_id"))
        self.confidence_threshold = float(value("confidence_threshold"))
        self.mask_threshold = float(value("mask_threshold"))
        self.suppress_duplicates = bool(value("suppress_duplicate_detections"))
        self.duplicate_mask_iou_threshold = float(
            value("duplicate_mask_iou_threshold")
        )
        self.duplicate_bbox_iou_threshold = float(
            value("duplicate_bbox_iou_threshold")
        )
        self.duplicate_bbox_ioa_threshold = float(
            value("duplicate_bbox_ioa_threshold")
        )
        self.enable_walkway_detection = bool(value("enable_walkway_detection"))
        self.walkway_prompt = str(value("walkway_prompt"))
        self.walkway_threshold = float(value("walkway_confidence_threshold"))
        self.walkway_interval = max(1.0, float(value("walkway_inference_interval_s")))
        self.show_walkway_overlay = bool(value("show_walkway_overlay"))
        self.enable_approximate_stop_zone = bool(value("enable_approximate_stop_zone"))
        self.approximate_stop_foot_overlap_ratio = max(
            0.0, min(1.0, float(value("approximate_stop_foot_overlap_ratio")))
        )
        self.approximate_stop_polygons = {
            "left": list(value("left_approximate_stop_polygon_normalized")),
            "right": list(value("right_approximate_stop_polygon_normalized")),
        }
        self.walkway_cache = {}
        self.backend = None
        if self.backend_name == "sam3":
            self.get_logger().info("Loading one shared SAM3 image model...")
            self.backend = _Sam3Backend(
                str(value("device")),
                self.model_id,
                self.confidence_threshold,
                self.mask_threshold,
                int(value("max_people")),
                bool(value("local_files_only")),
            )
        elif self.backend_name != "disabled":
            raise ValueError("backend must be 'sam3' or 'disabled'")
        self.trackers = {
            "left": PersonTrackManager(
                "L",
                value("max_missed_frames"),
                value("minimum_match_score"),
                value("match_mask_iou_weight"),
                value("match_bbox_iou_weight"),
                value("match_foot_distance_weight"),
                value("match_size_similarity_weight"),
                value("match_max_foot_distance_ratio"),
                value("match_velocity_smoothing"),
            ),
            "right": PersonTrackManager(
                "R",
                value("max_missed_frames"),
                value("minimum_match_score"),
                value("match_mask_iou_weight"),
                value("match_bbox_iou_weight"),
                value("match_foot_distance_weight"),
                value("match_size_similarity_weight"),
                value("match_max_foot_distance_ratio"),
                value("match_velocity_smoothing"),
            ),
        }
        self.latest, self.lock, self.busy = {}, threading.Lock(), False
        self.paused = False
        image_type = CompressedImage if self.compressed else Image
        self.create_subscription(
            image_type,
            str(value("left_image_topic")),
            lambda msg: self._receive("left", msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            image_type,
            str(value("right_image_topic")),
            lambda msg: self._receive("right", msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            "/future_risk/pipeline_paused",
            self._set_paused,
            pipeline_pause_qos(),
        )
        self.tracks_pub = self.create_publisher(String, "/future_risk/person_tracks", 10)
        self.status_pub = self.create_publisher(String, "/future_risk/sam3_status", 10)
        self.preview_pubs = {
            side: self.create_publisher(
                CompressedImage,
                f"/future_risk/{side}/sam3/compressed",
                qos_profile_sensor_data,
            )
            for side in ("left", "right")
        }
        self.create_timer(1.0 / max(0.1, float(value("inference_hz"))), self._start)
        self.get_logger().info(f"SAM3 tracker ready: backend={self.backend_name}")

    @staticmethod
    def _stamp(message):
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def _decode(self, message):
        if self.compressed:
            return cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
        return self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    def _receive(self, side, message):
        try:
            item = (self._stamp(message), self._decode(message))
            with self.lock:
                self.latest[side] = item
        except Exception as exc:
            self.get_logger().error(f"{side} image decode failed: {exc}")

    def _set_paused(self, message):
        with self.lock:
            self.paused = bool(message.data)
            busy = self.busy
        timing = "after current inference" if busy and self.paused else "immediately"
        self.get_logger().info(f"SAM3 inference {'paused' if self.paused else 'resumed'} {timing}")

    def _start(self):
        with self.lock:
            if self.paused or self.busy or set(self.latest) != {"left", "right"}:
                return
            self.busy = True
            frames = {side: (stamp, image.copy()) for side, (stamp, image) in self.latest.items()}
        threading.Thread(target=self._process, args=(frames,), daemon=True).start()

    def _refresh_walkway(self, side, stamp, frame):
        if (
            self.backend is None
            or not self.enable_walkway_detection
            or (
                side in self.walkway_cache
                and self.walkway_cache[side][1].shape == frame.shape[:2]
                and 0.0 <= stamp - self.walkway_cache[side][0] < self.walkway_interval
            )
        ):
            return
        try:
            floor_results = self.backend.infer(
                frame,
                text=self.walkway_prompt,
                threshold=self.walkway_threshold,
                max_results=8,
            )
            floor_mask = np.zeros(frame.shape[:2], dtype=bool)
            for _, _, mask in floor_results:
                floor_mask |= np.asarray(mask, dtype=bool)
            polygon = []
            if floor_mask.any():
                cleaned = cv2.morphologyEx(
                    floor_mask.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    np.ones((11, 11), dtype=np.uint8),
                )
                contours, _ = cv2.findContours(
                    cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.015 * cv2.arcLength(contour, True)
                    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
                    polygon = [[int(x), int(y)] for x, y in polygon]
            self.walkway_cache[side] = (stamp, floor_mask, polygon)
        except Exception as exc:
            # Floor segmentation is optional and must never interrupt person safety.
            self.get_logger().warning(f"{side} walkway segmentation failed: {exc}")

    def _process(self, frames):
        started = time.perf_counter()
        report = {
            "schema_version": "1.0",
            "backend": self.backend_name,
            "person_detection_enabled": self.enable_person_detection,
            "tracking_enabled": self.enable_tracking,
            "model_id": self.model_id if self.backend is not None else None,
            "model_revision": None if self.backend is None else self.backend.revision,
            "text_prompt": "person",
            "confidence_threshold": self.confidence_threshold,
            "mask_threshold": self.mask_threshold,
            "cameras": {},
        }
        try:
            for side in ("left", "right"):
                stamp, frame = frames[side]
                raw = [] if self.backend is None else self.backend.infer(frame)
                self._refresh_walkway(side, stamp, frame)
                detections = []
                for box, score, mask in raw:
                    try:
                        foot = footpoint_from_mask(mask)
                    except ValueError:
                        foot = ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))
                    detections.append(Detection(tuple(float(v) for v in box), foot, score, mask))
                raw_detection_count = len(detections)
                if self.suppress_duplicates:
                    detections = suppress_duplicate_detections(
                        detections,
                        self.duplicate_mask_iou_threshold,
                        self.duplicate_bbox_iou_threshold,
                        self.duplicate_bbox_ioa_threshold,
                    )
                if not self.enable_tracking:
                    self.trackers[side].reset()
                tracks = self.trackers[side].update(detections, frame.shape, stamp)
                serialized = []
                preview = frame.copy()
                walkway_mask = self.walkway_cache.get(
                    side, (stamp, np.zeros(frame.shape[:2], dtype=bool), [])
                )[1]
                walkway_polygon = self.walkway_cache.get(side, (stamp, None, []))[2]
                if self.show_walkway_overlay and walkway_mask.any():
                    overlay = np.zeros_like(preview)
                    overlay[walkway_mask] = (255, 150, 0)
                    preview = cv2.addWeighted(preview, 1.0, overlay, 0.22, 0.0)
                stop_points = normalized_polygon_to_image(
                    self.approximate_stop_polygons[side], frame.shape[1], frame.shape[0]
                )
                stop_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(stop_mask, [np.int32(stop_points)], 1)
                stop_mask = stop_mask.astype(bool)
                if self.enable_approximate_stop_zone:
                    red = np.zeros_like(preview)
                    cv2.fillPoly(red, [np.int32(stop_points)], (0, 0, 255))
                    preview = cv2.addWeighted(preview, 1.0, red, 0.18, 0.0)
                    cv2.polylines(
                        preview, [np.int32(stop_points)], True, (0, 0, 255), 3, cv2.LINE_AA
                    )
                    anchor = tuple(np.int32(stop_points[-1]))
                    cv2.putText(
                        preview, "APPROX 1m STOP", anchor,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA,
                    )
                for track in tracks:
                    foot_overlap_ratio = (
                        mask_foot_zone_overlap_ratio(track.mask, stop_mask)
                        if track.visible
                        else 0.0
                    )
                    inside_stop = bool(
                        self.enable_approximate_stop_zone
                        and track.visible
                        and (
                            point_in_normalized_polygon(
                                track.footpoint_uv,
                                self.approximate_stop_polygons[side],
                                frame.shape[1],
                                frame.shape[0],
                            )
                            or foot_overlap_ratio
                            >= self.approximate_stop_foot_overlap_ratio
                        )
                    )
                    item = {
                        "track_id": track.track_id,
                        "bbox_xyxy": [round(float(v), 2) for v in track.bbox],
                        "footpoint_uv": [round(float(v), 2) for v in track.footpoint_uv],
                        "confidence": round(float(track.confidence), 3),
                        "visible": bool(track.visible),
                        "missed_frames": int(track.missed_frames),
                        "age_frames": int(track.age_frames),
                        "inside_approximate_stop_zone": inside_stop,
                        "stop_zone_foot_overlap_ratio": round(foot_overlap_ratio, 3),
                    }
                    serialized.append(item)
                    if track.visible:
                        color = (0, 220, 0)
                        overlay = np.zeros_like(preview)
                        overlay[track.mask] = color
                        preview = cv2.addWeighted(preview, 1.0, overlay, 0.32, 0.0)
                        x1, y1, x2, y2 = np.int32(track.bbox)
                        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            preview,
                            f"{track.track_id} {track.confidence:.2f}",
                            (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                report["cameras"][side] = {
                    "stamp_s": stamp,
                    "frame_width": frame.shape[1],
                    "frame_height": frame.shape[0],
                    "raw_detection_count": raw_detection_count,
                    "deduplicated_detection_count": len(detections),
                    "duplicates_suppressed": raw_detection_count - len(detections),
                    "walkway_detection_enabled": self.enable_walkway_detection,
                    "walkway_prompt": self.walkway_prompt,
                    "walkway_polygon_uv": walkway_polygon,
                    "approximate_stop_zone_enabled": self.enable_approximate_stop_zone,
                    "approximate_stop_zone_metric": False,
                    "approximate_stop_polygon_uv": [
                        [round(float(x), 2), round(float(y), 2)] for x, y in stop_points
                    ],
                    "tracks": serialized,
                }
                ok, encoded = cv2.imencode(".jpg", preview)
                if ok:
                    message = CompressedImage(format="jpeg", data=encoded.tobytes())
                    self.preview_pubs[side].publish(message)
            report["inference_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            message = String()
            message.data = json.dumps(report)
            self.tracks_pub.publish(message)
            status = String()
            status.data = json.dumps(
                {
                    "state": "ready",
                    "backend": self.backend_name,
                    "model_id": self.model_id if self.backend is not None else None,
                    "model_revision": (None if self.backend is None else self.backend.revision),
                    "paused": self.paused,
                    "inference_ms": report["inference_ms"],
                }
            )
            self.status_pub.publish(status)
        except Exception as exc:
            self.get_logger().error(f"SAM3 processing failed: {exc}")
            status = String()
            status.data = json.dumps({"state": "error", "error": str(exc)})
            self.status_pub.publish(status)
        finally:
            with self.lock:
                self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = Sam3PersonTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
