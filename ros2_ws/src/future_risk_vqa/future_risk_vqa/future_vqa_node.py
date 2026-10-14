#!/usr/bin/env python3
"""Ask the VLM only for future semantic event interpretation."""

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
from std_msgs.msg import Bool, String

from .model_client import LlavaWorkerClient
from .motion import estimate_camera_motion_hint
from .prompt import build_future_event_prompt, wrap_vicuna_prompt
from .qos import pipeline_pause_qos
from .schema import extract_future_event_report, validate_future_event_report
from .semantic_quality import (
    apply_effective_confidence,
    apply_tracking_motion,
    detect_output_collapse,
)
from .storyboard import (
    FrameBuffer,
    TrackHistoryBuffer,
    build_hybrid_storyboard,
    build_storyboard,
    build_track_crop_storyboard,
    draw_detection_snapshot,
)


def active_agents(geometry):
    """Return people visible in the VQA snapshot, excluding retained missed tracks."""
    return [
        item
        for item in geometry.get("tracked_agents", [])
        if bool(item.get("visible", int(item.get("missed_frames", 0)) == 0))
    ]


def compressed_message(jpeg_bytes, stamp_s):
    message = CompressedImage(format="jpeg", data=jpeg_bytes)
    seconds = int(float(stamp_s))
    nanoseconds = int(round((float(stamp_s) - seconds) * 1e9))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    message.header.stamp.sec = seconds
    message.header.stamp.nanosec = nanoseconds
    return message


class FutureVQANode(Node):
    def __init__(self):
        super().__init__("future_vqa")
        defaults = {
            "enable_future_vqa": True,
            "include_tracking_context": True,
            "show_sam3_detection_overlay": True,
            "trajectory_context_mode": "none",
            "storyboard_mode": "hybrid",
            "vqa_inference_mode": "joint",
            "per_track_max_new_tokens": 128,
            "track_history_tolerance_s": 0.75,
            "track_crop_padding_ratio": 0.30,
            "track_crop_cell_size": 224,
            "track_crop_max_agents": 6,
            "motion_source": "tracking_hint",
            "motion_scale_change_threshold": 0.12,
            "motion_lateral_displacement_threshold": 0.35,
            "motion_stationary_scale_threshold": 0.04,
            "motion_stationary_lateral_threshold": 0.12,
            "enable_collapse_detection": False,
            "collapse_min_identical_tracks": 3,
            "enable_effective_confidence": True,
            "unreliable_tracking_confidence_scale": 0.50,
            "left_image_topic": "/left_camera/rgb",
            "right_image_topic": "/right_camera/rgb",
            "use_compressed_image": False,
            "backend": "llava_worker",
            "controller_url": "http://localhost:21001",
            "model_name": "",
            "request_timeout_s": 90.0,
            "request_retry_count": 2,
            "request_retry_backoff_s": 1.0,
            "max_new_tokens": 512,
            "use_cache": False,
            "vqa_hz": 0.5,
            "temporal_frame_count": 3,
            "temporal_window_s": 2.0,
            "jpeg_quality": 90,
            "work_context": "A robot may operate inside the red swept-volume zone.",
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        value = lambda name: self.get_parameter(name).value
        self.bridge = CvBridge()
        self.enable_future_vqa = bool(value("enable_future_vqa"))
        self.include_tracking_context = bool(value("include_tracking_context"))
        self.show_sam3_detection_overlay = bool(value("show_sam3_detection_overlay"))
        self.trajectory_context_mode = str(value("trajectory_context_mode")).lower()
        if self.trajectory_context_mode not in {"none", "overlay", "numeric", "both"}:
            raise ValueError("trajectory_context_mode must be none, overlay, numeric, or both")
        self.storyboard_mode = str(value("storyboard_mode")).lower()
        if self.storyboard_mode not in {"global", "hybrid", "track_crops"}:
            raise ValueError("storyboard_mode must be global, hybrid, or track_crops")
        self.vqa_inference_mode = str(value("vqa_inference_mode")).lower()
        if self.vqa_inference_mode not in {"joint", "per_track"}:
            raise ValueError("vqa_inference_mode must be joint or per_track")
        if self.vqa_inference_mode == "per_track" and self.storyboard_mode != "track_crops":
            raise ValueError("per_track inference requires storyboard_mode=track_crops")
        self.per_track_max_new_tokens = max(
            128, int(value("per_track_max_new_tokens"))
        )
        self.track_history_tolerance_s = max(
            0.0, float(value("track_history_tolerance_s"))
        )
        self.track_crop_padding_ratio = max(
            0.0, float(value("track_crop_padding_ratio"))
        )
        self.track_crop_cell_size = max(96, int(value("track_crop_cell_size")))
        self.track_crop_max_agents = max(1, int(value("track_crop_max_agents")))
        self.motion_source = str(value("motion_source")).lower()
        if self.motion_source not in {"vlm", "tracking_hint"}:
            raise ValueError("motion_source must be vlm or tracking_hint")
        self.motion_thresholds = {
            "scale_change_threshold": float(value("motion_scale_change_threshold")),
            "lateral_displacement_threshold": float(
                value("motion_lateral_displacement_threshold")
            ),
            "stationary_scale_threshold": float(
                value("motion_stationary_scale_threshold")
            ),
            "stationary_lateral_threshold": float(
                value("motion_stationary_lateral_threshold")
            ),
        }
        self.enable_collapse_detection = bool(value("enable_collapse_detection"))
        self.collapse_min_identical_tracks = max(
            2, int(value("collapse_min_identical_tracks"))
        )
        self.enable_effective_confidence = bool(value("enable_effective_confidence"))
        self.unreliable_tracking_confidence_scale = min(
            1.0,
            max(0.0, float(value("unreliable_tracking_confidence_scale"))),
        )
        self.compressed = bool(value("use_compressed_image"))
        self.backend = str(value("backend")).lower()
        if not self.enable_future_vqa:
            self.backend = "disabled"
        if self.backend not in {"llava_worker", "disabled"}:
            raise ValueError("backend must be 'llava_worker' or 'disabled'")
        self.client = LlavaWorkerClient(
            value("controller_url"),
            value("model_name"),
            value("request_timeout_s"),
            value("max_new_tokens"),
            value("use_cache"),
            value("request_retry_count"),
            value("request_retry_backoff_s"),
        )
        self.count = int(value("temporal_frame_count"))
        self.window = float(value("temporal_window_s"))
        self.quality = int(value("jpeg_quality"))
        self.work_context = str(value("work_context"))
        self.buffers = {"left": FrameBuffer(), "right": FrameBuffer()}
        self.track_history = TrackHistoryBuffer()
        self.geometry, self.lock, self.busy = None, threading.Lock(), False
        self.paused = False
        self.last_geometry_stamp_started = None
        image_type = CompressedImage if self.compressed else Image
        self.create_subscription(
            image_type,
            str(value("left_image_topic")),
            lambda msg: self._image("left", msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            image_type,
            str(value("right_image_topic")),
            lambda msg: self._image("right", msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(String, "/future_risk/geometry_prediction", self._geometry, 10)
        self.create_subscription(String, "/future_risk/person_tracks", self._tracks, 10)
        self.create_subscription(
            Bool,
            "/future_risk/pipeline_paused",
            self._set_paused,
            pipeline_pause_qos(),
        )
        self.events_pub = self.create_publisher(String, "/future_risk/vlm_events", 10)
        self.raw_pub = self.create_publisher(String, "/future_risk/vlm_raw_response", 10)
        self.status_pub = self.create_publisher(String, "/future_risk/vqa_status", 10)
        self.storyboard_pub = self.create_publisher(
            CompressedImage, "/future_risk/storyboard/compressed", qos_profile_sensor_data
        )
        self.snapshot_pubs = {
            side: self.create_publisher(
                CompressedImage,
                f"/future_risk/vqa_input/{side}_now/compressed",
                qos_profile_sensor_data,
            )
            for side in ("left", "right")
        }
        self.create_timer(1.0 / max(0.05, float(value("vqa_hz"))), self._start)
        self.get_logger().info(
            f"Future VQA ready: backend={self.backend}, {self.count} frames/view "
            f"over {self.window:.1f}s, storyboard={self.storyboard_mode}, "
            f"inference={self.vqa_inference_mode}, motion={self.motion_source}"
        )

    @staticmethod
    def _stamp(message):
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def _image(self, side, message):
        try:
            if self.compressed:
                frame = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            now = self.get_clock().now().nanoseconds * 1e-9
            with self.lock:
                self.buffers[side].append(self._stamp(message), now, frame)
        except Exception as exc:
            self.get_logger().error(f"{side} temporal image failed: {exc}")

    def _geometry(self, message):
        try:
            geometry = json.loads(message.data)
            with self.lock:
                self.geometry = geometry
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Invalid geometry JSON: {exc}")

    def _tracks(self, message):
        try:
            report = json.loads(message.data)
            with self.lock:
                self.track_history.append_report(report)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.get_logger().error(f"Invalid person-track history JSON: {exc}")

    def _set_paused(self, message):
        with self.lock:
            self.paused = bool(message.data)
            busy = self.busy
        timing = "after current inference" if busy and self.paused else "immediately"
        self.get_logger().info(f"VQA inference {'paused' if self.paused else 'resumed'} {timing}")

    def _start(self):
        with self.lock:
            if self.paused or self.busy or self.geometry is None:
                return
            geometry_stamp = self.geometry.get("stamp_s")
            if geometry_stamp == self.last_geometry_stamp_started:
                return
            camera_stamps = self.geometry.get("camera_stamps_s", {})
            left = self.buffers["left"].sample(
                self.count, self.window, camera_stamps.get("left", geometry_stamp)
            )
            right = self.buffers["right"].sample(
                self.count, self.window, camera_stamps.get("right", geometry_stamp)
            )
            if len(left) != self.count or len(right) != self.count:
                return
            self.busy = True
            self.last_geometry_stamp_started = geometry_stamp
            geometry = json.loads(json.dumps(self.geometry))
            track_samples = self._track_samples(left, right, geometry)
        threading.Thread(
            target=self._run,
            args=(left, right, geometry, track_samples),
            daemon=True,
        ).start()

    def _track_samples(self, left, right, geometry):
        """Resolve the bbox for each ID at each selected frame timestamp."""
        if self.storyboard_mode not in {"hybrid", "track_crops"}:
            return {}
        frames_by_camera = {"left": left, "right": right}
        samples_by_track = {}
        for agent in active_agents(geometry):
            track_id = str(agent.get("track_id"))
            camera = str(agent.get("camera"))
            frames = frames_by_camera.get(camera, [])
            samples = [
                self.track_history.sample(
                    camera,
                    track_id,
                    frame.stamp_s,
                    self.track_history_tolerance_s,
                )
                for frame in frames
            ]
            # Geometry and NOW are generated from the same tracker report. Use its
            # bbox as a deterministic last-resort when ROS callback order means the
            # history subscription has not received that report yet.
            bbox = agent.get("bbox_xyxy", [])
            if (
                frames
                and samples[-1] is None
                and isinstance(bbox, (list, tuple))
                and len(bbox) == 4
            ):
                samples[-1] = {
                    "bbox_xyxy": tuple(float(value) for value in bbox),
                    "visible": True,
                    "confidence": float(agent.get("detection_confidence", 0.0)),
                    "report_stamp_s": frames[-1].stamp_s,
                    "frame_delta_s": 0.0,
                    "source": "geometry_now_fallback",
                }
            samples_by_track[track_id] = samples
        return samples_by_track

    def _motion_hints(self, left, right, geometry, track_samples):
        frames_by_camera = {"left": left, "right": right}
        hints = {}
        for agent in active_agents(geometry):
            track_id = str(agent.get("track_id"))
            camera = str(agent.get("camera"))
            frames = frames_by_camera.get(camera, [])
            frame_shape = frames[-1].image.shape if frames else (1, 1, 3)
            hints[track_id] = estimate_camera_motion_hint(
                track_samples.get(track_id, []),
                frame_shape,
                **self.motion_thresholds,
            )
        return hints

    def _semantic_context(
        self, geometry, storyboard_mode=None, motion_hints=None
    ):
        zone_logic_enabled = bool(geometry.get("zone_logic_enabled", False))
        agents = active_agents(geometry)
        context = {
            "zone_logic_enabled": zone_logic_enabled,
            "trajectory_context_mode": self.trajectory_context_mode,
            "storyboard_mode": storyboard_mode or self.storyboard_mode,
            "tracks": [
                {
                    "track_id": item.get("track_id"),
                    "camera": item.get("camera"),
                    **(
                        {
                            "trusted_camera_motion_hint": motion_hints.get(
                                str(item.get("track_id")), {}
                            )
                        }
                        if motion_hints is not None
                        else {}
                    ),
                    **(
                        {
                            "tracking_reliable": item.get("tracking_reliable"),
                            "track_age_frames": item.get("track_age_frames"),
                        }
                        if self.include_tracking_context
                        else {}
                    ),
                }
                for item in agents
            ],
        }
        if self.trajectory_context_mode in {"numeric", "both"}:
            by_id = {item.get("track_id"): item for item in agents}
            for track in context["tracks"]:
                agent = by_id.get(track["track_id"], {})
                track["trusted_motion_prediction"] = {
                    "velocity": agent.get("velocity"),
                    "future_path": agent.get("future_path", []),
                }
        if zone_logic_enabled:
            context["calibration_mode"] = geometry.get("calibration_mode")
            context["units"] = geometry.get("units")
            context["prediction_horizon_s"] = geometry.get("prediction_horizon_s")
            by_id = {item.get("track_id"): item for item in agents}
            for track in context["tracks"]:
                agent = by_id.get(track["track_id"], {})
                track.update(
                    {
                        "current_zone": agent.get("current_zone"),
                        "trajectory_intersects_hazard": agent.get("trajectory_intersects_hazard"),
                        "time_to_entry_s": agent.get("time_to_entry_s"),
                    }
                )
        return context

    @staticmethod
    def _fallback(geometry, error):
        agents = active_agents(geometry)
        return {
            "schema_version": "3.1",
            "availability": "UNAVAILABLE",
            "unavailable_reason": str(error),
            "critical_track_id": None,
            "assessments": [
                {
                    "track_id": item.get("track_id"),
                    "camera": item.get("camera"),
                    "current_observation": {
                        "body_orientation": "UNCERTAIN",
                        "body_posture": "UNCERTAIN",
                        "current_activity": "UNCERTAIN",
                        "motion_trend": "UNCERTAIN",
                        "confidence": None,
                        "evidence": "semantic model unavailable",
                    },
                    "future_prediction": {
                        "future_motion": "UNCERTAIN",
                        "confidence": None,
                        "evidence": "semantic model unavailable",
                    },
                    "confidence": None,
                    "evidence": "semantic model unavailable",
                }
                for item in agents
            ],
            "scene_summary": "Semantic future interpretation is unavailable.",
            "validation_errors": [],
            "normalizations": [],
        }

    def _encode_storyboard(self, image):
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        if not ok:
            raise RuntimeError("could not encode VLM storyboard")
        return encoded.tobytes()

    @staticmethod
    def _one_agent_geometry(geometry, agent):
        selected = json.loads(json.dumps(geometry))
        selected["tracked_agents"] = [json.loads(json.dumps(agent))]
        return selected

    def _infer_joint(self, geometry, jpeg, storyboard_mode, motion_hints):
        track_ids = [item.get("track_id") for item in active_agents(geometry)]
        user_prompt = build_future_event_prompt(
            self.work_context,
            self._semantic_context(
                geometry,
                storyboard_mode,
                motion_hints if self.motion_source == "tracking_hint" else None,
            ),
        )
        raw, metadata = self.client.generate(wrap_vicuna_prompt(user_prompt), jpeg)
        report = validate_future_event_report(
            extract_future_event_report(raw), track_ids
        )
        return report, raw, metadata

    def _infer_per_track(self, left, right, geometry, track_samples, motion_hints):
        reports = []
        raw_responses = []
        request_metadata = []
        for agent in active_agents(geometry):
            track_id = str(agent.get("track_id"))
            selected_geometry = self._one_agent_geometry(geometry, agent)
            row, crop_metadata = build_track_crop_storyboard(
                left,
                right,
                selected_geometry,
                {track_id: track_samples.get(track_id, [])},
                crop_padding_ratio=self.track_crop_padding_ratio,
                cell_size=self.track_crop_cell_size,
                maximum_agents=1,
            )
            prompt_context = self._semantic_context(
                selected_geometry,
                "track_crops",
                (
                    {track_id: motion_hints.get(track_id, {})}
                    if self.motion_source == "tracking_hint"
                    else None
                ),
            )
            raw, metadata = self.client.generate(
                wrap_vicuna_prompt(
                    build_future_event_prompt(self.work_context, prompt_context)
                ),
                self._encode_storyboard(row),
                max_new_tokens=self.per_track_max_new_tokens,
            )
            validated = validate_future_event_report(
                extract_future_event_report(raw), [track_id]
            )
            reports.append(validated)
            raw_responses.append({"track_id": track_id, "response": raw})
            request_metadata.append(
                {
                    "track_id": track_id,
                    **metadata,
                    "crop_coverage": crop_metadata.get(
                        "track_crop_coverage", {}
                    ).get(track_id, 0),
                }
            )

        aggregate = {
            "schema_version": "3.1",
            "critical_track_id": None,
            "assessments": [
                assessment
                for item in reports
                for assessment in item.get("assessments", [])
            ],
            "scene_summary": " | ".join(
                item.get("scene_summary", "")
                for item in reports
                if item.get("scene_summary")
            )[:500],
            "validation_errors": [
                f"{track_id}: {error}"
                for track_id, item in zip(
                    [str(agent.get("track_id")) for agent in active_agents(geometry)],
                    reports,
                )
                for error in item.get("validation_errors", [])
            ],
            "normalizations": [
                f"{track_id}: {normalization}"
                for track_id, item in zip(
                    [str(agent.get("track_id")) for agent in active_agents(geometry)],
                    reports,
                )
                for normalization in item.get("normalizations", [])
            ],
        }
        raw_envelope = json.dumps(
            {
                "inference_mode": "per_track",
                "responses": raw_responses,
            },
            ensure_ascii=False,
        )
        metadata = {
            "request_count": len(request_metadata),
            "per_track_requests": request_metadata,
            "round_trip_ms": round(
                sum(float(item.get("round_trip_ms", 0.0)) for item in request_metadata),
                3,
            ),
            "models": sorted(
                {
                    str(item.get("model"))
                    for item in request_metadata
                    if item.get("model")
                }
            ),
        }
        return aggregate, raw_envelope, metadata

    def _apply_quality_controls(
        self, report, geometry, evidence, motion_hints
    ):
        agents = {str(item.get("track_id")): item for item in active_agents(geometry)}
        coverage = evidence.get("track_crop_coverage", {})
        expected = max(1, int(evidence.get("sample_count_per_track", self.count)))
        for assessment in report.get("assessments", []):
            track_id = str(assessment.get("track_id"))
            agent = agents.get(track_id, {})
            apply_tracking_motion(
                assessment,
                motion_hints.get(track_id, {}),
                mode=self.motion_source,
            )
            apply_effective_confidence(
                assessment,
                float(coverage.get(track_id, expected)) / expected,
                bool(agent.get("tracking_reliable", False)),
                enabled=self.enable_effective_confidence,
                unreliable_tracking_scale=self.unreliable_tracking_confidence_scale,
            )

        warnings = []
        if self.enable_collapse_detection:
            for track_ids in detect_output_collapse(
                report.get("assessments", []),
                self.collapse_min_identical_tracks,
            ):
                warning = {
                    "code": "semantic_output_collapse",
                    "track_ids": track_ids,
                    "explanation": (
                        "Multiple IDs received an identical model-originated semantic, "
                        "confidence, and evidence signature."
                    ),
                }
                warnings.append(warning)
        report["quality_warnings"] = warnings
        return report

    def _run(self, left, right, geometry, track_samples):
        started = time.perf_counter()
        encoded_storyboard = None
        snapshots = {}
        evidence = {}
        agents = active_agents(geometry)
        try:
            storyboard_geometry = json.loads(json.dumps(geometry))
            storyboard_geometry["show_prediction_overlay"] = self.trajectory_context_mode in {
                "overlay",
                "both",
            }
            storyboard_geometry["show_sam3_detection_overlay"] = (
                self.show_sam3_detection_overlay
            )
            if self.storyboard_mode == "hybrid" and agents:
                storyboard, evidence = build_hybrid_storyboard(
                    left,
                    right,
                    storyboard_geometry,
                    track_samples,
                    crop_padding_ratio=self.track_crop_padding_ratio,
                    cell_size=self.track_crop_cell_size,
                    maximum_agents=self.track_crop_max_agents,
                )
            elif self.storyboard_mode == "track_crops" and agents:
                storyboard, evidence = build_track_crop_storyboard(
                    left,
                    right,
                    storyboard_geometry,
                    track_samples,
                    crop_padding_ratio=self.track_crop_padding_ratio,
                    cell_size=self.track_crop_cell_size,
                    maximum_agents=(
                        max(self.track_crop_max_agents, len(agents))
                        if self.vqa_inference_mode == "per_track"
                        else self.track_crop_max_agents
                    ),
                )
            else:
                storyboard, evidence = build_storyboard(
                    left, right, storyboard_geometry
                )
                evidence["storyboard_mode"] = "global"
            encoded_storyboard = self._encode_storyboard(storyboard)
            motion_hints = self._motion_hints(
                left, right, geometry, track_samples
            )
            for side, frames in (("left", left), ("right", right)):
                now_image = draw_detection_snapshot(
                    frames[-1].image, side, storyboard_geometry
                )
                snapshot_ok, snapshot = cv2.imencode(
                    ".jpg", now_image, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                if not snapshot_ok:
                    raise RuntimeError(f"could not encode {side} VQA snapshot")
                snapshots[side] = (snapshot.tobytes(), frames[-1].stamp_s)
            track_ids = [item.get("track_id") for item in agents]
            metadata = {}
            if not track_ids:
                report = {
                    "schema_version": "3.1",
                    "availability": "READY",
                    "critical_track_id": None,
                    "assessments": [],
                    "scene_summary": "No visible person is detected.",
                    "validation_errors": [],
                    "normalizations": [],
                }
                metadata["model_skipped"] = "no_visible_person"
                skipped = String()
                skipped.data = json.dumps({"model_skipped": "no_visible_person"})
                self.raw_pub.publish(skipped)
            elif self.backend == "disabled":
                raise RuntimeError("future VQA backend is disabled")
            else:
                if self.vqa_inference_mode == "per_track":
                    report, raw, metadata = self._infer_per_track(
                        left,
                        right,
                        geometry,
                        track_samples,
                        motion_hints,
                    )
                else:
                    report, raw, metadata = self._infer_joint(
                        geometry,
                        encoded_storyboard,
                        evidence.get("storyboard_mode", "global"),
                        motion_hints,
                    )
                raw_message = String()
                raw_message.data = raw
                self.raw_pub.publish(raw_message)
                report = self._apply_quality_controls(
                    report, geometry, evidence, motion_hints
                )
                report["availability"] = "READY"
            report["metadata"] = {
                **metadata,
                **evidence,
                "pipeline_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "geometry_stamp_s": geometry.get("stamp_s"),
                "snapshot_stamps_s": {
                    side: round(float(stamp_s), 6)
                    for side, (_, stamp_s) in snapshots.items()
                },
                "prompt_contract": "compact_hybrid_posture_future_v3_2",
                "configured_storyboard_mode": self.storyboard_mode,
                "vqa_inference_mode": self.vqa_inference_mode,
                "motion_source": self.motion_source,
                "motion_hints": motion_hints if track_ids else {},
                "effective_confidence_enabled": self.enable_effective_confidence,
                "use_cache": self.client.use_cache,
                "collapse_detection_enabled": self.enable_collapse_detection,
                "input_tracks": [
                    {
                        "track_id": item.get("track_id"),
                        "camera": item.get("camera"),
                        "visible": True,
                        "tracking_reliable": item.get("tracking_reliable", False),
                        "missed_frames": 0,
                    }
                    for item in agents
                ],
            }
            if report["validation_errors"]:
                state = "schema_warning"
            elif report.get("quality_warnings"):
                state = "quality_warning"
            else:
                state = "ready"
        except Exception as exc:
            report = self._fallback(geometry, exc)
            report["metadata"] = {
                "geometry_stamp_s": geometry.get("stamp_s"),
                "prompt_contract": "camera_relative_posture_current_future_flat_v3_1",
                "configured_storyboard_mode": self.storyboard_mode,
                "vqa_inference_mode": self.vqa_inference_mode,
                "motion_source": self.motion_source,
                "use_cache": self.client.use_cache,
                "pipeline_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "input_tracks": [
                    {
                        "track_id": item.get("track_id"),
                        "camera": item.get("camera"),
                        "visible": True,
                        "tracking_reliable": item.get("tracking_reliable", False),
                        "missed_frames": 0,
                    }
                    for item in active_agents(geometry)
                ],
            }
            state = "unavailable"
            self.get_logger().warning(f"Future VQA unavailable: {exc}")
        for side, (jpeg_bytes, stamp_s) in snapshots.items():
            self.snapshot_pubs[side].publish(compressed_message(jpeg_bytes, stamp_s))
        if encoded_storyboard is not None:
            storyboard_stamp = max(left[-1].stamp_s, right[-1].stamp_s)
            self.storyboard_pub.publish(
                compressed_message(encoded_storyboard, storyboard_stamp)
            )
        output = String()
        output.data = json.dumps(report, ensure_ascii=False)
        self.events_pub.publish(output)
        status = String()
        status.data = json.dumps(
            {
                "state": state,
                "backend": self.backend,
                "paused": self.paused,
                "trajectory_context_mode": self.trajectory_context_mode,
                "vqa_inference_mode": self.vqa_inference_mode,
                "motion_source": self.motion_source,
                "storyboard_mode": evidence.get(
                    "storyboard_mode", self.storyboard_mode
                ),
                "pipeline_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        self.status_pub.publish(status)
        with self.lock:
            self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = FutureVQANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
