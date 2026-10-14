#!/usr/bin/env python3
"""Append research outputs to a run-scoped JSONL log without changing decisions."""

from datetime import datetime, timezone
import json
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

from .qos import pipeline_pause_qos


class ExperimentLoggerNode(Node):
    def __init__(self):
        super().__init__("future_risk_experiment_logger")
        self.declare_parameter("output_root", "/workspace/results/future_risk_vqa")
        self.declare_parameter("run_name", "video_demo")
        self.declare_parameter("save_storyboards", True)
        root = str(self.get_parameter("output_root").value)
        name = str(self.get_parameter("run_name").value)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(root, f"{stamp}_{name}")
        os.makedirs(os.path.join(self.run_dir, "storyboards"), exist_ok=False)
        self.file = open(
            os.path.join(self.run_dir, "events.jsonl"), "a", encoding="utf-8", buffering=1
        )
        self.counts = {}
        for label, topic in (
            ("person_tracks", "/future_risk/person_tracks"),
            ("geometry_prediction", "/future_risk/geometry_prediction"),
            ("vlm_events", "/future_risk/vlm_events"),
            ("vlm_raw_response", "/future_risk/vlm_raw_response"),
            ("decision", "/future_risk/decision"),
            ("sam3_status", "/future_risk/sam3_status"),
            ("vqa_status", "/future_risk/vqa_status"),
        ):
            self.create_subscription(
                String, topic, lambda message, key=label: self._record(key, message.data), 50
            )
        if bool(self.get_parameter("save_storyboards").value):
            self.create_subscription(
                CompressedImage,
                "/future_risk/storyboard/compressed",
                self._storyboard,
                qos_profile_sensor_data,
            )
        self.create_subscription(
            Bool,
            "/future_risk/pipeline_paused",
            lambda message: self._record(
                "pipeline_paused", json.dumps({"paused": bool(message.data)})
            ),
            pipeline_pause_qos(),
        )
        with open(os.path.join(self.run_dir, "manifest.json"), "w", encoding="utf-8") as manifest:
            json.dump(
                {
                    "schema_version": "1.0",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "run_name": name,
                    "events_file": "events.jsonl",
                },
                manifest,
                indent=2,
            )
        self.get_logger().info(f"Experiment log: {self.run_dir}")

    def _record(self, event_type, payload):
        self.counts[event_type] = self.counts.get(event_type, 0) + 1
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            decoded = payload
        record = {
            "receipt_time_s": self.get_clock().now().nanoseconds * 1e-9,
            "event_type": event_type,
            "sequence": self.counts[event_type],
            "payload": decoded,
        }
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _storyboard(self, message):
        sequence = self.counts.get("storyboard", 0) + 1
        self.counts["storyboard"] = sequence
        filename = f"storyboards/{sequence:06d}.jpg"
        with open(os.path.join(self.run_dir, filename), "wb") as image_file:
            image_file.write(bytes(message.data))
        self._record("storyboard_saved", json.dumps({"filename": filename}))

    def destroy_node(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
