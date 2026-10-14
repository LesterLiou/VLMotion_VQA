#!/usr/bin/env python3
"""Publish two prerecorded videos on one synchronized ROS clock."""

import json
import os

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from .qos import pipeline_pause_qos


class _Reader:
    def __init__(self, path, offset_s):
        self.path = path
        self.offset_s = max(0.0, float(offset_s))
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_s = count / self.fps - self.offset_s
        self.reset()

    def reset(self):
        self.next_index = int(round(self.offset_s * self.fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.next_index)

    def read_at(self, time_s):
        target = int(round((self.offset_s + time_s) * self.fps))
        while self.next_index < target:
            if not self.cap.grab():
                return None
            self.next_index += 1
        ok, frame = self.cap.read()
        self.next_index += 1
        return frame if ok else None


class DualVideoSourceNode(Node):
    def __init__(self):
        super().__init__("future_risk_video_source")
        defaults = {
            "left_video_path": "/workspace/example/video/0610_take3n4.mp4",
            "right_video_path": "/workspace/example/video/0610x9_v0629.mp4",
            "left_image_topic": "/left_camera/rgb",
            "right_image_topic": "/right_camera/rgb",
            "left_frame_id": "left_camera_optical_frame",
            "right_frame_id": "right_camera_optical_frame",
            "use_compressed_image": False,
            "loop_videos": True,
            "playback_rate": 1.0,
            "output_fps": 10.0,
            "output_width": 960,
            "output_height": 540,
            "left_start_offset_s": 0.0,
            "right_start_offset_s": 0.0,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        value = lambda name: self.get_parameter(name).value
        self.left_path, self.right_path = str(value("left_video_path")), str(
            value("right_video_path")
        )
        for path in (self.left_path, self.right_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"video does not exist: {path}")
        self.left = _Reader(self.left_path, value("left_start_offset_s"))
        self.right = _Reader(self.right_path, value("right_start_offset_s"))
        self.duration = min(self.left.duration_s, self.right.duration_s)
        self.rate = max(0.05, float(value("playback_rate")))
        self.fps = max(1.0, float(value("output_fps")))
        self.width, self.height = int(value("output_width")), int(value("output_height"))
        self.loop = bool(value("loop_videos"))
        self.compressed = bool(value("use_compressed_image"))
        self.bridge = CvBridge()
        msg_type = CompressedImage if self.compressed else Image
        self.left_pub = self.create_publisher(
            msg_type, str(value("left_image_topic")), qos_profile_sensor_data
        )
        self.right_pub = self.create_publisher(
            msg_type, str(value("right_image_topic")), qos_profile_sensor_data
        )
        self.status_pub = self.create_publisher(String, "/future_risk/video_status", 10)
        self.pause_pub = self.create_publisher(
            Bool, "/future_risk/pipeline_paused", pipeline_pause_qos()
        )
        self.create_service(SetBool, "/future_risk/set_video_paused", self._pause)
        self.timeline, self.index, self.paused = 0.0, 0, False
        initial_pause = Bool()
        initial_pause.data = False
        self.pause_pub.publish(initial_pause)
        self.timer = self.create_timer(1.0 / self.fps, self._tick)
        self.get_logger().info(
            f"Video source: left={self.left_path}, right={self.right_path}, "
            f"{self.fps:.1f} FPS, duration={self.duration:.2f}s"
        )

    def _pause(self, request, response):
        self.paused = bool(request.data)
        pause_message = Bool()
        pause_message.data = self.paused
        self.pause_pub.publish(pause_message)
        self._publish_status("paused" if self.paused else "running")
        response.success = True
        response.message = (
            "Video paused; the in-flight inference may finish, then results freeze."
            if self.paused
            else "Video and inference resumed."
        )
        return response

    def _publish_status(self, state):
        status = String()
        status.data = json.dumps(
            {
                "state": state,
                "timeline_s": self.timeline,
                "index": self.index,
                "paused": self.paused,
                "inference_gate": "paused_after_in_flight" if self.paused else "running",
            }
        )
        self.status_pub.publish(status)

    def _reset(self):
        self.left.reset()
        self.right.reset()
        self.timeline, self.index = 0.0, 0

    def _publish(self, pub, frame, stamp, frame_id):
        if self.width > 0 and self.height > 0:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.compressed:
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                return
            msg = CompressedImage(format="jpeg", data=encoded.tobytes())
        else:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp, msg.header.frame_id = stamp, frame_id
        pub.publish(msg)

    def _tick(self):
        if self.paused:
            return
        if self.timeline >= self.duration:
            if not self.loop:
                self.timer.cancel()
                return
            self._reset()
        left, right = self.left.read_at(self.timeline), self.right.read_at(self.timeline)
        if left is None or right is None:
            self._reset()
            return
        stamp = self.get_clock().now().to_msg()
        self._publish(self.left_pub, left, stamp, "left_camera_optical_frame")
        self._publish(self.right_pub, right, stamp, "right_camera_optical_frame")
        self.index += 1
        self.timeline += self.rate / self.fps
        self._publish_status("running")


def main(args=None):
    rclpy.init(args=args)
    node = DualVideoSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.left.cap.release()
        node.right.cap.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
