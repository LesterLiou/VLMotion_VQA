#!/usr/bin/env python3
"""Map tracked foot points to ground coordinates and predict short trajectories."""

import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .geometry import (
    classify_zone,
    distance_to_polygon,
    homography_matrix,
    image_to_ground,
    point_in_polygon,
)
from .motion import KalmanCV2D


def _pairs(values):
    numbers = [float(value) for value in values]
    if len(numbers) < 6 or len(numbers) % 2:
        raise ValueError("polygon parameter needs at least three x,y pairs")
    return [numbers[index : index + 2] for index in range(0, len(numbers), 2)]


class GroundTrajectoryNode(Node):
    def __init__(self):
        super().__init__("ground_trajectory")
        defaults = {
            "enable_motion_prediction": True,
            "enable_zone_logic": False,
            "show_track_id_overlay": True,
            "show_prediction_overlay": False,
            "calibration_mode": "normalized_debug",
            "calibration_valid": True,
            "left_homography": [1.0 / 960.0, 0.0, 0.0, 0.0, -1.0 / 540.0, 1.0, 0.0, 0.0, 1.0],
            "right_homography": [1.0 / 960.0, 0.0, 0.0, 0.0, -1.0 / 540.0, 1.0, 0.0, 0.0, 1.0],
            "hazard_polygon": [0.35, 0.0, 0.65, 0.0, 0.65, 0.28, 0.35, 0.28],
            "warning_polygon": [0.20, 0.0, 0.80, 0.0, 0.80, 0.48, 0.20, 0.48],
            "prediction_horizon_s": 3.0,
            "prediction_step_s": 0.2,
            "process_noise": 0.5,
            "measurement_noise": 0.08,
            "minimum_track_age_frames": 2,
            "max_track_gap_s": 1.5,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        value = lambda name: self.get_parameter(name).value
        self.motion_prediction_enabled = bool(value("enable_motion_prediction"))
        self.zone_logic_enabled = bool(value("enable_zone_logic"))
        self.show_track_id_overlay = bool(value("show_track_id_overlay"))
        self.show_prediction_overlay = bool(value("show_prediction_overlay"))
        self.mode = str(value("calibration_mode"))
        self.valid = bool(value("calibration_valid"))
        self.homographies = {
            "left": homography_matrix(value("left_homography")),
            "right": homography_matrix(value("right_homography")),
        }
        self.hazard = _pairs(value("hazard_polygon"))
        self.warning = _pairs(value("warning_polygon"))
        self.horizon = float(value("prediction_horizon_s"))
        self.step = float(value("prediction_step_s"))
        self.process_noise = float(value("process_noise"))
        self.measurement_noise = float(value("measurement_noise"))
        self.minimum_age = int(value("minimum_track_age_frames"))
        self.max_gap = float(value("max_track_gap_s"))
        self.filters, self.last_seen = {}, {}
        self.publisher = self.create_publisher(String, "/future_risk/geometry_prediction", 10)
        self.create_subscription(String, "/future_risk/person_tracks", self._tracks, 10)
        self.get_logger().info(
            f"Ground trajectory ready: mode={self.mode}, horizon={self.horizon:.1f}s, "
            f"zone_logic={self.zone_logic_enabled}"
        )

    def _tracks(self, message):
        try:
            source = json.loads(message.data)
            agents = []
            newest_stamp = 0.0
            camera_stamps = {}
            approximate_stop_polygons_uv = {}
            walkway_polygons_uv = {}
            approximate_stop_zone_enabled = False
            active_ids = set()
            for camera, camera_data in source.get("cameras", {}).items():
                if camera not in self.homographies:
                    continue
                stamp = float(camera_data.get("stamp_s", 0.0))
                camera_stamps[camera] = stamp
                approximate_stop_polygons_uv[camera] = camera_data.get(
                    "approximate_stop_polygon_uv", []
                )
                walkway_polygons_uv[camera] = camera_data.get("walkway_polygon_uv", [])
                approximate_stop_zone_enabled = approximate_stop_zone_enabled or bool(
                    camera_data.get("approximate_stop_zone_enabled", False)
                )
                newest_stamp = max(newest_stamp, stamp)
                for track in camera_data.get("tracks", []):
                    track_id = str(track.get("track_id"))
                    active_ids.add(track_id)
                    position = image_to_ground(
                        self.homographies[camera], track.get("footpoint_uv", [0, 0])
                    )
                    if track_id not in self.filters:
                        self.filters[track_id] = KalmanCV2D(
                            position, stamp, self.process_noise, self.measurement_noise
                        )
                    elif track.get("visible", False):
                        self.filters[track_id].update(position, stamp)
                    else:
                        self.filters[track_id].predict_to(stamp)
                    if track.get("visible", False):
                        self.last_seen[track_id] = stamp
                    agents.append(self._agent(camera, track, stamp))
            for track_id in list(self.filters):
                if (
                    track_id not in active_ids
                    and newest_stamp - self.last_seen.get(track_id, 0) > self.max_gap
                ):
                    del self.filters[track_id]
                    self.last_seen.pop(track_id, None)
            report = {
                "schema_version": "1.0",
                "stamp_s": newest_stamp,
                "camera_stamps_s": camera_stamps,
                "calibration_mode": self.mode,
                "localization_valid": self.valid,
                "metric_coordinates": self.mode == "metric_homography",
                "units": "m" if self.mode == "metric_homography" else "normalized_debug",
                "zone_logic_enabled": self.zone_logic_enabled,
                "motion_prediction_enabled": self.motion_prediction_enabled,
                "show_track_id_overlay": self.show_track_id_overlay,
                "show_prediction_overlay": self.show_prediction_overlay,
                "homographies": {
                    side: matrix.reshape(-1).tolist() for side, matrix in self.homographies.items()
                },
                "hazard_polygon": self.hazard if self.zone_logic_enabled else [],
                "warning_polygon": self.warning if self.zone_logic_enabled else [],
                "approximate_stop_zone_enabled": approximate_stop_zone_enabled,
                "approximate_stop_zone_metric": False,
                "approximate_stop_polygons_uv": approximate_stop_polygons_uv,
                "walkway_polygons_uv": walkway_polygons_uv,
                "prediction_horizon_s": self.horizon,
                "tracked_agents": agents,
                "source_backend": source.get("backend", "unknown"),
            }
            output = String()
            output.data = json.dumps(report)
            self.publisher.publish(output)
        except Exception as exc:
            self.get_logger().error(f"Ground trajectory update failed: {exc}")

    def _agent(self, camera, track, stamp):
        track_id = str(track["track_id"])
        motion = self.filters[track_id]
        path_with_time = (
            motion.future_path(self.horizon, self.step)
            if self.motion_prediction_enabled
            else [(0.0, motion.position)]
        )
        path = [list(point) for _, point in path_with_time]
        entries = (
            [tau for tau, point in path_with_time if point_in_polygon(point, self.hazard)]
            if self.zone_logic_enabled
            else []
        )
        tte = min(entries) if entries else None
        position = motion.position
        covariance_trace = float(motion.covariance.trace())
        reliable = (
            bool(track.get("visible", False))
            and int(track.get("age_frames", 0)) >= self.minimum_age
            and stamp - self.last_seen.get(track_id, stamp) <= self.max_gap
            and math.isfinite(covariance_trace)
        )
        current_zone = (
            classify_zone(position, self.hazard, self.warning)
            if self.zone_logic_enabled
            else "DISABLED"
        )
        distance = (
            round(distance_to_polygon(position, self.hazard), 5)
            if self.zone_logic_enabled
            else None
        )
        return {
            "track_id": track_id,
            "camera": camera,
            "visible": bool(track.get("visible", False)),
            "bbox_xyxy": [round(float(value), 2) for value in track.get("bbox_xyxy", [])],
            "detection_confidence": round(float(track.get("confidence", 0.0)), 4),
            "footpoint_uv": [
                round(float(value), 2) for value in track.get("footpoint_uv", [])
            ],
            "position": [round(value, 5) for value in position],
            "velocity": (
                [round(value, 5) for value in motion.velocity]
                if self.motion_prediction_enabled
                else None
            ),
            "current_zone": current_zone,
            "distance_to_hazard": distance,
            "future_path": (
                [[round(value, 5) for value in point] for point in path]
                if self.motion_prediction_enabled
                else []
            ),
            "trajectory_intersects_hazard": bool(entries) if self.zone_logic_enabled else False,
            "time_to_entry_s": None if tte is None else round(float(tte), 3),
            "tracking_reliable": reliable,
            "track_age_frames": int(track.get("age_frames", 0)),
            "missed_frames": int(track.get("missed_frames", 0)),
            "filter_covariance_trace": round(covariance_trace, 5),
            "inside_approximate_stop_zone": bool(
                track.get("inside_approximate_stop_zone", False)
            ),
        }


def main(args=None):
    rclpy.init(args=args)
    node = GroundTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
