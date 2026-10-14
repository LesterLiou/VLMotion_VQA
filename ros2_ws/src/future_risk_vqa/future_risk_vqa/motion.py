"""Constant-velocity Kalman filtering and short-horizon path prediction."""

from collections import deque

import numpy as np


class KalmanCV2D:
    """Four-state [X, Y, Vx, Vy] Kalman filter."""

    def __init__(self, position_xy, timestamp_s, process_noise=0.5, measurement_noise=0.08):
        self.state = np.array([float(position_xy[0]), float(position_xy[1]), 0.0, 0.0])
        self.covariance = np.diag([0.2, 0.2, 2.0, 2.0])
        self.timestamp_s = float(timestamp_s)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.history = deque(maxlen=30)
        self.history.append((self.timestamp_s, tuple(self.state[:2])))

    @staticmethod
    def transition(dt):
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def predict_to(self, timestamp_s):
        timestamp_s = float(timestamp_s)
        dt = max(0.0, min(2.0, timestamp_s - self.timestamp_s))
        if dt <= 0.0:
            return
        transition = self.transition(dt)
        acceleration = self.process_noise
        q = acceleration**2
        process = q * np.array(
            [
                [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
                [dt**3 / 2.0, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2.0, 0.0, dt**2],
            ]
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.timestamp_s = timestamp_s

    def update(self, position_xy, timestamp_s):
        self.predict_to(timestamp_s)
        measurement = np.asarray(position_xy, dtype=float)
        observation = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        noise = np.eye(2) * self.measurement_noise**2
        innovation = measurement - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        identity = np.eye(4)
        self.covariance = (identity - gain @ observation) @ self.covariance
        self.history.append((self.timestamp_s, tuple(self.state[:2])))

    @property
    def position(self):
        return tuple(float(value) for value in self.state[:2])

    @property
    def velocity(self):
        return tuple(float(value) for value in self.state[2:])

    def future_path(self, horizon_s=3.0, step_s=0.2):
        times = np.arange(0.0, float(horizon_s) + step_s * 0.5, float(step_s))
        path = []
        for tau in times:
            future = self.transition(float(tau)) @ self.state
            path.append((float(tau), (float(future[0]), float(future[1]))))
        return path


def estimate_camera_motion_hint(
    samples,
    frame_shape,
    scale_change_threshold=0.12,
    lateral_displacement_threshold=0.35,
    stationary_scale_threshold=0.04,
    stationary_lateral_threshold=0.12,
):
    """Estimate apparent camera-relative motion from timestamped person boxes.

    This is a monocular image-space hint, not metric depth. Increasing bbox height is
    treated as approaching the camera; decreasing height is treated as moving away.
    Horizontal center displacement is normalized by the person's mean bbox width.
    """
    valid = [
        item
        for item in samples
        if isinstance(item, dict)
        and isinstance(item.get("bbox_xyxy"), (list, tuple))
        and len(item["bbox_xyxy"]) == 4
    ]
    coverage = len(valid) / max(1, len(samples))
    base = {
        "source": "timestamped_bbox_change",
        "motion_trend": "UNCERTAIN",
        "reliable": False,
        "confidence": 0.0,
        "observation_count": len(valid),
        "expected_observation_count": len(samples),
        "coverage_ratio": round(coverage, 3),
        "metric_depth": False,
    }
    if len(valid) < 2:
        return base

    first = [float(value) for value in valid[0]["bbox_xyxy"]]
    last = [float(value) for value in valid[-1]["bbox_xyxy"]]
    first_height = max(1.0, first[3] - first[1])
    last_height = max(1.0, last[3] - last[1])
    first_width = max(1.0, first[2] - first[0])
    last_width = max(1.0, last[2] - last[0])
    mean_width = max(1.0, (first_width + last_width) / 2.0)
    scale_change = (last_height - first_height) / first_height
    first_center_x = (first[0] + first[2]) / 2.0
    last_center_x = (last[0] + last[2]) / 2.0
    lateral_change = (last_center_x - first_center_x) / mean_width

    scale_strength = abs(scale_change) / max(1e-6, float(scale_change_threshold))
    lateral_strength = abs(lateral_change) / max(
        1e-6, float(lateral_displacement_threshold)
    )
    if scale_change >= float(scale_change_threshold):
        trend = "TOWARD_CAMERA"
        strength = scale_strength
    elif scale_change <= -float(scale_change_threshold):
        trend = "AWAY_FROM_CAMERA"
        strength = scale_strength
    elif abs(lateral_change) >= float(lateral_displacement_threshold):
        trend = "LATERAL_ACROSS_VIEW"
        strength = lateral_strength
    elif (
        abs(scale_change) <= float(stationary_scale_threshold)
        and abs(lateral_change) <= float(stationary_lateral_threshold)
    ):
        trend = "STATIONARY"
        strength = 1.0
    else:
        trend = "UNCERTAIN"
        strength = max(scale_strength, lateral_strength) * 0.5

    height, width = frame_shape[:2]
    confidence = min(1.0, coverage * min(1.0, max(0.0, strength)))
    return {
        **base,
        "motion_trend": trend,
        "reliable": trend != "UNCERTAIN" and coverage >= 2.0 / 3.0,
        "confidence": round(confidence, 3),
        "bbox_height_change_ratio": round(scale_change, 4),
        "horizontal_change_body_widths": round(lateral_change, 4),
        "frame_width": int(width),
        "frame_height": int(height),
    }
