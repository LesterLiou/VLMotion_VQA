"""Mask-IoU and foot-point matching for per-camera SAM3 detections."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Detection:
    bbox: tuple
    footpoint_uv: tuple
    confidence: float
    mask: np.ndarray


@dataclass
class TrackObservation:
    track_id: str
    bbox: tuple
    footpoint_uv: tuple
    confidence: float
    mask: np.ndarray
    visible: bool
    missed_frames: int
    age_frames: int


@dataclass
class _Track:
    track_id: str
    bbox: tuple
    footpoint_uv: tuple
    confidence: float
    mask: np.ndarray
    missed_frames: int = 0
    age_frames: int = 1
    timestamp_s: float = 0.0
    velocity_uv: tuple = (0.0, 0.0)


def bbox_iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_intersection_over_smaller(first, second):
    """Return overlap divided by the smaller box area for nested-box detection."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0.0 else 0.0


def mask_iou(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        return 0.0
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)


def suppress_duplicate_detections(
    detections,
    mask_iou_threshold=0.65,
    bbox_iou_threshold=0.70,
    bbox_ioa_threshold=0.85,
):
    """Keep the highest-confidence detection among strongly overlapping proposals."""
    kept = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for selected in kept:
            larger_width = max(
                detection.bbox[2] - detection.bbox[0],
                selected.bbox[2] - selected.bbox[0],
            )
            larger_height = max(
                detection.bbox[3] - detection.bbox[1],
                selected.bbox[3] - selected.bbox[1],
            )
            foot_distance = math.hypot(
                detection.footpoint_uv[0] - selected.footpoint_uv[0],
                detection.footpoint_uv[1] - selected.footpoint_uv[1],
            )
            nested_same_foot = (
                bbox_intersection_over_smaller(detection.bbox, selected.bbox)
                >= float(bbox_ioa_threshold)
                and foot_distance
                <= 0.25 * max(1.0, math.hypot(larger_width, larger_height))
            )
            if (
                mask_iou(detection.mask, selected.mask) >= float(mask_iou_threshold)
                or bbox_iou(detection.bbox, selected.bbox) >= float(bbox_iou_threshold)
                or nested_same_foot
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


class PersonTrackManager:
    """Maintain local L*/R* IDs without adding another learned model."""

    def __init__(
        self,
        prefix,
        max_missed_frames=3,
        minimum_match_score=0.20,
        mask_iou_weight=0.15,
        bbox_iou_weight=0.20,
        foot_distance_weight=0.50,
        size_similarity_weight=0.15,
        max_foot_distance_ratio=0.20,
        velocity_smoothing=0.60,
    ):
        self.prefix = str(prefix)
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.minimum_match_score = float(minimum_match_score)
        self.mask_iou_weight = float(mask_iou_weight)
        self.bbox_iou_weight = float(bbox_iou_weight)
        self.foot_distance_weight = float(foot_distance_weight)
        self.size_similarity_weight = float(size_similarity_weight)
        self.max_foot_distance_ratio = max(0.01, float(max_foot_distance_ratio))
        self.velocity_smoothing = min(0.95, max(0.0, float(velocity_smoothing)))
        self._tracks = {}
        self._next_id = 1

    def reset(self):
        self._tracks.clear()
        self._next_id = 1

    @staticmethod
    def _shifted_bbox(bbox, delta_u, delta_v):
        x1, y1, x2, y2 = bbox
        return (x1 + delta_u, y1 + delta_v, x2 + delta_u, y2 + delta_v)

    def _score(self, track, detection, frame_shape, timestamp_s):
        height, width = frame_shape[:2]
        diagonal = max(1.0, math.hypot(width, height))
        dt = max(0.0, min(2.0, float(timestamp_s) - track.timestamp_s))
        delta_u = track.velocity_uv[0] * dt
        delta_v = track.velocity_uv[1] * dt
        predicted_foot = (
            track.footpoint_uv[0] + delta_u,
            track.footpoint_uv[1] + delta_v,
        )
        predicted_bbox = self._shifted_bbox(track.bbox, delta_u, delta_v)
        distance = math.hypot(
            predicted_foot[0] - detection.footpoint_uv[0],
            predicted_foot[1] - detection.footpoint_uv[1],
        )
        maximum_distance = self.max_foot_distance_ratio * diagonal
        if distance > maximum_distance:
            return 0.0
        distance_score = max(0.0, 1.0 - distance / maximum_distance)
        predicted_area = max(1.0, (predicted_bbox[2] - predicted_bbox[0])) * max(
            1.0, predicted_bbox[3] - predicted_bbox[1]
        )
        detection_area = max(1.0, detection.bbox[2] - detection.bbox[0]) * max(
            1.0, detection.bbox[3] - detection.bbox[1]
        )
        size_score = min(predicted_area, detection_area) / max(
            predicted_area, detection_area
        )
        return (
            self.mask_iou_weight * mask_iou(track.mask, detection.mask)
            + self.bbox_iou_weight * bbox_iou(predicted_bbox, detection.bbox)
            + self.foot_distance_weight * distance_score
            + self.size_similarity_weight * size_score
        )

    @staticmethod
    def _assignment(scores):
        if scores.size == 0:
            return []
        try:
            from scipy.optimize import linear_sum_assignment

            rows, columns = linear_sum_assignment(1.0 - scores)
            return list(zip(rows.tolist(), columns.tolist()))
        except ImportError:
            candidates = [
                (float(scores[row, column]), row, column)
                for row in range(scores.shape[0])
                for column in range(scores.shape[1])
            ]
            candidates.sort(reverse=True)
            used_rows = set()
            used_columns = set()
            matches = []
            for _, row, column in candidates:
                if row not in used_rows and column not in used_columns:
                    matches.append((row, column))
                    used_rows.add(row)
                    used_columns.add(column)
            return matches

    def update(self, detections, frame_shape, timestamp_s=None):
        timestamp_s = (
            float(timestamp_s)
            if timestamp_s is not None
            else max((track.timestamp_s for track in self._tracks.values()), default=0.0) + 1.0
        )
        track_ids = list(self._tracks)
        scores = np.zeros((len(track_ids), len(detections)), dtype=float)
        for row, track_id in enumerate(track_ids):
            for column, detection in enumerate(detections):
                scores[row, column] = self._score(
                    self._tracks[track_id], detection, frame_shape, timestamp_s
                )

        matched_tracks = set()
        matched_detections = set()
        for row, column in self._assignment(scores):
            if scores[row, column] < self.minimum_match_score:
                continue
            track_id = track_ids[row]
            detection = detections[column]
            track = self._tracks[track_id]
            dt = max(1e-3, timestamp_s - track.timestamp_s)
            measured_velocity = (
                (detection.footpoint_uv[0] - track.footpoint_uv[0]) / dt,
                (detection.footpoint_uv[1] - track.footpoint_uv[1]) / dt,
            )
            keep = self.velocity_smoothing
            track.velocity_uv = (
                keep * track.velocity_uv[0] + (1.0 - keep) * measured_velocity[0],
                keep * track.velocity_uv[1] + (1.0 - keep) * measured_velocity[1],
            )
            track.bbox = detection.bbox
            track.footpoint_uv = detection.footpoint_uv
            track.confidence = detection.confidence
            track.mask = detection.mask
            track.missed_frames = 0
            track.age_frames += 1
            track.timestamp_s = timestamp_s
            matched_tracks.add(track_id)
            matched_detections.add(column)

        for track_id in list(track_ids):
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = f"{self.prefix}{self._next_id}"
            self._next_id += 1
            self._tracks[track_id] = _Track(
                track_id=track_id,
                bbox=detection.bbox,
                footpoint_uv=detection.footpoint_uv,
                confidence=detection.confidence,
                mask=detection.mask,
                timestamp_s=timestamp_s,
            )

        observations = []
        for track_id in sorted(self._tracks):
            track = self._tracks[track_id]
            observations.append(
                TrackObservation(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    footpoint_uv=track.footpoint_uv,
                    confidence=track.confidence,
                    mask=track.mask,
                    visible=track.missed_frames == 0,
                    missed_frames=track.missed_frames,
                    age_frames=track.age_frames,
                )
            )
        return observations
