"""Optical time-to-contact estimation for YOLO/ByteTrack person tracks."""

from dataclasses import dataclass
import math


def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


@dataclass
class _PersonTrack:
    track_id: int
    bbox: tuple
    confidence: float
    timestamp_s: float
    log_scale: float
    growth_rate: float = 0.0
    age: int = 1
    missed: int = 0


class PersonTTCMonitor:
    """Estimate optical TTC for YOLO person tracks with persistent IDs."""

    def __init__(
        self,
        warning_ttc_s=3.0,
        critical_ttc_s=1.0,
        min_growth_rate=0.04,
        growth_ema_alpha=0.45,
        min_confirmed_age=3,
        max_missed=15,
        min_area_ratio=0.002,
        critical_area_ratio=0.08,
        roi=(0.2, 0.15, 0.8, 0.95),
    ):
        self.warning_ttc_s = float(warning_ttc_s)
        self.critical_ttc_s = float(critical_ttc_s)
        self.min_growth_rate = float(min_growth_rate)
        self.growth_ema_alpha = float(growth_ema_alpha)
        self.min_confirmed_age = int(min_confirmed_age)
        self.max_missed = int(max_missed)
        self.min_area_ratio = float(min_area_ratio)
        self.critical_area_ratio = float(critical_area_ratio)
        self.roi = tuple(float(value) for value in roi)
        self._tracks = {}

    def reset(self):
        self._tracks.clear()

    @staticmethod
    def _log_scale(box):
        return math.log(math.sqrt(max(1.0, bbox_area(box))))

    def _inside_roi(self, box, frame_shape):
        height, width = frame_shape[:2]
        cx, cy = bbox_center(box)
        x1, y1, x2, y2 = self.roi
        return x1 * width <= cx <= x2 * width and y1 * height <= cy <= y2 * height

    def _new_track(self, detection, timestamp_s):
        box = tuple(float(value) for value in detection['bbox'])
        return _PersonTrack(
            track_id=int(detection['track_id']),
            bbox=box,
            confidence=float(detection['confidence']),
            timestamp_s=float(timestamp_s),
            log_scale=self._log_scale(box),
        )

    def _update_track(self, track, detection, timestamp_s):
        box = tuple(float(value) for value in detection['bbox'])
        log_scale = self._log_scale(box)
        dt = float(timestamp_s) - track.timestamp_s
        if dt > 1e-4:
            instantaneous = (log_scale - track.log_scale) / dt
            alpha = self.growth_ema_alpha
            track.growth_rate = alpha * instantaneous + (1.0 - alpha) * track.growth_rate
        track.bbox = box
        track.confidence = float(detection['confidence'])
        track.timestamp_s = float(timestamp_s)
        track.log_scale = log_scale
        track.age += 1
        track.missed = 0

    def _result(self, track, frame_shape):
        height, width = frame_shape[:2]
        area_ratio = bbox_area(track.bbox) / max(1.0, width * height)
        in_roi = self._inside_roi(track.bbox, frame_shape)
        confirmed = track.age >= self.min_confirmed_age
        approaching = confirmed and track.growth_rate >= self.min_growth_rate
        ttc_s = 1.0 / track.growth_rate if approaching else None

        action = 'CONTINUE'
        reason = 'no confirmed person approach'
        if confirmed and in_roi and area_ratio >= self.critical_area_ratio:
            action = 'STOP'
            reason = 'person occupies a critical portion of the safety ROI'
        elif in_roi and area_ratio >= self.min_area_ratio and ttc_s is not None:
            if ttc_s <= self.critical_ttc_s:
                action = 'STOP'
                reason = 'person optical TTC is below critical threshold'
            elif ttc_s <= self.warning_ttc_s:
                action = 'SLOW_DOWN'
                reason = 'person optical TTC is below warning threshold'
            else:
                reason = 'approaching person is outside warning horizon'

        return {
            'track_id': track.track_id,
            'class_name': 'person',
            'confidence': round(track.confidence, 3),
            'bbox': [round(value, 1) for value in track.bbox],
            'age_frames': track.age,
            'area_ratio': round(area_ratio, 5),
            'growth_rate_per_s': round(track.growth_rate, 4),
            'ttc_s': round(ttc_s, 3) if ttc_s is not None else None,
            'approaching': approaching,
            'inside_roi': in_roi,
            'action': action,
            'reason': reason,
        }

    def update(self, detections, timestamp_s, frame_shape):
        """Update YOLO/ByteTrack person detections and return safety output."""
        seen_ids = set()
        for detection in detections:
            track_id = int(detection['track_id'])
            seen_ids.add(track_id)
            if track_id in self._tracks:
                self._update_track(self._tracks[track_id], detection, timestamp_s)
            else:
                self._tracks[track_id] = self._new_track(detection, timestamp_s)

        for track_id, track in list(self._tracks.items()):
            if track_id not in seen_ids:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self._tracks[track_id]

        visible = [self._tracks[track_id] for track_id in sorted(seen_ids)]
        results = [self._result(track, frame_shape) for track in visible]
        priority = {'CONTINUE': 0, 'SLOW_DOWN': 1, 'STOP': 2}
        action = max(
            (item['action'] for item in results),
            key=lambda value: priority[value],
            default='CONTINUE',
        )
        urgent = [item for item in results if item['action'] == action]
        urgent.sort(key=lambda item: item['ttc_s'] if item['ttc_s'] is not None else math.inf)
        return {
            'action': action,
            'person_count': len(results),
            'most_urgent_track': urgent[0]['track_id'] if urgent and action != 'CONTINUE' else None,
            'tracks': results,
        }
