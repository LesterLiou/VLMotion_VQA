"""Temporal frame buffers and geometry-aware storyboard rendering."""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import ground_to_image


@dataclass(frozen=True)
class TimedFrame:
    stamp_s: float
    receipt_s: float
    image: np.ndarray


class FrameBuffer:
    def __init__(self, max_frames=120):
        self._frames = deque(maxlen=max(2, int(max_frames)))

    def append(self, stamp_s, receipt_s, image):
        self._frames.append(TimedFrame(float(stamp_s), float(receipt_s), image.copy()))

    @property
    def latest(self):
        return self._frames[-1] if self._frames else None

    def sample(self, count, window_s, end_stamp_s=None):
        if not self._frames:
            return []
        latest_available = self._frames[-1].stamp_s
        end = (
            latest_available
            if end_stamp_s is None
            else min(float(end_stamp_s), latest_available)
        )
        start = end - max(0.0, float(window_s))
        candidates = [item for item in self._frames if start <= item.stamp_s <= end]
        if not candidates:
            candidates = [min(self._frames, key=lambda item: abs(item.stamp_s - end))]
        targets = np.linspace(start, end, max(1, int(count)))
        return [min(candidates, key=lambda item: abs(item.stamp_s - target)) for target in targets]


class TrackHistoryBuffer:
    """Keep timestamped camera-local bounding boxes for target-centric sampling."""

    def __init__(self, max_reports=120):
        self._reports = {
            "left": deque(maxlen=max(2, int(max_reports))),
            "right": deque(maxlen=max(2, int(max_reports))),
        }

    def append_report(self, report):
        for camera, camera_data in report.get("cameras", {}).items():
            if camera not in self._reports:
                continue
            tracks = {}
            for item in camera_data.get("tracks", []):
                track_id = str(item.get("track_id", ""))
                bbox = item.get("bbox_xyxy", [])
                if not track_id or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                tracks[track_id] = {
                    "bbox_xyxy": tuple(float(value) for value in bbox),
                    "visible": bool(item.get("visible", False)),
                    "confidence": float(item.get("confidence", 0.0)),
                }
            self._reports[camera].append(
                (float(camera_data.get("stamp_s", 0.0)), tracks)
            )

    def sample(self, camera, track_id, stamp_s, tolerance_s):
        reports = self._reports.get(camera, ())
        if not reports:
            return None
        report_stamp, tracks = min(reports, key=lambda item: abs(item[0] - stamp_s))
        if abs(report_stamp - float(stamp_s)) > float(tolerance_s):
            return None
        item = tracks.get(str(track_id))
        if not item or not item["visible"]:
            return None
        return {
            **item,
            "report_stamp_s": report_stamp,
            "frame_delta_s": round(report_stamp - float(stamp_s), 6),
        }


def _fit(frame, width, height):
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized = cv2.resize(frame, (int(source_w * scale), int(source_h * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = resized.shape[:2]
    canvas[(height - h) // 2 : (height - h) // 2 + h, (width - w) // 2 : (width - w) // 2 + w] = (
        resized
    )
    return canvas


def _crop_with_padding(frame, bbox, padding_ratio):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    padding = max(0.0, float(padding_ratio))
    x1 = max(0, int(round(x1 - box_width * padding)))
    x2 = min(width, int(round(x2 + box_width * padding)))
    y1 = max(0, int(round(y1 - box_height * padding)))
    y2 = min(height, int(round(y2 + box_height * padding)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _track_color(track_id):
    palette = (
        (0, 220, 255),
        (255, 120, 0),
        (80, 220, 80),
        (220, 80, 220),
        (80, 170, 255),
        (255, 100, 120),
    )
    number = sum(ord(character) for character in str(track_id))
    return palette[number % len(palette)]


def build_track_crop_storyboard(
    left_frames,
    right_frames,
    geometry,
    track_samples,
    crop_padding_ratio=0.30,
    cell_size=224,
    maximum_agents=6,
):
    """Build one OLD/MID/NOW row per active ID using timestamp-matched boxes."""
    frames_by_camera = {"left": left_frames, "right": right_frames}
    agents = [
        item
        for item in geometry.get("tracked_agents", [])
        if bool(item.get("visible", int(item.get("missed_frames", 0)) == 0))
    ]
    agents.sort(key=lambda item: (str(item.get("camera")), str(item.get("track_id"))))
    if not agents:
        raise ValueError("at least one active track is required for a crop storyboard")
    if len(agents) > int(maximum_agents):
        image, evidence = build_storyboard(left_frames, right_frames, geometry)
        evidence.update(
            {
                "storyboard_mode": "global_fallback",
                "track_crop_fallback_reason": "too_many_active_tracks",
                "active_track_count": len(agents),
            }
        )
        return image, evidence

    rows = []
    coverage = {}
    sample_stamps = {}
    for agent in agents:
        track_id = str(agent.get("track_id"))
        camera = str(agent.get("camera"))
        frames = frames_by_camera.get(camera, [])
        samples = track_samples.get(track_id, [])
        if not frames or len(samples) != len(frames):
            raise ValueError(f"missing temporal track samples for {track_id}")
        latest = frames[-1].stamp_s
        cells = []
        found = 0
        sample_stamps[track_id] = []
        color = _track_color(track_id)
        for index, (frame, sample) in enumerate(zip(frames, samples)):
            age = frame.stamp_s - latest
            stage = "NOW" if index == len(frames) - 1 else f"T-{abs(age):.1f}s"
            source = None
            if sample is not None:
                source = _crop_with_padding(
                    frame.image, sample["bbox_xyxy"], crop_padding_ratio
                )
                sample_stamps[track_id].append(sample["report_stamp_s"])
            else:
                sample_stamps[track_id].append(None)
            if source is None:
                cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                cv2.putText(
                    cell,
                    "TRACK NOT FOUND",
                    (18, cell_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (150, 150, 150),
                    2,
                    cv2.LINE_AA,
                )
            else:
                found += 1
                cell = _fit(source, cell_size, cell_size)
            cv2.rectangle(cell, (0, 0), (cell_size - 1, cell_size - 1), color, 4)
            cv2.rectangle(cell, (0, 0), (cell_size, 34), (0, 0, 0), -1)
            cv2.putText(
                cell,
                f"{track_id} | {camera.upper()} | {stage}",
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cells.append(cell)
        coverage[track_id] = found
        rows.append(np.hstack(cells))
    return np.vstack(rows), {
        "storyboard_mode": "track_crops",
        "sample_count_per_track": len(left_frames),
        "track_crop_coverage": coverage,
        "track_crop_report_stamps_s": sample_stamps,
        "active_track_count": len(agents),
    }


def build_hybrid_storyboard(
    left_frames,
    right_frames,
    geometry,
    track_samples,
    crop_padding_ratio=0.30,
    cell_size=224,
    maximum_agents=6,
):
    """Combine two NOW context views with a compact two-column crop timeline grid."""
    if not left_frames or not right_frames:
        raise ValueError("left and right temporal frames are required")
    agents = [
        item
        for item in geometry.get("tracked_agents", [])
        if bool(item.get("visible", int(item.get("missed_frames", 0)) == 0))
    ]
    if not agents:
        raise ValueError("at least one active track is required for a hybrid storyboard")

    # The prompt must assess every active ID. Keep all rows even when the configured
    # visual target is exceeded, and expose that fact in metadata for benchmarking.
    crop_board, crop_metadata = build_track_crop_storyboard(
        left_frames,
        right_frames,
        geometry,
        track_samples,
        crop_padding_ratio=crop_padding_ratio,
        cell_size=cell_size,
        maximum_agents=max(int(maximum_agents), len(agents)),
    )
    strip_width = cell_size * len(left_frames)
    context_height = cell_size * 2
    context_cells = []
    for camera, frames, color in (
        ("left", left_frames, (255, 120, 0)),
        ("right", right_frames, (0, 165, 255)),
    ):
        source = draw_detection_snapshot(frames[-1].image, camera, geometry)
        cell = _fit(source, strip_width, context_height)
        cv2.rectangle(cell, (0, 0), (strip_width - 1, context_height - 1), color, 4)
        cv2.rectangle(cell, (0, 0), (strip_width, 38), (0, 0, 0), -1)
        cv2.putText(
            cell,
            f"GLOBAL CONTEXT | {camera.upper()} | NOW",
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        context_cells.append(cell)

    crop_strips = [
        crop_board[index : index + cell_size]
        for index in range(0, crop_board.shape[0], cell_size)
    ]
    grid_rows = []
    for index in range(0, len(crop_strips), 2):
        pair = crop_strips[index : index + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        grid_rows.append(np.hstack(pair))

    metadata = {
        **crop_metadata,
        "storyboard_mode": "hybrid",
        "global_context_camera_count": 2,
        "crop_grid_columns": 2,
        "configured_track_crop_max_agents": int(maximum_agents),
        "track_crop_limit_exceeded": len(agents) > int(maximum_agents),
    }
    return np.vstack([np.hstack(context_cells), *grid_rows]), metadata


def draw_geometry(frame, camera, geometry):
    """Draw only enabled research overlays on the newest temporal frame."""
    result = frame.copy()
    # These two overlays already use image coordinates, so they remain useful
    # before a real ground-plane homography has been calibrated.
    walkway = geometry.get("walkway_polygons_uv", {}).get(camera, [])
    if len(walkway) >= 3:
        blue = np.zeros_like(result)
        cv2.fillPoly(blue, [np.int32(walkway)], (255, 150, 0))
        result = cv2.addWeighted(result, 1.0, blue, 0.15, 0.0)
        cv2.polylines(result, [np.int32(walkway)], True, (255, 150, 0), 2, cv2.LINE_AA)
    approximate_stop = geometry.get("approximate_stop_polygons_uv", {}).get(camera, [])
    if geometry.get("approximate_stop_zone_enabled", False) and len(approximate_stop) >= 3:
        red = np.zeros_like(result)
        cv2.fillPoly(red, [np.int32(approximate_stop)], (0, 0, 255))
        result = cv2.addWeighted(result, 1.0, red, 0.18, 0.0)
        cv2.polylines(
            result, [np.int32(approximate_stop)], True, (0, 0, 255), 3, cv2.LINE_AA
        )
        cv2.putText(
            result,
            "APPROX 1m STOP",
            tuple(np.int32(approximate_stop[-1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    homographies = geometry.get("homographies", {})
    matrix = homographies.get(camera)
    if not matrix:
        return result
    if geometry.get("zone_logic_enabled", False):
        for key, color in (
            ("warning_polygon", (0, 190, 255)),
            ("hazard_polygon", (0, 0, 255)),
        ):
            polygon = geometry.get(key, [])
            points = [ground_to_image(matrix, point) for point in polygon]
            if points:
                cv2.polylines(result, [np.int32(points)], True, color, 3, cv2.LINE_AA)
    for agent in geometry.get("tracked_agents", []):
        if agent.get("camera") != camera:
            continue
        if not bool(agent.get("visible", int(agent.get("missed_frames", 0)) == 0)):
            continue
        if geometry.get("show_prediction_overlay", False):
            path = [ground_to_image(matrix, point) for point in agent.get("future_path", [])]
            if len(path) > 1:
                cv2.polylines(result, [np.int32(path)], False, (255, 80, 255), 3, cv2.LINE_AA)
        if geometry.get("show_track_id_overlay", True):
            point = ground_to_image(matrix, agent.get("position", [0.0, 0.0]))
            point = tuple(np.int32(point))
            cv2.circle(result, point, 7, (0, 255, 255), -1)
            cv2.putText(
                result,
                str(agent.get("track_id", "?")),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return result


def draw_detection_snapshot(frame, camera, geometry):
    """Draw SAM3 detections on the timestamp-aligned VQA NOW snapshot."""
    result = draw_geometry(frame, camera, geometry)
    if not geometry.get("show_sam3_detection_overlay", True):
        return result
    height, width = result.shape[:2]
    for agent in geometry.get("tracked_agents", []):
        if agent.get("camera") != camera:
            continue
        if not bool(agent.get("visible", int(agent.get("missed_frames", 0)) == 0)):
            continue
        bbox = agent.get("bbox_xyxy", [])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        track_id = str(agent.get("track_id", "?"))
        confidence = float(agent.get("detection_confidence", 0.0))
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(
            result,
            f"{track_id} {confidence:.2f}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return result


def build_storyboard(left_frames, right_frames, geometry, cell_width=448, cell_height=300):
    if not left_frames or not right_frames or len(left_frames) != len(right_frames):
        raise ValueError("matching left/right temporal frames are required")
    rows = []
    for camera, frames, color in (
        ("left", left_frames, (255, 120, 0)),
        ("right", right_frames, (0, 165, 255)),
    ):
        latest = frames[-1].stamp_s
        cells = []
        for index, item in enumerate(frames):
            source = (
                draw_geometry(item.image, camera, geometry)
                if index == len(frames) - 1
                else item.image.copy()
            )
            cell = _fit(source, cell_width, cell_height)
            age = item.stamp_s - latest
            stage = "NOW" if index == len(frames) - 1 else f"T-{abs(age):.1f}s"
            cv2.rectangle(cell, (0, 0), (cell_width - 1, cell_height - 1), color, 3)
            cv2.rectangle(cell, (0, 0), (cell_width, 34), (0, 0, 0), -1)
            cv2.putText(
                cell,
                f"{camera.upper()} | {stage}",
                (9, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cells.append(cell)
        rows.append(np.hstack(cells))
    image = np.vstack(rows)
    return image, {
        "storyboard_mode": "global",
        "sample_count_per_camera": len(left_frames),
        "left_stamps_s": [round(item.stamp_s, 6) for item in left_frames],
        "right_stamps_s": [round(item.stamp_s, 6) for item in right_frames],
    }
