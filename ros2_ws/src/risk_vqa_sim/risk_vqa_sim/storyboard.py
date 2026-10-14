"""Timestamped frame buffers and a dual-camera temporal storyboard."""

from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np


@dataclass(frozen=True)
class TimedFrame:
    stamp_s: float
    receipt_s: float
    image: np.ndarray


class FrameBuffer:
    """Keep recent frames and select evenly spaced temporal evidence."""

    def __init__(self, max_frames=120):
        self._frames = deque(maxlen=max(2, int(max_frames)))

    def append(self, stamp_s, receipt_s, image):
        self._frames.append(
            TimedFrame(float(stamp_s), float(receipt_s), image.copy())
        )

    def __len__(self):
        return len(self._frames)

    @property
    def latest(self):
        return self._frames[-1] if self._frames else None

    def sample(self, count, window_s):
        """Return nearest frames at evenly spaced targets ending at latest."""
        if not self._frames:
            return []
        count = max(1, int(count))
        latest_s = self._frames[-1].stamp_s
        start_s = latest_s - max(0.0, float(window_s))
        candidates = [frame for frame in self._frames if frame.stamp_s >= start_s]
        if not candidates:
            candidates = [self._frames[-1]]
        if count == 1:
            return [candidates[-1]]
        targets = np.linspace(start_s, latest_s, count)
        return [
            min(candidates, key=lambda frame: abs(frame.stamp_s - target))
            for target in targets
        ]


def _fit_frame(frame, width, height):
    if frame is None or frame.size == 0:
        raise ValueError('storyboard frame is empty')
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized
    return canvas


def build_storyboard(left_frames, right_frames, cell_width=448, cell_height=336):
    """Build a labeled 2xN BGR storyboard and return image plus metadata."""
    if not left_frames or not right_frames:
        raise ValueError('both camera frame lists are required')
    if len(left_frames) != len(right_frames):
        raise ValueError('left and right sample counts must match')
    count = len(left_frames)
    rows = []
    colors = {'LEFT CAMERA': (255, 120, 0), 'RIGHT CAMERA': (0, 165, 255)}
    for row_name, timed_frames in (
        ('LEFT CAMERA', left_frames),
        ('RIGHT CAMERA', right_frames),
    ):
        cells = []
        newest_s = timed_frames[-1].stamp_s
        for index, timed_frame in enumerate(timed_frames):
            cell = _fit_frame(timed_frame.image, cell_width, cell_height)
            label = 'NOW' if index == count - 1 else (
                'OLDEST' if index == 0 else f'MID {index}'
            )
            delta_s = timed_frame.stamp_s - newest_s
            cv2.rectangle(cell, (0, 0), (cell_width - 1, cell_height - 1), colors[row_name], 3)
            cv2.rectangle(cell, (0, 0), (cell_width, 34), (0, 0, 0), -1)
            cv2.putText(
                cell,
                f'{row_name} | {label} | {delta_s:+.1f}s',
                (9, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cells.append(cell)
        rows.append(np.hstack(cells))
    storyboard = np.vstack(rows)
    metadata = {
        'sample_count_per_camera': count,
        'left_stamps_s': [round(frame.stamp_s, 6) for frame in left_frames],
        'right_stamps_s': [round(frame.stamp_s, 6) for frame in right_frames],
        'storyboard_width': int(storyboard.shape[1]),
        'storyboard_height': int(storyboard.shape[0]),
    }
    return storyboard, metadata
