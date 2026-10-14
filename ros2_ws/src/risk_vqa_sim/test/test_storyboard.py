import numpy as np
import pytest

pytest.importorskip('cv2')
from risk_vqa_sim.storyboard import FrameBuffer, build_storyboard  # noqa: E402


def test_frame_buffer_samples_window_in_time_order():
    buffer = FrameBuffer(max_frames=20)
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    for index in range(11):
        buffer.append(index * 0.2, index * 0.2, image)
    samples = buffer.sample(3, 2.0)
    assert [frame.stamp_s for frame in samples] == [0.0, 1.0, 2.0]


def test_storyboard_has_two_rows_and_metadata():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    left = FrameBuffer(max_frames=10)
    right = FrameBuffer(max_frames=10)
    for index in range(3):
        left.append(float(index), float(index), image)
        right.append(float(index), float(index), image)
    storyboard, metadata = build_storyboard(
        left.sample(3, 2.0),
        right.sample(3, 2.0),
        cell_width=100,
        cell_height=80,
    )
    assert storyboard.shape == (160, 300, 3)
    assert metadata['sample_count_per_camera'] == 3
    assert metadata['left_stamps_s'] == [0.0, 1.0, 2.0]
