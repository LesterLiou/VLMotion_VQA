from types import SimpleNamespace

import numpy as np

from future_risk_vqa.storyboard import (
    FrameBuffer,
    TrackHistoryBuffer,
    build_hybrid_storyboard,
    build_track_crop_storyboard,
    draw_detection_snapshot,
    draw_geometry,
)


def test_temporal_sampling_is_anchored_to_inference_camera_stamp():
    buffer = FrameBuffer()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    for stamp in (1.0, 2.0, 3.0):
        buffer.append(stamp, stamp, image)
    sampled = buffer.sample(count=2, window_s=1.0, end_stamp_s=2.0)
    assert [item.stamp_s for item in sampled] == [1.0, 2.0]


def test_detection_snapshot_restores_sam3_bbox_on_aligned_frame():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    geometry = {
        "homographies": {"left": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        "show_track_id_overlay": False,
        "tracked_agents": [
            {
                "track_id": "L3",
                "camera": "left",
                "visible": True,
                "bbox_xyxy": [10, 20, 60, 80],
                "detection_confidence": 0.91,
            }
        ],
    }
    rendered = draw_detection_snapshot(image, "left", geometry)
    assert rendered[20, 10, 1] > 0
    assert rendered[80, 60, 1] > 0


def test_image_space_stop_zone_draws_without_homography():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    geometry = {
        "homographies": {},
        "approximate_stop_zone_enabled": True,
        "approximate_stop_polygons_uv": {
            "left": [[10, 90], [110, 90], [80, 50], [40, 50]]
        },
        "walkway_polygons_uv": {"left": [[5, 95], [115, 95], [60, 30]]},
    }
    rendered = draw_geometry(image, "left", geometry)
    assert rendered.sum() > 0
    assert rendered[70, 60, 2] > 0


def test_track_history_uses_bbox_from_nearest_timestamp():
    history = TrackHistoryBuffer()
    history.append_report(
        {
            "cameras": {
                "left": {
                    "stamp_s": 2.0,
                    "tracks": [
                        {
                            "track_id": "L1",
                            "bbox_xyxy": [10, 12, 30, 38],
                            "visible": True,
                            "confidence": 0.9,
                        }
                    ],
                }
            }
        }
    )
    sample = history.sample("left", "L1", 2.1, tolerance_s=0.2)
    assert sample["bbox_xyxy"] == (10.0, 12.0, 30.0, 38.0)
    assert history.sample("left", "L1", 2.3, tolerance_s=0.2) is None


def test_track_crop_storyboard_has_one_old_mid_now_row_per_id():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[10:70, 20:50] = (30, 90, 180)
    left = [
        SimpleNamespace(stamp_s=stamp, image=image.copy())
        for stamp in (1.0, 2.0, 3.0)
    ]
    geometry = {
        "tracked_agents": [
            {"track_id": "L1", "camera": "left", "visible": True}
        ]
    }
    samples = {
        "L1": [
            {
                "bbox_xyxy": (20.0, 10.0, 50.0, 70.0),
                "report_stamp_s": stamp,
            }
            for stamp in (1.0, 2.0, 3.0)
        ]
    }
    board, metadata = build_track_crop_storyboard(
        left,
        [],
        geometry,
        samples,
        crop_padding_ratio=0.2,
        cell_size=128,
    )
    assert board.shape == (128, 384, 3)
    assert metadata["storyboard_mode"] == "track_crops"
    assert metadata["track_crop_coverage"] == {"L1": 3}


def test_hybrid_storyboard_has_global_context_and_two_column_crop_grid():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    frames = [
        SimpleNamespace(stamp_s=stamp, image=image.copy())
        for stamp in (1.0, 2.0, 3.0)
    ]
    agents = [
        {"track_id": track_id, "camera": camera, "visible": True}
        for track_id, camera in (("L1", "left"), ("R1", "right"), ("R2", "right"))
    ]
    geometry = {"tracked_agents": agents, "show_track_id_overlay": False}
    samples = {
        item["track_id"]: [
            {"bbox_xyxy": (20.0, 10.0, 55.0, 72.0), "report_stamp_s": stamp}
            for stamp in (1.0, 2.0, 3.0)
        ]
        for item in agents
    }
    board, metadata = build_hybrid_storyboard(
        frames, frames, geometry, samples, cell_size=96
    )
    assert board.shape == (384, 576, 3)
    assert metadata["storyboard_mode"] == "hybrid"
    assert metadata["crop_grid_columns"] == 2
    assert metadata["track_crop_coverage"] == {"L1": 3, "R1": 3, "R2": 3}
