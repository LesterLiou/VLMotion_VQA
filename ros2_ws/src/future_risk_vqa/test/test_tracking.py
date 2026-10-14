import numpy as np

from future_risk_vqa.tracking import (
    Detection,
    PersonTrackManager,
    suppress_duplicate_detections,
)


def test_duplicate_suppression_keeps_highest_confidence_nested_detection():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 20:80] = True
    inner_mask = np.zeros_like(mask)
    inner_mask[20:80, 25:75] = True
    detections = [
        Detection((20, 10, 80, 90), (50, 89), 0.92, mask),
        Detection((25, 20, 75, 80), (50, 79), 0.71, inner_mask),
    ]
    kept = suppress_duplicate_detections(detections)
    assert len(kept) == 1
    assert kept[0].confidence == 0.92


def test_duplicate_suppression_keeps_separate_people():
    first_mask = np.zeros((100, 100), dtype=bool)
    second_mask = np.zeros_like(first_mask)
    first_mask[10:80, 5:35] = True
    second_mask[10:80, 65:95] = True
    detections = [
        Detection((5, 10, 35, 80), (20, 79), 0.9, first_mask),
        Detection((65, 10, 95, 80), (80, 79), 0.88, second_mask),
    ]
    assert len(suppress_duplicate_detections(detections)) == 2


def test_nested_boxes_with_different_masks_and_feet_are_not_suppressed():
    outer_mask = np.zeros((120, 120), dtype=bool)
    inner_mask = np.zeros_like(outer_mask)
    outer_mask[10:110, 10:35] = True
    inner_mask[20:65, 55:80] = True
    detections = [
        Detection((5, 5, 100, 115), (20, 109), 0.9, outer_mask),
        Detection((50, 15, 85, 70), (68, 69), 0.88, inner_mask),
    ]
    assert len(suppress_duplicate_detections(detections)) == 2


def detection(x):
    mask = np.zeros((40, 80), dtype=bool)
    mask[10:31, x : x + 10] = True
    return Detection((x, 10, x + 10, 31), (x + 4.5, 30), 0.9, mask)


def test_track_id_persists_and_expires():
    tracker = PersonTrackManager("L", max_missed_frames=1)
    first = tracker.update([detection(10)], (40, 80, 3))
    second = tracker.update([detection(12)], (40, 80, 3))
    assert first[0].track_id == second[0].track_id == "L1"
    missing = tracker.update([], (40, 80, 3))
    assert not missing[0].visible
    assert tracker.update([], (40, 80, 3)) == []


def test_velocity_prediction_keeps_id_when_boxes_no_longer_overlap():
    tracker = PersonTrackManager("L", max_missed_frames=2)
    first = tracker.update([detection(5)], (40, 80, 3), timestamp_s=1.0)
    second = tracker.update([detection(15)], (40, 80, 3), timestamp_s=2.0)
    third = tracker.update([detection(28)], (40, 80, 3), timestamp_s=3.0)

    assert first[0].track_id == "L1"
    assert second[0].track_id == "L1"
    assert third[0].track_id == "L1"
    assert third[0].age_frames == 3
