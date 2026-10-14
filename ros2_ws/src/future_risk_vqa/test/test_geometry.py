import numpy as np

from future_risk_vqa.geometry import (
    classify_zone,
    footpoint_from_mask,
    image_to_ground,
    mask_foot_zone_overlap_ratio,
    normalized_polygon_to_image,
    point_in_normalized_polygon,
    point_in_polygon,
)


def test_mask_footpoint_uses_bottom_band_median():
    mask = np.zeros((10, 12), dtype=bool)
    mask[3:9, 4:8] = True
    assert footpoint_from_mask(mask) == (5.5, 8.0)


def test_mask_foot_overlap_detects_one_foot_crossing_zone_boundary():
    person = np.zeros((20, 20), dtype=bool)
    person[4:19, 5:15] = True
    zone = np.zeros_like(person)
    zone[15:, 13:] = True
    ratio = mask_foot_zone_overlap_ratio(person, zone)
    assert 0.0 < ratio < 0.5


def test_homography_and_zone_boundary():
    matrix = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]])
    assert image_to_ground(matrix, (20, 30)) == (2.0, 3.0)
    hazard = [(0, 0), (2, 0), (2, 2), (0, 2)]
    warning = [(-1, -1), (3, -1), (3, 3), (-1, 3)]
    assert point_in_polygon((2, 1), hazard)
    assert classify_zone((1, 1), hazard, warning) == "HAZARD_ZONE"
    assert classify_zone((2.5, 1), hazard, warning) == "WARNING_ZONE"


def test_normalized_image_stop_polygon_scales_with_resolution():
    polygon = [0.1, 1.0, 0.9, 1.0, 0.7, 0.5, 0.3, 0.5]
    points = normalized_polygon_to_image(polygon, 640, 480)
    assert points[0] == (64.0, 479.0)
    assert point_in_normalized_polygon((320, 400), polygon, 640, 480)
    assert not point_in_normalized_polygon((320, 100), polygon, 640, 480)
