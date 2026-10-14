from future_risk_vqa.motion import KalmanCV2D, estimate_camera_motion_hint


def test_constant_velocity_prediction_moves_forward():
    motion = KalmanCV2D((0.0, 0.0), 0.0, measurement_noise=0.01)
    motion.update((1.0, 0.0), 1.0)
    motion.update((2.0, 0.0), 2.0)
    path = motion.future_path(1.0, 0.5)
    assert motion.velocity[0] > 0.5
    assert path[-1][1][0] > motion.position[0]


def samples(*boxes):
    return [{"bbox_xyxy": box} for box in boxes]


def test_bbox_growth_produces_toward_camera_hint():
    hint = estimate_camera_motion_hint(
        samples((10, 20, 30, 60), (9, 15, 33, 65), (7, 8, 37, 68)),
        (100, 160, 3),
    )
    assert hint["motion_trend"] == "TOWARD_CAMERA"
    assert hint["reliable"]


def test_horizontal_bbox_shift_produces_lateral_hint():
    hint = estimate_camera_motion_hint(
        samples((10, 20, 30, 60), (20, 20, 40, 60), (32, 20, 52, 60)),
        (100, 160, 3),
    )
    assert hint["motion_trend"] == "LATERAL_ACROSS_VIEW"
