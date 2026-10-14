from future_risk_vqa.semantic_quality import (
    apply_effective_confidence,
    apply_tracking_motion,
    detect_output_collapse,
)


def assessment(track_id, evidence="person standing"):
    return {
        "track_id": track_id,
        "current_observation": {
            "body_orientation": "FACING_CAMERA",
            "current_activity": "STANDING",
            "motion_trend": "TOWARD_CAMERA",
            "confidence": 0.9,
            "evidence": evidence,
        },
        "future_prediction": {
            "future_activity": "WAITING",
            "future_motion": "TOWARD_CAMERA",
            "confidence": 0.9,
            "evidence": "person waiting",
        },
        "confidence": 0.9,
    }


def test_three_identical_model_signatures_are_flagged():
    groups = detect_output_collapse(
        [assessment("L1"), assessment("L2"), assessment("R1")], 3
    )
    assert groups == [["L1", "L2", "R1"]]


def test_distinct_evidence_prevents_exact_collapse_flag():
    groups = detect_output_collapse(
        [assessment("L1", "back visible"), assessment("L2", "side visible")], 2
    )
    assert groups == []


def test_tracking_motion_is_auditable_and_changes_effective_motion():
    item = assessment("L1")
    apply_tracking_motion(
        item,
        {
            "motion_trend": "AWAY_FROM_CAMERA",
            "reliable": True,
            "confidence": 0.8,
        },
    )
    assert item["current_observation"]["vlm_motion_trend"] == "TOWARD_CAMERA"
    assert item["current_observation"]["motion_trend"] == "AWAY_FROM_CAMERA"
    assert item["future_prediction"]["vlm_future_motion"] == "TOWARD_CAMERA"
    assert item["future_prediction"]["future_motion"] == "AWAY_FROM_CAMERA"
    assert "future_event" not in item


def test_unreliable_tracking_cannot_become_directional_motion():
    item = assessment("R9")
    apply_tracking_motion(
        item,
        {"motion_trend": "UNCERTAIN", "reliable": False, "confidence": 0.293},
    )
    assert item["future_prediction"]["vlm_future_motion"] == "TOWARD_CAMERA"
    assert item["future_prediction"]["future_motion"] == "UNCERTAIN"
    assert "future_event" not in item


def test_tracking_stationary_seated_worker_remains_stationary():
    item = assessment("R5")
    item["current_observation"].update(
        {"body_posture": "SITTING", "current_activity": "WORKING"}
    )
    apply_tracking_motion(
        item,
        {"motion_trend": "STATIONARY", "reliable": True, "confidence": 1.0},
    )
    assert item["future_prediction"]["future_motion"] == "STATIONARY"
    assert "future_event" not in item


def test_seated_non_transitioning_person_overrides_uncertain_tracking_to_stationary():
    item = assessment("R5")
    item["current_observation"].update(
        {
            "body_posture": "SITTING",
            "current_activity": "WORKING",
            "motion_trend": "UNCERTAIN",
        }
    )
    item["future_prediction"]["future_motion"] = "UNCERTAIN"
    apply_tracking_motion(
        item,
        {"motion_trend": "UNCERTAIN", "reliable": False, "confidence": 0.0},
    )
    assert item["current_observation"]["motion_trend"] == "STATIONARY"
    assert item["future_prediction"]["future_motion"] == "STATIONARY"
    assert item["future_prediction"]["motion_source"] == "seated_stationary_rule"


def test_effective_confidence_penalizes_incomplete_crop_history():
    item = assessment("L1")
    apply_effective_confidence(item, 2.0 / 3.0, True)
    assert "vlm_self_reported_confidence" not in item
    assert item["confidence"] == 0.6
