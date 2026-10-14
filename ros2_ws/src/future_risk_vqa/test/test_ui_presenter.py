from future_risk_vqa.ui_presenter import camera_behavior_summary, explain_decision


def test_camera_summary_separates_left_and_right_tracks():
    report = {
        "scene_visibility": "CLEAR",
        "assessments": [
            {"track_id": "L1", "future_motion": "TOWARD_CAMERA", "confidence": 0.8},
            {"track_id": "R2", "future_motion": "AWAY_FROM_CAMERA", "confidence": 0.9},
        ],
    }
    left, left_level = camera_behavior_summary(report, "L")
    right, right_level = camera_behavior_summary(report, "R")
    assert "L1 | toward camera" in left
    assert left_level == "warning"
    assert "R2 | away from camera" in right
    assert right_level == "clear"


def test_decision_explanation_names_critical_track():
    report = {
        "action": "DANGER_WARNING",
        "reason_code": "vqa_future_approaching_warning",
        "critical_tracks": ["L1"],
    }
    assert "L1" in explain_decision(report)
    assert "toward" in explain_decision(report)


def test_mixed_warning_explanation_only_calls_approaching_track_approaching():
    decision = {
        "action": "DANGER_WARNING",
        "reason_code": "vqa_future_approaching_warning",
        "critical_tracks": ["R13", "R8", "R9"],
    }
    semantics = {
        "assessments": [
            {"track_id": "R13", "future_prediction": {"future_motion": "UNCERTAIN"}},
            {"track_id": "R8", "future_prediction": {"future_motion": "UNCERTAIN"}},
            {"track_id": "R9", "future_prediction": {"future_motion": "TOWARD_CAMERA"}},
        ]
    }
    text = explain_decision(decision, semantics)
    assert "R9 is moving toward" in text
    assert "R13, R8 have another non-safe future motion" in text
    assert "R13, R8, R9 is moving toward" not in text


def test_schema_error_is_shown_instead_of_no_person():
    report = {
        "scene_visibility": "CLEAR",
        "assessments": [],
        "validation_errors": ["assessments[0].track_id: unknown track"],
    }
    text, level = camera_behavior_summary(report, "L")
    assert "SCHEMA WARNING" in text
    assert "unknown track" in text
    assert level == "warning"


def test_camera_without_input_tracks_shows_no_response_despite_global_schema_error():
    report = {
        "scene_visibility": "UNRELIABLE",
        "assessments": [],
        "validation_errors": ["assessments: must be a list"],
        "metadata": {
            "input_tracks": [
                {"track_id": "R1", "visible": True, "missed_frames": 0},
                {"track_id": "R2", "visible": True, "missed_frames": 0},
            ]
        },
    }
    left, left_level = camera_behavior_summary(report, "L")
    right, right_level = camera_behavior_summary(report, "R")
    assert left == "NO RESPONSE\nNo person detected"
    assert left_level == "unknown"
    assert "VQA SCHEMA WARNING" in right
    assert right_level == "warning"


def test_no_response_explains_no_visible_person():
    report = {"action": "NO_RESPONSE", "reason_code": "no_person_detected"}
    assert "No visible person" in explain_decision(report)


def test_approximate_image_stop_zone_explanation_is_explicit():
    report = {
        "action": "STOP",
        "reason_code": "approximate_image_stop_zone",
        "critical_tracks": ["R2"],
    }
    text = explain_decision(report)
    assert "R2" in text
    assert "approximate image-space 1 m stop zone" in text


def test_model_unavailable_uses_na_instead_of_zero_confidence_or_schema_warning():
    report = {
        "availability": "UNAVAILABLE",
        "unavailable_reason": "network busy",
        "scene_visibility": "UNRELIABLE",
        "assessments": [
            {"track_id": "L3", "future_motion": "UNCERTAIN", "confidence": None}
        ],
        "validation_errors": ["network busy"],
    }
    text, level = camera_behavior_summary(report, "L")
    assert "VQA UNAVAILABLE" in text
    assert "Confidence: N/A" in text
    assert "confidence 0%" not in text
    assert "SCHEMA WARNING" not in text
    assert level == "warning"


def test_unavailable_decision_explains_model_error():
    decision = {
        "action": "DANGER_WARNING",
        "reason_code": "future_behavior_unavailable",
        "critical_tracks": ["L3"],
    }
    semantics = {"unavailable_reason": "network busy"}
    text = explain_decision(decision, semantics)
    assert "Model error: network busy" in text


def test_two_layer_summary_shows_now_next_without_map():
    report = {
        "scene_visibility": "CLEAR",
        "assessments": [
            {
                "track_id": "L3",
                "current_observation": {
                    "body_orientation": "FACING_CAMERA",
                    "body_posture": "STANDING",
                    "current_activity": "WALKING",
                    "motion_trend": "TOWARD_CAMERA",
                    "confidence": 0.82,
                },
                "future_prediction": {
                    "future_activity": "WALKING",
                    "future_motion": "TOWARD_CAMERA",
                    "confidence": 0.76,
                },
                "confidence": 0.76,
            }
        ],
    }
    text, level = camera_behavior_summary(report, "L")
    assert "L3 NOW: standing posture, walking, toward camera, facing camera" in text
    assert "L3 NEXT: toward camera" in text
    assert "MAP:" not in text
    assert "%" not in text
    assert level == "warning"


def test_now_next_summary_has_no_workspace_mapping_layer():
    report = {
        "assessments": [
            {
                "track_id": "R1",
                "current_observation": {
                    "body_orientation": "FACING_CAMERA",
                    "body_posture": "SITTING",
                    "current_activity": "WORKING",
                    "motion_trend": "STATIONARY",
                },
                "future_prediction": {
                    "future_activity": "WORKING",
                    "future_motion": "STATIONARY",
                },
            }
        ],
    }
    text, level = camera_behavior_summary(report, "R")
    assert "R1 NOW:" in text
    assert "R1 NEXT:" in text
    assert "MAP:" not in text
    assert level == "clear"


def test_output_collapse_is_diagnostic_info_only():
    report = {
        "scene_visibility": "CLEAR",
        "assessments": [
            {"track_id": "L1", "future_motion": "UNCERTAIN", "confidence": 0.0}
        ],
        "quality_warnings": [
            {"code": "semantic_output_collapse", "track_ids": ["L1", "L2", "R1"]}
        ],
    }
    text, level = camera_behavior_summary(report, "L")
    assert "QUALITY INFO: semantic_output_collapse" in text
    assert level == "warning"  # the assessment motion itself is UNCERTAIN

    decision = {
        "action": "DANGER_WARNING",
        "reason_code": "future_behavior_output_collapse",
        "critical_tracks": ["L1", "L2", "R1"],
    }
    assert "copied an identical answer" in explain_decision(decision)
