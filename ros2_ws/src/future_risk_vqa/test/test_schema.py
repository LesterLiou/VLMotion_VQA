from future_risk_vqa.schema import (
    extract_future_event_report,
    validate_future_event_report,
)


def test_joint_extractor_recovers_truncated_per_track_wrappers():
    raw = r'''[
    {"track\_id":"R1","scene\_visibility":"CLEAR","assessments":[
      {"track\_id":"R1","body\_posture":"STANDING","current\_activity":"OTHER",
       "future\_activity":"WAITING","future\_motion":"STATIONARY"},
    {"track\_id":"R2","scene\_visibility":"CLEAR","assessments":[
      {"track\_id":"R2","body\_posture":"SITTING","current\_activity":"WORKING",
       "future\_activity":"WORKING","future\_motion":"STATIONARY"}]
    }'''
    report = extract_future_event_report(raw)
    assert set(report) == {"assessments"}
    assert [item["track_id"] for item in report["assessments"]] == ["R1", "R2"]


def test_joint_extractor_flattens_complete_wrapper_array():
    raw = '''[
      {"scene_visibility":"CLEAR","assessments":[{"track_id":"R1"}]},
      {"scene_visibility":"CLEAR","assessments":[{"track_id":"R2"}]}
    ]'''
    report = extract_future_event_report(raw)
    assert [item["track_id"] for item in report["assessments"]] == ["R1", "R2"]


def test_vlm_geometry_field_is_rejected_but_report_remains_safe():
    raw = {
        "scene_visibility": "CLEAR",
        "assessments": [
            {
                "track_id": "L1",
                "current_observation": {
                    "body_orientation": "SIDE_ON",
                    "current_activity": "WALKING",
                    "motion_trend": "LATERAL_ACROSS_VIEW",
                    "confidence": 0.9,
                    "evidence": "moves left to right",
                },
                "future_prediction": {
                    "future_activity": "WALKING",
                    "future_motion": "LATERAL_ACROSS_VIEW",
                    "confidence": 0.85,
                    "evidence": "continues laterally",
                },
                "time_to_entry_s": 0.2,
            }
        ],
    }
    report = validate_future_event_report(raw, ["L1"])
    assert report["assessments"][0]["future_prediction"]["future_motion"] == "LATERAL_ACROSS_VIEW"
    assert any("forbidden geometry" in item for item in report["validation_errors"])
    assert "time_to_entry_s" not in report["assessments"][0]


def test_missing_current_track_is_a_schema_warning():
    report = validate_future_event_report(
        {"scene_visibility": "CLEAR", "assessments": []}, ["L9"]
    )
    assert report["assessments"] == []
    assert any("missing current track IDs" in item for item in report["validation_errors"])


def test_camera_relative_motion_is_preserved_without_workspace_mapping():
    report = validate_future_event_report(
        {
            "scene_visibility": "CLEAR",
            "assessments": [
                {
                    "track_id": "L9",
                    "current_observation": {
                        "body_orientation": "BACK_TO_CAMERA",
                        "current_activity": "WALKING",
                        "motion_trend": "AWAY_FROM_CAMERA",
                        "confidence": 0.8,
                        "evidence": "L9 becomes smaller from OLD to NOW",
                    },
                    "future_prediction": {
                        "future_activity": "WALKING",
                        "future_motion": "AWAY_FROM_CAMERA",
                        "confidence": 0.73,
                        "evidence": "L9 continues away from the camera",
                    },
                }
            ],
            "scene_summary": "L9 is moving away.",
        },
        ["L9"],
    )
    assert report["validation_errors"] == []
    assessment = report["assessments"][0]
    assert assessment["future_prediction"]["future_motion"] == "AWAY_FROM_CAMERA"
    assert "future_event" not in assessment


def test_zero_confidence_is_a_schema_warning_for_visible_track():
    report = validate_future_event_report(
        {
            "scene_visibility": "CLEAR",
            "assessments": [
                {
                    "track_id": "L9",
                    "current_observation": {
                        "body_orientation": "UNCERTAIN",
                        "current_activity": "UNCERTAIN",
                        "motion_trend": "UNCERTAIN",
                        "confidence": 0.0,
                        "evidence": "motion is unclear",
                    },
                    "future_prediction": {
                        "future_activity": "UNCERTAIN",
                        "future_motion": "UNCERTAIN",
                        "confidence": 0.0,
                        "evidence": "future motion is unclear",
                    },
                }
            ],
        },
        ["L9"],
    )
    assert any("greater than zero" in item for item in report["validation_errors"])


def test_uncalibrated_crossing_camera_plane_is_rejected():
    raw = {
        "scene_visibility": "CLEAR",
        "assessments": [
            {
                "track_id": "R2",
                "current_observation": {
                    "body_orientation": "FACING_CAMERA",
                    "current_activity": "WALKING",
                    "motion_trend": "TOWARD_CAMERA",
                    "confidence": 0.9,
                    "evidence": "R2 approaches",
                },
                "future_prediction": {
                    "future_activity": "WALKING",
                    "future_motion": "CROSSING_CAMERA_PLANE",
                    "confidence": 0.8,
                    "evidence": "R2 is likely to pass the camera plane",
                },
            }
        ],
    }
    report = validate_future_event_report(raw, ["R2"])
    assessment = report["assessments"][0]
    assert assessment["future_prediction"]["future_motion"] == "UNCERTAIN"
    assert "future_event" not in assessment
    assert any("future_motion: invalid value" in error for error in report["validation_errors"])


def test_removed_future_activity_is_ignored_without_schema_warning():
    raw = {
        "scene_visibility": "CLEAR",
        "assessments": [
            {
                "track_id": "L1",
                "current_observation": {
                    "body_orientation": "FACING_CAMERA",
                    "body_posture": "STANDING",
                    "current_activity": "OTHER",
                    "motion_trend": "TOWARD_CAMERA",
                    "confidence": 0.9,
                    "evidence": "L1 is visible",
                },
                "future_prediction": {
                    "future_activity": None,
                    "future_motion": "UNCERTAIN",
                    "confidence": 0.9,
                    "evidence": "future behavior is unclear",
                },
            }
        ],
    }
    report = validate_future_event_report(raw, ["L1"])
    assessment = report["assessments"][0]
    assert report["validation_errors"] == []
    assert "future_activity" not in assessment["future_prediction"]
    assert assessment["future_prediction"]["future_motion"] == "UNCERTAIN"


def test_body_posture_is_independent_from_current_activity():
    raw = {
        "scene_visibility": "CLEAR",
        "assessments": [
            {
                "track_id": "L2",
                "body_orientation": "SIDE_ON",
                "body_posture": "SQUATTING",
                "current_activity": "REACHING_OR_HANDLING_OBJECT",
                "motion_trend": "STATIONARY",
                "observation_confidence": 0.91,
                "observation_evidence": "L2 is squatting while handling an object",
                "future_activity": "WORKING",
                "future_motion": "STATIONARY",
                "prediction_confidence": 0.72,
                "prediction_evidence": "L2 is likely to continue the task",
            }
        ],
    }
    report = validate_future_event_report(raw, ["L2"])
    current = report["assessments"][0]["current_observation"]
    assert report["validation_errors"] == []
    assert current["body_posture"] == "SQUATTING"
    assert current["current_activity"] == "REACHING_OR_HANDLING_OBJECT"


def test_legacy_posture_in_activity_is_migrated():
    raw = {
        "scene_visibility": "CLEAR",
        "assessments": [
            {
                "track_id": "R3",
                "body_orientation": "FACING_CAMERA",
                "current_activity": "SITTING",
                "motion_trend": "STATIONARY",
                "observation_confidence": 0.8,
                "observation_evidence": "R3 is seated",
                "future_activity": "WAITING",
                "future_motion": "STATIONARY",
                "prediction_confidence": 0.7,
                "prediction_evidence": "R3 remains idle",
            }
        ],
    }
    report = validate_future_event_report(raw, ["R3"])
    current = report["assessments"][0]["current_observation"]
    assert report["validation_errors"] == []
    assert current["body_posture"] == "SITTING"
    assert current["current_activity"] == "OTHER"
    assert any("legacy current_activity posture migrated" in item for item in report["normalizations"])


def test_flat_small_model_contract_is_normalized_to_two_layers():
    raw = {
        "scene_visibility": "CLEAR",
        "assessments": [
            {
                "track_id": "R1",
                "body_orientation": "BACK_TO_CAMERA",
                "current_activity": "WALKING",
                "motion_trend": "AWAY_FROM_CAMERA",
                "observation_confidence": 0.88,
                "observation_evidence": "R1 gets smaller from OLD to NOW",
                "future_activity": "WALKING",
                "future_motion": "AWAY_FROM_CAMERA",
                "prediction_confidence": 0.79,
                "prediction_evidence": "R1 is likely to continue away",
            }
        ],
    }
    report = validate_future_event_report(raw, ["R1"])
    assessment = report["assessments"][0]
    assert report["validation_errors"] == []
    assert assessment["current_observation"]["confidence"] == 0.88
    assert assessment["future_prediction"]["confidence"] == 0.79
    assert assessment["future_prediction"]["future_motion"] == "AWAY_FROM_CAMERA"
    assert "future_event" not in assessment


def test_removed_stationary_future_activity_is_ignored():
    raw = {
        "assessments": [
            {
                "track_id": "L1",
                "body_orientation": "SIDE_ON",
                "body_posture": "STANDING",
                "current_activity": "OTHER",
                "observation_confidence": 0.8,
                "future_activity": "STATIONARY",
                "future_motion": "STATIONARY",
                "prediction_confidence": 0.7,
            }
        ]
    }
    report = validate_future_event_report(raw, ["L1"])
    assert report["validation_errors"] == []
    assert "future_activity" not in report["assessments"][0]["future_prediction"]
    assert report["assessments"][0]["future_prediction"]["future_motion"] == "STATIONARY"


def test_removed_motion_value_in_future_activity_is_ignored():
    raw = {
        "assessments": [
            {
                "track_id": "R6",
                "body_orientation": "FACING_CAMERA",
                "body_posture": "STANDING",
                "current_activity": "OTHER",
                "future_activity": "TOWARD_CAMERA",
                "future_motion": "TOWARD_CAMERA",
            }
        ]
    }
    report = validate_future_event_report(raw, ["R6"])
    assert report["validation_errors"] == []
    assert "future_activity" not in report["assessments"][0]["future_prediction"]
    assert report["assessments"][0]["future_prediction"]["future_motion"] == "TOWARD_CAMERA"


def test_seated_stationary_worker_keeps_stationary_motion():
    raw = {
        "assessments": [
            {
                "track_id": "R5",
                "body_orientation": "FACING_CAMERA",
                "body_posture": "SITTING",
                "current_activity": "WORKING",
                "future_activity": "WAITING",
                "future_motion": "STATIONARY",
            }
        ]
    }
    report = validate_future_event_report(raw, ["R5"])
    assert report["validation_errors"] == []
    assert report["assessments"][0]["future_prediction"]["future_motion"] == "STATIONARY"
    assert "future_event" not in report["assessments"][0]


def test_duplicate_posture_in_current_activity_is_normalized():
    raw = {
        "assessments": [
            {
                "track_id": "R1",
                "body_orientation": "FACING_CAMERA",
                "body_posture": "STANDING",
                "current_activity": "STANDING",
                "future_activity": "WAITING",
                "future_motion": "STATIONARY",
            }
        ]
    }
    report = validate_future_event_report(raw, ["R1"])
    assert report["validation_errors"] == []
    assert report["assessments"][0]["current_observation"]["body_posture"] == "STANDING"
    assert report["assessments"][0]["current_observation"]["current_activity"] == "OTHER"
    assert "posture value STANDING normalized to OTHER" in report["normalizations"][0]


def test_removed_posture_in_future_activity_is_ignored():
    raw = {
        "assessments": [
            {
                "track_id": "R2",
                "body_orientation": "SIDE_ON",
                "body_posture": "SITTING",
                "current_activity": "OTHER",
                "future_activity": "SITTING",
                "future_motion": "STATIONARY",
            }
        ]
    }
    report = validate_future_event_report(raw, ["R2"])
    assert report["validation_errors"] == []
    assert "future_activity" not in report["assessments"][0]["future_prediction"]


def test_removed_future_activity_alias_is_ignored():
    raw = {
        "assessments": [
            {
                "track_id": "R1",
                "body_orientation": "FACING_CAMERA",
                "body_posture": "SITTING",
                "current_activity": "WORKING",
                "future_activity": "CONTINUE_WORKING",
                "future_motion": "AWAY_FROM_CAMERA",
            }
        ]
    }
    report = validate_future_event_report(raw, ["R1"])
    assert report["validation_errors"] == []
    assert "future_activity" not in report["assessments"][0]["future_prediction"]
    assert report["assessments"][0]["future_prediction"]["future_motion"] == "AWAY_FROM_CAMERA"
