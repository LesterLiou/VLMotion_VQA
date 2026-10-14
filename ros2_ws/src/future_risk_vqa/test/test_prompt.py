from future_risk_vqa.prompt import build_future_event_prompt
from future_risk_vqa.schema import validate_future_event_report


def test_future_only_prompt_contains_no_zone_or_tte_contract():
    prompt = build_future_event_prompt(
        "Predict visible future behavior.",
        {
            "zone_logic_enabled": False,
            "tracks": [{"track_id": "L1", "camera": "left"}],
        },
    )
    lowered = prompt.lower()
    assert "zone" not in lowered
    assert "hazard" not in lowered
    assert "warning zone" not in lowered
    assert "stop zone" not in lowered
    assert "time_to_entry" not in lowered
    assert "current observation" in lowered
    assert "future prediction" in lowered
    assert "toward_camera" in lowered
    assert "away_from_camera" in lowered
    assert "workspace-relative event" not in lowered


def test_two_layer_camera_relative_vocabulary_validates_without_map_layer():
    report = validate_future_event_report(
        {
            "scene_visibility": "CLEAR",
            "assessments": [
                {
                    "track_id": "L1",
                    "current_observation": {
                        "body_orientation": "FACING_CAMERA",
                        "current_activity": "WALKING",
                        "motion_trend": "TOWARD_CAMERA",
                        "confidence": 0.8,
                        "evidence": "L1 moves closer from OLD to NOW",
                    },
                    "future_prediction": {
                        "future_activity": "WALKING",
                        "future_motion": "TOWARD_CAMERA",
                        "confidence": 0.75,
                        "evidence": "L1 continues the same motion",
                    },
                }
            ],
        },
        ["L1"],
    )
    assert report["validation_errors"] == []
    assessment = report["assessments"][0]
    assert assessment["future_prediction"]["future_motion"] == "TOWARD_CAMERA"
    assert "future_event" not in assessment
    assert assessment["confidence"] == 0.75


def test_trajectory_context_modes_change_vqa_contract():
    base = {
        "zone_logic_enabled": False,
        "tracks": [{"track_id": "L1", "camera": "left"}],
    }
    overlay = build_future_event_prompt(
        "Predict visible future behavior.",
        {**base, "trajectory_context_mode": "overlay"},
    )
    numeric = build_future_event_prompt(
        "Predict visible future behavior.",
        {
            **base,
            "trajectory_context_mode": "numeric",
            "tracks": [
                {
                    "track_id": "L1",
                    "camera": "left",
                    "trusted_motion_prediction": {"future_path": [[0.1, 0.2]]},
                }
            ],
        },
    )
    assert "purple path overlay" in overlay.lower()
    assert "trusted_motion_prediction" in numeric


def test_prompt_contract_uses_current_ids_without_default_answers():
    prompt = build_future_event_prompt(
        "Predict visible future behavior.",
        {
            "zone_logic_enabled": False,
            "tracks": [
                {"track_id": "L9", "camera": "left"},
                {"track_id": "R4", "camera": "right"},
            ],
        },
    )
    assert 'Current track IDs are exactly: ["L9", "R4"]' in prompt
    assert '"track_id": "L3"' not in prompt
    assert '"confidence": 0.0' not in prompt
    assert "Replace with visible" not in prompt
    assert "observation_confidence" not in prompt
    assert "prediction_confidence" not in prompt
    assert "body_posture" in prompt
    assert "future_activity" not in prompt
    assert "SQUATTING" in prompt
    assert "Keep posture and activity independent" in prompt
    assert "keep each assessment flat" in prompt.lower()
    assert "do not output motion_trend" in prompt.lower()
    assert "scene_summary" not in prompt


def test_hybrid_prompt_separates_global_context_from_track_crops():
    prompt = build_future_event_prompt(
        "Predict visible future behavior.",
        {
            "storyboard_mode": "hybrid",
            "tracks": [{"track_id": "R2", "camera": "right"}],
        },
    )
    lowered = prompt.lower()
    assert "full-scene context" in lowered
    assert "old/mid/now crop strip" in lowered
    assert "never transfer an answer between ids" in lowered


def test_track_crop_prompt_explains_per_id_rows_and_missing_history():
    prompt = build_future_event_prompt(
        "Predict visible future behavior.",
        {
            "storyboard_mode": "track_crops",
            "tracks": [{"track_id": "R2", "camera": "right"}],
        },
    )
    lowered = prompt.lower()
    assert "each horizontal row belongs to exactly one labeled track id" in lowered
    assert "old through mid to now" in lowered
    assert "track not found" in lowered
    assert "do not substitute another person" in lowered


def test_prompt_explains_trusted_tracking_motion_hint():
    prompt = build_future_event_prompt(
        "Predict visible future behavior.",
        {
            "storyboard_mode": "track_crops",
            "tracks": [
                {
                    "track_id": "L1",
                    "camera": "left",
                    "trusted_camera_motion_hint": {
                        "motion_trend": "AWAY_FROM_CAMERA",
                        "reliable": True,
                    },
                }
            ],
        },
    )
    assert "trusted_camera_motion_hint" in prompt
    assert "image-space observation" in prompt
