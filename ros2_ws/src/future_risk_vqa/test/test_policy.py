from future_risk_vqa.policy import FutureBehaviorPolicy, FutureRiskPolicy


def agent(zone="OUTSIDE", tte=None, intersects=False, reliable=True):
    return {
        "track_id": "L1",
        "current_zone": zone,
        "time_to_entry_s": tte,
        "trajectory_intersects_hazard": intersects,
        "tracking_reliable": reliable,
    }


def semantics(motion="LATERAL_ACROSS_VIEW", confidence=0.9, visibility="CLEAR"):
    return {
        "scene_visibility": visibility,
        "assessments": [
            {
                "track_id": "L1",
                "future_prediction": {"future_motion": motion},
                "confidence": confidence,
            }
        ],
    }


def geometry(item):
    return {"localization_valid": True, "tracked_agents": [item]}


def test_geometry_stop_has_priority_over_vlm_safe_claim():
    result = FutureRiskPolicy().evaluate(
        geometry(agent(zone="HAZARD_ZONE")), semantics("AWAY_FROM_CAMERA")
    )
    assert result.action == "STOP"


def test_future_intersection_returns_danger_warning():
    result = FutureRiskPolicy().evaluate(geometry(agent(tte=2.0, intersects=True)), semantics())
    assert result.action == "DANGER_WARNING"


def test_continue_needs_two_distinct_evaluations():
    policy = FutureRiskPolicy(continue_confirmations_required=2)
    assert policy.evaluate(geometry(agent()), semantics()).action == "DANGER_WARNING"
    assert policy.evaluate(geometry(agent()), semantics()).action == "CONTINUE"


def test_unreliable_nearby_track_warns():
    result = FutureRiskPolicy().evaluate(
        geometry(agent(zone="WARNING_ZONE", reliable=False)), semantics()
    )
    assert result.action == "DANGER_WARNING"


def test_semantic_schema_warning_is_fail_safe():
    report = semantics()
    report["validation_errors"] = ["forbidden geometry field"]
    result = FutureRiskPolicy().evaluate(geometry(agent()), report)
    assert result.action == "DANGER_WARNING"
    assert result.reason_code == "semantic_schema_invalid"


def behavior_agents(reliable=True):
    return [{"track_id": "L1", "tracking_reliable": reliable}]


def test_future_behavior_has_no_uncalibrated_motion_stop_by_default():
    policy = FutureBehaviorPolicy()
    result = policy.evaluate(semantics("CROSSING_CAMERA_PLANE", confidence=0.85), behavior_agents())
    assert result.action == "DANGER_WARNING"
    assert result.critical_tracks == ("L1",)


def test_future_behavior_approaching_returns_danger_warning():
    policy = FutureBehaviorPolicy()
    result = policy.evaluate(
        semantics("TOWARD_CAMERA", confidence=0.75), behavior_agents()
    )
    assert result.action == "DANGER_WARNING"


def test_future_behavior_quality_score_does_not_override_safe_motion():
    policy = FutureBehaviorPolicy(continue_confirmations_required=1)
    result = policy.evaluate(semantics("LATERAL_ACROSS_VIEW", confidence=0.5), behavior_agents())
    assert result.action == "CONTINUE"


def test_future_behavior_continue_requires_confirmation():
    policy = FutureBehaviorPolicy(continue_confirmations_required=2)
    assert policy.evaluate(semantics(), behavior_agents()).action == "DANGER_WARNING"
    assert policy.evaluate(semantics(), behavior_agents()).action == "CONTINUE"


def test_no_visible_person_returns_no_response_without_vqa():
    policy = FutureBehaviorPolicy()
    assert policy.evaluate(None, []).action == "NO_RESPONSE"
    hidden = [{"track_id": "L1", "visible": False, "missed_frames": 1}]
    assert policy.evaluate(semantics(), hidden).action == "NO_RESPONSE"


def test_model_unavailable_is_not_mislabeled_as_schema_invalid():
    policy = FutureBehaviorPolicy()
    report = {
        "availability": "UNAVAILABLE",
        "validation_errors": ["network busy"],
    }
    result = policy.evaluate(report, behavior_agents())
    assert result.action == "DANGER_WARNING"
    assert result.reason_code == "future_behavior_unavailable"


def test_output_collapse_quality_warning_does_not_override_safe_motion():
    policy = FutureBehaviorPolicy(continue_confirmations_required=1)
    report = semantics("STATIONARY", confidence=0.9)
    report["quality_warnings"] = [
        {"code": "semantic_output_collapse", "track_ids": ["L1", "L2", "R1"]}
    ]
    result = policy.evaluate(report, behavior_agents())
    assert result.action == "CONTINUE"
    assert result.reason_code == "vqa_future_clear_confirmed"


def test_zone_policy_no_person_returns_no_response():
    result = FutureRiskPolicy().evaluate(
        {"localization_valid": True, "tracked_agents": []}, semantics()
    )
    assert result.action == "NO_RESPONSE"
