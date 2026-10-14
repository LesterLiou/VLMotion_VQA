"""Post-validation quality controls for VLM behavior assessments."""

from collections import defaultdict
import re

def _normalized_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def output_signature(assessment):
    """Return the model-originated semantic signature used for collapse checks."""
    current = assessment.get("current_observation", {})
    future = assessment.get("future_prediction", {})
    return (
        current.get("body_orientation"),
        current.get("body_posture"),
        current.get("current_activity"),
        current.get("vlm_motion_trend", current.get("motion_trend")),
        round(float(current.get("confidence") or 0.0), 3),
        _normalized_text(current.get("evidence", "")),
        future.get("vlm_future_motion", future.get("future_motion")),
        round(float(future.get("confidence") or 0.0), 3),
        _normalized_text(future.get("evidence", "")),
    )


def detect_output_collapse(assessments, minimum_identical_tracks=3):
    """Flag exact multi-ID template copies without claiming all equal labels are wrong."""
    groups = defaultdict(list)
    for item in assessments:
        if isinstance(item, dict):
            groups[output_signature(item)].append(str(item.get("track_id")))
    minimum = max(2, int(minimum_identical_tracks))
    return [track_ids for track_ids in groups.values() if len(track_ids) >= minimum]


def apply_tracking_motion(assessment, hint, mode="tracking_hint"):
    """Keep raw VLM motion, then optionally use image-space tracking for policy motion."""
    assessment["tracking_motion_hint"] = hint
    if mode != "tracking_hint":
        return assessment
    current = assessment.get("current_observation", {})
    future = assessment.get("future_prediction", {})
    current["vlm_motion_trend"] = current.get("motion_trend", "UNCERTAIN")
    future["vlm_future_motion"] = future.get("future_motion", "UNCERTAIN")
    seated_and_not_transitioning = (
        str(current.get("body_posture", "")).upper() == "SITTING"
        and str(current.get("current_activity", "")).upper()
        not in {"WALKING", "CHANGING_POSTURE"}
    )
    if seated_and_not_transitioning:
        current["motion_trend"] = "STATIONARY"
        current["motion_source"] = "seated_stationary_rule"
        future["future_motion"] = "STATIONARY"
        future["motion_source"] = "seated_stationary_rule"
        return assessment
    if not hint.get("reliable", False):
        current["motion_trend"] = "UNCERTAIN"
        current["motion_source"] = "tracking_hint_unreliable"
        future["future_motion"] = "UNCERTAIN"
        future["motion_source"] = "tracking_hint_unreliable"
        return assessment
    current["motion_trend"] = hint["motion_trend"]
    current["motion_source"] = "tracking_hint"
    future_motion = hint["motion_trend"]
    source = "tracking_motion_persistence"
    future["future_motion"] = future_motion
    future["motion_source"] = source
    return assessment


def apply_effective_confidence(
    assessment,
    crop_coverage_ratio,
    tracking_reliable,
    enabled=True,
    unreliable_tracking_scale=0.5,
):
    """Derive a policy quality value from crop coverage and tracking reliability."""
    base_quality = float(assessment.get("confidence", 1.0))
    coverage = min(1.0, max(0.0, float(crop_coverage_ratio)))
    tracking_scale = 1.0 if tracking_reliable else float(unreliable_tracking_scale)
    effective = base_quality
    if enabled:
        effective *= coverage * tracking_scale
    assessment["confidence"] = round(min(1.0, max(0.0, effective)), 3)
    assessment["confidence_control"] = {
        "enabled": bool(enabled),
        "crop_coverage_ratio": round(coverage, 3),
        "tracking_reliable": bool(tracking_reliable),
        "tracking_scale": round(tracking_scale, 3),
        "effective_confidence": assessment["confidence"],
    }
    return assessment
