"""Closed schema for camera-relative current observation and future prediction."""

import ast
import json
import math


BODY_ORIENTATIONS = {"FACING_CAMERA", "BACK_TO_CAMERA", "SIDE_ON", "UNCERTAIN"}
BODY_POSTURES = {
    "STANDING",
    "SQUATTING",
    "SITTING",
    "KNEELING",
    "LYING_DOWN",
    "BENDING_OR_CROUCHING",
    "UNCERTAIN",
}
ACTIVITIES = {
    "WALKING",
    "CHANGING_POSTURE",
    "REACHING_OR_HANDLING_OBJECT",
    "WORKING",
    "OTHER",
    "UNCERTAIN",
}
ACTIVITY_ALIASES = {
    "STOPPING": "OTHER",
    "WAITING": "OTHER",
    "CONTINUE_WORKING": "WORKING",
    "CONTINUING_WORK": "WORKING",
    "CONTINUING_TO_WORK": "WORKING",
    "KEEP_WORKING": "WORKING",
}
MOTION_TRENDS = {
    "TOWARD_CAMERA",
    "AWAY_FROM_CAMERA",
    "LATERAL_ACROSS_VIEW",
    "STATIONARY",
    "UNCERTAIN",
}
FUTURE_MOTIONS = MOTION_TRENDS
def _camera_name(track_id):
    return "left" if str(track_id).upper().startswith("L") else "right"


def extract_json_object(text):
    if not isinstance(text, str):
        raise ValueError("model response must be text")
    normalized = text.replace("\\_", "_")
    decoder = json.JSONDecoder()
    for index, character in enumerate(normalized):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(normalized[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    for start, character in enumerate(normalized):
        if character != "{":
            continue
        for end in range(len(normalized) - 1, start, -1):
            if normalized[end] != "}":
                continue
            try:
                value = ast.literal_eval(normalized[start : end + 1])
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("model response does not contain an object")


def extract_future_event_report(text):
    """Recover the report shape emitted by small VLMs in joint mode.

    The preferred contract is one top-level object. Some small models instead emit a
    top-level list, wrap every assessment in its own report, or truncate those wrappers
    while leaving the flat assessment objects intact. Preserve the strict validator,
    but normalize those transport-level shapes before validation.
    """
    if not isinstance(text, str):
        raise ValueError("model response must be text")
    normalized = text.replace("\\_", "_")
    decoder = json.JSONDecoder()
    stripped = normalized.strip()
    root = None
    if stripped.startswith(("{", "[")):
        try:
            root, _ = decoder.raw_decode(stripped)
        except json.JSONDecodeError:
            root = None
    decoded = []
    for index, character in enumerate(normalized):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(normalized[index:])
        except json.JSONDecodeError:
            continue
        decoded.append(value)

    # Always prefer the actual root report object when the model followed the contract.
    if isinstance(root, dict) and isinstance(root.get("assessments"), list):
        return root

    def assessment_items(value):
        if not isinstance(value, list):
            return []
        items = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            nested = entry.get("assessments")
            if isinstance(nested, list):
                items.extend(item for item in nested if isinstance(item, dict))
            elif "track_id" in entry:
                items.append(entry)
        return items

    # Accept a complete top-level array and flatten per-track report wrappers.
    root_values = [root] if isinstance(root, list) else []
    for value in root_values:
        items = assessment_items(value)
        if items:
            return {"assessments": items}

    # Last-resort recovery for a truncated array of wrappers: inner flat assessment
    # objects remain independently decodable even when the outer JSON is incomplete.
    flat_markers = {
        "body_orientation",
        "body_posture",
        "current_activity",
        "motion_trend",
        "future_motion",
    }
    recovered = []
    seen = set()
    for value in decoded:
        if not isinstance(value, dict) or "track_id" not in value:
            continue
        if not flat_markers.intersection(value):
            continue
        track_id = str(value.get("track_id", "")).strip()
        if not track_id or track_id in seen:
            continue
        seen.add(track_id)
        recovered.append(value)
    if recovered:
        return {"assessments": recovered}

    # Retain the old failure behavior and error message for completely unusable text.
    return extract_json_object(text)


def _enum(value, allowed, fallback, field, errors):
    normalized = str(value).strip().upper()
    if normalized not in allowed:
        errors.append(f"{field}: invalid value {value!r}")
        return fallback
    return normalized


def _observable_enum(value, allowed, field, errors, normalizations):
    """Treat model null as an explicit unknown without hiding invalid non-null values."""
    if value is None:
        normalizations.append(f"{field}: null normalized to UNCERTAIN")
        return "UNCERTAIN"
    return _enum(value, allowed, "UNCERTAIN", field, errors)


def _confidence(value, field, errors):
    if isinstance(value, bool):
        errors.append(f"{field}: must be numeric")
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field}: must be numeric")
        return 0.0
    if not math.isfinite(number):
        errors.append(f"{field}: must be finite")
        return 0.0
    return round(min(1.0, max(0.0, number)), 3)


def validate_future_event_report(raw_report, known_track_ids):
    """Validate camera-relative current observations and future predictions."""
    if not isinstance(raw_report, dict):
        raise ValueError("future event report must be an object")
    known = {str(track_id) for track_id in known_track_ids}
    errors = []
    normalizations = []
    raw_assessments = raw_report.get("assessments")
    if not isinstance(raw_assessments, list):
        errors.append("assessments: must be a list")
        raw_assessments = []
    assessments = []
    seen = set()
    forbidden = {
        "position",
        "position_xy",
        "velocity",
        "velocity_xy",
        "time_to_entry_s",
        "tte",
        "distance",
        "inside_hazard_zone",
    }
    for index, raw in enumerate(raw_assessments[:24]):
        field = f"assessments[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{field}: must be an object")
            continue
        leaked = sorted(forbidden.intersection(raw))
        if leaked:
            errors.append(f"{field}: forbidden geometry fields {leaked}")
        track_id = str(raw.get("track_id", "")).strip()
        if not track_id or track_id not in known or track_id in seen:
            errors.append(f"{field}.track_id: unknown, missing, or duplicated")
            continue
        seen.add(track_id)
        current = raw.get("current_observation")
        future = raw.get("future_prediction")
        flat_current_keys = {
            "body_orientation",
            "body_posture",
            "current_activity",
            "motion_trend",
            "observation_confidence",
            "observation_evidence",
        }
        flat_future_keys = {
            "future_motion",
            "prediction_confidence",
            "prediction_evidence",
        }
        if not isinstance(current, dict) and flat_current_keys.intersection(raw):
            current = {
                "body_orientation": raw.get("body_orientation"),
                "body_posture": raw.get("body_posture"),
                "current_activity": raw.get("current_activity"),
                "motion_trend": raw.get("motion_trend"),
                "confidence": raw.get("observation_confidence"),
                "evidence": raw.get("observation_evidence"),
            }
        if not isinstance(future, dict) and flat_future_keys.intersection(raw):
            future = {
                "future_motion": raw.get("future_motion"),
                "confidence": raw.get("prediction_confidence"),
                "evidence": raw.get("prediction_evidence"),
            }
        if not isinstance(current, dict):
            errors.append(f"{field}.current_observation: must be an object")
            current = {}
        if not isinstance(future, dict):
            errors.append(f"{field}.future_prediction: must be an object")
            future = {}
        # Confidence is not part of the compact VLM contract. Keep accepting legacy
        # values, but use a neutral base before deterministic quality controls.
        observation_confidence = (
            None
            if current.get("confidence") is None
            else _confidence(
                current.get("confidence"),
                f"{field}.current_observation.confidence",
                errors,
            )
        )
        prediction_confidence = (
            None
            if future.get("confidence") is None
            else _confidence(
                future.get("confidence"),
                f"{field}.future_prediction.confidence",
                errors,
            )
        )
        if observation_confidence is not None and observation_confidence <= 0.0:
            errors.append(
                f"{field}.current_observation.confidence: must be greater than zero"
            )
        if prediction_confidence is not None and prediction_confidence <= 0.0:
            errors.append(
                f"{field}.future_prediction.confidence: must be greater than zero"
            )
        body_orientation = _observable_enum(
            current.get("body_orientation"),
            BODY_ORIENTATIONS,
            f"{field}.current_observation.body_orientation",
            errors,
            normalizations,
        )
        body_posture_value = current.get("body_posture")
        current_activity_value = current.get("current_activity")
        raw_current_activity = str(current_activity_value).strip().upper()
        if raw_current_activity in ACTIVITY_ALIASES:
            current_activity_value = ACTIVITY_ALIASES[raw_current_activity]
            normalizations.append(
                f"{field}.current_observation.current_activity: "
                f"{raw_current_activity} normalized to {current_activity_value}"
            )
        if body_posture_value is None and str(current_activity_value).upper() in {
            "STANDING",
            "SQUATTING",
            "SITTING",
        }:
            body_posture_value = str(current_activity_value).upper()
            current_activity_value = "OTHER"
            normalizations.append(
                f"{field}: legacy current_activity posture migrated to body_posture"
            )
        elif str(current_activity_value).strip().upper() in BODY_POSTURES - {
            "UNCERTAIN"
        }:
            misplaced_posture = str(current_activity_value).strip().upper()
            current_activity_value = "OTHER"
            normalizations.append(
                f"{field}.current_observation.current_activity: posture value "
                f"{misplaced_posture} normalized to OTHER"
            )
        body_posture = _observable_enum(
            body_posture_value,
            BODY_POSTURES,
            f"{field}.current_observation.body_posture",
            errors,
            normalizations,
        )
        current_activity = _observable_enum(
            current_activity_value,
            ACTIVITIES,
            f"{field}.current_observation.current_activity",
            errors,
            normalizations,
        )
        # Current motion comes from tracking, not the compact VLM contract. Preserve
        # legacy model output when present without requiring the field.
        motion_trend = (
            "UNCERTAIN"
            if current.get("motion_trend") is None
            else _observable_enum(
                current.get("motion_trend"),
                MOTION_TRENDS,
                f"{field}.current_observation.motion_trend",
                errors,
                normalizations,
            )
        )
        future_motion = _observable_enum(
            future.get("future_motion"),
            FUTURE_MOTIONS,
            f"{field}.future_prediction.future_motion",
            errors,
            normalizations,
        )
        camera = _camera_name(track_id)
        assessments.append(
            {
                "track_id": track_id,
                "camera": camera,
                "current_observation": {
                    "body_orientation": body_orientation,
                    "body_posture": body_posture,
                    "current_activity": current_activity,
                    "motion_trend": motion_trend,
                    "confidence": observation_confidence,
                    "evidence": str(current.get("evidence", "")).strip()[:240],
                },
                "future_prediction": {
                    "future_motion": future_motion,
                    "confidence": prediction_confidence,
                    "evidence": str(future.get("evidence", "")).strip()[:240],
                },
                "confidence": (
                    min(observation_confidence, prediction_confidence)
                    if observation_confidence is not None
                    and prediction_confidence is not None
                    else 1.0
                ),
                "evidence": str(future.get("evidence", "")).strip()[:240],
            }
        )
    missing = sorted(known - seen)
    if missing:
        errors.append(f"assessments: missing current track IDs {missing}")
    critical_track_id = raw_report.get("critical_track_id")
    if critical_track_id is not None:
        critical_track_id = str(critical_track_id).strip()
        if critical_track_id not in known:
            errors.append("critical_track_id: unknown track")
            critical_track_id = None
    return {
        "schema_version": "3.1",
        "critical_track_id": critical_track_id,
        "assessments": assessments,
        "scene_summary": str(raw_report.get("scene_summary", "")).strip()[:500],
        "validation_errors": errors,
        "normalizations": normalizations,
    }
