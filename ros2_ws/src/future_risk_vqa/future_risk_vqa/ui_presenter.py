"""Human-readable presentation helpers for the research GUI."""


ORIENTATION_LABELS = {
    "FACING_CAMERA": "facing camera",
    "BACK_TO_CAMERA": "back to camera",
    "SIDE_ON": "side-on",
    "UNCERTAIN": "orientation uncertain",
}

POSTURE_LABELS = {
    "STANDING": "standing posture",
    "SQUATTING": "squatting",
    "SITTING": "sitting",
    "KNEELING": "kneeling",
    "LYING_DOWN": "lying down",
    "BENDING_OR_CROUCHING": "bending / crouching",
    "UNCERTAIN": "posture uncertain",
}

ACTIVITY_LABELS = {
    "WALKING": "walking",
    "CHANGING_POSTURE": "changing posture",
    "REACHING_OR_HANDLING_OBJECT": "handling object",
    "WORKING": "working",
    "OTHER": "other activity",
    "UNCERTAIN": "activity uncertain",
}

MOTION_LABELS = {
    "TOWARD_CAMERA": "toward camera",
    "AWAY_FROM_CAMERA": "away from camera",
    "LATERAL_ACROSS_VIEW": "across camera view",
    "STATIONARY": "stationary",
    "UNCERTAIN": "motion uncertain",
}

MOTION_LEVELS = {
    "TOWARD_CAMERA": "warning",
    "UNCERTAIN": "warning",
    "AWAY_FROM_CAMERA": "clear",
    "LATERAL_ACROSS_VIEW": "clear",
    "STATIONARY": "clear",
}


def camera_behavior_summary(semantics, prefix):
    """Return readable per-camera VQA assessments and the most severe level."""
    if not isinstance(semantics, dict):
        return "Waiting for VQA future-behavior result", "unknown"
    metadata = semantics.get("metadata", {})
    input_tracks = metadata.get("input_tracks") if isinstance(metadata, dict) else None
    if isinstance(input_tracks, list):
        camera_input_tracks = [
            item
            for item in input_tracks
            if isinstance(item, dict)
            and str(item.get("track_id", "")).upper().startswith(prefix)
            and bool(item.get("visible", int(item.get("missed_frames", 0)) == 0))
        ]
        if not camera_input_tracks:
            return "NO RESPONSE\nNo person detected", "unknown"
    errors = [str(value) for value in semantics.get("validation_errors", [])]
    quality_warnings = [
        item
        for item in semantics.get("quality_warnings", [])
        if isinstance(item, dict)
        and (
            not isinstance(item.get("track_ids"), list)
            or any(str(value).upper().startswith(prefix) for value in item["track_ids"])
        )
    ]
    assessments = [
        item
        for item in semantics.get("assessments", [])
        if isinstance(item, dict) and str(item.get("track_id", "")).upper().startswith(prefix)
    ]
    if str(semantics.get("availability", "READY")).upper() == "UNAVAILABLE":
        track_ids = [str(item.get("track_id", "?")) for item in assessments]
        reason = str(semantics.get("unavailable_reason") or (errors[0] if errors else "unknown"))
        affected = ", ".join(track_ids) if track_ids else "none"
        return (
            f"VQA UNAVAILABLE\nAffected tracks: {affected}\nConfidence: N/A\n{reason}",
            "warning",
        )
    if not assessments:
        if errors:
            return f"VQA SCHEMA WARNING\n{errors[0]}", "warning"
        return "NO RESPONSE\nNo person detected", "unknown"

    levels = []
    lines = []
    for item in assessments:
        track_id = str(item.get("track_id", "?"))
        current = item.get("current_observation", {})
        future = item.get("future_prediction", {})
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        if isinstance(current, dict) and isinstance(future, dict) and current and future:
            orientation = ORIENTATION_LABELS.get(
                str(current.get("body_orientation", "UNCERTAIN")).upper(),
                "orientation unknown",
            )
            posture = POSTURE_LABELS.get(
                str(current.get("body_posture", "UNCERTAIN")).upper(),
                "posture unknown",
            )
            activity = ACTIVITY_LABELS.get(
                str(current.get("current_activity", "UNCERTAIN")).upper(),
                "activity unknown",
            )
            motion = MOTION_LABELS.get(
                str(current.get("motion_trend", "UNCERTAIN")).upper(),
                "motion unknown",
            )
            future_motion = MOTION_LABELS.get(
                str(future.get("future_motion", "UNCERTAIN")).upper(),
                "motion unknown",
            )
            lines.extend(
                [
                    f"{track_id} NOW: {posture}, {activity}, {motion}, {orientation}",
                    f"{track_id} NEXT: {future_motion}",
                ]
            )
            levels.append(
                MOTION_LEVELS.get(
                    str(future.get("future_motion", "UNCERTAIN")).upper(), "warning"
                )
            )
        else:
            motion = str(item.get("future_motion", "UNCERTAIN")).upper()
            label = MOTION_LABELS.get(motion, f"Unknown motion: {motion}")
            lines.append(f"{track_id} | {label} | confidence {confidence:.0%}")
            levels.append(MOTION_LEVELS.get(motion, "warning"))
    if errors:
        lines.append(f"SCHEMA WARNING: {errors[0]}")
        levels.append("warning")
    if quality_warnings:
        warning = quality_warnings[0]
        tracks = ", ".join(str(value) for value in warning.get("track_ids", []))
        diagnostic_only = warning.get("code") == "semantic_output_collapse"
        lines.append(
            f"QUALITY {'INFO' if diagnostic_only else 'WARNING'}: "
            f"{warning.get('code', 'unknown')} | IDs: {tracks or '--'}"
        )
        if not diagnostic_only:
            levels.append("warning")
    level = "stop" if "stop" in levels else "warning" if "warning" in levels else "clear"
    return "\n".join(lines), level


def explain_decision(report, semantics=None):
    """Explain the four-state action in concise English."""
    action = str(report.get("action", "DANGER_WARNING")).upper()
    reason = str(report.get("reason_code", ""))
    tracks = [str(value) for value in report.get("critical_tracks", [])]
    track_text = ", ".join(tracks) if tracks else "the current scene"

    if action == "STOP":
        if reason == "approximate_image_stop_zone":
            return (
                f"{track_text} is inside the approximate image-space 1 m stop zone, "
                "so the system must stop."
            )
        return f"{track_text} is in or about to enter the calibrated hazard zone, so the system must stop."
    if action == "CONTINUE":
        return (
            "All visible people are moving away, moving laterally, or stationary, and the "
            "result passed consecutive safety confirmation."
        )
    if action == "OBSERVE":
        return "Action output is disabled; only the VQA observation is displayed."
    if action == "NO_RESPONSE":
        return "No visible person is detected, so the system issues no action response."

    messages = {
        "tracking_report_stale": "The person-tracking report is stale, so safety is unknown.",
        "future_behavior_unavailable": "No fresh VQA future-behavior result is available.",
        "future_behavior_schema_invalid": "The VQA output does not satisfy the required schema.",
        "future_behavior_output_collapse": (
            f"The VQA copied an identical answer signature across {track_text}."
        ),
        "future_behavior_view_unreliable": "The camera evidence is not reliable enough.",
        "future_behavior_track_unreliable": f"Tracking is not yet reliable for {track_text}.",
        "future_behavior_assessment_missing": (
            f"VQA did not return a future-behavior assessment for {track_text}."
        ),
        "vqa_future_approaching_warning": (
            f"{track_text} is moving toward the robot-mounted camera."
        ),
        "vqa_future_uncertain_warning": (
            f"{track_text} has uncertain or insufficient future evidence."
        ),
        "vqa_future_not_proven_safe": (
            "The current evidence does not prove it is safe to continue."
        ),
        "awaiting_vqa_continue_confirmation": (
            "The first result is clear, but another confirmation is required."
        ),
        "awaiting_continue_confirmation": (
            "The first result is clear, but another confirmation is required."
        ),
    }
    if reason == "vqa_future_approaching_warning" and isinstance(semantics, dict):
        motion_by_track = {
            str(item.get("track_id")): str(
                item.get("future_prediction", {}).get("future_motion", "UNCERTAIN")
            ).upper()
            for item in semantics.get("assessments", [])
            if isinstance(item, dict)
        }
        approaching = [
            track_id
            for track_id in tracks
            if motion_by_track.get(track_id) == "TOWARD_CAMERA"
        ]
        other_warnings = [track_id for track_id in tracks if track_id not in approaching]
        parts = []
        if approaching:
            parts.append(f"{', '.join(approaching)} is moving toward the robot-mounted camera")
        if other_warnings:
            verb = "has" if len(other_warnings) == 1 else "have"
            parts.append(
                f"{', '.join(other_warnings)} {verb} another non-safe future motion"
            )
        detail = "; ".join(parts) or "A tracked person may move toward the robot"
        return f"{detail}. Therefore, the system shows DANGER WARNING."
    if reason in messages:
        detail = messages[reason]
        if reason == "future_behavior_unavailable" and isinstance(semantics, dict):
            unavailable = str(semantics.get("unavailable_reason", "")).strip()
            if unavailable:
                detail = f"{detail} Model error: {unavailable}"
        if reason == "future_behavior_schema_invalid" and isinstance(semantics, dict):
            errors = [str(value) for value in semantics.get("validation_errors", [])]
            if errors:
                detail = f"{detail} Validation error: {errors[0]}"
        return f"{detail} Therefore, the system shows DANGER WARNING."

    if isinstance(semantics, dict):
        motions = {
            str(item.get("track_id")): MOTION_LABELS.get(
                str(
                    item.get("future_prediction", {}).get("future_motion", "UNCERTAIN")
                ).upper(),
                "motion uncertain",
            )
            for item in semantics.get("assessments", [])
            if isinstance(item, dict)
        }
        details = [
            f"{track_id}: {motions[track_id]}" for track_id in tracks if track_id in motions
        ]
        if details:
            return (
                f"{'; '.join(details)}. The safe-to-continue condition is not met, so the "
                "system shows DANGER WARNING."
            )
    return "Future behavior is not sufficiently confirmed, so the system shows DANGER WARNING."
