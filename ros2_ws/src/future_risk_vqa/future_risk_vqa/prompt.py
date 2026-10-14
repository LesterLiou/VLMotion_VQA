"""Prompt contracts for future-behavior VQA research."""

import json


SYSTEM_PROMPT = (
    "Observe each tracked person's current posture/activity and predict their next "
    "activity/motion. Use camera-relative directions. Return only the requested JSON."
)


def _future_only_prompt(work_context, tracking_context):
    trajectory_mode = str(tracking_context.get("trajectory_context_mode", "none"))
    storyboard_mode = str(tracking_context.get("storyboard_mode", "global"))
    visible_context = {"tracks": tracking_context.get("tracks", [])}
    track_ids = [
        str(item.get("track_id"))
        for item in visible_context.get("tracks", [])
        if item.get("track_id") is not None
    ]
    context_json = json.dumps(visible_context, ensure_ascii=False, separators=(",", ":"))
    trajectory_descriptions = {
        "none": "No predicted path or numerical motion prior is provided.",
        "overlay": (
            "A purple path overlay is an algorithm-provided motion prior. Compare it with "
            "visible OLD-to-NOW motion."
        ),
        "numeric": (
            "trusted_motion_prediction in the context is an algorithm-provided numerical "
            "motion prior. Compare it with visible OLD-to-NOW motion."
        ),
        "both": (
            "Purple path overlays and trusted_motion_prediction are algorithm-provided "
            "motion priors. Compare them with visible OLD-to-NOW motion."
        ),
    }
    trajectory_description = trajectory_descriptions.get(
        trajectory_mode, trajectory_descriptions["none"]
    )
    has_motion_hint = any(
        isinstance(item.get("trusted_camera_motion_hint"), dict)
        for item in visible_context.get("tracks", [])
    )
    motion_hint_description = (
        "trusted_camera_motion_hint is computed from timestamp-matched OLD/MID/NOW "
        "bounding-box change. It is an image-space observation, not metric depth. "
        "The tracking layer supplies current motion separately; use visible activity "
        "to judge whether that motion is likely to continue."
        if has_motion_hint
        else "No trusted camera-motion hint is provided."
    )
    if storyboard_mode == "track_crops":
        storyboard_description = (
            "The image is a target-centric temporal storyboard. Each horizontal row "
            "belongs to exactly one labeled track ID and camera. Within every row, "
            "frames run from OLD through MID to NOW from left to right. The tracked "
            "person is enlarged using that ID's timestamp-matched bounding box in "
            "every cell. TRACK NOT FOUND means that ID has no reliable observation "
            "at that time; do not substitute another person."
        )
    elif storyboard_mode == "hybrid":
        storyboard_description = (
            "The top row contains LEFT NOW and RIGHT NOW full-scene context with "
            "track-ID detections. Below it, each labeled track has one OLD/MID/NOW "
            "crop strip. Crop strips are packed two per visual row. Use the full-scene "
            "cells for environment context and use only the matching labeled crop "
            "strip for that person's posture and activity. Never transfer an answer "
            "between IDs."
        )
    else:
        storyboard_description = (
            "The image is a dual-camera temporal storyboard. Top row is LEFT, bottom "
            "row is RIGHT. Frames run from OLD to NOW. Only the NOW cells contain "
            "track-ID markers."
        )
    return f"""<image>
Research task: {work_context}

{storyboard_description}
{trajectory_description}
{motion_hint_description}

Trusted tracking context:
{context_json}

For each track, use NOW for current observation and OLD-to-NOW for future prediction.
TOWARD_CAMERA and AWAY_FROM_CAMERA refer to that track's camera.

Closed vocabularies:
- body_orientation: FACING_CAMERA, BACK_TO_CAMERA, SIDE_ON, UNCERTAIN
- body_posture: STANDING, SQUATTING, SITTING, KNEELING, LYING_DOWN,
  BENDING_OR_CROUCHING, UNCERTAIN
- current_activity: WALKING, CHANGING_POSTURE,
  REACHING_OR_HANDLING_OBJECT, WORKING, OTHER, UNCERTAIN
- future_motion: TOWARD_CAMERA, AWAY_FROM_CAMERA, STATIONARY, UNCERTAIN

Rules:
- Current track IDs are exactly: {json.dumps(track_ids)}
- Return one assessment per ID, once, in this order. Analyze only its labeled row.
- Report visible state and camera-relative behavior, not geometry, robot action,
  identity, role, or intent.
- Keep posture and activity independent. body_posture describes the current body
  configuration; current_activity describes the action. Posture words (such as
  STANDING or SITTING) belong only in body_posture.
- Direction words (TOWARD_CAMERA, AWAY_FROM_CAMERA, STATIONARY) are future_motion
  only. For current_activity, choose only from the activity vocabulary above.
- Use WORKING for continued work; never invent variants such as CONTINUE_WORKING.
- Use UNCERTAIN, never null, when evidence is insufficient or the target is occluded.
- Return exactly {{"assessments":[...]}} with no Markdown or prose. Keep each assessment flat
  and containing exactly: track_id,
  body_orientation, body_posture, current_activity, future_motion.
- Do not output motion_trend, evidence, nested objects, or extra keys.
- assessments must be an empty list only when Current track IDs is empty.
"""


def build_future_event_prompt(work_context, geometry_context):
    """Always keep VLM semantics camera-relative; geometry stays in the rule layer."""
    return _future_only_prompt(work_context, geometry_context)


def wrap_vicuna_prompt(user_prompt):
    return f"{SYSTEM_PROMPT}###USER: {user_prompt}###ASSISTANT:"
