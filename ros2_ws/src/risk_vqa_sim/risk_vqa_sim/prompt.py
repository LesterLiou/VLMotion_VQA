"""Prompt for a dual-camera, temporally ordered storyboard."""

import json


SYSTEM_PROMPT = (
    'A chat between a curious user and an artificial intelligence assistant. '
    'The assistant follows the requested machine-readable schema exactly.'
)


def build_observation_prompt(work_context, sample_count, sample_window_s):
    """Build a closed-vocabulary prompt for the single storyboard image."""
    example = {
        'left': {
            'visibility': 'CLEAR',
            'visibility_confidence': 0.95,
            'people': [
                {
                    'person_id': 'L1',
                    'position': 'APPROACH_ZONE',
                    'motion': 'APPROACHING',
                    'intent': 'ENTER',
                    'entry_time': '0_TO_2_SECONDS',
                    'confidence': 0.82,
                    'evidence': 'person becomes larger and moves toward the marked entrance',
                }
            ],
        },
        'right': {
            'visibility': 'CLEAR',
            'visibility_confidence': 0.96,
            'people': [],
        },
        'scene_summary': 'One person approaches from the left; right side is clear.',
    }
    return f"""<image>
You are observing a temporal storyboard from two fixed safety cameras.
Current work: {work_context}

Storyboard layout:
- The top row is LEFT CAMERA. The bottom row is RIGHT CAMERA.
- Each row has {sample_count} frames ordered from OLDEST on the left to NOW on the right.
- The frames span approximately {sample_window_s:.1f} seconds.
- Camera labels and OLD/NOW labels are overlays, not physical scene objects.

Perform these layers in order:
1. Visibility: decide whether each hazard-zone approach is CLEAR, OCCLUDED,
   OUT_OF_VIEW, or UNCERTAIN. CLEAR means the relevant approach is actually visible;
   it does not mean no person is present.
2. Observation: list each visible person separately for each camera. Use local IDs
   L1, L2... in the left row and R1, R2... in the right row.
3. Temporal interpretation: compare that same person's position across the ordered
   frames and classify position, observed motion, likely next intent, and entry-time band.

Closed vocabularies:
- position: OUTSIDE, APPROACH_ZONE, WAITING_ZONE, INSIDE_HAZARD_ZONE, UNKNOWN
- motion: APPROACHING, CROSSING, STATIONARY, LEAVING, PASSING_BY, UNKNOWN
- intent: ENTER, WAIT, PASS_BY, MOVE_AWAY, UNCERTAIN
- entry_time: NOW, 0_TO_2_SECONDS, 2_TO_5_SECONDS, MORE_THAN_5_SECONDS,
  NOT_APPROACHING, UNKNOWN

Rules:
- Use only visible evidence from this storyboard. Do not infer identity, age, role,
  permission, or exact metric distance.
- Motion must be based on change across frames. A single pose or facing direction is
  insufficient. If matching a person across frames is unreliable, use UNKNOWN and
  lower confidence.
- ENTER means the observed path is likely to cross the hazard-zone boundary. A person
  merely visible in the scene is not automatically ENTER.
- PASS_BY means the path is visibly parallel to or away from the entrance.
- If a view is blocked, never convert lack of visible people into a clear scene.
- person_id is local to this answer and does not assert cross-camera identity.
- confidence is a number from 0 to 1 for the complete person classification.
- evidence is one short English phrase describing only visible temporal evidence.
- Return one JSON object only, with no Markdown or additional prose.
- Use exactly these keys and this structure:
{json.dumps(example, ensure_ascii=False)}
"""


def wrap_vicuna_prompt(user_prompt):
    """Wrap the request for the repository's LLaVA/Vicuna model worker."""
    return f'{SYSTEM_PROMPT}###USER: {user_prompt}###ASSISTANT:'
