"""Prompt construction for visual risk assessment."""

import json


SYSTEM_PROMPT = (
    'A chat between a curious user and an artificial intelligence assistant. '
    'The assistant gives helpful, detailed, and polite answers to the user questions.'
)


def build_risk_prompt(work_context, horizon_s, allowed_actions, user_question=''):
    """Build a strict, single-frame risk-assessment prompt."""
    actions = ', '.join(allowed_actions)
    schema_example = {
        'summary': 'short overall assessment',
        'factors': [
            {
                'factor': 'visible condition that may affect the work',
                'impact': 'how it will affect the work',
                'time_to_impact_s': 2.5,
                'severity': 'high',
                'confidence': 0.8,
                'action': 'STOP',
            }
        ],
    }
    question_section = ''
    if user_question.strip():
        question_section = f'User question: {user_question.strip()}\n\n'
    return f"""<image>
You are a visual safety observer. Assess only evidence visible in the current image.

Current work: {work_context}
Prediction horizon: the next {horizon_s:.1f} seconds.
{question_section}

Required task:
1. Identify every visible factor likely to affect the current work within the horizon.
2. For every factor, estimate how many seconds from now it will first affect the work.
   Use 0 when it is already affecting the work. time_to_impact_s is mandatory.
3. Explain the expected impact briefly.
4. Optionally select one suggested action from: {actions}. Use null if no action is justified.

Rules:
- Do not output coordinates, bounding boxes, candidate labels, or keypoints.
- Do not describe irrelevant background objects.
- A single image cannot establish velocity precisely, so lower confidence when motion is uncertain.
- severity must be one of: low, medium, high, critical.
- confidence must be between 0 and 1.
- Write every descriptive JSON value in English, even if the user question is not English.
- Return JSON only, with no Markdown fences or extra text.
- Use this exact structure:
{json.dumps(schema_example, ensure_ascii=False)}
- If no factor is likely to affect the work, return an empty factors list.
"""


def wrap_vicuna_prompt(user_prompt):
    """Wrap one image question using the worker's default conversation format."""
    return f'{SYSTEM_PROMPT}###USER: {user_prompt}###ASSISTANT:'
