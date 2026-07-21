"""Low-frequency VLM prompts and schemas for person-related semantic risk."""

import math


SEVERITIES = {'low', 'medium', 'high', 'critical'}
RESPONSE_MODES = {'compact', 'detailed'}


def _validate_response_mode(response_mode):
    mode = str(response_mode).strip().lower()
    if mode not in RESPONSE_MODES:
        raise ValueError('response_mode must be compact or detailed')
    return mode


def _common_prompt(work_context, allowed_actions, user_question):
    actions = ', '.join(allowed_actions)
    action_meanings = {
        'CONTINUE': 'no task change is needed',
        'SLOW_DOWN': 'reduce motion as a precaution',
        'WAIT': 'pause for a brief, temporary obstruction',
        'STOP': 'stop for an immediate unsafe conflict',
        'REPLAN': 'choose another path or task plan',
        'ALERT_HUMAN': 'request human attention to the situation',
    }
    action_guide = '; '.join(
        f"{str(action).upper()}={action_meanings[str(action).upper()]}"
        for action in allowed_actions if str(action).upper() in action_meanings
    )
    question = user_question.strip() or (
        'How could the visible people affect the current work?'
    )
    return f"""<image>
Current work: {work_context}
User question: {question}
Allowed actions: {actions}
Rules:
- This is one RGB image. Use only directly visible evidence.
- Do not estimate TTC, distance, numeric speed, future seconds, duration, intent, identity, age, coordinates, boxes, or keypoints.
- Describe qualitative movement only when visible walking or running posture supports it; otherwise mark it uncertain.
- Decide in this order: visible activity, relationship to the current work, impact, then action.
- Do not claim path, workspace, or target occupation unless it is identifiable from the image and Current work.
- CONTINUE requires visible evidence that the activity is unrelated to the work. No obvious collision is not enough.
- Use SLOW_DOWN for task-relevant ambiguity, WAIT for visible shared-area occupation, STOP for visible direct conflict,
  and null when neither intervention nor safe continuation is supported. Never default to CONTINUE or WAIT.
- Use WAIT when a person visibly faces the camera/robot and their walking or running posture clearly suggests
  fast movement toward it. Require visible body-posture evidence; facing the camera alone is not enough.
- Action meanings: {action_guide}.
- severity is low, medium, high, or critical. confidence is visible-evidence certainty from 0 to 1.
- Return one valid JSON object only. No Markdown and no explanation.
"""


def _build_compact_prompt(work_context, allowed_actions, user_question):
    return _common_prompt(work_context, allowed_actions, user_question) + """
Report only the highest-priority visible person-related factor using short English phrases.
Use exactly this JSON shape:
{"factor":"short category","activity":"visible activity","task_relation":"relation to current work","impact":"effect on work","severity":"low|medium|high|critical","confidence":0.0,"action":null}
Replace every placeholder with image-specific evidence. action is one allowed action or null.
If there is no relevant factor, return:
{"factor":"","activity":"","task_relation":"","impact":"","severity":"low","confidence":0.0,"action":null}
Start the response with { and end it with }.
"""


def _build_detailed_prompt(work_context, allowed_actions, user_question):
    return _common_prompt(work_context, allowed_actions, user_question) + """
Assess every visible person-related factor that could affect the current work.
Write a short overall summary and concise full-sentence descriptions.
Return one JSON object containing:
- "summary": a brief overall assessment string
- "people_present": a boolean
- "factors": a JSON array with one object per relevant person
Each factor object must contain "person", "activity", "impact", "severity",
"confidence", and "action", using the same definitions above. It must also contain "factor",
a short category, and "task_relation", the visible relationship to the current work.
If no person-related factor is visible, return an empty factors list and set
people_present appropriately. Do not invent a person or an action.
"""


def build_person_semantic_prompt(
    work_context,
    allowed_actions,
    user_question='',
    response_mode='compact',
):
    """Build either the latency-oriented or descriptive semantic prompt."""
    mode = _validate_response_mode(response_mode)
    if mode == 'detailed':
        return _build_detailed_prompt(
            work_context, allowed_actions, user_question
        )
    return _build_compact_prompt(work_context, allowed_actions, user_question)


def _number(value, field):
    if isinstance(value, bool):
        raise ValueError(f'{field} must be numeric')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be numeric') from exc
    if not math.isfinite(number):
        raise ValueError(f'{field} must be finite')
    return number


def validate_person_semantic_report(
    raw_report,
    allowed_actions,
    response_mode='compact',
):
    """Validate VLM output and normalize both modes to the factors API."""
    mode = _validate_response_mode(response_mode)
    if not isinstance(raw_report, dict):
        raise ValueError('semantic report must be a JSON object')
    if 'activity' in raw_report or 'impact' in raw_report:
        raw_factors = [raw_report]
    elif isinstance(raw_report.get('risk'), dict):
        raw_factors = [raw_report['risk']]
    else:
        raw_factors = raw_report.get('factors', [])
    if not isinstance(raw_factors, list):
        raise ValueError('factors must be a list')

    allowed = {str(action).upper() for action in allowed_actions}
    factors = []
    errors = []
    for index, raw in enumerate(raw_factors):
        try:
            if not isinstance(raw, dict):
                raise ValueError('factor must be an object')
            person = str(raw.get('person', '')).strip()
            factor = str(raw.get('factor', '')).strip()
            activity = str(raw.get('activity', '')).strip()
            task_relation = str(raw.get('task_relation', '')).strip()
            impact = str(raw.get('impact', '')).strip()
            if not factor and not activity and not task_relation and not impact:
                continue
            if not activity or not impact:
                raise ValueError('activity and impact are required')
            # Small VLMs do not always reproduce every requested JSON key.
            # Keep an otherwise useful legacy response instead of dropping the
            # whole factor, while making the schema omission observable.
            if not factor:
                factor = 'person-related factor'
                errors.append(f'factor[{index}]: factor was missing; fallback used')
            if not task_relation:
                task_relation = 'relationship to the current work was not stated'
                errors.append(
                    f'factor[{index}]: task_relation was missing; fallback used'
                )
            if mode == 'detailed' and not person:
                raise ValueError('person is required in detailed mode')
            severity = str(raw.get('severity', 'medium')).lower().strip()
            if severity not in SEVERITIES:
                raise ValueError('invalid severity')
            confidence = min(
                1.0,
                max(0.0, _number(raw.get('confidence', 0.5), 'confidence')),
            )
            if confidence <= 0.0:
                raise ValueError(
                    'confidence must be greater than 0 for a reported factor'
                )
            action = raw.get('action')
            if action is not None:
                action = str(action).replace('\\_', '_').upper().strip()
                if action not in allowed:
                    action = None
            factor = {
                'factor': factor,
                'activity': activity,
                'task_relation': task_relation,
                'impact': impact,
                'severity': severity,
                'confidence': round(confidence, 3),
                'action': action,
            }
            if person:
                factor['person'] = person
            factors.append(factor)
        except ValueError as exc:
            errors.append(f'factor[{index}]: {exc}')

    if mode == 'compact' and factors:
        severity_priority = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        factors = [max(
            factors,
            key=lambda item: (
                severity_priority[item['severity']], item['confidence']
            ),
        )]

    raw_people_present = raw_report.get('people_present', bool(factors))
    if isinstance(raw_people_present, str):
        people_present = raw_people_present.strip().lower() in {
            '1', 'true', 'yes'
        }
    else:
        people_present = bool(raw_people_present)
    if factors:
        people_present = True

    summary = ''
    if mode == 'detailed':
        summary = str(raw_report.get('summary', '')).strip()
    return {
        'analysis_mode': 'person_semantic',
        'semantic_response_mode': mode,
        'summary': summary,
        'people_present': people_present,
        'factors': factors,
        'validation_errors': errors,
    }


def render_person_semantic_en(report):
    factors = report.get('factors', [])
    if not factors:
        if report.get('validation_errors'):
            return 'The VLM person assessment was missing required fields.'
        return 'The VLM did not identify a visible person-related semantic risk.'

    sentences = []
    summary = str(report.get('summary', '')).strip()
    if summary:
        sentences.append(summary)
    for item in factors:
        if item.get('person'):
            sentence = (
                f"{item['person']} {item['activity']} "
                f"Task relation: {item['task_relation']}. "
                f"Possible task impact: {item['impact']} "
                f"(semantic severity: {item['severity']}, "
                f"confidence: {item['confidence']:.0%})"
            )
        else:
            sentence = (
                f"Visible factor: {item['factor']}. "
                f"Activity: {item['activity']}. "
                f"Task relation: {item['task_relation']}. "
                f"Possible task impact: {item['impact']} "
                f"(semantic severity: {item['severity']}, "
                f"confidence: {item['confidence']:.0%})"
            )
        if item.get('action'):
            sentence += f". Advisory action: {item['action']}"
        sentences.append(sentence.rstrip('.') + '.')
    return '\n'.join(sentences)
