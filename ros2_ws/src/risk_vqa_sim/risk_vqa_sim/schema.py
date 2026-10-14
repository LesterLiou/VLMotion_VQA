"""Strict schema and parser for dual-camera VQA observations."""

import ast
import json
import math


VISIBILITIES = {'CLEAR', 'OCCLUDED', 'OUT_OF_VIEW', 'UNCERTAIN'}
POSITIONS = {
    'OUTSIDE',
    'APPROACH_ZONE',
    'WAITING_ZONE',
    'INSIDE_HAZARD_ZONE',
    'UNKNOWN',
}
MOTIONS = {
    'APPROACHING',
    'CROSSING',
    'STATIONARY',
    'LEAVING',
    'PASSING_BY',
    'UNKNOWN',
}
INTENTS = {'ENTER', 'WAIT', 'PASS_BY', 'MOVE_AWAY', 'UNCERTAIN'}
ENTRY_TIMES = {
    'NOW',
    '0_TO_2_SECONDS',
    '2_TO_5_SECONDS',
    'MORE_THAN_5_SECONDS',
    'NOT_APPROACHING',
    'UNKNOWN',
}


def extract_json_object(text):
    """Extract the first JSON object, with a safe Python-literal fallback."""
    if not isinstance(text, str):
        raise ValueError('model response must be text')
    normalized = text.replace('\\_', '_')
    decoder = json.JSONDecoder()
    for index, char in enumerate(normalized):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(normalized[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    for start, char in enumerate(normalized):
        if char != '{':
            continue
        for end in range(len(normalized) - 1, start, -1):
            if normalized[end] != '}':
                continue
            try:
                value = ast.literal_eval(normalized[start:end + 1])
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    raise ValueError('model response does not contain a valid object')


def _enum(value, allowed, fallback, field, errors):
    normalized = str(value).strip().upper()
    if normalized not in allowed:
        errors.append(f'{field}: invalid value {value!r}')
        return fallback
    return normalized


def _confidence(value, field, errors):
    if isinstance(value, bool):
        errors.append(f'{field}: confidence must be numeric')
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f'{field}: confidence must be numeric')
        return 0.0
    if not math.isfinite(number):
        errors.append(f'{field}: confidence must be finite')
        return 0.0
    return round(min(1.0, max(0.0, number)), 3)


def _camera_placeholder():
    return {
        'visibility': 'UNCERTAIN',
        'visibility_confidence': 0.0,
        'people': [],
    }


def _validate_person(raw, side, index, errors):
    field = f'{side}.people[{index}]'
    if not isinstance(raw, dict):
        errors.append(f'{field}: person must be an object')
        return {
            'person_id': f'{side[0].upper()}{index + 1}',
            'position': 'UNKNOWN',
            'motion': 'UNKNOWN',
            'intent': 'UNCERTAIN',
            'entry_time': 'UNKNOWN',
            'confidence': 0.0,
            'evidence': '',
        }
    person_id = str(raw.get('person_id', '')).strip()
    if not person_id:
        person_id = f'{side[0].upper()}{index + 1}'
        errors.append(f'{field}.person_id: missing; assigned {person_id}')
    return {
        'person_id': person_id[:32],
        'position': _enum(
            raw.get('position'), POSITIONS, 'UNKNOWN', f'{field}.position', errors
        ),
        'motion': _enum(
            raw.get('motion'), MOTIONS, 'UNKNOWN', f'{field}.motion', errors
        ),
        'intent': _enum(
            raw.get('intent'), INTENTS, 'UNCERTAIN', f'{field}.intent', errors
        ),
        'entry_time': _enum(
            raw.get('entry_time'),
            ENTRY_TIMES,
            'UNKNOWN',
            f'{field}.entry_time',
            errors,
        ),
        'confidence': _confidence(
            raw.get('confidence'), f'{field}.confidence', errors
        ),
        'evidence': str(raw.get('evidence', '')).strip()[:240],
    }


def _validate_camera(raw, side, errors):
    if not isinstance(raw, dict):
        errors.append(f'{side}: camera observation is missing or invalid')
        return _camera_placeholder()
    visibility = _enum(
        raw.get('visibility'),
        VISIBILITIES,
        'UNCERTAIN',
        f'{side}.visibility',
        errors,
    )
    visibility_confidence = _confidence(
        raw.get('visibility_confidence'),
        f'{side}.visibility_confidence',
        errors,
    )
    if 'people' not in raw:
        errors.append(f'{side}.people: field is missing')
        return _camera_placeholder()
    raw_people = raw.get('people')
    if not isinstance(raw_people, list):
        errors.append(f'{side}.people: must be a list')
        return _camera_placeholder()
    if len(raw_people) > 12:
        errors.append(f'{side}.people: too many entries to assess reliably')
        return _camera_placeholder()
    return {
        'visibility': visibility,
        'visibility_confidence': visibility_confidence,
        'people': [
            _validate_person(person, side, index, errors)
            for index, person in enumerate(raw_people)
        ],
    }


def validate_observation(raw_report):
    """Validate and conservatively normalize a VLM observation report."""
    if not isinstance(raw_report, dict):
        raise ValueError('observation report must be an object')
    errors = []
    report = {
        'schema_version': '1.0',
        'left': _validate_camera(raw_report.get('left'), 'left', errors),
        'right': _validate_camera(raw_report.get('right'), 'right', errors),
        'scene_summary': str(raw_report.get('scene_summary', '')).strip()[:500],
        'validation_errors': errors,
    }
    return report
