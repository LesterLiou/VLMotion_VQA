"""Validation and natural-language rendering for risk VQA output."""

import ast
import json
import math


SEVERITIES = {'low', 'medium', 'high', 'critical'}


def extract_json_object(text):
    """Extract a JSON object, accepting a safe Python-dict fallback from small VLMs."""
    if not isinstance(text, str):
        raise ValueError('model response must be text')
    # Some LLaVA/Vicuna responses Markdown-escape underscores inside JSON
    # keys or values (for example task\_relation and SLOW\_DOWN). ``\_`` is
    # not a valid JSON escape, so normalize it before either parser runs.
    normalized_text = text.replace('\\_', '_')
    decoder = json.JSONDecoder()
    for index, char in enumerate(normalized_text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(normalized_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    # LLaVA 7B sometimes follows the requested structure but emits a Python
    # literal with single quotes. literal_eval parses data literals only and
    # does not execute model-generated code.
    for start, char in enumerate(normalized_text):
        if char != '{':
            continue
        for end in range(len(normalized_text) - 1, start, -1):
            if normalized_text[end] != '}':
                continue
            try:
                candidate = normalized_text[start:end + 1]
                value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    raise ValueError('model response does not contain a valid JSON object')


def _finite_number(value, field):
    if isinstance(value, bool):
        raise ValueError(f'{field} must be numeric')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be numeric') from exc
    if not math.isfinite(number):
        raise ValueError(f'{field} must be finite')
    return number


def validate_report(raw_report, horizon_s, allowed_actions):
    """Normalize a model report and reject factors missing the required time."""
    if not isinstance(raw_report, dict):
        raise ValueError('risk report must be a JSON object')
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
            factor = str(raw.get('factor', '')).strip()
            impact = str(raw.get('impact', '')).strip()
            if not factor or not impact:
                raise ValueError('factor and impact are required')

            time_s = _finite_number(raw.get('time_to_impact_s'), 'time_to_impact_s')
            if time_s < 0:
                raise ValueError('time_to_impact_s cannot be negative')
            if time_s > horizon_s:
                raise ValueError('time_to_impact_s exceeds the configured horizon')

            severity = str(raw.get('severity', 'medium')).lower().strip()
            if severity not in SEVERITIES:
                raise ValueError('invalid severity')
            confidence = _finite_number(raw.get('confidence', 0.5), 'confidence')
            confidence = min(1.0, max(0.0, confidence))

            action = raw.get('action')
            if action is not None:
                action = str(action).upper().strip()
                if action not in allowed:
                    action = None

            factors.append({
                'factor': factor,
                'impact': impact,
                'time_to_impact_s': round(time_s, 2),
                'severity': severity,
                'confidence': round(confidence, 3),
                'action': action,
            })
        except ValueError as exc:
            errors.append(f'factor[{index}]: {exc}')

    factors.sort(key=lambda item: (item['time_to_impact_s'], -item['confidence']))
    return {
        'summary': str(raw_report.get('summary', '')).strip(),
        'factors': factors,
        'validation_errors': errors,
    }


def render_zh_tw(report):
    """Render a validated report as Traditional Chinese natural language."""
    factors = report.get('factors', [])
    if not factors:
        if report.get('validation_errors'):
            return '模型回傳的風險因子缺少必要欄位，目前無法形成可靠判斷。'
        return '目前影像中未發現會在預測時間內明顯影響工作的因子。'

    severity_names = {
        'low': '低',
        'medium': '中',
        'high': '高',
        'critical': '嚴重',
    }
    action_names = {
        'CONTINUE': '繼續作業',
        'SLOW_DOWN': '降低速度',
        'WAIT': '等待',
        'STOP': '停止動作',
        'REPLAN': '重新規劃',
        'ALERT_HUMAN': '通知人員',
    }
    sentences = []
    for item in factors:
        time_s = item['time_to_impact_s']
        timing = '目前已經' if time_s == 0 else f'預計約 {time_s:g} 秒後'
        sentence = (
            f"{item['factor']}，{timing}影響工作：{item['impact']}"
            f"（風險：{severity_names[item['severity']]}，信心：{item['confidence']:.0%}）"
        )
        action = item.get('action')
        if action:
            sentence += f"。建議：{action_names.get(action, action)}"
        sentences.append(sentence + '。')
    return '\n'.join(sentences)


def render_en(report):
    """Render a validated report in English without requiring CJK fonts."""
    factors = report.get('factors', [])
    if not factors:
        if report.get('validation_errors'):
            return 'The model response was missing required risk fields, so no reliable assessment can be made.'
        return 'No visible factor is expected to significantly affect the work within the prediction horizon.'

    action_names = {
        'CONTINUE': 'continue the work',
        'SLOW_DOWN': 'slow down',
        'WAIT': 'wait',
        'STOP': 'stop the current motion',
        'REPLAN': 'replan the task',
        'ALERT_HUMAN': 'alert a human operator',
    }
    sentences = []
    for item in factors:
        time_s = item['time_to_impact_s']
        timing = 'is already affecting the work' if time_s == 0 else (
            f'is expected to affect the work in about {time_s:g} seconds'
        )
        sentence = (
            f"{item['factor']} {timing}: {item['impact']} "
            f"(risk: {item['severity']}, confidence: {item['confidence']:.0%})"
        )
        action = item.get('action')
        if action:
            sentence += f". Suggested action: {action_names.get(action, action)}"
        sentences.append(sentence + '.')
    return '\n'.join(sentences)


def select_action(report):
    """Return only the model's highest-priority suggestion; never execute it."""
    priority = {
        'STOP': 6,
        'ALERT_HUMAN': 5,
        'REPLAN': 4,
        'WAIT': 3,
        'SLOW_DOWN': 2,
        'CONTINUE': 1,
    }
    actions = [item.get('action') for item in report.get('factors', [])]
    actions = [action for action in actions if action]
    return max(actions, key=lambda action: priority.get(action, 0)) if actions else 'NO_ACTION'
