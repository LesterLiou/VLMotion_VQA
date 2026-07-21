import pytest

from risk_vqa.prompt import build_risk_prompt
from risk_vqa.schema import (
    extract_json_object,
    render_en,
    render_zh_tw,
    select_action,
    validate_report,
)


ACTIONS = ['CONTINUE', 'SLOW_DOWN', 'WAIT', 'STOP', 'REPLAN', 'ALERT_HUMAN']


def test_prompt_contains_question_and_required_time():
    prompt = build_risk_prompt('搬運箱子', 10, ACTIONS, '是否有人接近？')
    assert '是否有人接近？' in prompt
    assert 'time_to_impact_s is mandatory' in prompt
    assert 'Do not output coordinates' in prompt


def test_extract_json_from_fenced_response():
    result = extract_json_object('```json\n{"summary": "ok", "factors": []}\n```')
    assert result['summary'] == 'ok'


def test_extract_python_dict_from_small_vlm_response():
    result = extract_json_object(
        "{'factor': 'visible activity', 'action': 'SLOW\\_DOWN'}"
    )
    assert result['factor'] == 'visible activity'
    assert result['action'] == 'SLOW_DOWN'


def test_extract_json_with_markdown_escaped_underscores():
    result = extract_json_object(
        '{"task\\_relation":"same aisle","action":"SLOW\\_DOWN"}'
    )
    assert result == {
        'task_relation': 'same aisle',
        'action': 'SLOW_DOWN',
    }


def test_validate_requires_time_to_impact():
    report = validate_report({
        'factors': [
            {'factor': 'person', 'impact': 'blocks path', 'severity': 'high'},
        ],
    }, 10.0, ACTIONS)
    assert report['factors'] == []
    assert 'time_to_impact_s must be numeric' in report['validation_errors'][0]
    assert '無法形成可靠判斷' in render_zh_tw(report)


def test_validate_render_and_action_selection():
    report = validate_report({
        'summary': 'risk',
        'factors': [{
            'factor': '人員進入工作區',
            'impact': '阻擋移動路徑',
            'time_to_impact_s': 2,
            'severity': 'high',
            'confidence': 0.8,
            'action': 'stop',
        }],
    }, 10.0, ACTIONS)
    assert report['factors'][0]['time_to_impact_s'] == 2.0
    assert select_action(report) == 'STOP'
    assert '約 2 秒後' in render_zh_tw(report)
    assert 'about 2 seconds' in render_en(report)


def test_reject_time_beyond_horizon_and_unlisted_action():
    report = validate_report({
        'factors': [
            {
                'factor': 'a',
                'impact': 'b',
                'time_to_impact_s': 11,
                'severity': 'low',
                'confidence': 0.5,
                'action': 'FLY',
            },
            {
                'factor': 'c',
                'impact': 'd',
                'time_to_impact_s': 1,
                'severity': 'medium',
                'confidence': 2,
                'action': 'FLY',
            },
        ],
    }, 10.0, ACTIONS)
    assert len(report['factors']) == 1
    assert report['factors'][0]['confidence'] == 1.0
    assert report['factors'][0]['action'] is None


def test_extract_json_raises_for_plain_text():
    with pytest.raises(ValueError):
        extract_json_object('no structured result')
