from risk_vqa.person_semantic import (
    build_person_semantic_prompt,
    render_person_semantic_en,
    validate_person_semantic_report,
)
from risk_vqa.schema import select_action


ACTIONS = ['CONTINUE', 'SLOW_DOWN', 'WAIT', 'STOP', 'REPLAN', 'ALERT_HUMAN']


def test_person_prompt_excludes_ttc_estimation():
    prompt = build_person_semantic_prompt('Stocking shelves', ACTIONS)
    assert 'Do not estimate TTC' in prompt
    assert 'visible person-related factor' in prompt
    assert '"activity"' in prompt
    assert '"factor"' in prompt
    assert '"task_relation"' in prompt
    assert '"summary"' not in prompt
    assert '"person"' not in prompt
    assert '"factors"' not in prompt
    assert '"action":"WAIT"' not in prompt
    assert 'Never default to CONTINUE or WAIT' in prompt
    assert 'This is one RGB image' in prompt
    assert 'Do not estimate TTC' in prompt
    assert 'No obvious collision is not enough' in prompt
    assert 'fast movement toward it' in prompt
    assert 'facing the camera alone is not enough' in prompt
    assert 'Start the response with { and end it with }' in prompt
    assert 'If there is no relevant factor, return:' in prompt
    assert 'confidence is visible-evidence certainty' in prompt


def test_detailed_prompt_preserves_descriptive_schema():
    prompt = build_person_semantic_prompt(
        'Stocking shelves', ACTIONS, response_mode='detailed'
    )
    assert '"summary"' in prompt
    assert '"person"' in prompt
    assert '"factors"' in prompt
    assert 'Assess every visible person-related factor' in prompt
    assert 'one object per relevant person' in prompt
    assert '"confidence":0.7' not in prompt


def test_person_semantic_validation_and_rendering():
    report = validate_person_semantic_report({
        'people_present': True,
        'factors': [{
            'person': 'A customer near the shelf',
            'factor': 'shared shelf use',
            'activity': 'reaching into the robot work area',
            'task_relation': 'using the shelf required by the current work',
            'impact': 'the robot path may be obstructed',
            'severity': 'high',
            'confidence': 0.8,
            'action': 'wait',
            'time_to_impact_s': 0.1,
        }],
    }, ACTIONS, response_mode='detailed')
    assert report['people_present'] is True
    assert report['semantic_response_mode'] == 'detailed'
    assert report['factors'][0]['person'] == 'A customer near the shelf'
    assert 'time_to_impact_s' not in report['factors'][0]
    assert select_action(report) == 'WAIT'
    assert 'reaching into the robot work area' in render_person_semantic_en(report)


def test_compact_person_semantic_output():
    report = validate_person_semantic_report({
        'factor': 'workspace interaction',
        'activity': 'reaching into workspace',
        'task_relation': 'using the area required by the current work',
        'impact': 'path obstruction',
        'severity': 'high',
        'confidence': 0.8,
        'action': 'wait',
    }, ACTIONS)
    assert report['people_present'] is True
    assert report['summary'] == ''
    assert report['semantic_response_mode'] == 'compact'
    assert report['factors'] == [{
        'factor': 'workspace interaction',
        'activity': 'reaching into workspace',
        'task_relation': 'using the area required by the current work',
        'impact': 'path obstruction',
        'severity': 'high',
        'confidence': 0.8,
        'action': 'WAIT',
    }]
    assert select_action(report) == 'WAIT'


def test_compact_empty_risk_returns_no_action():
    report = validate_person_semantic_report({
        'factor': '',
        'activity': '',
        'task_relation': '',
        'impact': '',
        'severity': 'low',
        'confidence': 0.0,
        'action': None,
    }, ACTIONS)
    assert report['factors'] == []
    assert report['people_present'] is False
    assert select_action(report) == 'NO_ACTION'


def test_nonempty_factor_with_zero_confidence_is_rejected():
    report = validate_person_semantic_report({
        'factor': 'aisle use',
        'activity': 'shopping',
        'task_relation': 'using the same visible aisle',
        'impact': 'aisle may be occupied',
        'severity': 'low',
        'confidence': 0.0,
        'action': 'WAIT',
    }, ACTIONS)
    assert report['factors'] == []
    assert select_action(report) == 'NO_ACTION'
    assert 'confidence must be greater than 0' in report['validation_errors'][0]


def test_legacy_multiple_factors_keep_highest_severity_only():
    report = validate_person_semantic_report({
        'factors': [
            {
                'factor': 'nearby activity',
                'activity': 'walking nearby',
                'task_relation': 'visibly separate from the task area',
                'impact': 'minor distraction',
                'severity': 'low',
                'confidence': 0.9,
                'action': 'CONTINUE',
            },
            {
                'factor': 'task-area occupation',
                'activity': 'blocking the robot path',
                'task_relation': 'occupying the required visible aisle',
                'impact': 'task cannot continue safely',
                'severity': 'critical',
                'confidence': 0.7,
                'action': 'STOP',
            },
        ],
    }, ACTIONS)
    assert len(report['factors']) == 1
    assert report['factors'][0]['severity'] == 'critical'
    assert select_action(report) == 'STOP'


def test_detailed_mode_keeps_summary_people_and_multiple_factors():
    report = validate_person_semantic_report({
        'summary': 'Two people may affect shelf stocking.',
        'people_present': True,
        'factors': [
            {
                'person': 'Customer beside the shelf.',
                'factor': 'shared shelf use',
                'activity': 'The customer is browsing products.',
                'task_relation': 'The customer uses the shelf required by the task.',
                'impact': 'The aisle remains partly occupied.',
                'severity': 'low',
                'confidence': 0.9,
                'action': None,
            },
            {
                'person': 'Worker in front of the robot.',
                'factor': 'visible area occupation',
                'activity': 'The worker is moving a carton.',
                'task_relation': 'The worker occupies the required visible work area.',
                'impact': 'The planned path could be temporarily blocked.',
                'severity': 'high',
                'confidence': 0.8,
                'action': 'REPLAN',
            },
        ],
    }, ACTIONS, response_mode='detailed')
    assert report['summary'] == 'Two people may affect shelf stocking.'
    assert len(report['factors']) == 2
    assert report['factors'][1]['person'] == 'Worker in front of the robot.'
    assert select_action(report) == 'REPLAN'
    text = render_person_semantic_en(report)
    assert 'Two people may affect shelf stocking.' in text
    assert 'Customer beside the shelf.' in text


def test_detailed_mode_rejects_factor_without_person_description():
    report = validate_person_semantic_report({
        'factors': [{
            'factor': 'nearby activity',
            'activity': 'walking nearby',
            'task_relation': 'relation to the current work is visible',
            'impact': 'temporary obstruction',
            'severity': 'medium',
            'confidence': 0.5,
            'action': None,
        }],
    }, ACTIONS, response_mode='detailed')
    assert report['factors'] == []
    assert 'person is required' in report['validation_errors'][0]


def test_legacy_factor_without_task_relation_is_preserved_with_warning():
    report = validate_person_semantic_report({
        'factor': 'shelf use',
        'activity': 'reaching toward a product',
        'task_relation': '',
        'impact': 'may affect the task',
        'severity': 'medium',
        'confidence': 0.7,
        'action': 'WAIT',
    }, ACTIONS)
    assert len(report['factors']) == 1
    assert report['factors'][0]['action'] == 'WAIT'
    assert report['factors'][0]['task_relation'] == (
        'relationship to the current work was not stated'
    )
    assert 'task_relation was missing' in report['validation_errors'][0]


def test_legacy_factor_without_new_fields_is_preserved():
    report = validate_person_semantic_report({
        'activity': 'standing beside a shelf',
        'impact': 'may affect shelf access',
        'severity': 'medium',
        'confidence': 0.7,
        'action': 'SLOW_DOWN',
    }, ACTIONS)
    assert len(report['factors']) == 1
    assert report['factors'][0]['factor'] == 'person-related factor'
    assert report['factors'][0]['action'] == 'SLOW_DOWN'
    assert len(report['validation_errors']) == 2


def test_markdown_escaped_action_is_normalized():
    report = validate_person_semantic_report({
        'factor': 'visible aisle occupation',
        'activity': 'standing in the aisle',
        'task_relation': 'using the same aisle as the current work',
        'impact': 'may obstruct access',
        'severity': 'medium',
        'confidence': 0.7,
        'action': 'SLOW\\_DOWN',
    }, ACTIONS)
    assert select_action(report) == 'SLOW_DOWN'
