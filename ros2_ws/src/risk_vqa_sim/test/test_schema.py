import pytest

from risk_vqa_sim.schema import extract_json_object, validate_observation


def test_extract_json_from_wrapped_model_text():
    value = extract_json_object('answer: {"left": {}, "right": {}} trailing')
    assert 'left' in value


def test_extract_json_rejects_non_object():
    with pytest.raises(ValueError):
        extract_json_object('there is no machine-readable response')


def test_missing_cameras_become_uncertain():
    report = validate_observation({})
    assert report['left']['visibility'] == 'UNCERTAIN'
    assert report['right']['visibility_confidence'] == 0.0
    assert report['validation_errors']


def test_invalid_person_fields_are_conservative():
    raw = {
        'left': {
            'visibility': 'CLEAR',
            'visibility_confidence': 0.9,
            'people': [{
                'person_id': 'L1',
                'position': 'somewhere',
                'motion': 'fast',
                'intent': 'maybe',
                'entry_time': 'soon',
                'confidence': 'bad',
            }],
        },
        'right': {
            'visibility': 'CLEAR',
            'visibility_confidence': 0.9,
            'people': [],
        },
    }
    report = validate_observation(raw)
    person = report['left']['people'][0]
    assert person['position'] == 'UNKNOWN'
    assert person['motion'] == 'UNKNOWN'
    assert person['intent'] == 'UNCERTAIN'
    assert person['entry_time'] == 'UNKNOWN'
    assert person['confidence'] == 0.0
    assert len(report['validation_errors']) == 5


def test_invalid_people_container_cannot_become_clear_empty_view():
    report = validate_observation({
        'left': {
            'visibility': 'CLEAR',
            'visibility_confidence': 0.9,
            'people': 'none',
        },
        'right': {
            'visibility': 'CLEAR',
            'visibility_confidence': 0.9,
            'people': [],
        },
    })
    assert report['left']['visibility'] == 'UNCERTAIN'
    assert report['left']['visibility_confidence'] == 0.0
