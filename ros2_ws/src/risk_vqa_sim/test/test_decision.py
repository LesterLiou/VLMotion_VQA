from risk_vqa_sim.decision import RiskDecisionEngine
from risk_vqa_sim.schema import validate_observation


def _report(left_people=None, right_people=None, left_visibility='CLEAR'):
    return validate_observation({
        'left': {
            'visibility': left_visibility,
            'visibility_confidence': 0.95,
            'people': left_people or [],
        },
        'right': {
            'visibility': 'CLEAR',
            'visibility_confidence': 0.95,
            'people': right_people or [],
        },
    })


def _person(**overrides):
    person = {
        'person_id': 'L1',
        'position': 'WAITING_ZONE',
        'motion': 'STATIONARY',
        'intent': 'WAIT',
        'entry_time': 'NOT_APPROACHING',
        'confidence': 0.9,
        'evidence': 'person stays at the waiting point',
    }
    person.update(overrides)
    return person


def test_clear_requires_two_confirmations():
    engine = RiskDecisionEngine(clear_confirmations_required=2)
    assert engine.evaluate(_report()).state == 'HOLD'
    assert engine.evaluate(_report()).state == 'GO'


def test_occlusion_forces_unknown_hold():
    engine = RiskDecisionEngine()
    result = engine.evaluate(_report(left_visibility='OCCLUDED'))
    assert result.state == 'UNKNOWN'
    assert result.command == 'HOLD'


def test_crossing_or_imminent_entry_stops():
    engine = RiskDecisionEngine()
    person = _person(
        position='APPROACH_ZONE',
        motion='APPROACHING',
        intent='ENTER',
        entry_time='0_TO_2_SECONDS',
    )
    result = engine.evaluate(_report(left_people=[person]))
    assert result.state == 'STOP'
    assert result.relevant_people == ('left:L1',)


def test_confirmed_inside_person_stops_even_if_intent_is_uncertain():
    engine = RiskDecisionEngine()
    person = _person(
        position='INSIDE_HAZARD_ZONE',
        motion='UNKNOWN',
        intent='UNCERTAIN',
        entry_time='NOW',
    )
    assert engine.evaluate(_report(left_people=[person])).state == 'STOP'


def test_approach_outside_imminent_horizon_holds():
    engine = RiskDecisionEngine()
    person = _person(
        position='APPROACH_ZONE',
        motion='APPROACHING',
        intent='ENTER',
        entry_time='2_TO_5_SECONDS',
    )
    assert engine.evaluate(_report(left_people=[person])).state == 'HOLD'


def test_waiting_person_is_watch():
    engine = RiskDecisionEngine()
    result = engine.evaluate(_report(left_people=[_person()]))
    assert result.state == 'WATCH'
    assert result.command == 'WATCH'


def test_low_confidence_person_holds():
    engine = RiskDecisionEngine()
    result = engine.evaluate(_report(left_people=[_person(confidence=0.2)]))
    assert result.reason_code == 'person_classification_uncertain'
    assert result.command == 'HOLD'
