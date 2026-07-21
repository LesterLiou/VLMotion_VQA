import pytest

from risk_vqa.ttc_tracker import PersonTTCMonitor


def detection(track_id, bbox, confidence=0.9):
    return {'track_id': track_id, 'bbox': bbox, 'confidence': confidence}


def test_expanding_yolo_person_track_produces_ttc():
    monitor = PersonTTCMonitor(
        warning_ttc_s=3.0,
        critical_ttc_s=1.2,
        min_growth_rate=0.01,
        growth_ema_alpha=1.0,
        min_confirmed_age=2,
        roi=(0.0, 0.0, 1.0, 1.0),
    )
    monitor.update([detection(7, (45, 45, 55, 55))], 0.0, (100, 100, 3))
    result = monitor.update(
        [detection(7, (40, 40, 60, 60), 0.92)],
        1.0,
        (100, 100, 3),
    )
    track = result['tracks'][0]
    assert result['person_count'] == 1
    assert track['track_id'] == 7
    assert track['class_name'] == 'person'
    assert track['confidence'] == 0.92
    assert track['ttc_s'] == pytest.approx(1.443, abs=0.002)
    assert result['action'] == 'SLOW_DOWN'

    result = monitor.update(
        [detection(7, (30, 30, 70, 70))],
        1.5,
        (100, 100, 3),
    )
    assert result['action'] == 'STOP'


def test_static_person_stays_continue():
    monitor = PersonTTCMonitor(
        min_confirmed_age=2,
        roi=(0.0, 0.0, 1.0, 1.0),
    )
    monitor.update([detection(1, (20, 20, 40, 40))], 0.0, (100, 100, 3))
    result = monitor.update([detection(1, (20, 20, 40, 40))], 1.0, (100, 100, 3))
    assert result['tracks'][0]['ttc_s'] is None
    assert result['action'] == 'CONTINUE'


def test_close_static_person_triggers_area_based_stop():
    monitor = PersonTTCMonitor(
        min_confirmed_age=2,
        critical_area_ratio=0.08,
        roi=(0.0, 0.0, 1.0, 1.0),
    )
    close_person = detection(2, (35, 20, 65, 50))  # 9% of the frame
    monitor.update([close_person], 0.0, (100, 100, 3))
    result = monitor.update([close_person], 1.0, (100, 100, 3))
    assert result['tracks'][0]['area_ratio'] == pytest.approx(0.09)
    assert result['action'] == 'STOP'


def test_person_outside_roi_does_not_trigger():
    monitor = PersonTTCMonitor(
        min_growth_rate=0.01,
        growth_ema_alpha=1.0,
        min_confirmed_age=2,
        roi=(0.4, 0.4, 0.9, 0.9),
    )
    monitor.update([detection(3, (0, 0, 10, 10))], 0.0, (100, 100, 3))
    result = monitor.update([detection(3, (0, 0, 20, 20))], 1.0, (100, 100, 3))
    assert result['tracks'][0]['inside_roi'] is False
    assert result['action'] == 'CONTINUE'


def test_multiple_people_preserve_bytetrack_ids():
    monitor = PersonTTCMonitor(min_confirmed_age=2)
    result = monitor.update([
        detection(11, (10, 10, 30, 50)),
        detection(22, (70, 10, 90, 50)),
    ], 0.0, (100, 100, 3))
    assert [track['track_id'] for track in result['tracks']] == [11, 22]
