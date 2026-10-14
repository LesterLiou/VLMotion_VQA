"""Deterministic and conservative decision policy over validated VQA output."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Decision:
    state: str
    command: str
    reason_code: str
    explanation: str
    relevant_people: tuple = ()
    clear_streak: int = 0

    def to_dict(self):
        value = asdict(self)
        value['relevant_people'] = list(self.relevant_people)
        value['decision_authority'] = 'deterministic_rules'
        return value


class RiskDecisionEngine:
    """Convert semantic observations to a small fail-conservative state set."""

    def __init__(
        self,
        person_confidence_threshold=0.55,
        visibility_confidence_threshold=0.55,
        clear_confirmations_required=2,
    ):
        self.person_confidence_threshold = float(person_confidence_threshold)
        self.visibility_confidence_threshold = float(
            visibility_confidence_threshold
        )
        self.clear_confirmations_required = max(
            1, int(clear_confirmations_required)
        )
        self.clear_streak = 0

    def reset(self):
        self.clear_streak = 0

    @staticmethod
    def stale(reason='observation_timeout'):
        return Decision(
            state='UNKNOWN',
            command='HOLD',
            reason_code=reason,
            explanation='No fresh validated dual-camera observation is available.',
        )

    def evaluate(self, report):
        """Evaluate one report. The input must already use the validated schema."""
        cameras = [report.get('left', {}), report.get('right', {})]
        blocked = []
        for side, camera in zip(('left', 'right'), cameras):
            visibility = camera.get('visibility', 'UNCERTAIN')
            confidence = float(camera.get('visibility_confidence', 0.0))
            if visibility != 'CLEAR' or confidence < self.visibility_confidence_threshold:
                blocked.append(f'{side}:{visibility}')
        if blocked:
            self.clear_streak = 0
            return Decision(
                state='UNKNOWN',
                command='HOLD',
                reason_code='camera_view_not_reliable',
                explanation='Cannot prove both approaches are visible: ' + ', '.join(blocked),
            )

        people = []
        for side, camera in zip(('left', 'right'), cameras):
            for person in camera.get('people', []):
                people.append((side, person))

        if not people:
            self.clear_streak += 1
            if self.clear_streak >= self.clear_confirmations_required:
                return Decision(
                    state='GO',
                    command='GO',
                    reason_code='confirmed_clear',
                    explanation='Both views are clear and contain no people.',
                    clear_streak=self.clear_streak,
                )
            return Decision(
                state='HOLD',
                command='HOLD',
                reason_code='awaiting_clear_confirmation',
                explanation='One clear observation received; waiting for temporal confirmation.',
                clear_streak=self.clear_streak,
            )

        self.clear_streak = 0
        imminent = []
        for side, person in people:
            if (
                float(person.get('confidence', 0.0))
                >= self.person_confidence_threshold
                and (
                    person['position'] == 'INSIDE_HAZARD_ZONE'
                    or person['motion'] == 'CROSSING'
                    or person['entry_time'] in {'NOW', '0_TO_2_SECONDS'}
                )
            ):
                imminent.append(f'{side}:{person["person_id"]}')
        if imminent:
            return Decision(
                state='STOP',
                command='STOP',
                reason_code='entry_or_crossing_imminent',
                explanation=(
                    'A person is inside, crossing, or expected to enter '
                    'within two seconds.'
                ),
                relevant_people=tuple(imminent),
            )

        uncertain = [
            f'{side}:{person["person_id"]}'
            for side, person in people
            if float(person.get('confidence', 0.0)) < self.person_confidence_threshold
            or person.get('position') == 'UNKNOWN'
            or person.get('motion') == 'UNKNOWN'
            or person.get('intent') == 'UNCERTAIN'
        ]
        if uncertain:
            return Decision(
                state='HOLD',
                command='HOLD',
                reason_code='person_classification_uncertain',
                explanation='A visible person cannot be classified reliably.',
                relevant_people=tuple(uncertain),
            )

        approaching = []
        for side, person in people:
            if (
                person['intent'] == 'ENTER'
                or person['motion'] == 'APPROACHING'
                or person['position'] == 'APPROACH_ZONE'
                or person['entry_time'] == '2_TO_5_SECONDS'
            ):
                approaching.append(f'{side}:{person["person_id"]}')
        if approaching:
            return Decision(
                state='HOLD',
                command='HOLD',
                reason_code='person_approaching_or_intends_to_enter',
                explanation='A person is approaching or is likely to enter the hazard zone.',
                relevant_people=tuple(approaching),
            )

        visible = tuple(f'{side}:{person["person_id"]}' for side, person in people)
        return Decision(
            state='WATCH',
            command='WATCH',
            reason_code='person_visible_without_current_conflict',
            explanation='People are visible but are waiting, passing by, or moving away.',
            relevant_people=visible,
        )
