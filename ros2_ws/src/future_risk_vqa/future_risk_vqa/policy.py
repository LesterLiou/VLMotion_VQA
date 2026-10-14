"""Explainable Future Risk Decision Policy."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FutureDecision:
    action: str
    reason_code: str
    explanation: str
    critical_tracks: tuple = ()
    continue_streak: int = 0

    def to_dict(self):
        report = asdict(self)
        report["critical_tracks"] = list(self.critical_tracks)
        report["decision_authority"] = "future_risk_deterministic_policy"
        report["priority"] = "STOP > DANGER_WARNING > CONTINUE; NO_RESPONSE if no person"
        return report


class FutureBehaviorPolicy:
    """Map camera-relative future motion to deterministic action hints."""

    def __init__(
        self,
        stop_motions=(),
        warning_motions=("TOWARD_CAMERA", "UNCERTAIN"),
        continue_motions=("AWAY_FROM_CAMERA", "LATERAL_ACROSS_VIEW", "STATIONARY"),
        continue_confirmations_required=2,
    ):
        self.stop_motions = {str(value).upper() for value in stop_motions}
        self.warning_motions = {str(value).upper() for value in warning_motions}
        self.continue_motions = {str(value).upper() for value in continue_motions}
        self.continue_confirmations_required = max(1, int(continue_confirmations_required))
        self.continue_streak = 0

    def reset(self):
        self.continue_streak = 0

    def _warning(self, reason, explanation, tracks=()):
        self.reset()
        return FutureDecision(
            action="DANGER_WARNING",
            reason_code=reason,
            explanation=explanation,
            critical_tracks=tuple(tracks),
        )

    def unavailable(self, reason, explanation):
        return self._warning(reason, explanation)

    def no_response(self):
        self.reset()
        return FutureDecision(
            action="NO_RESPONSE",
            reason_code="no_person_detected",
            explanation="No visible person is currently detected, so no action is issued.",
        )

    def evaluate(self, semantics, tracked_agents=()):
        tracked_agents = [
            agent
            for agent in tracked_agents
            if bool(agent.get("visible", int(agent.get("missed_frames", 0)) == 0))
        ]
        if not tracked_agents:
            return self.no_response()
        if not isinstance(semantics, dict):
            return self._warning(
                "future_behavior_unavailable",
                "A fresh VQA future-behavior report is required.",
            )
        if str(semantics.get("availability", "READY")).upper() == "UNAVAILABLE":
            return self._warning(
                "future_behavior_unavailable",
                "The VQA model request is unavailable.",
                [str(agent.get("track_id")) for agent in tracked_agents],
            )
        if semantics.get("validation_errors"):
            return self._warning(
                "future_behavior_schema_invalid",
                "The VQA future-behavior report failed schema validation.",
            )
        agent_ids = [str(agent.get("track_id")) for agent in tracked_agents]
        assessments = {
            str(item.get("track_id")): item
            for item in semantics.get("assessments", [])
            if isinstance(item, dict)
        }
        missing = [track_id for track_id in agent_ids if track_id not in assessments]
        if missing:
            return self._warning(
                "future_behavior_assessment_missing",
                "At least one active person ID has no VQA future assessment.",
                missing,
            )
        stop_tracks = []
        warning_tracks = []
        approaching_tracks = []
        safe_tracks = []
        for track_id in agent_ids:
            assessment = assessments[track_id]
            future = assessment.get("future_prediction", {})
            motion = str(future.get("future_motion", "UNCERTAIN")).upper()
            if motion in self.stop_motions:
                stop_tracks.append(track_id)
            elif motion in self.warning_motions:
                warning_tracks.append(track_id)
                if motion == "TOWARD_CAMERA":
                    approaching_tracks.append(track_id)
            elif motion in self.continue_motions:
                safe_tracks.append(track_id)
            else:
                warning_tracks.append(track_id)

        if stop_tracks:
            self.reset()
            return FutureDecision(
                action="STOP",
                reason_code="vqa_future_crossing",
                explanation="A tracked person is predicted to cross the camera plane.",
                critical_tracks=tuple(stop_tracks),
            )
        if warning_tracks:
            reason = (
                "vqa_future_approaching_warning"
                if approaching_tracks
                else "vqa_future_uncertain_warning"
            )
            explanation = (
                "A tracked person is moving toward the robot-mounted camera."
                if approaching_tracks
                else "A future behavior is uncertain or has insufficient evidence."
            )
            return self._warning(reason, explanation, warning_tracks)
        if len(safe_tracks) == len(agent_ids):
            return self._confirmed_continue(tuple(safe_tracks))
        return self._warning(
            "vqa_future_not_proven_safe",
            "The VQA future behavior is not sufficient for a continue hint.",
            agent_ids,
        )

    def _confirmed_continue(self, tracks):
        self.continue_streak += 1
        if self.continue_streak < self.continue_confirmations_required:
            return FutureDecision(
                action="DANGER_WARNING",
                reason_code="awaiting_vqa_continue_confirmation",
                explanation="A clear future report was received; waiting for confirmation.",
                critical_tracks=tuple(tracks),
                continue_streak=self.continue_streak,
            )
        return FutureDecision(
            action="CONTINUE",
            reason_code="vqa_future_clear_confirmed",
            explanation="All tracked future behaviors are clear across confirmations.",
            critical_tracks=tuple(tracks),
            continue_streak=self.continue_streak,
        )


class FutureRiskPolicy:
    """Combine geometry, camera-relative motion, and robot plan without a risk score."""

    SAFE_MOTIONS = {"AWAY_FROM_CAMERA", "LATERAL_ACROSS_VIEW", "STATIONARY"}

    def __init__(
        self,
        stop_tte_s=1.0,
        warning_tte_s=3.0,
        semantic_confidence_threshold=0.55,
        continue_confirmations_required=2,
        require_vlm_for_continue=True,
    ):
        self.stop_tte_s = float(stop_tte_s)
        self.warning_tte_s = float(warning_tte_s)
        self.semantic_confidence_threshold = float(semantic_confidence_threshold)
        self.continue_confirmations_required = max(1, int(continue_confirmations_required))
        self.require_vlm_for_continue = bool(require_vlm_for_continue)
        self.continue_streak = 0

    def reset(self):
        self.continue_streak = 0

    def unavailable(self, reason_code="pipeline_data_unavailable"):
        self.reset()
        return FutureDecision(
            action="DANGER_WARNING",
            reason_code=reason_code,
            explanation="Fresh geometry and semantic future evidence are required.",
        )

    def no_response(self):
        self.reset()
        return FutureDecision(
            action="NO_RESPONSE",
            reason_code="no_person_detected",
            explanation="No visible person is currently detected, so no action is issued.",
        )

    def evaluate(self, geometry, semantics, robot_plan=None):
        robot_plan = robot_plan or {}
        if not geometry.get("localization_valid", False):
            return self.unavailable("ground_localization_invalid")
        agents = [
            agent
            for agent in geometry.get("tracked_agents", [])
            if bool(agent.get("visible", int(agent.get("missed_frames", 0)) == 0))
        ]
        if not agents:
            return self.no_response()
        assessments = (
            {
                item.get("track_id"): item
                for item in semantics.get("assessments", [])
                if isinstance(item, dict)
            }
            if isinstance(semantics, dict)
            else {}
        )

        stop_tracks = []
        for agent in agents:
            tte = agent.get("time_to_entry_s")
            if agent.get("current_zone") == "HAZARD_ZONE" or (
                tte is not None and float(tte) <= self.stop_tte_s
            ):
                stop_tracks.append(str(agent.get("track_id")))
        if stop_tracks:
            self.reset()
            return FutureDecision(
                action="STOP",
                reason_code="immediate_hazard_or_tte",
                explanation=(
                    "A tracked person is in the hazard zone or will enter within one second."
                ),
                critical_tracks=tuple(stop_tracks),
            )

        unreliable = [
            str(agent.get("track_id"))
            for agent in agents
            if not agent.get("tracking_reliable", False)
            and agent.get("current_zone") in {"WARNING_ZONE", "HAZARD_ZONE"}
        ]
        if unreliable:
            self.reset()
            return FutureDecision(
                action="DANGER_WARNING",
                reason_code="nearby_track_unreliable",
                explanation="A nearby person track is interrupted or unreliable.",
                critical_tracks=tuple(unreliable),
            )

        if self.require_vlm_for_continue and not isinstance(semantics, dict):
            return self.unavailable("semantic_report_unavailable")
        if isinstance(semantics, dict) and semantics.get("validation_errors"):
            return self.unavailable("semantic_schema_invalid")
        uncertain = []
        for agent in agents:
            track_id = str(agent.get("track_id"))
            assessment = assessments.get(track_id)
            near = agent.get("current_zone") in {"WARNING_ZONE", "HAZARD_ZONE"}
            if not near:
                continue
            if (
                assessment is None
                or (float(assessment.get("confidence", 0.0)) < self.semantic_confidence_threshold)
                or assessment.get("future_prediction", {}).get("future_motion")
                == "UNCERTAIN"
            ):
                uncertain.append(track_id)
        if uncertain:
            self.reset()
            return FutureDecision(
                action="DANGER_WARNING",
                reason_code="nearby_semantics_uncertain",
                explanation=(
                    "A person near the warning zone lacks reliable semantic interpretation."
                ),
                critical_tracks=tuple(uncertain),
            )

        warning_tracks = []
        for agent in agents:
            track_id = str(agent.get("track_id"))
            tte = agent.get("time_to_entry_s")
            assessment = assessments.get(track_id, {})
            geometry_risk = bool(agent.get("trajectory_intersects_hazard"))
            within_horizon = (
                tte is not None and self.stop_tte_s < float(tte) <= self.warning_tte_s
            )
            semantic_risk = (
                assessment.get("future_prediction", {}).get("future_motion")
                == "TOWARD_CAMERA"
            )
            if geometry_risk or within_horizon or semantic_risk:
                warning_tracks.append(track_id)
        if warning_tracks:
            self.reset()
            return FutureDecision(
                action="DANGER_WARNING",
                reason_code="future_intersection_or_semantic_approach",
                explanation=(
                    "A predicted path may enter the hazard zone, or the person is moving "
                    "toward the robot-mounted camera. "
                    "A VLM contradiction cannot override geometry to CONTINUE."
                ),
                critical_tracks=tuple(warning_tracks),
            )

        semantics_safe = all(
            assessments.get(str(agent.get("track_id")), {})
            .get("future_prediction", {})
            .get("future_motion")
            in self.SAFE_MOTIONS
            for agent in agents
        )
        if semantics_safe or not self.require_vlm_for_continue:
            self.continue_streak += 1
            if self.continue_streak >= self.continue_confirmations_required:
                return FutureDecision(
                    action="CONTINUE",
                    reason_code="confirmed_future_clear",
                    explanation="Geometry and camera-relative motion remain clear across confirmations.",
                    continue_streak=self.continue_streak,
                )
            return FutureDecision(
                action="DANGER_WARNING",
                reason_code="awaiting_continue_confirmation",
                explanation="One clear future assessment received; waiting for confirmation.",
                continue_streak=self.continue_streak,
            )
        return self.unavailable("future_state_not_proven_safe")
