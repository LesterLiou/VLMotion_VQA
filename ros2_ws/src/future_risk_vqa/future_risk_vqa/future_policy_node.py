#!/usr/bin/env python3
"""ROS wrapper for the deterministic Future Risk Decision Policy."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .policy import FutureBehaviorPolicy, FutureDecision, FutureRiskPolicy
from .qos import pipeline_pause_qos


class FuturePolicyNode(Node):
    def __init__(self):
        super().__init__("future_risk_policy")
        defaults = {
            "decision_mode": "future_behavior_policy",
            "enable_action_output": True,
            # ROS 2 Humble cannot represent a typed empty array parameter.
            # DISABLED is removed below before constructing the policy.
            "behavior_stop_motions": ["DISABLED"],
            "behavior_warning_motions": ["TOWARD_CAMERA", "UNCERTAIN"],
            "behavior_continue_motions": [
                "AWAY_FROM_CAMERA",
                "LATERAL_ACROSS_VIEW",
                "STATIONARY",
            ],
            "behavior_continue_confirmations_required": 1,
            "stop_tte_s": 1.0,
            "warning_tte_s": 3.0,
            "semantic_confidence_threshold": 0.55,
            "continue_confirmations_required": 2,
            "require_vlm_for_continue": True,
            "geometry_timeout_s": 2.0,
            "semantics_timeout_s": 20.0,
            "heartbeat_hz": 5.0,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        value = lambda name: self.get_parameter(name).value
        stop_motions = [
            item
            for item in value("behavior_stop_motions")
            if str(item).strip().upper() not in {"", "DISABLED", "NONE"}
        ]
        self.decision_mode = str(value("decision_mode")).lower()
        self.enable_action_output = bool(value("enable_action_output"))
        if self.decision_mode not in {
            "future_behavior_policy",
            "observe_only",
            "zone_safety_policy",
        }:
            raise ValueError(
                "decision_mode must be future_behavior_policy, observe_only, "
                "or zone_safety_policy"
            )
        self.behavior_policy = FutureBehaviorPolicy(
            stop_motions=stop_motions,
            warning_motions=value("behavior_warning_motions"),
            continue_motions=value("behavior_continue_motions"),
            continue_confirmations_required=value(
                "behavior_continue_confirmations_required"
            ),
        )
        self.policy = FutureRiskPolicy(
            value("stop_tte_s"),
            value("warning_tte_s"),
            value("semantic_confidence_threshold"),
            value("continue_confirmations_required"),
            value("require_vlm_for_continue"),
        )
        self.policy_parameters = {
            "stop_tte_s": float(value("stop_tte_s")),
            "warning_tte_s": float(value("warning_tte_s")),
            "semantic_confidence_threshold": float(value("semantic_confidence_threshold")),
            "continue_confirmations_required": int(
                value("continue_confirmations_required")
            ),
            "require_vlm_for_continue": bool(value("require_vlm_for_continue")),
        }
        self.behavior_policy_parameters = {
            "stop_motions": list(stop_motions),
            "warning_motions": list(value("behavior_warning_motions")),
            "continue_motions": list(value("behavior_continue_motions")),
            "continue_confirmations_required": int(
                value("behavior_continue_confirmations_required")
            ),
        }
        self.geometry_timeout = float(value("geometry_timeout_s"))
        self.semantics_timeout = float(value("semantics_timeout_s"))
        self.behavior_policy_parameters["semantics_timeout_s"] = self.semantics_timeout
        self.paused = False
        self.geometry = self.semantics = self.robot_plan = None
        self.geometry_received = self.semantics_received = 0.0
        self.last_key = "__uninitialized__"
        self.last_decision = self.behavior_policy.no_response()
        self.decision_pub = self.create_publisher(String, "/future_risk/decision", 10)
        self.action_pub = self.create_publisher(String, "/future_risk/action", 10)
        self.create_subscription(String, "/future_risk/geometry_prediction", self._geometry, 10)
        self.create_subscription(String, "/future_risk/vlm_events", self._semantics, 10)
        self.create_subscription(String, "/future_risk/robot_plan", self._robot_plan, 10)
        self.create_subscription(
            Bool, "/future_risk/pipeline_paused", self._pipeline_paused, pipeline_pause_qos()
        )
        self.create_timer(1.0 / max(0.2, float(value("heartbeat_hz"))), self._tick)
        self.get_logger().info(f"Future policy ready: mode={self.decision_mode}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _geometry(self, message):
        try:
            self.geometry = json.loads(message.data)
            self.geometry_received = self._now()
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Invalid geometry JSON: {exc}")

    def _semantics(self, message):
        try:
            self.semantics = json.loads(message.data)
            self.semantics_received = self._now()
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Invalid semantic JSON: {exc}")

    def _robot_plan(self, message):
        try:
            self.robot_plan = json.loads(message.data)
        except json.JSONDecodeError:
            self.robot_plan = {"raw": message.data}

    def _pipeline_paused(self, message):
        self.paused = bool(message.data)

    def _tick(self):
        now = self._now()
        geometry_fresh = (
            self.geometry is not None
            and (self.paused or now - self.geometry_received <= self.geometry_timeout)
        )
        semantics_fresh = (
            self.semantics is not None
            and (self.paused or now - self.semantics_received <= self.semantics_timeout)
        )
        if self.decision_mode == "observe_only" or not self.enable_action_output:
            semantics = self.semantics if semantics_fresh else None
            key = None if semantics is None else semantics_received_key(semantics)
            if key != self.last_key:
                self.last_decision = self._future_observation(semantics)
                self.last_key = key
        elif self.decision_mode == "future_behavior_policy":
            semantics = self.semantics if semantics_fresh else None
            current_agents = active_visible_agents(self.geometry)
            key = (
                None if self.geometry is None else self.geometry.get("stamp_s"),
                None if semantics is None else semantics_received_key(semantics),
            )
            if key != self.last_key:
                if self.geometry is None:
                    self.last_decision = self.behavior_policy.no_response()
                elif not current_agents:
                    # An explicit zero-person report is authoritative even when
                    # VQA is absent or the report later becomes stale.
                    self.last_decision = self.behavior_policy.no_response()
                elif any(
                    bool(agent.get("inside_approximate_stop_zone", False))
                    for agent in current_agents
                ):
                    stop_tracks = [
                        str(agent.get("track_id"))
                        for agent in current_agents
                        if bool(agent.get("inside_approximate_stop_zone", False))
                    ]
                    self.behavior_policy.reset()
                    self.last_decision = FutureDecision(
                        action="STOP",
                        reason_code="approximate_image_stop_zone",
                        explanation=(
                            "A visible person foot point is inside the configured "
                            "approximate near-field stop zone."
                        ),
                        critical_tracks=tuple(stop_tracks),
                    )
                elif not geometry_fresh:
                    self.last_decision = self.behavior_policy.unavailable(
                        "tracking_report_stale",
                        "A fresh person-tracking report is required for an action hint.",
                    )
                else:
                    input_agents = (
                        semantics.get("metadata", {}).get("input_tracks")
                        if isinstance(semantics, dict)
                        else None
                    )
                    self.last_decision = self.behavior_policy.evaluate(
                        semantics, input_agents or current_agents
                    )
                self.last_key = key
        elif not geometry_fresh:
            self.last_decision = self.policy.unavailable("geometry_report_stale")
            self.last_key = None
        else:
            semantics = self.semantics if semantics_fresh else None
            key = (
                self.geometry.get("stamp_s"),
                (
                    None
                    if semantics is None
                    else semantics.get("metadata", {}).get("geometry_stamp_s")
                ),
                None if semantics is None else semantics_received_key(semantics),
            )
            if key != self.last_key:
                self.last_decision = self.policy.evaluate(
                    self.geometry, semantics, self.robot_plan
                )
                self.last_key = key
        report = self.last_decision.to_dict()
        report["stamp_s"] = now
        report["geometry_age_s"] = (
            None if self.geometry is None else round(now - self.geometry_received, 3)
        )
        report["semantics_age_s"] = (
            None if self.semantics is None else round(now - self.semantics_received, 3)
        )
        report["calibration_mode"] = (
            None if self.geometry is None else self.geometry.get("calibration_mode")
        )
        report["metric_coordinates"] = (
            False if self.geometry is None else self.geometry.get("metric_coordinates", False)
        )
        report["approximate_stop_zone_enabled"] = (
            False
            if self.geometry is None
            else self.geometry.get("approximate_stop_zone_enabled", False)
        )
        report["approximate_stop_zone_metric"] = (
            False
            if self.geometry is None
            else self.geometry.get("approximate_stop_zone_metric", False)
        )
        if self.decision_mode == "future_behavior_policy":
            report["policy_parameters"] = self.behavior_policy_parameters
        elif self.decision_mode == "zone_safety_policy":
            report["policy_parameters"] = self.policy_parameters
        else:
            report["policy_parameters"] = {"action_output_enabled": False}
        report["decision_mode"] = self.decision_mode
        report["zone_logic_enabled"] = self.decision_mode == "zone_safety_policy"
        report["action_output_enabled"] = self.enable_action_output
        if self.decision_mode == "observe_only" or not self.enable_action_output:
            report["decision_authority"] = "vqa_future_observation_mode"
            report["priority"] = "not_applicable"
        elif self.decision_mode == "future_behavior_policy":
            report["decision_authority"] = "vqa_future_behavior_policy"
            report["priority"] = (
                "STOP > DANGER_WARNING > CONTINUE; NO_RESPONSE if no person"
            )
        message = String()
        message.data = json.dumps(report)
        self.decision_pub.publish(message)
        action = String()
        action.data = report["action"]
        self.action_pub.publish(action)

    @staticmethod
    def _future_observation(semantics):
        if not isinstance(semantics, dict):
            return FutureDecision(
                action="OBSERVE",
                reason_code="awaiting_vqa_future_report",
                explanation="Action output is disabled; waiting for a future-behavior report.",
            )
        errors = semantics.get("validation_errors", [])
        assessments = semantics.get("assessments", [])
        if errors:
            return FutureDecision(
                action="OBSERVE",
                reason_code="vqa_future_report_uncertain",
                explanation=(
                    "Action output is disabled; the current future-behavior report is "
                    "schema-invalid."
                ),
            )
        critical = semantics.get("critical_track_id")
        return FutureDecision(
            action="OBSERVE",
            reason_code="vqa_future_report_ready",
            explanation=(
                f"Action output is disabled; {len(assessments)} tracked future-behavior "
                "assessment(s) are ready for research inspection."
            ),
            critical_tracks=() if critical is None else (str(critical),),
        )


def semantics_received_key(report):
    """Stable content key so heartbeats do not count as clear confirmations."""
    return json.dumps(
        {
            "assessments": report.get("assessments", []),
            "errors": report.get("validation_errors", []),
            "quality_warnings": report.get("quality_warnings", []),
        },
        sort_keys=True,
    )


def active_visible_agents(geometry):
    if not isinstance(geometry, dict):
        return []
    return [
        item
        for item in geometry.get("tracked_agents", [])
        if bool(item.get("visible", int(item.get("missed_frames", 0)) == 0))
    ]


def main(args=None):
    rclpy.init(args=args)
    node = FuturePolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
