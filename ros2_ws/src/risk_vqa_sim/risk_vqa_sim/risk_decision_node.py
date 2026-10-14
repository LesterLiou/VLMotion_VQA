#!/usr/bin/env python3
"""ROS 2 node applying deterministic rules to dual-camera VQA observations."""

import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from risk_vqa_sim.decision import RiskDecisionEngine
from risk_vqa_sim.schema import validate_observation


class RiskDecisionNode(Node):
    """Publish a fail-conservative action while keeping the VLM advisory-only."""

    def __init__(self):
        super().__init__('risk_decision')
        self._declare_parameters()

        def value(name):
            return self.get_parameter(name).value

        self.observation_topic = str(value('observation_topic'))
        self.decision_topic = str(value('decision_topic'))
        self.command_topic = str(value('command_topic'))
        self.input_timeout_s = max(0.5, float(value('input_timeout_s')))
        self.max_evidence_age_s = max(
            0.5, float(value('max_evidence_age_s'))
        )
        self.heartbeat_period_s = max(0.1, float(value('heartbeat_period_s')))
        self.engine = RiskDecisionEngine(
            person_confidence_threshold=float(
                value('person_confidence_threshold')
            ),
            visibility_confidence_threshold=float(
                value('visibility_confidence_threshold')
            ),
            clear_confirmations_required=int(
                value('clear_confirmations_required')
            ),
        )
        self._last_receipt_s = None
        self._source_evidence_age_s = None
        self._last_decision = self.engine.stale('waiting_for_first_observation')
        self._decision_pub = self.create_publisher(String, self.decision_topic, 10)
        self._command_pub = self.create_publisher(String, self.command_topic, 10)
        self.create_subscription(
            String, self.observation_topic, self._observation_callback, 10
        )
        self.create_timer(self.heartbeat_period_s, self._heartbeat)
        self.get_logger().info(
            f'Decision rules ready: observation={self.observation_topic}, '
            f'command={self.command_topic}'
        )

    def _declare_parameters(self):
        self.declare_parameter('observation_topic', '/risk_vqa_sim/observation')
        self.declare_parameter('decision_topic', '/risk_vqa_sim/decision')
        self.declare_parameter('command_topic', '/risk_vqa_sim/command')
        self.declare_parameter('input_timeout_s', 5.0)
        self.declare_parameter('max_evidence_age_s', 5.0)
        self.declare_parameter('heartbeat_period_s', 1.0)
        self.declare_parameter('person_confidence_threshold', 0.55)
        self.declare_parameter('visibility_confidence_threshold', 0.55)
        self.declare_parameter('clear_confirmations_required', 2)

    def _observation_callback(self, message):
        try:
            raw = json.loads(message.data)
            metadata = raw.get('metadata', {})
            evidence_age_s = float(
                metadata.get('latest_frame_age_at_publish_s', 0.0)
            )
            if not math.isfinite(evidence_age_s) or evidence_age_s < 0.0:
                raise ValueError('latest_frame_age_at_publish_s is invalid')
            self._source_evidence_age_s = evidence_age_s
            if evidence_age_s > self.max_evidence_age_s:
                self.engine.reset()
                self._last_decision = self.engine.stale(
                    'visual_evidence_stale'
                )
                self._last_receipt_s = time.monotonic()
                self._publish(self._last_decision, 0.0)
                return
            report = validate_observation(raw)
            self._last_decision = self.engine.evaluate(report)
            self._last_receipt_s = time.monotonic()
            self._publish(self._last_decision, 0.0)
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            self.engine.reset()
            self._last_receipt_s = time.monotonic()
            self._source_evidence_age_s = 0.0
            self._last_decision = self.engine.stale('invalid_observation')
            self.get_logger().error(f'Invalid observation: {exc}')
            self._publish(self._last_decision, 0.0)

    def _heartbeat(self):
        if self._last_receipt_s is None:
            self._publish(self._last_decision, None)
            return
        age_s = time.monotonic() - self._last_receipt_s
        evidence_age_s = (self._source_evidence_age_s or 0.0) + age_s
        if evidence_age_s > self.max_evidence_age_s:
            self.engine.reset()
            self._last_decision = self.engine.stale('visual_evidence_stale')
        elif age_s > self.input_timeout_s:
            self.engine.reset()
            self._last_decision = self.engine.stale()
        self._publish(self._last_decision, age_s)

    def _publish(self, decision, age_s):
        report = decision.to_dict()
        if age_s is None:
            report['observation_age_s'] = None
        else:
            source_age_s = self._source_evidence_age_s or 0.0
            report['observation_age_s'] = round(
                max(0.0, source_age_s + age_s), 3
            )
        report['safety_boundary'] = (
            'advisory_only; do not connect directly to safety-rated actuation'
        )
        message = String()
        message.data = json.dumps(report, ensure_ascii=False)
        self._decision_pub.publish(message)
        command = String()
        command.data = decision.command
        self._command_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = RiskDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
