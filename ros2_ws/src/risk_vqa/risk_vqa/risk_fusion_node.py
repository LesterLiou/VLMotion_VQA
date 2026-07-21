#!/usr/bin/env python3
"""Fuse deterministic real-time risk with asynchronous VLM suggestions."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


PRIORITY = {
    'NO_ACTION': 0,
    'CONTINUE': 1,
    'SLOW_DOWN': 2,
    'WAIT': 3,
    'REPLAN': 4,
    'ALERT_HUMAN': 5,
    'STOP': 6,
}


class RiskFusionNode(Node):
    """Publish a fused suggestion while preserving the deterministic action."""

    def __init__(self):
        super().__init__('risk_fusion')
        self.declare_parameter('realtime_action_topic', '/risk_vqa/realtime_action')
        self.declare_parameter('semantic_action_topic', '/risk_vqa/action_suggestion')
        self.declare_parameter('fused_action_topic', '/risk_vqa/fused_action_suggestion')
        self.declare_parameter('fusion_report_topic', '/risk_vqa/fusion_report')
        self.declare_parameter('semantic_timeout_s', 5.0)

        value = lambda name: self.get_parameter(name).value
        self.semantic_timeout_s = float(value('semantic_timeout_s'))
        self.realtime_action = 'CONTINUE'
        self.semantic_action = 'NO_ACTION'
        self.semantic_updated_s = None

        self.fused_pub = self.create_publisher(String, str(value('fused_action_topic')), 10)
        self.report_pub = self.create_publisher(String, str(value('fusion_report_topic')), 10)
        self.create_subscription(
            String,
            str(value('realtime_action_topic')),
            self._realtime_callback,
            10,
        )
        self.create_subscription(
            String,
            str(value('semantic_action_topic')),
            self._semantic_callback,
            10,
        )

    def _now_s(self):
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    @staticmethod
    def _normalize(action, default):
        action = str(action).upper().strip()
        return action if action in PRIORITY else default

    def _realtime_callback(self, message):
        self.realtime_action = self._normalize(message.data, 'CONTINUE')
        self._publish_fusion()

    def _semantic_callback(self, message):
        self.semantic_action = self._normalize(message.data, 'NO_ACTION')
        self.semantic_updated_s = self._now_s()
        self._publish_fusion()

    def _publish_fusion(self):
        semantic_fresh = (
            self.semantic_updated_s is not None
            and self._now_s() - self.semantic_updated_s <= self.semantic_timeout_s
        )
        semantic = self.semantic_action if semantic_fresh else 'NO_ACTION'
        fused = max(
            (self.realtime_action, semantic),
            key=lambda action: PRIORITY[action],
        )
        report = {
            'realtime_action': self.realtime_action,
            'semantic_action_suggestion': semantic,
            'semantic_fresh': semantic_fresh,
            'fused_action_suggestion': fused,
            'control_authority': 'realtime_action_only',
            'note': 'VLM output is advisory and must not directly command the robot',
        }
        action_message = String()
        action_message.data = fused
        self.fused_pub.publish(action_message)
        report_message = String()
        report_message.data = json.dumps(report)
        self.report_pub.publish(report_message)


def main(args=None):
    rclpy.init(args=args)
    node = RiskFusionNode()
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
