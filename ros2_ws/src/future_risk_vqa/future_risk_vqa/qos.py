"""Shared ROS 2 QoS profiles for pipeline control state."""

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def pipeline_pause_qos():
    """Keep the last pause state so late model nodes cannot miss it."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
