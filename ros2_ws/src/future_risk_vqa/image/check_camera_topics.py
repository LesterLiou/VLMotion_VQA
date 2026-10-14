"""Run with ROS 2 Humble system Python on either computer; never commands motion."""
import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, CompressedImage, JointState

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seconds', type=float, default=10)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
rclpy.init(args=[])
node = rclpy.create_node('x9_camera_connectivity_check')
stats = {}
subscriptions = []
topics = [('/joint_states', JointState)]
for side in ('front', 'left', 'right'):
    for suffix, msg_type in [('image_raw', Image), ('camera_info', CameraInfo),
                             ('image_raw/compressed', CompressedImage)]:
        topics.append((f'/camera/{side}/color/{suffix}', msg_type))


def receive(topic, msg):
    entry = stats[topic]
    entry['received'] += 1
    if isinstance(msg, Image):
        entry.update(width=msg.width, height=msg.height, encoding=msg.encoding, bytes=len(msg.data))
    elif isinstance(msg, CompressedImage):
        entry.update(format=msg.format, bytes=len(msg.data))


try:
    for topic, msg_type in topics:
        stats[topic] = {'received': 0}
        subscriptions.append(node.create_subscription(
            msg_type, topic, lambda msg, topic=topic: receive(topic, msg), qos_profile_sensor_data))
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    for topic in stats:
        stats[topic]['publishers'] = node.count_publishers(topic)
    report = {'seconds': args.seconds, 'topics': stats}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n')
finally:
    node.destroy_node()
    rclpy.shutdown()
