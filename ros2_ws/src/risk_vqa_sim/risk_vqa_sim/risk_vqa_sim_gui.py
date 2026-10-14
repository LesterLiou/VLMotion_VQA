#!/usr/bin/env python3
"""PyQt5 monitor for dual-camera temporal VQA and deterministic decisions."""

import json
import os
import sys
import threading

import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import SetBool

# This QWidget-only monitor does not need GLX. Disabling the xcb GL bridge
# avoids nouveau/DIR3 errors in GPU-enabled Docker containers.
os.environ.setdefault('QT_XCB_GL_INTEGRATION', 'none')


class GuiSignals(QObject):
    left_frame_ready = pyqtSignal(QImage)
    right_frame_ready = pyqtSignal(QImage)
    storyboard_ready = pyqtSignal(QImage)
    observation_ready = pyqtSignal(str, str)
    decision_ready = pyqtSignal(str, str, str, str)
    vqa_status_ready = pyqtSignal(str)
    video_status_ready = pyqtSignal(str)
    raw_response_ready = pyqtSignal(str)
    playback_result = pyqtSignal(bool, bool, str)


class RiskVQASimGuiNode(Node):
    """Subscribe to all monitoring topics and bridge them to Qt signals."""

    def __init__(self, signals):
        super().__init__('risk_vqa_sim_gui')
        self.signals = signals
        self._declare_parameters()

        def value(name):
            return self.get_parameter(name).value

        self.use_compressed_image = bool(value('use_compressed_image'))
        image_type = CompressedImage if self.use_compressed_image else Image
        self.create_subscription(
            image_type,
            str(value('left_image_topic')),
            self._left_frame_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            image_type,
            str(value('right_image_topic')),
            self._right_frame_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CompressedImage,
            str(value('storyboard_topic')),
            self._storyboard_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            str(value('observation_topic')),
            self._observation_callback,
            10,
        )
        self.create_subscription(
            String,
            str(value('decision_topic')),
            self._decision_callback,
            10,
        )
        self.create_subscription(
            String,
            str(value('vqa_status_topic')),
            lambda message: self.signals.vqa_status_ready.emit(message.data),
            10,
        )
        self.create_subscription(
            String,
            str(value('video_status_topic')),
            self._video_status_callback,
            10,
        )
        self.create_subscription(
            String,
            str(value('raw_response_topic')),
            lambda message: self.signals.raw_response_ready.emit(message.data),
            10,
        )
        self._pause_client = self.create_client(
            SetBool, str(value('pause_service'))
        )

    def _declare_parameters(self):
        self.declare_parameter('left_image_topic', '/left_camera/rgb')
        self.declare_parameter('right_image_topic', '/right_camera/rgb')
        self.declare_parameter('use_compressed_image', False)
        self.declare_parameter(
            'storyboard_topic', '/risk_vqa_sim/storyboard/compressed'
        )
        self.declare_parameter('observation_topic', '/risk_vqa_sim/observation')
        self.declare_parameter('decision_topic', '/risk_vqa_sim/decision')
        self.declare_parameter('vqa_status_topic', '/risk_vqa_sim/status')
        self.declare_parameter(
            'video_status_topic', '/risk_vqa_sim/video_source_status'
        )
        self.declare_parameter(
            'raw_response_topic', '/risk_vqa_sim/raw_model_response'
        )
        self.declare_parameter(
            'pause_service', '/risk_vqa_sim/set_video_paused'
        )

    def _to_qimage(self, message):
        if self.use_compressed_image:
            image = QImage.fromData(bytes(message.data))
            if image.isNull():
                raise ValueError('Qt could not decode compressed image')
            return image.copy()
        encoding = str(message.encoding).lower()
        if encoding not in {'rgb8', 'bgr8'}:
            raise ValueError(
                f'GUI supports rgb8 or bgr8 raw images, got {message.encoding!r}'
            )
        height = int(message.height)
        width = int(message.width)
        step = int(message.step)
        row_bytes = width * 3
        if height <= 0 or width <= 0 or step < row_bytes:
            raise ValueError('raw image dimensions or row step are invalid')
        raw = np.frombuffer(message.data, dtype=np.uint8)
        if raw.size < height * step:
            raise ValueError('raw image data is shorter than its declared size')
        rows = raw[:height * step].reshape(height, step)
        frame = rows[:, :row_bytes].reshape(height, width, 3)
        if encoding == 'bgr8':
            frame = frame[:, :, ::-1]
        frame = np.ascontiguousarray(frame)
        image = QImage(
            frame.data,
            width,
            height,
            frame.strides[0],
            QImage.Format_RGB888,
        )
        return image.copy()

    def _left_frame_callback(self, message):
        try:
            self.signals.left_frame_ready.emit(self._to_qimage(message))
        except Exception as exc:
            self.get_logger().error(f'Left GUI image conversion failed: {exc}')

    def _right_frame_callback(self, message):
        try:
            self.signals.right_frame_ready.emit(self._to_qimage(message))
        except Exception as exc:
            self.get_logger().error(f'Right GUI image conversion failed: {exc}')

    def _storyboard_callback(self, message):
        image = QImage.fromData(bytes(message.data))
        if not image.isNull():
            self.signals.storyboard_ready.emit(image.copy())

    def _observation_callback(self, message):
        try:
            report = json.loads(message.data)
            summary = self._format_observation(report)
            pretty = json.dumps(report, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            summary = f'Invalid observation JSON: {exc}'
            pretty = message.data
        self.signals.observation_ready.emit(summary, pretty)

    def _decision_callback(self, message):
        try:
            report = json.loads(message.data)
            state = str(report.get('state', 'UNKNOWN')).upper()
            command = str(report.get('command', 'HOLD')).upper()
            detail = self._format_decision(report)
            pretty = json.dumps(report, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            state = 'UNKNOWN'
            command = 'HOLD'
            detail = f'Invalid decision JSON: {exc}'
            pretty = message.data
        self.signals.decision_ready.emit(state, command, detail, pretty)

    def _video_status_callback(self, message):
        try:
            report = json.loads(message.data)
            state = str(report.get('state', 'unknown'))
            timeline_s = float(report.get('timeline_s', 0.0))
            paused = bool(report.get('paused', False))
            suffix = 'paused' if paused else state
            text = f'{suffix} | video time {timeline_s:.1f} s'
        except (json.JSONDecodeError, TypeError, ValueError):
            text = message.data
        self.signals.video_status_ready.emit(text)

    @staticmethod
    def _format_observation(report):
        lines = []
        for side in ('left', 'right'):
            camera = report.get(side, {})
            visibility = str(camera.get('visibility', 'UNCERTAIN'))
            visibility_confidence = float(
                camera.get('visibility_confidence', 0.0)
            )
            people = camera.get('people', [])
            lines.append(
                f'{side.upper()}  {visibility} ({visibility_confidence:.0%})  '
                f'people={len(people)}'
            )
            if not people:
                lines.append('  No visible person reported')
            for person in people:
                lines.append(
                    '  {person_id}: {position} | {motion} | intent={intent} | '
                    'entry={entry_time} | confidence={confidence:.0%}'.format(
                        person_id=person.get('person_id', '?'),
                        position=person.get('position', 'UNKNOWN'),
                        motion=person.get('motion', 'UNKNOWN'),
                        intent=person.get('intent', 'UNCERTAIN'),
                        entry_time=person.get('entry_time', 'UNKNOWN'),
                        confidence=float(person.get('confidence', 0.0)),
                    )
                )
                evidence = str(person.get('evidence', '')).strip()
                if evidence:
                    lines.append(f'    evidence: {evidence}')
            lines.append('')
        scene_summary = str(report.get('scene_summary', '')).strip()
        if scene_summary:
            lines.append(f'SCENE: {scene_summary}')
        metadata = report.get('metadata', {})
        if metadata:
            model = metadata.get('model', '--')
            pipeline_ms = float(metadata.get('pipeline_ms', 0.0))
            evidence_age = float(
                metadata.get('latest_frame_age_at_publish_s', 0.0)
            )
            lines.append(
                f'MODEL: {model} | pipeline={pipeline_ms:.0f} ms | '
                f'evidence age={evidence_age:.1f} s'
            )
        errors = report.get('validation_errors', [])
        if errors:
            lines.append('VALIDATION: ' + '; '.join(str(item) for item in errors))
        return '\n'.join(lines)

    @staticmethod
    def _format_decision(report):
        reason = str(report.get('reason_code', '--'))
        explanation = str(report.get('explanation', ''))
        people = ', '.join(report.get('relevant_people', [])) or '--'
        age = report.get('observation_age_s')
        age_text = '--' if age is None else f'{float(age):.1f} s'
        streak = int(report.get('clear_streak', 0))
        return (
            f'Reason: {reason}\n'
            f'{explanation}\n'
            f'Relevant people: {people}\n'
            f'Observation age: {age_text} | clear streak: {streak}'
        )

    def set_paused(self, paused):
        if not self._pause_client.service_is_ready():
            self.signals.playback_result.emit(
                False,
                paused,
                'Video pause service is not ready.',
            )
            return
        request = SetBool.Request()
        request.data = paused
        future = self._pause_client.call_async(request)
        future.add_done_callback(
            lambda completed: self._pause_done(completed, paused)
        )

    def _pause_done(self, future, paused):
        try:
            response = future.result()
            self.signals.playback_result.emit(
                bool(response.success), paused, str(response.message)
            )
        except Exception as exc:
            self.signals.playback_result.emit(False, paused, str(exc))


class RiskVQASimWindow(QMainWindow):
    """Display source views, VLM evidence, and the rule-based decision."""

    COLORS = {
        'GO': '#198754',
        'WATCH': '#1473a8',
        'HOLD': '#d17b00',
        'STOP': '#c62828',
        'UNKNOWN': '#60666c',
    }

    def __init__(self, ros_node, signals, shutdown_callback):
        super().__init__()
        self.ros_node = ros_node
        self.shutdown_callback = shutdown_callback
        self.images = {'left': None, 'right': None, 'storyboard': None}
        self.setWindowTitle('Dual-Camera Risk VQA Monitor')
        self.resize(1660, 960)
        self._build_ui()
        self._connect_signals(signals)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        header = QHBoxLayout()
        title = QLabel('Dual-Camera Risk VQA Monitor')
        title.setFont(QFont('Arial', 19, QFont.Bold))
        header.addWidget(title)
        flow = QLabel('LEFT + RIGHT → TEMPORAL VQA → RULES → COMMAND')
        flow.setStyleSheet('color: #555; font-weight: bold;')
        header.addStretch()
        header.addWidget(flow)
        root.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)
        splitter.addWidget(self._build_visual_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([1060, 600])

    def _build_visual_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        camera_grid = QGridLayout()
        camera_grid.addWidget(self._section_title('LEFT CAMERA'), 0, 0)
        camera_grid.addWidget(self._section_title('RIGHT CAMERA'), 0, 1)
        self.left_view = self._image_label('Waiting for left video...')
        self.right_view = self._image_label('Waiting for right video...')
        self.left_view.setMinimumSize(450, 260)
        self.right_view.setMinimumSize(450, 260)
        camera_grid.addWidget(self.left_view, 1, 0)
        camera_grid.addWidget(self.right_view, 1, 1)
        layout.addLayout(camera_grid, 3)

        layout.addWidget(self._section_title('MODEL INPUT — OLD → MID → NOW'))
        self.storyboard_view = self._image_label(
            'Waiting for the first temporal storyboard...'
        )
        self.storyboard_view.setMinimumHeight(300)
        layout.addWidget(self.storyboard_view, 2)
        return panel

    def _build_result_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        cards = QGridLayout()
        cards.addWidget(QLabel('Decision State'), 0, 0)
        cards.addWidget(QLabel('Advisory Command'), 0, 1)
        self.state_label = self._state_label('UNKNOWN')
        self.command_label = self._state_label('HOLD')
        cards.addWidget(self.state_label, 1, 0)
        cards.addWidget(self.command_label, 1, 1)
        layout.addLayout(cards)

        self.decision_detail = QLabel('Waiting for decision heartbeat...')
        self.decision_detail.setWordWrap(True)
        self.decision_detail.setMinimumHeight(92)
        self.decision_detail.setStyleSheet(
            'QLabel { background: #f3f4f5; border: 1px solid #ccc; padding: 8px; }'
        )
        layout.addWidget(self.decision_detail)

        controls = QHBoxLayout()
        self.pause_button = QPushButton('Pause Videos')
        self.continue_button = QPushButton('Continue Videos')
        self.pause_button.setMinimumHeight(38)
        self.continue_button.setMinimumHeight(38)
        self.pause_button.clicked.connect(lambda: self._request_pause(True))
        self.continue_button.clicked.connect(lambda: self._request_pause(False))
        controls.addWidget(self.pause_button)
        controls.addWidget(self.continue_button)
        layout.addLayout(controls)

        self.video_status = QLabel('Video: waiting')
        self.vqa_status = QLabel('VQA: waiting')
        layout.addWidget(self.video_status)
        layout.addWidget(self.vqa_status)

        tabs = QTabWidget()
        self.summary_text = self._text_widget(
            'Waiting for the first validated observation...'
        )
        self.observation_json = self._text_widget('Waiting for observation JSON...')
        self.decision_json = self._text_widget('Waiting for decision JSON...')
        self.raw_response = self._text_widget('Waiting for raw model response...')
        tabs.addTab(self.summary_text, 'Layer Summary')
        tabs.addTab(self.observation_json, 'Observation JSON')
        tabs.addTab(self.decision_json, 'Decision JSON')
        tabs.addTab(self.raw_response, 'Raw VLM')
        layout.addWidget(tabs, 1)

        note = QLabel(
            'Research advisory only. UNKNOWN and stale evidence map to HOLD; '
            'this UI is not a safety-rated control interface.'
        )
        note.setWordWrap(True)
        note.setStyleSheet('color: #7a1f1f; font-weight: bold;')
        layout.addWidget(note)
        return panel

    @staticmethod
    def _section_title(text):
        label = QLabel(text)
        label.setStyleSheet('font-weight: bold; color: #333;')
        return label

    @staticmethod
    def _image_label(initial):
        label = QLabel(initial)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            'QLabel { background: #202124; color: #ddd; border: 1px solid #555; }'
        )
        return label

    @staticmethod
    def _text_widget(initial):
        widget = QTextEdit()
        widget.setReadOnly(True)
        widget.setLineWrapMode(QTextEdit.NoWrap)
        widget.setPlainText(initial)
        return widget

    def _state_label(self, state):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(48)
        self._set_state(label, state)
        return label

    def _set_state(self, label, state):
        state = str(state).upper().strip() or 'UNKNOWN'
        color = self.COLORS.get(state, self.COLORS['UNKNOWN'])
        label.setText(state)
        label.setStyleSheet(
            f'QLabel {{ background: {color}; color: white; padding: 9px; '
            'font-weight: bold; font-size: 16px; border-radius: 4px; }'
        )

    def _connect_signals(self, signals):
        signals.left_frame_ready.connect(
            lambda image: self._show_image('left', image)
        )
        signals.right_frame_ready.connect(
            lambda image: self._show_image('right', image)
        )
        signals.storyboard_ready.connect(
            lambda image: self._show_image('storyboard', image)
        )
        signals.observation_ready.connect(self._show_observation)
        signals.decision_ready.connect(self._show_decision)
        signals.vqa_status_ready.connect(
            lambda status: self.vqa_status.setText(f'VQA: {status}')
        )
        signals.video_status_ready.connect(
            lambda status: self.video_status.setText(f'Video: {status}')
        )
        signals.raw_response_ready.connect(self.raw_response.setPlainText)
        signals.playback_result.connect(self._playback_result)

    def _show_image(self, name, image):
        self.images[name] = image
        self._scale_image(name)

    def _scale_image(self, name):
        image = self.images.get(name)
        if image is None:
            return
        target = {
            'left': self.left_view,
            'right': self.right_view,
            'storyboard': self.storyboard_view,
        }[name]
        pixmap = QPixmap.fromImage(image).scaled(
            target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        target.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for name in self.images:
            self._scale_image(name)

    def _show_observation(self, summary, pretty):
        self.summary_text.setPlainText(summary)
        self.observation_json.setPlainText(pretty)

    def _show_decision(self, state, command, detail, pretty):
        self._set_state(self.state_label, state)
        self._set_state(self.command_label, command)
        self.decision_detail.setText(detail)
        self.decision_json.setPlainText(pretty)

    def _request_pause(self, paused):
        self.pause_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        self.video_status.setText('Video: changing playback state...')
        self.ros_node.set_paused(paused)

    def _playback_result(self, success, paused, message):
        self.pause_button.setEnabled(True)
        self.continue_button.setEnabled(True)
        if success:
            state = 'paused' if paused else 'running'
            self.video_status.setText(f'Video: {state}')
        else:
            self.video_status.setText(f'Playback error: {message}')

    def closeEvent(self, event):
        self.shutdown_callback()
        event.accept()


def main(args=None):
    raw_args = args if args is not None else sys.argv
    qt_args = remove_ros_args(args=raw_args)
    rclpy.init(args=raw_args)
    app = QApplication(qt_args)
    signals = GuiSignals()
    node = RiskVQASimGuiNode(signals)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin_ros():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    spin_thread = threading.Thread(target=spin_ros, daemon=True)
    spin_thread.start()
    stopped = False

    def shutdown():
        nonlocal stopped
        if stopped:
            return
        stopped = True
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    window = RiskVQASimWindow(node, signals, shutdown)
    window.show()
    result = app.exec_()
    shutdown()
    spin_thread.join(timeout=2.0)
    return result


if __name__ == '__main__':
    sys.exit(main())
