#!/usr/bin/env python3
"""Qt monitor for the video real-time and asynchronous VLM risk layers."""

import json
import sys
import threading

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
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import SetBool


class GuiSignals(QObject):
    frame_ready = pyqtSignal(QImage)
    realtime_ready = pyqtSignal(str, str)
    vlm_text_ready = pyqtSignal(str)
    vlm_report_ready = pyqtSignal(str)
    vlm_action_ready = pyqtSignal(str)
    fusion_ready = pyqtSignal(str, str)
    vlm_status_ready = pyqtSignal(str)
    video_status_ready = pyqtSignal(str)
    playback_result = pyqtSignal(bool, bool, str)


class RiskVideoGuiNode(Node):
    """Subscribe to risk topics and expose playback control to the Qt window."""

    def __init__(self, signals):
        super().__init__('risk_video_gui')
        self.signals = signals
        self.create_subscription(
            CompressedImage,
            '/risk_vqa/video/annotated/compressed',
            self._frame_callback,
            2,
        )
        self.create_subscription(
            String, '/risk_vqa/realtime_report', self._realtime_callback, 10
        )
        self.create_subscription(String, '/risk_vqa/text', self._vlm_text_callback, 10)
        self.create_subscription(String, '/risk_vqa/report', self._vlm_report_callback, 10)
        self.create_subscription(
            String, '/risk_vqa/action_suggestion', self._vlm_action_callback, 10
        )
        self.create_subscription(
            String, '/risk_vqa/fusion_report', self._fusion_callback, 10
        )
        self.create_subscription(String, '/risk_vqa/status', self._vlm_status_callback, 10)
        self.create_subscription(
            String, '/risk_vqa/video_status', self._video_status_callback, 10
        )
        self.pause_client = self.create_client(SetBool, '/risk_vqa/set_paused')

    def _frame_callback(self, message):
        image = QImage.fromData(bytes(message.data))
        if image.isNull():
            return
        self.signals.frame_ready.emit(image.copy())

    def _realtime_callback(self, message):
        try:
            report = json.loads(message.data)
            action = str(report.get('action', 'CONTINUE'))
            people = int(report.get('person_count', 0))
            processing_ms = float(report.get('processing_ms', 0.0))
            tracks = report.get('tracks', [])
            ttc_values = [
                float(track['ttc_s'])
                for track in tracks
                if track.get('ttc_s') is not None
            ]
            ttc = f'{min(ttc_values):.2f} s' if ttc_values else '--'
            detail = (
                f'People: {people}   Nearest optical TTC: {ttc}   '
                f'Inference: {processing_ms:.1f} ms'
            )
            self.signals.realtime_ready.emit(action, detail)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.signals.realtime_ready.emit('UNKNOWN', f'Invalid real-time report: {exc}')

    def _vlm_text_callback(self, message):
        self.signals.vlm_text_ready.emit(message.data)

    def _vlm_report_callback(self, message):
        self.signals.vlm_report_ready.emit(self._pretty_json(message.data))

    def _vlm_action_callback(self, message):
        self.signals.vlm_action_ready.emit(message.data)

    def _fusion_callback(self, message):
        pretty = self._pretty_json(message.data)
        try:
            fused = str(json.loads(message.data).get('fused_action_suggestion', 'NO_ACTION'))
        except (TypeError, ValueError, json.JSONDecodeError):
            fused = 'UNKNOWN'
        self.signals.fusion_ready.emit(fused, pretty)

    def _vlm_status_callback(self, message):
        self.signals.vlm_status_ready.emit(message.data)

    def _video_status_callback(self, message):
        self.signals.video_status_ready.emit(message.data)

    @staticmethod
    def _pretty_json(text):
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except (TypeError, ValueError, json.JSONDecodeError):
            return text

    def set_paused(self, paused):
        if not self.pause_client.service_is_ready():
            self.signals.playback_result.emit(
                False,
                paused,
                'Playback service is not ready. Check that video_ttc_node is running.',
            )
            return
        request = SetBool.Request()
        request.data = paused
        future = self.pause_client.call_async(request)
        future.add_done_callback(
            lambda completed: self._playback_done(completed, paused)
        )

    def _playback_done(self, future, paused):
        try:
            response = future.result()
            self.signals.playback_result.emit(
                bool(response.success), paused, str(response.message)
            )
        except Exception as exc:
            self.signals.playback_result.emit(False, paused, str(exc))


class RiskVideoWindow(QMainWindow):
    """Display annotated video and both risk-decision layers."""

    ACTION_COLORS = {
        'CONTINUE': '#1b8f3a',
        'SLOW_DOWN': '#c88700',
        'WAIT': '#8a6500',
        'REPLAN': '#7656b5',
        'ALERT_HUMAN': '#b85c00',
        'STOP': '#c62828',
        'NO_ACTION': '#5f6368',
        'UNKNOWN': '#5f6368',
    }

    def __init__(self, ros_node, signals, shutdown_callback):
        super().__init__()
        self.ros_node = ros_node
        self.shutdown_callback = shutdown_callback
        self.current_image = None
        self.setWindowTitle('Risk Video VQA Monitor')
        self.resize(1420, 850)
        self._build_ui()
        self._connect_signals(signals)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        title = QLabel('Risk Video VQA Monitor')
        title.setFont(QFont('Arial', 18, QFont.Bold))
        root.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        self.video_label = QLabel('Waiting for annotated video...')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 500)
        self.video_label.setStyleSheet(
            'QLabel { background: #202124; color: #ddd; border: 1px solid #555; }'
        )
        video_layout.addWidget(self.video_label, 1)

        controls = QHBoxLayout()
        self.pause_button = QPushButton('Pause')
        self.continue_button = QPushButton('Continue')
        self.pause_button.setMinimumHeight(42)
        self.continue_button.setMinimumHeight(42)
        self.pause_button.clicked.connect(lambda: self._request_pause(True))
        self.continue_button.clicked.connect(lambda: self._request_pause(False))
        controls.addWidget(self.pause_button)
        controls.addWidget(self.continue_button)
        self.playback_label = QLabel('Video: waiting')
        controls.addWidget(self.playback_label, 1)
        video_layout.addLayout(controls)
        self.realtime_detail = QLabel('Waiting for real-time report...')
        video_layout.addWidget(self.realtime_detail)
        splitter.addWidget(video_panel)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        cards = QGridLayout()
        cards.addWidget(QLabel('Real-time Action'), 0, 0)
        cards.addWidget(QLabel('VLM Advisory'), 0, 1)
        cards.addWidget(QLabel('Fused Advisory'), 0, 2)
        self.realtime_action = self._action_label('CONTINUE')
        self.vlm_action = self._action_label('NO_ACTION')
        self.fused_action = self._action_label('NO_ACTION')
        cards.addWidget(self.realtime_action, 1, 0)
        cards.addWidget(self.vlm_action, 1, 1)
        cards.addWidget(self.fused_action, 1, 2)
        result_layout.addLayout(cards)

        self.vlm_status = QLabel('VLM: waiting for image')
        result_layout.addWidget(self.vlm_status)

        tabs = QTabWidget()
        self.vlm_text = self._read_only_text('Waiting for the first VLM answer...')
        self.vlm_report = self._read_only_text('Waiting for structured VLM report...')
        self.fusion_report = self._read_only_text('Waiting for fusion report...')
        tabs.addTab(self.vlm_text, 'VLM Answer')
        tabs.addTab(self.vlm_report, 'VLM JSON')
        tabs.addTab(self.fusion_report, 'Fusion')
        result_layout.addWidget(tabs, 1)
        note = QLabel(
            'Safety control authority remains with the real-time action. '
            'VLM and fused actions are advisory.'
        )
        note.setWordWrap(True)
        result_layout.addWidget(note)
        splitter.addWidget(result_panel)
        splitter.setSizes([850, 570])

    @staticmethod
    def _read_only_text(initial):
        widget = QTextEdit()
        widget.setReadOnly(True)
        widget.setPlainText(initial)
        return widget

    def _action_label(self, action):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(44)
        self._set_action(label, action)
        return label

    def _set_action(self, label, action):
        action = str(action).upper().strip() or 'NO_ACTION'
        color = self.ACTION_COLORS.get(action, self.ACTION_COLORS['UNKNOWN'])
        label.setText(action)
        label.setStyleSheet(
            f'QLabel {{ background: {color}; color: white; padding: 8px; '
            'font-weight: bold; border-radius: 4px; }}'
        )

    def _connect_signals(self, signals):
        signals.frame_ready.connect(self._show_frame)
        signals.realtime_ready.connect(self._show_realtime)
        signals.vlm_text_ready.connect(self.vlm_text.setPlainText)
        signals.vlm_report_ready.connect(self.vlm_report.setPlainText)
        signals.vlm_action_ready.connect(
            lambda action: self._set_action(self.vlm_action, action)
        )
        signals.fusion_ready.connect(self._show_fusion)
        signals.vlm_status_ready.connect(
            lambda status: self.vlm_status.setText(f'VLM: {status}')
        )
        signals.video_status_ready.connect(
            lambda status: self.playback_label.setText(f'Video: {status}')
        )
        signals.playback_result.connect(self._playback_result)

    def _show_frame(self, image):
        self.current_image = image
        self._scale_current_image()

    def _scale_current_image(self):
        if self.current_image is None:
            return
        pixmap = QPixmap.fromImage(self.current_image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_current_image()

    def _show_realtime(self, action, detail):
        self._set_action(self.realtime_action, action)
        self.realtime_detail.setText(detail)

    def _show_fusion(self, action, report):
        self._set_action(self.fused_action, action)
        self.fusion_report.setPlainText(report)

    def _request_pause(self, paused):
        self.pause_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        self.playback_label.setText('Video: changing state...')
        self.ros_node.set_paused(paused)

    def _playback_result(self, success, paused, message):
        self.pause_button.setEnabled(True)
        self.continue_button.setEnabled(True)
        state = 'paused' if paused else 'running'
        prefix = 'Video' if success else 'Playback error'
        self.playback_label.setText(f'{prefix}: {state if success else message}')

    def closeEvent(self, event):
        self.shutdown_callback()
        event.accept()


def main(args=None):
    raw_args = args if args is not None else sys.argv
    qt_args = remove_ros_args(args=raw_args)
    rclpy.init(args=raw_args)
    app = QApplication(qt_args)
    signals = GuiSignals()
    node = RiskVideoGuiNode(signals)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
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

    window = RiskVideoWindow(node, signals, shutdown)
    window.show()
    result = app.exec_()
    shutdown()
    spin_thread.join(timeout=2.0)
    return result


if __name__ == '__main__':
    sys.exit(main())
