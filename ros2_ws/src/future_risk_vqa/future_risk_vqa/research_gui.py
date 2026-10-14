#!/usr/bin/env python3
"""Qt research monitor for segmentation, prediction, semantics, and policy."""

import json
import os
import sys
import threading

os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")

from PyQt5.QtCore import QObject, Qt, pyqtSignal  # noqa: E402
from PyQt5.QtGui import QFont, QImage, QPixmap  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
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
import rclpy  # noqa: E402
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.utilities import remove_ros_args  # noqa: E402
from sensor_msgs.msg import CompressedImage  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import SetBool  # noqa: E402

from .ui_presenter import (  # noqa: E402
    camera_behavior_summary,
    explain_decision,
)


class Signals(QObject):
    left = pyqtSignal(QImage)
    right = pyqtSignal(QImage)
    storyboard = pyqtSignal(QImage)
    json_update = pyqtSignal(str, str)
    decision = pyqtSignal(str, str, str)
    behaviors = pyqtSignal(str, str, str, str)
    evidence_timing = pyqtSignal(str)
    status = pyqtSignal(str, str)
    pause_result = pyqtSignal(bool, str)


class ResearchGuiNode(Node):
    def __init__(self, signals):
        super().__init__("future_risk_research_gui")
        self.signals = signals
        self.latest_semantics = None
        self._image_sub("/future_risk/vqa_input/left_now/compressed", signals.left)
        self._image_sub("/future_risk/vqa_input/right_now/compressed", signals.right)
        self._image_sub("/future_risk/storyboard/compressed", signals.storyboard)
        for name, topic in (
            ("tracks", "/future_risk/person_tracks"),
            ("geometry", "/future_risk/geometry_prediction"),
            ("semantics", "/future_risk/vlm_events"),
            ("raw_vlm", "/future_risk/vlm_raw_response"),
        ):
            self.create_subscription(
                String, topic, lambda message, key=name: self._json(key, message.data), 10
            )
        for name, topic in (
            ("SAM3", "/future_risk/sam3_status"),
            ("VQA", "/future_risk/vqa_status"),
            ("VIDEO", "/future_risk/video_status"),
        ):
            self.create_subscription(
                String, topic, lambda message, key=name: signals.status.emit(key, message.data), 10
            )
        self.create_subscription(String, "/future_risk/decision", self._decision, 10)
        self.pause_client = self.create_client(SetBool, "/future_risk/set_video_paused")

    def _image_sub(self, topic, signal):
        self.create_subscription(
            CompressedImage,
            topic,
            lambda message: self._image(signal, message),
            qos_profile_sensor_data,
        )

    @staticmethod
    def _image(signal, message):
        image = QImage.fromData(bytes(message.data))
        if not image.isNull():
            signal.emit(image.copy())

    def _json(self, name, data):
        try:
            parsed = json.loads(data)
            pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError):
            parsed = None
            pretty = data
        if name == "semantics" and isinstance(parsed, dict):
            self.latest_semantics = parsed
            left_text, left_level = camera_behavior_summary(parsed, "L")
            right_text, right_level = camera_behavior_summary(parsed, "R")
            self.signals.behaviors.emit(left_text, left_level, right_text, right_level)
            metadata = parsed.get("metadata", {})
            stamps = metadata.get("snapshot_stamps_s", {})
            stamp_text = ", ".join(
                f"{side.upper()}={float(stamp):.3f}"
                for side, stamp in stamps.items()
                if isinstance(stamp, (int, float))
            )
            self.signals.evidence_timing.emit(
                "VQA EVIDENCE (not live): "
                f"{stamp_text or 'timestamp unavailable'} | "
                f"pipeline={float(metadata.get('pipeline_ms', 0.0)) / 1000.0:.2f}s | "
                f"mode={metadata.get('configured_storyboard_mode', '--')} + "
                f"{metadata.get('vqa_inference_mode', '--')}. "
                "Images and NOW/NEXT text belong to this completed inference; "
                "the Action uses the newest tracking report."
            )
        self.signals.json_update.emit(name, pretty)

    def _decision(self, message):
        try:
            report = json.loads(message.data)
            action = str(report.get("action", "DANGER_WARNING")).upper()
            if report.get("decision_mode") == "future_behavior_policy":
                mode_detail = "mode: VQA FUTURE ACTION | zone logic: disabled"
            elif report.get("decision_mode") == "observe_only":
                mode_detail = "mode: OBSERVE ONLY | action output: disabled"
            else:
                mode_detail = (
                    f"calibration: {report.get('calibration_mode', '--')} | "
                    f"metric={report.get('metric_coordinates', False)}"
                )
            detail = (
                f"{explain_decision(report, self.latest_semantics)}\n\n"
                f"Timing: tracking age={report.get('geometry_age_s', '--')}s | "
                f"VQA result age={report.get('semantics_age_s', '--')}s"
                f"{self._evidence_age_text(report)}\n\n"
                f"Technical reason: {report.get('reason_code', '--')}\n"
                f"Critical IDs: {', '.join(report.get('critical_tracks', [])) or '--'}\n"
                f"{mode_detail}"
            )
            pretty = json.dumps(report, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError):
            action, detail, pretty = "DANGER_WARNING", message.data, message.data
        self.signals.decision.emit(action, detail, pretty)

    def _evidence_age_text(self, report):
        if not isinstance(self.latest_semantics, dict):
            return ""
        result_age = report.get("semantics_age_s")
        pipeline_ms = self.latest_semantics.get("metadata", {}).get("pipeline_ms")
        if not isinstance(result_age, (int, float)) or not isinstance(
            pipeline_ms, (int, float)
        ):
            return ""
        return f" | displayed evidence age≈{result_age + pipeline_ms / 1000.0:.2f}s"

    def set_paused(self, paused):
        if not self.pause_client.service_is_ready():
            self.signals.pause_result.emit(False, "pause service is not ready")
            return
        request = SetBool.Request()
        request.data = paused
        future = self.pause_client.call_async(request)
        future.add_done_callback(self._pause_done)

    def _pause_done(self, future):
        try:
            response = future.result()
            self.signals.pause_result.emit(response.success, response.message)
        except Exception as exc:
            self.signals.pause_result.emit(False, str(exc))


class ImageView(QLabel):
    def __init__(self, waiting):
        super().__init__(waiting)
        self.source = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 240)
        self.setStyleSheet("background:#15191d;color:#9ca3aa;border:1px solid #3d444b;")

    def set_image(self, image):
        self.source = image
        self._scale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale()

    def _scale(self):
        if self.source is not None:
            self.setPixmap(
                QPixmap.fromImage(self.source).scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )


class ResearchWindow(QMainWindow):
    COLORS = {
        "OBSERVE": "#1473a8",
        "CONTINUE": "#198754",
        "DANGER_WARNING": "#e08a00",
        "NO_RESPONSE": "#6c757d",
        "STOP": "#c62828",
    }
    ACTION_TEXT = {
        "OBSERVE": "OBSERVE",
        "CONTINUE": "CONTINUE",
        "DANGER_WARNING": "DANGER WARNING",
        "NO_RESPONSE": "NO RESPONSE",
        "STOP": "STOP",
    }
    BEHAVIOR_COLORS = {
        "unknown": ("#f4f5f6", "#59636e"),
        "clear": ("#e5f5ea", "#176b38"),
        "warning": ("#fff0d8", "#9a5700"),
        "stop": ("#fde2e2", "#a61b1b"),
    }

    def __init__(self, node, signals, shutdown):
        super().__init__()
        self.node, self.shutdown = node, shutdown
        self.paused = False
        self.tabs = {}
        self.statuses = {}
        self.setWindowTitle("Future Behavior VQA - Research Monitor")
        self.resize(1680, 980)
        self._build()
        signals.left.connect(self.left.set_image)
        signals.right.connect(self.right.set_image)
        signals.storyboard.connect(self.storyboard.set_image)
        signals.json_update.connect(self._json)
        signals.decision.connect(self._decision)
        signals.behaviors.connect(self._behaviors)
        signals.evidence_timing.connect(self.evidence_timing.setText)
        signals.status.connect(self._status)
        signals.pause_result.connect(self._pause_result)

    def _build(self):
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)
        header = QHBoxLayout()
        title = QLabel("Future Behavior VQA - Research Monitor")
        title.setFont(QFont("Arial", 19, QFont.Bold))
        header.addWidget(title)
        flow = QLabel("SAM3 IDS -> OLD/MID/NOW -> CURRENT STATE -> FUTURE PREDICTION")
        flow.setStyleSheet("color:#55606b;font-weight:bold;")
        header.addStretch()
        header.addWidget(flow)
        root.addLayout(header)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)
        visual = QWidget()
        visual_layout = QVBoxLayout(visual)
        self.evidence_timing = QLabel(
            "VQA EVIDENCE (not live): waiting for the first completed inference..."
        )
        self.evidence_timing.setWordWrap(True)
        self.evidence_timing.setStyleSheet(
            "background:#fff3cd;color:#704f00;padding:6px;border-radius:4px;"
        )
        visual_layout.addWidget(self.evidence_timing)
        grid = QGridLayout()
        grid.addWidget(QLabel("LEFT - TIMESTAMPED VQA EVIDENCE + SAM3"), 0, 0)
        grid.addWidget(QLabel("RIGHT - TIMESTAMPED VQA EVIDENCE + SAM3"), 0, 1)
        self.left, self.right = ImageView("Waiting for left VQA snapshot..."), ImageView(
            "Waiting for right VQA snapshot..."
        )
        grid.addWidget(self.left, 1, 0)
        grid.addWidget(self.right, 1, 1)
        visual_layout.addLayout(grid, 3)
        visual_layout.addWidget(
            QLabel(
                "VLM INPUT USED BY THE COMPLETED VQA INFERENCE "
                "(SEE THE ACTUAL MODE AND TIMING ABOVE)"
            )
        )
        self.storyboard = ImageView("Waiting for storyboard...")
        self.storyboard.setMinimumHeight(320)
        visual_layout.addWidget(self.storyboard, 2)
        splitter.addWidget(visual)
        results = QWidget()
        layout = QVBoxLayout(results)
        self.action = QLabel("NO RESPONSE")
        self.action.setAlignment(Qt.AlignCenter)
        self.action.setFont(QFont("Arial", 30, QFont.Bold))
        layout.addWidget(self.action)
        behavior_title = QLabel("PER-CAMERA CURRENT OBSERVATION + FUTURE PREDICTION")
        behavior_title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(behavior_title)
        behavior_row = QHBoxLayout()
        self.left_behavior = QLabel("Waiting for left-camera VQA result")
        self.right_behavior = QLabel("Waiting for right-camera VQA result")
        for widget in (self.left_behavior, self.right_behavior):
            widget.setWordWrap(True)
            widget.setMinimumHeight(150)
            widget.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        behavior_row.addWidget(self._behavior_card("LEFT CAMERA", self.left_behavior))
        behavior_row.addWidget(self._behavior_card("RIGHT CAMERA", self.right_behavior))
        layout.addLayout(behavior_row)
        why_title = QLabel("WHY THIS DECISION?")
        why_title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(why_title)
        self.detail = QLabel("Waiting for fresh future-behavior evidence...")
        self.detail.setWordWrap(True)
        self.detail.setMinimumHeight(100)
        layout.addWidget(self.detail)
        self.status_label = QLabel("SAM3: -- | VQA: -- | VIDEO: --")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        controls = QHBoxLayout()
        self.pause = QPushButton("Pause after current inference")
        self.pause.clicked.connect(self._toggle_pause)
        controls.addWidget(self.pause)
        controls.addStretch()
        layout.addLayout(controls)
        tab_widget = QTabWidget()
        for name in ("decision", "tracks", "geometry", "semantics", "raw_vlm"):
            editor = QTextEdit()
            editor.setReadOnly(True)
            editor.setFont(QFont("Monospace", 9))
            tab_widget.addTab(editor, name)
            self.tabs[name] = editor
        layout.addWidget(tab_widget, 1)
        splitter.addWidget(results)
        splitter.setSizes([1080, 600])
        waiting = "Waiting for VQA future-behavior result"
        self._behaviors(waiting, "unknown", waiting, "unknown")
        self._decision(
            "NO_RESPONSE",
            "Waiting for the first person-detection report. No action is issued.",
            "{}",
        )

    @staticmethod
    def _behavior_card(title, content):
        card = QWidget()
        card_layout = QVBoxLayout(card)
        label = QLabel(title)
        label.setFont(QFont("Arial", 10, QFont.Bold))
        card_layout.addWidget(label)
        card_layout.addWidget(content)
        card.setStyleSheet("border:1px solid #c7ccd1;border-radius:6px;padding:4px;")
        return card

    def _decision(self, action, detail, pretty):
        color = self.COLORS.get(action, "#60666c")
        self.action.setText(self.ACTION_TEXT.get(action, action))
        self.action.setStyleSheet(
            f"background:{color};color:white;padding:14px;border-radius:8px;"
        )
        self.detail.setText(detail)
        self.tabs["decision"].setPlainText(pretty)

    def _behaviors(self, left_text, left_level, right_text, right_level):
        for widget, value, level in (
            (self.left_behavior, left_text, left_level),
            (self.right_behavior, right_text, right_level),
        ):
            background, foreground = self.BEHAVIOR_COLORS.get(
                level, self.BEHAVIOR_COLORS["unknown"]
            )
            widget.setText(value)
            widget.setStyleSheet(
                f"background:{background};color:{foreground};padding:8px;border-radius:4px;"
            )

    def _json(self, name, pretty):
        if name in self.tabs:
            self.tabs[name].setPlainText(pretty)

    def _status(self, name, value):
        try:
            state = json.loads(value).get("state", value)
        except json.JSONDecodeError:
            state = value
        self.statuses[name] = str(state)
        self.status_label.setText(
            " | ".join(
                f'{key}: {self.statuses.get(key, "--")}' for key in ("SAM3", "VQA", "VIDEO")
            )
        )

    def _toggle_pause(self):
        self.node.set_paused(not self.paused)

    def _pause_result(self, success, message):
        if success:
            self.paused = not self.paused
            self.pause.setText(
                "Resume video + inference" if self.paused else "Pause after current inference"
            )
        self.statusBar().showMessage(message, 4000)

    def closeEvent(self, event):
        self.shutdown()
        event.accept()


def main(args=None):
    ros_args = args if args is not None else sys.argv
    qt_args = remove_ros_args(args=ros_args)
    rclpy.init(args=ros_args)
    signals = Signals()
    node = ResearchGuiNode(signals)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stopped = threading.Event()

    def spin():
        try:
            executor.spin()
        except (ExternalShutdownException, KeyboardInterrupt):
            pass

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    app = QApplication(qt_args)

    def shutdown():
        if stopped.is_set():
            return
        stopped.set()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    window = ResearchWindow(node, signals, shutdown)
    window.show()
    code = app.exec_()
    shutdown()
    thread.join(timeout=2.0)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
