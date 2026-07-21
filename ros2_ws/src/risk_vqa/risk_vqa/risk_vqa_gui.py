#!/usr/bin/env python3
"""Standalone image-question GUI for Risk VQA."""

import argparse
import base64
import json
import os
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import requests

from risk_vqa.prompt import build_risk_prompt, wrap_vicuna_prompt
from risk_vqa.schema import (
    extract_json_object,
    render_en,
    select_action,
    validate_report,
)


DEFAULT_ACTIONS = [
    'CONTINUE',
    'SLOW_DOWN',
    'WAIT',
    'STOP',
    'REPLAN',
    'ALERT_HUMAN',
]


class RiskRequestThread(QThread):
    """Run controller and worker HTTP requests without blocking Qt."""

    result_ready = pyqtSignal(str, str, str)
    status_changed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        controller_url,
        model_name,
        image_bytes,
        question,
        work_context,
        horizon_s,
    ):
        super().__init__()
        self.controller_url = controller_url.rstrip('/')
        self.model_name = model_name.strip()
        self.image_bytes = image_bytes
        self.question = question
        self.work_context = work_context
        self.horizon_s = float(horizon_s)

    def _resolve_model(self):
        if self.model_name:
            return self.model_name
        response = requests.post(
            self.controller_url + '/list_models',
            json={},
            timeout=5,
        )
        response.raise_for_status()
        models = response.json().get('models', [])
        if not models:
            raise RuntimeError('The controller is connected, but no model worker is available')
        return str(models[0])

    def _worker_address(self, model_name):
        response = requests.post(
            self.controller_url + '/get_worker_address',
            json={'model': model_name},
            timeout=5,
        )
        response.raise_for_status()
        address = str(response.json().get('address', '')).rstrip('/')
        if not address:
            raise RuntimeError(f'No worker is available for model {model_name}')
        return address

    def run(self):
        try:
            self.status_changed.emit('Connecting to the controller...')
            model_name = self._resolve_model()
            worker_address = self._worker_address(model_name)
            self.status_changed.emit(f'Analyzing the image with {model_name}...')

            risk_prompt = build_risk_prompt(
                self.work_context,
                self.horizon_s,
                DEFAULT_ACTIONS,
                self.question,
            )
            prompt = wrap_vicuna_prompt(risk_prompt)
            payload = {
                'model': model_name,
                'prompt': prompt,
                'temperature': 0.0,
                'top_p': 1.0,
                'max_new_tokens': 768,
                'use_cache': False,
                'stop': '###',
                'images': [base64.b64encode(self.image_bytes).decode('ascii')],
                'task_type': 'risk_assessment',
                'use_sam3_candidates': False,
            }
            response = requests.post(
                worker_address + '/worker_generate_stream',
                headers={'User-Agent': 'risk_vqa standalone GUI'},
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
            completion = ''
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b'\0'):
                if not chunk:
                    continue
                data = json.loads(chunk.decode('utf-8'))
                if data.get('error_code', 1) != 0:
                    raise RuntimeError(data.get('text', 'Model worker error'))
                text = str(data.get('text', ''))
                completion = text[len(prompt):] if text.startswith(prompt) else text
            if not completion.strip():
                raise RuntimeError('The model worker returned an empty response')

            report = validate_report(
                extract_json_object(completion),
                self.horizon_s,
                DEFAULT_ACTIONS,
            )
            report.update({
                'prediction_horizon_s': self.horizon_s,
                'work_context': self.work_context,
                'question': self.question,
                'model': model_name,
            })
            self.result_ready.emit(
                render_en(report),
                select_action(report),
                json.dumps(report, ensure_ascii=False, indent=2),
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class RiskVQAGUI(QMainWindow):
    """Load one local image and ask risk-oriented VQA questions."""

    def __init__(self, controller_url, model_name, work_context, horizon_s):
        super().__init__()
        self.controller_url = controller_url
        self.model_name = model_name
        self.image_path = ''
        self.image_bytes = None
        self.worker_thread = None

        self.setWindowTitle('Risk VQA - Image Question Answering')
        self.resize(1180, 760)
        self._build_ui(work_context, horizon_s)

    def _build_ui(self, work_context, horizon_s):
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        title = QLabel('Risk VQA - Image Question Answering')
        title.setFont(QFont('Arial', 18, QFont.Bold))
        layout.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.image_label = QLabel('Click "Open Image" to begin')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(520, 420)
        self.image_label.setStyleSheet(
            'QLabel { background: #202124; color: #ddd; border: 1px solid #555; }'
        )
        left_layout.addWidget(self.image_label, 1)
        self.open_button = QPushButton('Open Image')
        self.open_button.clicked.connect(self.open_image)
        left_layout.addWidget(self.open_button)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel('Current Work Context'))
        self.context_input = QLineEdit(work_context)
        right_layout.addWidget(self.context_input)

        horizon_row = QHBoxLayout()
        horizon_row.addWidget(QLabel('Prediction Horizon (seconds)'))
        self.horizon_input = QSpinBox()
        self.horizon_input.setRange(1, 120)
        self.horizon_input.setValue(int(horizon_s))
        horizon_row.addWidget(self.horizon_input)
        horizon_row.addStretch()
        right_layout.addLayout(horizon_row)

        right_layout.addWidget(QLabel('VQA Question'))
        self.question_input = QTextEdit()
        self.question_input.setPlainText(
            'Identify factors that may affect the current work, estimate how many '
            'seconds until each factor has an impact, and suggest an action.'
        )
        self.question_input.setMaximumHeight(110)
        right_layout.addWidget(self.question_input)

        self.send_button = QPushButton('Ask Question')
        self.send_button.setMinimumHeight(42)
        self.send_button.clicked.connect(self.send_question)
        right_layout.addWidget(self.send_button)

        self.status_label = QLabel('Ready')
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(QLabel('Answer'))
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        right_layout.addWidget(self.answer_output, 1)
        splitter.addWidget(right)
        splitter.setSizes([580, 600])

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select a VQA Image',
            '/workspace',
            'Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)',
        )
        if not path:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, 'Open Failed', f'Cannot read image:\n{path}')
            return
        try:
            with open(path, 'rb') as image_file:
                self.image_bytes = image_file.read()
        except OSError as exc:
            QMessageBox.warning(self, 'Open Failed', str(exc))
            return
        self.image_path = path
        self._show_pixmap(pixmap)
        self.status_label.setText(f'Loaded: {os.path.basename(path)}')

    def _show_pixmap(self, pixmap):
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_path:
            self._show_pixmap(QPixmap(self.image_path))

    def send_question(self):
        question = self.question_input.toPlainText().strip()
        work_context = self.context_input.text().strip()
        if self.image_bytes is None:
            QMessageBox.information(self, 'Image Required', 'Open an image first.')
            return
        if not question or not work_context:
            QMessageBox.information(
                self,
                'Missing Input',
                'Enter both the work context and the VQA question.',
            )
            return
        if self.worker_thread is not None and self.worker_thread.isRunning():
            return

        self.answer_output.clear()
        self.send_button.setEnabled(False)
        self.worker_thread = RiskRequestThread(
            self.controller_url,
            self.model_name,
            self.image_bytes,
            question,
            work_context,
            self.horizon_input.value(),
        )
        self.worker_thread.status_changed.connect(self.status_label.setText)
        self.worker_thread.result_ready.connect(self._show_result)
        self.worker_thread.failed.connect(self._show_error)
        self.worker_thread.finished.connect(lambda: self.send_button.setEnabled(True))
        self.worker_thread.start()

    def _show_result(self, natural_text, action, report_json):
        self.status_label.setText('Completed')
        self.answer_output.setPlainText(
            f'{natural_text}\n\nAction Suggestion: {action}\n\nDetailed Report:\n{report_json}'
        )

    def _show_error(self, message):
        self.status_label.setText('Analysis Failed')
        self.answer_output.setPlainText(f'Error: {message}')


def main(args=None):
    parser = argparse.ArgumentParser(description='Risk VQA standalone image GUI')
    parser.add_argument('--controller-url', default='http://127.0.0.1:11000')
    parser.add_argument('--model-name', default='')
    parser.add_argument(
        '--work-context',
        default='Robot manipulation and navigation in a shared workspace',
    )
    parser.add_argument('--horizon-s', type=int, default=10)
    parsed, _ = parser.parse_known_args(args if args is not None else sys.argv[1:])

    app = QApplication([sys.argv[0]])
    window = RiskVQAGUI(
        parsed.controller_url,
        parsed.model_name,
        parsed.work_context,
        parsed.horizon_s,
    )
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
