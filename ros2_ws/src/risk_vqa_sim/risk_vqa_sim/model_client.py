"""HTTP client for the VLMotion controller and LLaVA model worker."""

import base64
import json
import time

import requests


class LlavaWorkerClient:
    """Resolve a worker through the controller and perform one-image VQA."""

    def __init__(
        self,
        controller_url,
        model_name='',
        request_timeout_s=60.0,
        max_new_tokens=512,
        use_cache=False,
    ):
        self.controller_url = str(controller_url).rstrip('/')
        self.model_name = str(model_name).strip()
        self.request_timeout_s = float(request_timeout_s)
        self.max_new_tokens = int(max_new_tokens)
        self.use_cache = bool(use_cache)

    def _resolve_model(self):
        if self.model_name:
            return self.model_name
        response = requests.post(
            self.controller_url + '/list_models', timeout=5.0
        )
        response.raise_for_status()
        models = response.json().get('models', [])
        if not models:
            raise RuntimeError('controller has no registered model workers')
        self.model_name = str(models[0])
        return self.model_name

    def _worker_address(self, model_name):
        response = requests.post(
            self.controller_url + '/get_worker_address',
            json={'model': model_name},
            timeout=5.0,
        )
        response.raise_for_status()
        address = str(response.json().get('address', '')).rstrip('/')
        if not address:
            raise RuntimeError(f'no worker is available for model {model_name}')
        return address

    def generate(self, prompt, jpeg):
        """Return completion text and timing/model metadata."""
        started = time.perf_counter()
        model_name = self._resolve_model()
        worker_address = self._worker_address(model_name)
        payload = {
            'model': model_name,
            'prompt': prompt,
            'temperature': 0.0,
            'top_p': 1.0,
            'max_new_tokens': self.max_new_tokens,
            'use_cache': self.use_cache,
            'stop': '###',
            'images': [base64.b64encode(jpeg).decode('ascii')],
            'task_type': 'risk_assessment',
            'use_sam3_candidates': False,
        }
        response = requests.post(
            worker_address + '/worker_generate_stream',
            headers={'User-Agent': 'risk_vqa_sim ROS 2 node'},
            json=payload,
            stream=True,
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()
        output = ''
        worker_latency = {}
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b'\0'):
            if not chunk:
                continue
            data = json.loads(chunk.decode('utf-8'))
            if data.get('error_code', 1) != 0:
                raise RuntimeError(data.get('text', 'model worker error'))
            text = str(data.get('text', ''))
            output = text[len(prompt):] if text.startswith(prompt) else text
            if isinstance(data.get('latency_ms'), dict):
                worker_latency = data['latency_ms']
        if not output.strip():
            raise RuntimeError('model worker returned an empty response')
        metadata = {
            'model': model_name,
            'worker_address': worker_address,
            'round_trip_ms': round((time.perf_counter() - started) * 1000.0, 3),
            'worker_latency_ms': worker_latency,
        }
        return output.strip(), metadata
