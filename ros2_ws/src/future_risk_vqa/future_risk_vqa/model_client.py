"""HTTP client for the existing VLMotion LLaVA controller/worker service."""

import base64
import json
import time

import requests


class LlavaWorkerClient:
    """Resolve a worker through the controller and run one-image VQA."""

    def __init__(
        self,
        controller_url,
        model_name="",
        request_timeout_s=90.0,
        max_new_tokens=768,
        use_cache=False,
        request_retry_count=2,
        request_retry_backoff_s=1.0,
    ):
        self.controller_url = str(controller_url).rstrip("/")
        self.model_name = str(model_name).strip()
        self.request_timeout_s = float(request_timeout_s)
        self.max_new_tokens = int(max_new_tokens)
        self.use_cache = bool(use_cache)
        self.request_retry_count = max(0, int(request_retry_count))
        self.request_retry_backoff_s = max(0.0, float(request_retry_backoff_s))

    def _resolve_model(self):
        if self.model_name:
            return self.model_name
        response = requests.post(self.controller_url + "/list_models", timeout=5.0)
        response.raise_for_status()
        models = response.json().get("models", [])
        if not models:
            raise RuntimeError("controller has no registered model workers")
        self.model_name = str(models[0])
        return self.model_name

    def generate(self, prompt, jpeg, max_new_tokens=None):
        last_error = None
        for attempt in range(self.request_retry_count + 1):
            try:
                if max_new_tokens is None:
                    output, metadata = self._generate_once(prompt, jpeg)
                else:
                    output, metadata = self._generate_once(
                        prompt, jpeg, max_new_tokens=max_new_tokens
                    )
                metadata["request_attempts"] = attempt + 1
                return output, metadata
            except (requests.RequestException, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.request_retry_count:
                    break
                time.sleep(self.request_retry_backoff_s * (attempt + 1))
        raise RuntimeError(
            f"VQA request failed after {self.request_retry_count + 1} attempt(s): {last_error}"
        ) from last_error

    def _generate_once(self, prompt, jpeg, max_new_tokens=None):
        started = time.perf_counter()
        model = self._resolve_model()
        response = requests.post(
            self.controller_url + "/get_worker_address",
            json={"model": model},
            timeout=5.0,
        )
        response.raise_for_status()
        worker = str(response.json().get("address", "")).rstrip("/")
        if not worker:
            raise RuntimeError(f"no worker is available for model {model}")
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": (
                self.max_new_tokens
                if max_new_tokens is None
                else max(64, int(max_new_tokens))
            ),
            "use_cache": self.use_cache,
            "stop": "###",
            "images": [base64.b64encode(jpeg).decode("ascii")],
            # Existing VLMotion worker treats this as its machine-readable mode.
            # Other task names append a natural-language instruction after our JSON prompt.
            "task_type": "risk_assessment",
            "use_sam3_candidates": False,
        }
        response = requests.post(
            worker + "/worker_generate_stream",
            headers={"User-Agent": "future_risk_vqa ROS 2 node"},
            json=payload,
            stream=True,
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()
        output = ""
        worker_latency = {}
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if not chunk:
                continue
            data = json.loads(chunk.decode("utf-8"))
            if data.get("error_code", 1) != 0:
                raise RuntimeError(data.get("text", "model worker error"))
            text = str(data.get("text", ""))
            output = text[len(prompt) :] if text.startswith(prompt) else text
            if isinstance(data.get("latency_ms"), dict):
                worker_latency = data["latency_ms"]
        if not output.strip():
            raise RuntimeError("model worker returned an empty response")
        return output.strip(), {
            "model": model,
            "worker_address": worker,
            "round_trip_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "worker_latency_ms": worker_latency,
            "use_cache": self.use_cache,
        }
