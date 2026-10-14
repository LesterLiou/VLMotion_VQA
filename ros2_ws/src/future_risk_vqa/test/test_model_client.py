from future_risk_vqa.model_client import LlavaWorkerClient


def test_model_client_retries_transient_worker_error(monkeypatch):
    client = LlavaWorkerClient(
        "http://controller",
        request_retry_count=2,
        request_retry_backoff_s=0.0,
    )
    attempts = []

    def generate_once(prompt, jpeg):
        attempts.append((prompt, jpeg))
        if len(attempts) < 2:
            raise RuntimeError("network busy")
        return "{}", {"model": "test"}

    monkeypatch.setattr(client, "_generate_once", generate_once)
    output, metadata = client.generate("prompt", b"jpeg")
    assert output == "{}"
    assert metadata["request_attempts"] == 2
    assert len(attempts) == 2
