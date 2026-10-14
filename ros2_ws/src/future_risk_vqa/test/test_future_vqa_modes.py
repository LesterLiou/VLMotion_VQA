import json
import re
from types import SimpleNamespace

import numpy as np

from future_risk_vqa.future_vqa_node import FutureVQANode


class FakeClient:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, jpeg, max_new_tokens=None):
        self.calls.append(
            {"prompt": prompt, "jpeg": jpeg, "max_new_tokens": max_new_tokens}
        )
        match = re.search(r"Current track IDs are exactly: \[\"([^\"]+)\"\]", prompt)
        track_id = match.group(1) if match else "L1"
        response = {
            "scene_visibility": "CLEAR",
            "assessments": [
                {
                    "track_id": track_id,
                    "body_orientation": "SIDE_ON",
                    "current_activity": "WALKING",
                    "motion_trend": "LATERAL_ACROSS_VIEW",
                    "observation_confidence": 0.8,
                    "observation_evidence": f"{track_id} changes position",
                    "future_activity": "WALKING",
                    "future_motion": "LATERAL_ACROSS_VIEW",
                    "prediction_confidence": 0.7,
                    "prediction_evidence": f"{track_id} continues moving",
                }
            ],
            "scene_summary": f"Assessment for {track_id}",
        }
        return json.dumps(response), {
            "model": "fake-vlm",
            "round_trip_ms": 1.0,
        }


def make_node():
    node = FutureVQANode.__new__(FutureVQANode)
    node.client = FakeClient()
    node.work_context = "Test current and future behavior."
    node.include_tracking_context = True
    node.trajectory_context_mode = "none"
    node.motion_source = "tracking_hint"
    node.track_crop_padding_ratio = 0.3
    node.track_crop_cell_size = 128
    node.per_track_max_new_tokens = 256
    node.quality = 90
    return node


def test_per_track_mode_sends_one_image_request_per_id_and_aggregates():
    node = make_node()
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    frames = [
        SimpleNamespace(stamp_s=stamp, image=image.copy())
        for stamp in (1.0, 2.0, 3.0)
    ]
    agents = [
        {"track_id": "L1", "camera": "left", "visible": True},
        {"track_id": "R1", "camera": "right", "visible": True},
    ]
    geometry = {"tracked_agents": agents, "zone_logic_enabled": False}
    samples = {
        track_id: [
            {"bbox_xyxy": (20.0, 10.0, 55.0, 72.0), "report_stamp_s": stamp}
            for stamp in (1.0, 2.0, 3.0)
        ]
        for track_id in ("L1", "R1")
    }
    hints = {
        track_id: {
            "motion_trend": "LATERAL_ACROSS_VIEW",
            "reliable": True,
            "confidence": 0.8,
        }
        for track_id in ("L1", "R1")
    }

    report, raw, metadata = node._infer_per_track(
        frames, frames, geometry, samples, hints
    )

    assert len(node.client.calls) == 2
    assert all(call["max_new_tokens"] == 256 for call in node.client.calls)
    assert [item["track_id"] for item in report["assessments"]] == ["L1", "R1"]
    assert json.loads(raw)["inference_mode"] == "per_track"
    assert metadata["request_count"] == 2


def test_joint_mode_preserves_one_request_for_all_ids():
    node = make_node()
    # The fake response contains one assessment, so validation will report the
    # second ID missing; this test only verifies the preserved request granularity.
    geometry = {
        "zone_logic_enabled": False,
        "tracked_agents": [
            {"track_id": "L1", "camera": "left", "visible": True},
            {"track_id": "R1", "camera": "right", "visible": True},
        ],
    }
    node._infer_joint(geometry, b"jpeg", "track_crops", {})
    assert len(node.client.calls) == 1
    assert node.client.calls[0]["max_new_tokens"] is None
