# Topic schema

目前使用 `std_msgs/msg/String` 承載 JSON，方便快速修改研究 schema。正式介面穩定後可轉成
自訂 ROS message。

## `/risk_vqa_sim/observation`

```json
{
  "schema_version": "1.0",
  "left": {
    "visibility": "CLEAR",
    "visibility_confidence": 0.95,
    "people": [
      {
        "person_id": "L1",
        "position": "APPROACH_ZONE",
        "motion": "APPROACHING",
        "intent": "ENTER",
        "entry_time": "0_TO_2_SECONDS",
        "confidence": 0.82,
        "evidence": "person becomes larger and moves toward the marked entrance"
      }
    ]
  },
  "right": {
    "visibility": "CLEAR",
    "visibility_confidence": 0.96,
    "people": []
  },
  "scene_summary": "One person approaches from the left.",
  "validation_errors": [],
  "metadata": {
    "model": "llava-v1.5-7b",
    "temporal_window_s": 2.0,
    "sample_count_per_camera": 3,
    "latest_frame_age_at_publish_s": 1.4,
    "pipeline_ms": 1380.5
  }
}
```

Enum：

- `visibility`：`CLEAR | OCCLUDED | OUT_OF_VIEW | UNCERTAIN`
- `position`：`OUTSIDE | APPROACH_ZONE | WAITING_ZONE | INSIDE_HAZARD_ZONE | UNKNOWN`
- `motion`：`APPROACHING | CROSSING | STATIONARY | LEAVING | PASSING_BY | UNKNOWN`
- `intent`：`ENTER | WAIT | PASS_BY | MOVE_AWAY | UNCERTAIN`
- `entry_time`：`NOW | 0_TO_2_SECONDS | 2_TO_5_SECONDS | MORE_THAN_5_SECONDS | NOT_APPROACHING | UNKNOWN`

若模型回傳未知 enum、缺少 camera 或非法 confidence，schema 層會產生 `validation_errors`
並用 `UNKNOWN/UNCERTAIN/0.0` 取代，讓 decision 層進入保守狀態。

## `/risk_vqa_sim/decision`

```json
{
  "state": "STOP",
  "command": "STOP",
  "reason_code": "entry_or_crossing_imminent",
  "explanation": "A person is inside, crossing, or expected to enter within two seconds.",
  "relevant_people": ["left:L1"],
  "clear_streak": 0,
  "decision_authority": "deterministic_rules",
  "observation_age_s": 0.0,
  "safety_boundary": "advisory_only; do not connect directly to safety-rated actuation"
}
```

`state=UNKNOWN` 時，`command` 固定是 `HOLD`。因此資料不足不會變成 `GO`。
