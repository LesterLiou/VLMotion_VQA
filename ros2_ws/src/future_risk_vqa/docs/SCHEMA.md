# Topic schemas

所有複合資料使用 `std_msgs/String` 承載 JSON，方便研究階段快速保存與修改 schema。
若介面穩定後，應改成自訂 ROS messages 並固定版本。person、geometry 與 VLM report 含
`schema_version`；decision 目前沒有同樣的頂層版本欄位，logger 額外保存 ROS receipt time。

## `/future_risk/person_tracks`

```json
{
  "schema_version": "1.0",
  "availability": "READY",
  "backend": "sam3",
  "cameras": {
    "left": {
      "stamp_s": 10.2,
      "frame_width": 960,
      "frame_height": 540,
      "tracks": [{
        "track_id": "L1",
        "bbox_xyxy": [10, 20, 100, 300],
        "footpoint_uv": [55, 300],
        "confidence": 0.91,
        "visible": true,
        "missed_frames": 0,
        "age_frames": 8
      }]
    }
  }
}
```

mask 不放入 JSON，避免 topic 尺寸暴增；mask 只在 SAM3 node 內做 matching 和 preview。

## `/future_risk/geometry_prediction`

`position`、`velocity`、`future_path` 與 `distance_to_hazard` 的單位由 `units` 決定。
只有 `calibration_mode=metric_homography` 時才是公尺。

目前 `zone_logic_enabled=false`，因此 `current_zone=DISABLED`、
`trajectory_intersects_hazard=false`、`time_to_entry_s=null`；這些欄位不會送入目前的
camera-relative current/future VQA prompt。主要使用`track_id`、camera、track age與tracking
reliability。

## `/future_risk/vlm_events`

```json
{
  "schema_version": "3.1",
  "availability": "READY",
  "assessments": [{
    "track_id": "L1",
    "camera": "left",
    "current_observation": {
      "body_orientation": "FACING_CAMERA",
      "body_posture": "STANDING",
      "current_activity": "WALKING",
      "motion_trend": "TOWARD_CAMERA",
      "vlm_motion_trend": "UNCERTAIN",
      "motion_source": "tracking_hint",
      "confidence": null,
      "evidence": ""
    },
    "future_prediction": {
      "future_motion": "TOWARD_CAMERA",
      "vlm_future_motion": "UNCERTAIN",
      "motion_source": "tracking_motion_persistence",
      "confidence": null,
      "evidence": ""
    },
    "confidence": 1.0,
    "evidence": ""
  }],
  "validation_errors": [],
  "metadata": {
    "prompt_contract": "compact_hybrid_posture_future_v3_2",
    "vqa_inference_mode": "joint",
    "motion_source": "tracking_hint",
    "storyboard_mode": "hybrid",
    "track_crop_coverage": {"L1": 3}
  }
}
```

`storyboard_mode=track_crops`時，`track_crop_coverage`表示每個ID在OLD/MID/NOW中實際找到
timestamp-matched bbox的格數。`track_crop_report_stamps_s`保存各crop使用的track report
timestamp，方便檢查輸入同步。多人超過`track_crop_max_agents`時mode會是
`global_fallback`，並附`track_crop_fallback_reason`。

為了讓`llava-v1.5-7b`較容易遵循格式，VLM raw response使用精簡flat contract：
`track_id`、`body_orientation`、`body_posture`、`current_activity`、
`future_motion`。不再要求模型輸出`future_activity`、confidence、`motion_trend`、evidence字串
或`scene_summary`；當前方向由timestamp對齊的
tracking hint補入。validator再建立`current_observation`與
`future_prediction`。系統不再建立workspace MAP或`future_event`；policy直接讀
`future_prediction.future_motion`。外層品質值再由crop coverage與tracking reliability調整。

`body_posture`與`current_activity`是正交欄位。姿勢 vocabulary 為`STANDING`、
`SQUATTING`、`SITTING`、`KNEELING`、`LYING_DOWN`、`BENDING_OR_CROUCHING`、
`UNCERTAIN`；活動欄位描述走路、等待、改變姿勢、工作或操作物品。舊版若把
`STANDING/SQUATTING/SITTING`放在`current_activity`，validator會遷移到`body_posture`並在
`normalizations`留下紀錄。

`motion_source=tracking_hint`時，validator後處理會把VLM原始方向保存為`vlm_motion_trend`及
`vlm_future_motion`，採用方向另附`motion_source`。外層`confidence`是提供policy的
coverage/reliability品質值，不是VLM自評機率。

per-track模式的`/future_risk/vlm_raw_response`是包裝後的JSON：

```json
{
  "inference_mode": "per_track",
  "responses": [
    {"track_id": "L1", "response": "{...raw model JSON...}"}
  ]
}
```

schema正確但偵測到答案坍縮時，top-level另有`quality_warnings`，受影響assessment的
`policy_eligible=false`、`confidence=0.0`。這和`validation_errors`分開，方便分別統計格式錯誤
與語意品質錯誤。

closed-vocabulary observable若為JSON `null`，會正規化成`UNCERTAIN`並寫入
`normalizations`，不列入`validation_errors`；未知的非null字串仍是schema error。兩個
confidence都必須是大於0的數值。完全沒有visible track時不呼叫VLM，而是輸出合法空
`assessments`。

若 model request 無法完成，node 會發布 synthetic fallback：`availability=UNAVAILABLE`、
`unavailable_reason=<原始錯誤>`、每筆 placeholder assessment 的 `confidence=null`。這種資料
不是模型 prediction，所以 UI 顯示 `N/A`；policy 會先判斷 availability，reason code 為
`future_behavior_unavailable`，而且 `validation_errors` 保持空陣列，不會誤標為
`future_behavior_schema_invalid`。

## `/future_risk/decision`

目前 `decision_mode=future_behavior_policy` 將 NEXT future motion 映射成
`STOP | DANGER_WARNING | CONTINUE | NO_RESPONSE` 提示，並回傳 `critical_tracks` 與
reason code。`NO_RESPONSE` 只表示目前沒有可見 person detection；sensor/VQA/schema
故障仍是 `DANGER_WARNING`，不會被誤當成空場景。
`observe_only` 才固定輸出 `OBSERVE`。這些都是研究提示；套件本身不直接驅動機器。
