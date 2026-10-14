# 功能開關、參數與調整指南

本頁於 2026-10-14 依兩份 source YAML 整併，取代重複的 `FUTURE_RISK_ISAAC_PARAMETERS.md`。完整實際值以 `config/future_risk_video_demo.yaml` 與 `config/future_risk_isaac.yaml` 為準；Python class 的預設值可能不同，啟動時以 YAML 加 launch 覆蓋後的結果生效。

## 修改與確認

影片修改 `future_risk_video_demo.yaml`；live／Isaac 修改 `future_risk_isaac.yaml`。大部分參數在節點啟動時讀取，修改後重啟，不要假設 `ros2 param set` 能觸發模型或排程重建。

```bash
# 容器內：直接讀 source，避免 install/share 的舊副本
ros2 launch future_risk_vqa future_risk_live_ui.launch.py \
  config:=/workspace/ros2_ws/src/future_risk_vqa/config/future_risk_isaac.yaml
ros2 param get /sam3_person_tracker device
ros2 param get /future_vqa storyboard_mode
ros2 param get /future_vqa vqa_inference_mode
ros2 param get /future_risk_policy decision_mode
```

使用安裝位置的預設設定時，重新建置 `colcon build --symlink-install --packages-select future_risk_vqa` 並 source workspace。host 路徑與容器 `/workspace` 路徑不可混用。

## Launch 覆蓋順序

- `future_risk_pipeline`：launch 覆蓋 tracker／VQA 的 image topics、compressed、backend；VQA 的 controller_url／model_name；四個核心節點的 use_sim_time。
- `live_ui`：預設 compressed topics，四個核心節點與 logger 使用 wall time；可關閉 pipeline／GUI／logger。
- `isaac`：核心節點使用 simulation time；相機／clock 由外部提供，logger 時間由 YAML 決定。
- `video_demo`：source 的影片路徑、播放速度、topics 由 YAML 提供；內含 pipeline 的 topic／compressed 預設為 raw `/left_camera/rgb`、`/right_camera/rgb`，若自行換 source topics 也要同步調整 launch 接線。
- 三個 demo/UI launch 會以 argument 覆蓋 logger 的 run_name／output_root；傳給 GUI 的參數另見 GUI guide。

支援 argument 用 `ros2 launch future_risk_vqa <launch檔名> --show-args` 查看。只改 YAML 裡 model_name，仍可能被 launch 的空字串覆蓋；啟動時應明確傳 `model_name:=...`。

## 調整時的依據

兩份設定目前都是 **CPU SAM3 + track_crops/per_track + tracking_hint**。CPU 首輪曾約 57 秒，未達成即時且模型格式未通過，詳見 `robopoint_runtime/SAM3_CPU_TRIAL.md`。頻率是目標排程，真正吞吐量需 benchmark。

- `storyboard_mode` 可用 `track_crops`、`hybrid`、`global`；`vqa_inference_mode` 可用 `per_track` 或 `joint`。逐 ID 請求隔離身分，但人數增加會增加總延遲；hybrid/joint 是另一組 ablation。
- `motion_source: tracking_hint` 使用 timestamped bbox 變化補入方向，原 VLM 答案保存於 audit 欄位。評估 VLM 本身時改用 `vlm` 並另存實驗結果。
- `trajectory_context_mode` 為 `none/overlay/numeric/both`；目前 `none`，避免把預測幾何當模型自行推理。
- `decision_mode` 為 `future_behavior_policy/observe_only/zone_safety_policy`。影片啟用 image-space 近場區 STOP，live／Isaac 關閉此區；兩者 metric zone/TTE 均關閉。`behavior_stop_motions: [DISABLED]` 只停用 motion→STOP 映射，不停用近場區 STOP。
- 去重先看 raw／deduplicated count，再調 mask IoU、bbox IoU／IoA，避免只拉高 confidence 而漏掉遠處人。四個 matching weight 建議合計 1。
- `track_history_tolerance_s` 應搭配實際追蹤延遲；找不到歷史時顯示 TRACK NOT FOUND，不借用別人的框。
- `enable_effective_confidence` 是輸入／追蹤品質診斷，不能解讀成預測準確率；目前 behavior policy 不用它改寫 motion action。
- timeout 需依 pipeline 實測評估，但把 timeout 拉長無法改善延遲。freshness 還需核對 capture／receipt time，詳見研究交接紀錄。

## 逐節點參數表

下表列出 source YAML 的值；`—` 表示該設定檔沒有此節點／參數。長 polygon 與 work_context 保留在 YAML，表內標示「見 YAML」。這些是設定檔快照，未表示各裝置實測可達到的效能。

### `sam3_person_tracker`

| 參數 | 影片 | Live／Isaac | 用途與調整 |
|---|---|---|---|
| `use_sim_time` | `false` | `true` | 是否使用 ROS /clock；launch 可能覆蓋。 |
| `enable_person_detection` | `true` | `true` | 是否執行人物偵測。`false` 會將 backend 強制切成 `disabled`。 |
| `enable_tracking` | `true` | `true` | 是否讓人物跨幀維持相同 ID。關閉後不適合分析人物歷史動作。 |
| `left_image_topic` | `/left_camera/rgb` | `/camera/left/color/image_raw` | 左影像來源，可能被 live launch 覆蓋。 |
| `right_image_topic` | `/right_camera/rgb` | `/camera/right/color/image_raw` | 右影像來源，可能被 live launch 覆蓋。 |
| `use_compressed_image` | `false` | `false` | `false` 使用 raw Image；`true` 使用 CompressedImage。 |
| `backend` | `sam3` | `sam3` | `sam3` 執行人物分割；`disabled` 停用偵測。 |
| `device` | `cpu` | `cpu` | SAM3 裝置；CPU 保留 GPU 顯存但延遲高，GPU 需先量測完整共用顯存。 |
| `model_id` | `facebook/sam3` | `facebook/sam3` | Hugging Face 模型 ID 或本機模型路徑。 |
| `local_files_only` | `true` | `true` | 只讀本機模型。模型尚未下載時需設 `false` 或先準備模型快取。 |
| `confidence_threshold` | `0.7` | `0.9` | 偵測信心門檻。提高可減少誤檢但增加漏檢；降低可找出更多遠處人物但可能產生假人。建議先試 `0.5–0.8`。 |
| `mask_threshold` | `0.5` | `0.5` | 二值 mask 門檻。提高會縮小 mask；降低會擴大 mask。常用 `0.4–0.6`。 |
| `max_people` | `12` | `6` | 每張影像最多保留的人數。降低可減少後續計算量。 |
| `suppress_duplicate_detections` | `true` | `true` | 是否移除同一人物的重複偵測。通常保持開啟。 |
| `duplicate_mask_iou_threshold` | `0.65` | `0.65` | mask IoU 超過此值時視為重複。降低會更積極合併。 |
| `duplicate_bbox_iou_threshold` | `0.7` | `0.7` | bounding box IoU 重複門檻。降低會更積極合併。 |
| `duplicate_bbox_ioa_threshold` | `0.85` | `0.85` | 小框被大框涵蓋的比例門檻。降低會更容易刪除被包含的偵測。 |
| `inference_hz` | `2.0` | `2.0` | 目標推論頻率，不代表實際吞吐量。 |
| `max_missed_frames` | `5` | `5` | 人物沒被偵測後保留 ID 的推論幀數。提高可承受遮擋，但舊 ID 殘留更久。 |
| `minimum_match_score` | `0.2` | `0.2` | 新偵測和舊 track 配對的最低分數。提高能減少錯接，但 ID 更容易斷線。 |
| `match_mask_iou_weight` | `0.15` | `0.15` | 配對分數中 mask 重疊的權重。 |
| `match_bbox_iou_weight` | `0.2` | `0.2` | 配對分數中 bounding box 重疊的權重。 |
| `match_foot_distance_weight` | `0.5` | `0.5` | 腳底位置距離的權重，目前是主要配對依據。 |
| `match_size_similarity_weight` | `0.15` | `0.15` | 人物框尺寸相似度的權重。 |
| `match_max_foot_distance_ratio` | `0.2` | `0.2` | 跨幀腳點允許移動的最大影像比例。快速移動、低 inference rate 時可提高；人物常錯接時降低。 |
| `match_velocity_smoothing` | `0.6` | `0.6` | 速度平滑程度。提高較穩但反應慢；降低較敏感但容易抖動。 |
| `enable_walkway_detection` | `true` | `true` | 低頻 floor 分割供顯示；不視為安全可通行區。 |
| `walkway_prompt` | `floor` | `floor` | floor 分割文字 prompt。 |
| `walkway_confidence_threshold` | `0.35` | `0.35` | floor proposal 的信心門檻。 |
| `walkway_inference_interval_s` | `5.0` | `5.0` | floor 推論最短間隔。 |
| `show_walkway_overlay` | `true` | `false` | 顯示 floor 候選遮罩。 |
| `enable_approximate_stop_zone` | `true` | `false` | 啟用影像近場區提示；不代表已標定公尺距離。 |
| `approximate_stop_foot_overlap_ratio` | `0.05` | `0.05` | 腳部 mask 與近場區的重疊門檻。 |
| `left_approximate_stop_polygon_normalized` | `[0.2, 1.0, 0.8, 1.0, 0.635, 0.76, 0.365, 0.76]` | `[0.2, 1.0, 0.8, 1.0, 0.635, 0.76, 0.365, 0.76]` | 左相機近場 polygon，影像正規化座標。 |
| `right_approximate_stop_polygon_normalized` | `[0.2, 1.0, 0.8, 1.0, 0.635, 0.76, 0.365, 0.76]` | `[0.2, 1.0, 0.8, 1.0, 0.635, 0.76, 0.365, 0.76]` | 右相機近場 polygon，影像正規化座標。 |

### `ground_trajectory`

| 參數 | 影片 | Live／Isaac | 用途與調整 |
|---|---|---|---|
| `use_sim_time` | `false` | `true` | 是否使用 ROS /clock；launch 可能覆蓋。 |
| `enable_motion_prediction` | `true` | `true` | 是否輸出速度和 future path。 |
| `enable_zone_logic` | `false` | `false` | 是否計算 hazard/warning zone、交會與 TTE。標定完成前必須保持 `false`。 |
| `show_track_id_overlay` | `true` | `true` | 是否要求視覺化 track ID。 |
| `show_prediction_overlay` | `false` | `false` | 是否繪出預測軌跡。適合除錯，但可能遮擋影像或干擾 VLM。 |
| `calibration_mode` | `normalized_debug` | `metric_homography` | `metric_homography` 預期公尺但須有效標定；`normalized_debug` 僅供非公尺除錯。 |
| `calibration_valid` | `true` | `false` | 影片的 true 只允許 normalized_debug 計算；metric 模式必須完成實際標定才可設 true。 |
| `left_homography` | `[0.0010416667, 0.0, 0.0, 0.0, -0.0018518519, 1.0, 0.0, 0.0, 1.0]` | `[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]` | 左相機像素 `(u,v)` 到共同地面 `(x,y)` 的 3×3 row-major 矩陣。identity 只是 placeholder。 |
| `right_homography` | `[0.0010416667, 0.0, 0.0, 0.0, -0.0018518519, 1.0, 0.0, 0.0, 1.0]` | `[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]` | 右相機 homography。必須輸出到與左相機相同的世界座標系。 |
| `hazard_polygon` | `[0.35, 0.0, 0.65, 0.0, 0.65, 0.28, 0.35, 0.28]` | `[-0.5, 0.0, 0.5, 0.0, 0.5, 1.0, -0.5, 1.0]` | 危險區域，格式為 `[x1,y1,x2,y2,...]`。metric 模式單位為公尺。 |
| `warning_polygon` | `[0.2, 0.0, 0.8, 0.0, 0.8, 0.48, 0.2, 0.48]` | `[-1.0, -0.5, 1.0, -0.5, 1.0, 1.5, -1.0, 1.5]` | 警告區域，通常應完整包含 hazard polygon。 |
| `prediction_horizon_s` | `3.0` | `3.0` | 預測未來幾秒。提高能提早預警，但遠期結果較不可靠。常用 `2–5 s`。 |
| `prediction_step_s` | `0.2` | `0.2` | 路徑取樣間隔。降低可提高 TTE 時間解析度，但增加資料量。 |
| `process_noise` | `0.5` | `0.5` | 運動模型允許突然轉向/加速的程度。提高反應更快但軌跡較抖。 |
| `measurement_noise` | `0.08` | `0.08` | 對偵測腳點雜訊的估計。提高會更平滑但延遲更大。 |
| `minimum_track_age_frames` | `2` | `2` | track 至少存在幾幀才標為可靠。提高會更保守。 |
| `max_track_gap_s` | `1.5` | `1.5` | 人物多久不可見後移除 filter。提高可承受遮擋，但殘留更久。 |

### `future_vqa`

| 參數 | 影片 | Live／Isaac | 用途與調整 |
|---|---|---|---|
| `use_sim_time` | `false` | `true` | 是否使用 ROS /clock；launch 可能覆蓋。 |
| `enable_future_vqa` | `true` | `true` | 是否執行 VQA。`false` 會將 backend 強制切成 `disabled`。 |
| `include_tracking_context` | `true` | `true` | prompt 是否包含人物 ID、位置與追蹤資訊。多人場景建議開啟。 |
| `show_sam3_detection_overlay` | `true` | `true` | storyboard 是否畫 SAM3 框、mask 與 ID。overlay 太多也可能干擾模型。 |
| `trajectory_context_mode` | `none` | `none` | 可選 `none`、`overlay`、`numeric`、`both`。標定無效時建議 `none`。 |
| `storyboard_mode` | `track_crops` | `hybrid` | `global` 為全景；`track_crops` 為人物 crop；`hybrid` 同時提供全景 NOW 和各 ID 的時間 crop。 |
| `vqa_inference_mode` | `per_track` | `joint` | `joint` 一次分析所有人；`per_track` 每個 ID 呼叫一次。`per_track` 必須搭配 `storyboard_mode: track_crops`。 |
| `per_track_max_new_tokens` | `128` | `128` | 每人獨立推論的最大 token，只在 `per_track` 生效，程式最低為 128。 |
| `track_history_tolerance_s` | `0.75` | `0.75` | 歷史影像與 track 記錄容許的時間差。提高較容易找到歷史框，但位置可能不準。 |
| `track_crop_padding_ratio` | `0.3` | `0.3` | 人物框周圍擴張比例。提高能保留環境，但人物會變小。建議 `0.15–0.4`。 |
| `track_crop_cell_size` | `224` | `224` | 每個人物 crop cell 的像素大小，最低為 96。提高細節也會增加資料量。 |
| `track_crop_max_agents` | `6` | `6` | 一次最多放入的人數。提高會增加 prompt 複雜度及答案混淆風險。 |
| `motion_source` | `tracking_hint` | `tracking_hint` | `tracking_hint` 用框的歷史位移提供運動提示；`vlm` 完全由模型自行判斷。 |
| `motion_scale_change_threshold` | `0.12` | `0.12` | 框尺度變化超過 12% 才明確判定接近或遠離。降低較敏感但容易受框抖動影響。 |
| `motion_lateral_displacement_threshold` | `0.35` | `0.35` | 正規化橫向位移超過門檻才判定明顯橫移。降低可提早辨識 crossing/passing。 |
| `motion_stationary_scale_threshold` | `0.04` | `0.04` | 尺度變化低於此值時可視為靜止。提高會把更多慢速移動判成靜止。 |
| `motion_stationary_lateral_threshold` | `0.12` | `0.12` | 橫向位移低於此值時可視為靜止。 |
| `enable_collapse_detection` | `false` | `false` | 偵測模型是否對多個人物產生完全相同答案。建議保持開啟。 |
| `collapse_min_identical_tracks` | `3` | `3` | 至少幾個 ID 答案相同才判定 collapse。降低到 2 更敏感，也更容易誤報。 |
| `enable_effective_confidence` | `true` | `true` | 產生品質診斷；不是校正過的預測機率。 |
| `unreliable_tracking_confidence_scale` | `0.5` | `0.5` | tracking 不可靠時 confidence 的乘數。越低越保守，程式限制為 `0–1`。 |
| `left_image_topic` | `/left_camera/rgb` | `/left_camera/rgb` | 左影像來源，可能被 live launch 覆蓋。 |
| `right_image_topic` | `/right_camera/rgb` | `/right_camera/rgb` | 右影像來源，可能被 live launch 覆蓋。 |
| `use_compressed_image` | `false` | `false` | `false` 使用 raw Image；`true` 使用 CompressedImage。 |
| `backend` | `llava_worker` | `llava_worker` | `llava_worker` 呼叫模型；`disabled` 只執行資料流程。 |
| `controller_url` | `http://127.0.0.1:11000` | `http://127.0.0.1:11000` | LLaVA controller 位址。注意容器內的 `127.0.0.1` 只代表該容器。 |
| `model_name` | `''` | `''` | worker 模型名稱。多模型時應明確指定；空字串由 controller/worker 選擇。 |
| `request_timeout_s` | `90.0` | `90.0` | 單次請求逾時。應設為正常推論時間的約 `1.5–2` 倍。 |
| `request_retry_count` | `2` | `2` | 失敗後重試次數。提高容錯，也會增加最壞等待時間。 |
| `request_retry_backoff_s` | `1.0` | `1.0` | 每次重試前等待秒數。 |
| `max_new_tokens` | `512` | `512` | joint 推論最大輸出 token。JSON 被截斷時提高；延遲太高時降低。 |
| `use_cache` | `false` | `false` | 是否使用快取。Live pipeline 通常保持關閉。 |
| `vqa_hz` | `0.5` | `0.5` | 每秒 VQA 次數。`0.5` 表示每 2 秒一次；提高會增加 VLM 負載。 |
| `temporal_frame_count` | `3` | `3` | 每台相機使用的時間影格數。提高可提供更完整歷史，但 storyboard 更複雜。 |
| `temporal_window_s` | `2.0` | `2.0` | 時間影格涵蓋的總秒數。慢動作可提高，快速動作可降低。 |
| `jpeg_quality` | `90` | `90` | JPEG 品質，通常使用 `80–95`。降低可減少資料量，但可能損失小物件細節。 |
| `work_context` | 見 YAML | 見 YAML | 加入 prompt 的工作場景說明。應描述可從畫面判斷的任務、工作區與風險。 |

### `future_risk_policy`

| 參數 | 影片 | Live／Isaac | 用途與調整 |
|---|---|---|---|
| `use_sim_time` | `false` | `true` | 是否使用 ROS /clock；launch 可能覆蓋。 |
| `decision_mode` | `future_behavior_policy` | `future_behavior_policy` | 可選 `future_behavior_policy`、`zone_safety_policy`、`observe_only`。未完成地面標定前不要使用 zone safety。 |
| `enable_action_output` | `true` | `true` | 是否啟用動作決策。研究與驗證初期建議設 `false`。 |
| `behavior_stop_motions` | `[DISABLED]` | `[DISABLED]` | motion→STOP 的映射；DISABLED 不停用 image-space 近場 STOP。 |
| `behavior_warning_motions` | `[TOWARD_CAMERA, UNCERTAIN]` | `[TOWARD_CAMERA, UNCERTAIN]` | 哪些NEXT motion觸發警告。 |
| `behavior_continue_motions` | `[AWAY_FROM_CAMERA, LATERAL_ACROSS_VIEW, STATIONARY]` | `[AWAY_FROM_CAMERA, LATERAL_ACROSS_VIEW, STATIONARY]` | 哪些NEXT motion可以允許CONTINUE。 |
| `behavior_continue_confirmations_required` | `1` | `1` | motion模式安全報告確認次數；目前單次安全即CONTINUE。 |
| `stop_tte_s` | `1.0` | `1.0` | 預測進入 hazard zone 的剩餘時間小於此值時 STOP。提高會更早停。 |
| `warning_tte_s` | `3.0` | `3.0` | TTE 小於此值時警告，必須大於 `stop_tte_s`。 |
| `semantic_confidence_threshold` | `0.55` | `0.55` | zone policy 接受 VLM 語意的最低 confidence。 |
| `require_vlm_for_continue` | `true` | `true` | 是否必須同時得到 VLM 確認才允許 CONTINUE。`true` 比較保守。 |
| `continue_confirmations_required` | `2` | `2` | 參照同名節點參數；改動後重啟並確認實際值。 |
| `geometry_timeout_s` | `2.0` | `2.0` | tracking/geometry 報告超過多久視為過期。應高於 tracker 正常週期。 |
| `semantics_timeout_s` | `20.0` | `20.0` | VQA 報告超過多久視為過期。太大會沿用舊判斷；太小會頻繁判定 unavailable。 |
| `heartbeat_hz` | `5.0` | `5.0` | 每秒發布幾次 policy 狀態。不會改變 SAM3 或 VQA 推論速度。 |

### `future_risk_video_source`

| 參數 | 影片 | Live／Isaac | 用途與調整 |
|---|---|---|---|
| `use_sim_time` | `false` | — | 是否使用 ROS /clock；launch 可能覆蓋。 |
| `left_video_path` | `/workspace/example/video/0610_take3n4.mp4` | — | 本地左影片絕對路徑；影片不納入 Git。 |
| `right_video_path` | `/workspace/example/video/0610x9_v0629.mp4` | — | 本地右影片絕對路徑。 |
| `left_image_topic` | `/left_camera/rgb` | — | 左影像來源，可能被 live launch 覆蓋。 |
| `right_image_topic` | `/right_camera/rgb` | — | 右影像來源，可能被 live launch 覆蓋。 |
| `use_compressed_image` | `false` | — | `false` 使用 raw Image；`true` 使用 CompressedImage。 |
| `loop_videos` | `true` | — | 是否循環播放。 |
| `playback_rate` | `1.0` | — | 影片時間縮放比。 |
| `output_fps` | `10.0` | — | 來源輸出頻率上限。 |
| `output_width` | `960` | — | 來源影像寬度。 |
| `output_height` | `540` | — | 來源影像高度。 |
| `left_start_offset_s` | `0.0` | — | 左影片起始偏移秒數。 |
| `right_start_offset_s` | `0.0` | — | 右影片起始偏移秒數。 |

### `future_risk_experiment_logger`

| 參數 | 影片 | Live／Isaac | 用途與調整 |
|---|---|---|---|
| `use_sim_time` | `false` | `false` | 是否使用 ROS /clock；launch 可能覆蓋。 |
| `output_root` | `/workspace/results/future_risk_vqa` | `/workspace/results/future_risk_vqa` | 實驗產物根目錄；demo launch argument 可能覆蓋。 |
| `run_name` | `video_demo` | `isaac_sim` | 實驗名稱；每次測試設唯一名稱。 |
| `save_storyboards` | `true` | `true` | 是否保存 VQA 輸入 storyboard。 |

## 延伸閱讀

[架構](../../../../docs/ARCHITECTURE.md)、[使用方式](../../../../docs/USAGE.md)、[GUI](GUI_GUIDE.md)、[校正](CALIBRATION.md)、[研究限制](RESEARCH_CONTINUATION.md)。
