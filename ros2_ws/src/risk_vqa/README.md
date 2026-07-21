# risk_vqa

`risk_vqa` 是一個 ROS 2 視覺風險分析 package，目前提供兩種用途：

| 模式 | 輸入 | 主要功能 |
|---|---|---|
| 影片雙層模式（推薦） | RGB 影片 | YOLO 人員偵測、ByteTrack、optical TTC、LLaVA 語意 VQA、監控 GUI |
| 圖片／相機 VQA | 單張圖片或 ROS image topic | 找出工作影響因子、估計 `time_to_impact_s`、提供 action 建議 |

介面與模型回答使用英文，避免執行環境缺少中文字型時顯示方框。

> Safety boundary：這是研究原型。所有 ROS topics 都不會直接控制機器人；VLM 與
> fusion 結果只作 advisory。它不能取代 safety-rated sensor、emergency stop 或正式
> safety controller。

## 1. 快速開始：影片 GUI + 即時層 + VLM

這是目前最完整的執行方式。系統會在每個影片影格執行 YOLO/TTC，並每隔 2 秒把
最新影格交給 LLaVA 做 person-semantic VQA。

### 1.1 Build

第一次使用或修改程式後執行：

```bash
cd /workspace/ros2_ws
colcon build --packages-select vlpoint risk_vqa --symlink-install
source install/setup.bash
```

每個新 terminal 都要執行：

```bash
cd /workspace/ros2_ws
source install/setup.bash
```

### 1.2 Terminal 1：Controller

```bash
ros2 launch vlpoint controller.launch.py
```

Controller 位於 `http://127.0.0.1:11000`，負責記錄 model worker 並把 VQA request
導向可用 worker；它本身不執行模型推論。

### 1.3 Terminal 2：LLaVA Model Worker

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ros2 launch vlpoint worker.launch.py \
  model_path:=liuhaotian/llava-v1.5-7b \
  quantization:=4bit \
  mm_use_sam3_conditioning:=false
```

這兩個 offline 環境變數強制使用 `/root/.cache/huggingface`，避免每次啟動重新下載。
如果 cache 不完整，worker 會直接回報缺少檔案。

`quantization` 可使用：

| 值 | Worker flag | 說明 |
|---|---|---|
| `4bit` | `--load-4bit` | 預設，最省 VRAM |
| `8bit` | `--load-8bit` | 較高精度，但使用更多 VRAM |
| `none` | 無 quantization flag | 以 FP16 載入，13B 需要大量 VRAM |

例如切換為無量化：

```bash
ros2 launch vlpoint worker.launch.py \
  model_path:=liuhaotian/llava-v1.5-7b \
  quantization:=none
```

#### Model 怎麼選

`model_path` 決定 worker 實際載入哪個 Hugging Face checkpoint；影片 GUI 的
`model_name` 則決定 request 要送到 controller 已註冊的哪個 worker。兩者用途不同：

| 設定 | 建議用途 | 取捨 |
|---|---|---|
| `liuhaotian/llava-v1.5-7b` + 4-bit | 12 GB GPU、影片實驗、latency benchmark | 較快且較不容易 OOM，建議先用 |
| `liuhaotian/llava-v1.5-13b` + 4-bit | GPU 記憶體足夠、重視語意回答品質 | 通常較慢且更容易 OOM |
| 其他 LLaVA-compatible checkpoint | 模型比較實驗 | 必須與目前 worker loader、conversation template 相容 |

如果 controller 只有一個 worker，GUI 的 `model_name` 保持空字串即可，程式會使用
`/list_models` 回傳的第一個模型。若有多個 worker，先查詢模型名稱：

確認 worker 已完成註冊：

```bash
curl -s -X POST http://127.0.0.1:11000/list_models
```

輸出例如：

```json
{"models":["llava-v1.5-7b"]}
```

接著把完全相同的名稱傳給影片 launch：

```bash
model_name:=llava-v1.5-7b
```

不要在 `model_name` 填入尚未出現在 `/list_models` 的名稱，否則 controller 找不到
對應 worker。切換 checkpoint 時必須停止舊 worker，再用新的 `model_path` 啟動；模型
已完整存在 Hugging Face cache 時不需要重新下載。

### 1.4 Terminal 3：影片監控 GUI

```bash
ros2 launch risk_vqa risk_video_gui_demo.launch.py \
  video_path:=/workspace/example/video/0610x9_v0629.mp4 \
  loop_video:=true \
  yolo_model:=/workspace/yolov8n.pt \
  yolo_device:=0 \
  enable_vlm:=true \
  evaluation_period_s:=2.0 \
  semantic_response_mode:=compact \
  max_new_tokens:=128 \
  use_cache:=true
```

`loop_video:=true` 會在影片播完後回到第一幀，並重設 frame index 與 tracking state，
適合長時間展示或收集 latency 樣本。只播放一次時改成 `loop_video:=false`（預設值）。

只啟動影片即時層、循環播放而不執行 VLM，可使用：

```bash
ros2 launch risk_vqa risk_video_gui_demo.launch.py \
  video_path:=/workspace/example/video/0610x9_v0629.mp4 \
  loop_video:=true \
  yolo_model:=/workspace/yolov8n.pt \
  yolo_device:=0 \
  enable_vlm:=false
```

需要原本包含整體 summary、人物描述與多個 factors 的長回答時，改用 detailed launch：

```bash
ros2 launch risk_vqa risk_video_gui_detailed_demo.launch.py \
  video_path:=/workspace/example/video/0610x9_v0629.mp4 \
  yolo_model:=/workspace/yolov8n.pt \
  yolo_device:=0 \
  enable_vlm:=true \
  evaluation_period_s:=2.0
```

這個 launch 固定 `semantic_response_mode:=detailed`，並將預設輸出上限保留為
`max_new_tokens:=384`。也可繼續使用一般 GUI launch，手動指定這兩個參數。

注意：這是在 Docker 容器內執行，因此要使用 `/workspace/...`，不要使用主機路徑
`/home/lester/VLMotion/...`。路徑包含空白時，必須把完整 launch argument 加上引號：

```bash
"video_path:=/workspace/example/Synthetic Videos/demo.mp4"
```

## 2. 影片 GUI

GUI 左側顯示標記後的影片，右側顯示兩層判斷結果：

- `Pause`：停止讀取下一個影片影格。
- `Continue`：恢復播放、YOLO tracking、TTC 與後續 VLM 分析。
- `Real-time Action`：即時層的 `CONTINUE`、`SLOW_DOWN` 或 `STOP`。
- `VLM Advisory`：LLaVA 根據人員活動提出的語意建議。
- `Fused Advisory`：即時 action 與 VLM action 的融合顯示。
- `VLM Answer`：VLM 英文自然語言回答。
- `VLM JSON`：驗證後的結構化 VLM report。
- `Fusion`：兩層 action、資料新鮮度與 control authority。

Pause 透過 `/risk_vqa/set_paused` service 真正停止影片讀幀，而不是只凍結 GUI。
沒有新影格時，VLM 不會重複分析同一張舊圖；已經送出的 request 仍可能完成並顯示。

關閉 GUI 會關閉同一個 video demo launch 中的影片、VQA 與 fusion nodes。

## 3. 影片雙層架構

即時層不應稱為「即時 VQA」。它是 deterministic person perception and risk
estimation；只有 LLaVA 路徑是 VQA。

```text
RGB video frame
  |
  +--> Real-time path (every frame)
  |      YOLO person detection
  |        -> ByteTrack
  |        -> bbox scale growth
  |        -> optical TTC
  |        -> ROI safety rules
  |        -> CONTINUE / SLOW_DOWN / STOP
  |
  +--> Semantic path (default: latest frame every 2 s)
         LLaVA person-semantic VQA
           -> activity
           -> impact on work
           -> severity / confidence
           -> advisory action

Real-time action + VLM advisory
  -> risk_fusion_node
  -> fused advisory (no robot control authority)
```

| 元件 | 執行頻率 | 工作 |
|---|---:|---|
| `video_ttc_node` | 每個影格 | 影片解碼、YOLO person、ByteTrack、TTC、規則與 overlay |
| `risk_vqa_node` | 預設每 2 秒 | 將最新未分析影格送到 LLaVA |
| `risk_fusion_node` | action 更新時 | 合併 real-time action 與仍在 timeout 內的 VLM advisory |
| `risk_video_gui` | topic 更新時 | 顯示影片、兩層結果並控制暫停／繼續 |
| Controller | 常駐 | 管理 worker address |
| Model worker | 每次 VQA request | 執行 LLaVA inference |

### 3.1 兩層的責任

| 問題 | YOLO/TTC 即時層 | LLaVA VQA 語意層 |
|---|---|---|
| 畫面是否有人 | 主要負責 | 語意確認 |
| 人的位置 | Bounding box / ROI | 禁止輸出座標 |
| 前後影格是否同一人 | ByteTrack ID | 不負責 |
| 人是否正在接近 | Bbox optical expansion | 不從單張圖判斷速度 |
| 約多久可能接觸 | Optical TTC 粗估 | 禁止估計 TTC |
| 人正在做什麼 | 不理解 | 主要負責 |
| 人如何影響工作 | 只依 ROI/TTC | 主要負責 |
| 更新頻率 | 每個影格 | 預設每 2 秒且不重疊 request |

簡單說：

```text
YOLO/TTC = fast: where is the person and are they approaching?
LLaVA    = slow: what is the person doing and why does it matter?
```

## 4. 即時層：YOLO、Track ID、TTC 與框色

### 4.1 Person detection 與 tracking

每個 RGB frame 會執行：

```text
YOLOv8n
  -> keep COCO class 0 (person) only
  -> confidence >= 0.35
  -> NMS IoU threshold 0.50
  -> ByteTrack with persistent state
  -> person bbox + confidence + temporary Track ID
```

Track ID 只是 ByteTrack 在連續影格中的短期識別，不是永久身分。遮擋、人物交錯或
快速移動時 ID 仍可能改變。

### 4.2 Pure-RGB optical TTC

系統沒有 depth，因此不計算公尺距離。它利用「人接近相機時 bbox 通常會變大」估計
optical time-to-contact：

```text
area        = bbox_width * bbox_height
scale       = sqrt(area)
growth_rate = d(log(scale)) / dt
TTC         = 1 / growth_rate
```

成長率會使用 exponential moving average 降低 bbox 抖動。Track 至少存在 3 個影格，
且 smoothed growth rate 大於 `0.04/s`，才會被視為正在接近。如果沒有足夠的 expansion
證據，TTC 顯示 `--`；這不代表絕對安全。

### 4.3 Safety ROI 與顏色

只有 bbox 中心位於 safety ROI，且 bbox 面積至少占畫面 `0.2%` 時，TTC 才會觸發
減速或停止規則。

| 顏色 | Action | 預設條件 |
|---|---|---|
| Green | `CONTINUE` | 不在 ROI、沒有確認接近，或 TTC 大於 3 秒 |
| Yellow | `SLOW_DOWN` | 在 ROI 且 `1 < TTC <= 3` 秒 |
| Red | `STOP` | 在 ROI 且 `TTC <= 1` 秒 |
| Red | `STOP` | 已確認的 ROI 內人員 bbox 占整張畫面至少 20% |
| Blue/Cyan | Safety ROI | 判斷區域，不是 detection |

顏色由 deterministic TTC 規則決定，不受 VLM 回答影響。

## 5. VLM 語意 VQA 與融合

影片模式有兩種可切換的 `person_semantic` response profile：

| Profile | 預設 tokens | 輸出內容 | 用途 |
|---|---:|---|---|
| `compact`（預設） | 128 | 最高優先的一個 risk、短詞組 | 降低生成延遲、即時展示 |
| `detailed` | 384 | summary、人物描述、多個 factors | 分析、報告與除錯 |

### 5.1 我要選哪一種 prompt

- 要跑影片、比較 latency、在 GUI 持續觀察：使用 `compact`。
- 要了解每個人在做什麼、保留 overall summary、分析多個人物：使用 `detailed`。
- 要測即時安全反應：不要依賴任一 VLM prompt；使用 YOLO/TTC 的
  `/risk_vqa/realtime_action`。
- 要估計幾秒後影響工作：由多影格 tracking/TTC 處理。單張 VLM prompt 不應估秒數。

同一個 GUI launch 可直接切換：

```bash
# 精簡、低延遲
semantic_response_mode:=compact max_new_tokens:=128

# 長句 summary + person + multiple factors
semantic_response_mode:=detailed max_new_tokens:=384
```

也可以直接使用 `risk_video_gui_detailed_demo.launch.py`，它會固定選擇 detailed。

### 5.2 Prompt 在哪裡設計

Prompt 實作位於：

- `risk_vqa/person_semantic.py::_common_prompt`：兩種模式共用的 evidence、confidence
  與 action 規則。
- `risk_vqa/person_semantic.py::_build_compact_prompt`：單一最高優先風險的短 JSON。
- `risk_vqa/person_semantic.py::_build_detailed_prompt`：summary、person 與多 factors。
- `risk_vqa/risk_vqa_node.py::_build_request`：依 `semantic_response_mode` 選 prompt，
  再組成送給 worker 的 request。

通常不需要修改 Python；可直接從 launch 覆寫問題與工作情境：

```bash
ros2 launch risk_vqa risk_video_gui_demo.launch.py \
  video_path:=/workspace/example/video/0610x9_v0629.mp4 \
  yolo_model:=/workspace/yolov8n.pt \
  enable_vlm:=true \
  model_name:=llava-v1.5-7b \
  semantic_response_mode:=compact \
  "work_context:=A mobile robot is travelling through a convenience-store aisle while stocking products on shelves." \
  "semantic_question:=Identify the most important visible human activity that could interfere with the current shelf-stocking task." \
  max_new_tokens:=128
```

`work_context` 應描述 robot 當下正在做的具體工作、工作區域及什麼算干擾；
`semantic_question` 應只詢問畫面可觀察的活動與 task impact。避免要求模型從單張 RGB
圖片猜測意圖、速度、距離或 TTC。

推薦的設計原則：

1. 使用具體工作，例如 shelf stocking，而不是只有 shared workspace。
2. 要求 observable activity，再判斷 workspace/path relation 與 task impact。
3. 清楚定義 action 條件，並選擇足夠但限制最小的 action。
4. 不提供完成的 JSON 數值範例，避免小模型固定複製 `WAIT` 或 confidence。
5. 把 confidence 定義成對 activity/impact 的把握度，而不是 severity。

### 5.3 Compact 與 detailed 輸出邏輯

Compact profile 每次只回傳最高優先的一個 person-related risk：

1. 人看起來正在進行什麼活動。
2. 這個活動可能如何影響目前工作。
3. Semantic severity、confidence 與可選 advisory action。

Prompt 明確禁止模型輸出座標、bbox、Track ID、keypoint、距離、速度與 TTC。單張圖片
無法可靠量測這些物理量。

Compact JSON 固定包含五個欄位：`activity`、`impact`、`severity`、`confidence`、
`action`，但 prompt 不提供任何完成的數值範例，避免 7B 模型固定複製範例答案。

Prompt 不再放入 `WAIT` 等非空 action 的完成範例，避免模型把範例當成固定答案；它也
明確要求不能把 `WAIT` 當預設值。只有畫面中的 activity 與 impact 足以支持介入時才能
選 action，否則使用 `null`。沒有明顯人員風險時使用空的 `activity` 與 `impact`。
`confidence` 表示模型對 activity 與 impact 判斷的把握度，不是 severity。只要模型
提出非空 factor，confidence 必須大於 0；若仍回傳 0，validator 會拒絕該 factor，
使它不能產生 VLM advisory 或進入 fusion。

Detailed profile 保留原始長格式：`summary`、`people_present`，以及每位人物的
`person`、`activity`、`impact`、`severity`、`confidence`、`action`。Validator 在
compact mode 只留下 severity 最高（同 severity 時 confidence 最高）的一項；在
detailed mode 則保留所有通過驗證的 factors。

允許的 action 及融合優先級為：

```text
STOP > ALERT_HUMAN > REPLAN > WAIT > SLOW_DOWN > CONTINUE > NO_ACTION
```

`risk_fusion_node` 會顯示最高優先級的建議，但 report 固定標記：

```json
{
  "control_authority": "realtime_action_only",
  "note": "VLM output is advisory and must not directly command the robot"
}
```

也就是 `/risk_vqa/fused_action_suggestion` 目前只供 GUI 與實驗觀察，真正的低延遲結果
是 `/risk_vqa/realtime_action`。目前兩者都沒有連接機器人控制器。

## 6. 其他啟動模式

### 6.1 只測即時 YOLO/TTC

不需要 controller 或 LLaVA worker：

```bash
ros2 launch risk_vqa risk_video_demo.launch.py \
  video_path:=/workspace/example/video/0610x9_v0629.mp4 \
  yolo_model:=/workspace/yolov8n.pt \
  yolo_device:=0 \
  enable_vlm:=false \
  show_window:=true
```

按 `q` 或 `Esc` 關閉 OpenCV 視窗。

### 6.2 無 GUI 的完整影片模式

先啟動 controller 與 worker，再執行：

```bash
ros2 launch risk_vqa risk_video_demo.launch.py \
  video_path:=/workspace/example/video/0610x9_v0629.mp4 \
  yolo_model:=/workspace/yolov8n.pt \
  enable_vlm:=true \
  evaluation_period_s:=2.0 \
  show_window:=false
```

使用 `ros2 topic echo` 觀察結果。

### 6.3 Standalone 圖片 Risk VQA GUI

先啟動 controller 與 worker：

```bash
ros2 launch risk_vqa risk_vqa_gui_standalone.launch.py
```

操作順序：

1. `Open Image` 選擇圖片。
2. 填寫 `Current Work Context`。
3. 設定 `Prediction Horizon (seconds)`。
4. 輸入 `VQA Question`。
5. 按 `Ask Question`。

此模式要求 `factor`、`impact` 和 `time_to_impact_s`，並顯示英文自然語言與完整 JSON。
它設定 `use_sam3_candidates=false`，不選 keypoint、不回傳座標，也不在圖片上畫標記。

### 6.4 自動相機 Risk VQA

```bash
ros2 launch risk_vqa risk_vqa.launch.py \
  image_topic:=/d435i/color/image_raw/compressed
```

它只保留最新影格，每隔 `evaluation_period_s` 分析一張新的圖片。前一個 request 尚未
完成時不會送出重疊 request。

### 6.5 White Point 一般圖片問答

```bash
ros2 launch white_point_pipeline white_point_gui_standalone.launch.py
```

這是通用圖片問答介面，不套用 `risk_vqa` 的固定 schema、秒數驗證或 action selection。

## 7. Standalone 圖片 VQA 問題設計

圖片模式的核心問題為：

```text
You are a visual safety observer. Assess only evidence visible in the current image.

Current work: <work_context>
Prediction horizon: the next <horizon_s> seconds.
User question: <user_question>

Required task:
1. Identify every visible factor likely to affect the current work within the horizon.
2. Estimate how many seconds from now each factor will first affect the work.
3. Explain the expected impact briefly.
4. Optionally select one action from the allowed action list.
```

模型必須輸出：

| 欄位 | 必要性 | 規則 |
|---|---|---|
| `factor` | Must have | 非空文字 |
| `impact` | Must have | 非空文字 |
| `time_to_impact_s` | Must have | `0` 到 prediction horizon 的有限數字 |
| `severity` | Must have | `low`, `medium`, `high`, `critical` |
| `confidence` | Must have | `0` 到 `1` |
| `action` | Good to have | 白名單 action 或 `null` |

範例：

```json
{
  "summary": "A person may enter the robot workspace.",
  "factors": [
    {
      "factor": "A person is approaching the robot workspace",
      "impact": "The person may block the robot path",
      "time_to_impact_s": 2.5,
      "severity": "high",
      "confidence": 0.8,
      "action": "STOP"
    }
  ]
}
```

模型輸出會先經 `validate_report()`：不合法 factor 會被拒絕並記錄於
`validation_errors`；通過的 factors 先按影響時間、再按 confidence 排序。圖片模式的
`time_to_impact_s` 是 VLM 從單張圖片做的粗略語意估計，不可用於安全控制。

## 8. ROS interfaces

### 8.1 影片模式 topics

| Topic | Type | 內容 |
|---|---|---|
| `/risk_vqa/video/image_raw/compressed` | `sensor_msgs/CompressedImage` | 給 VLM 的原始影格 |
| `/risk_vqa/video/annotated/compressed` | `sensor_msgs/CompressedImage` | 含 ROI、bbox 與 TTC 的 GUI 影格 |
| `/risk_vqa/video_status` | `std_msgs/String` | `ready`, `running`, `paused`, `finished` |
| `/risk_vqa/realtime_report` | `std_msgs/String` | Tracks、TTC、action、`processing_ms` JSON |
| `/risk_vqa/realtime_action` | `std_msgs/String` | `CONTINUE`, `SLOW_DOWN`, `STOP` |
| `/risk_vqa/report` | `std_msgs/String` | 驗證後的 VLM JSON |
| `/risk_vqa/text` | `std_msgs/String` | VLM 英文自然語言 |
| `/risk_vqa/action_suggestion` | `std_msgs/String` | VLM advisory action |
| `/risk_vqa/status` | `std_msgs/String` | VLM node 狀態或錯誤 |
| `/risk_vqa/fusion_report` | `std_msgs/String` | 兩層融合狀態 JSON |
| `/risk_vqa/fused_action_suggestion` | `std_msgs/String` | 融合後 advisory |

常用檢查：

```bash
ros2 topic echo /risk_vqa/realtime_report
ros2 topic echo /risk_vqa/text
ros2 topic echo /risk_vqa/report
ros2 topic echo /risk_vqa/fusion_report
```

### 8.2 播放控制 service

```text
/risk_vqa/set_paused    std_srvs/srv/SetBool
```

不透過 GUI 也可以控制：

```bash
ros2 service call /risk_vqa/set_paused std_srvs/srv/SetBool "{data: true}"
ros2 service call /risk_vqa/set_paused std_srvs/srv/SetBool "{data: false}"
```

## 9. Parameters

影片設定檔：[config/video_risk.yaml](config/video_risk.yaml)

| Parameter | Default | 功能 |
|---|---:|---|
| `video_path` | `/workspace/input.mp4` | 容器內影片絕對路徑 |
| `loop_video` | `false` | 播完後重新開始 |
| `playback_rate` | `1.0` | 播放速度倍率 |
| `show_window` | `true` | 顯示 OpenCV overlay；GUI launch 會設為 `false` |
| `yolo_model` | `yolov8n.pt` | YOLO 權重路徑 |
| `yolo_device` | `"0"` | 第一張 GPU；CPU 使用 `cpu` |
| `yolo_confidence` | `0.35` | Person confidence threshold |
| `yolo_iou` | `0.50` | YOLO NMS IoU threshold |
| `yolo_image_size` | `640` | YOLO inference resolution |
| `yolo_quantize` | `16` | GPU FP16；CPU 自動改成 `32` |
| `tracker_config` | `bytetrack.yaml` | Ultralytics tracker config |
| `roi` | `[0.20,0.15,0.80,0.95]` | Normalized `[x1,y1,x2,y2]` safety ROI |
| `warning_ttc_s` | `3.0` | 低於此值輸出 `SLOW_DOWN` |
| `critical_ttc_s` | `1.0` | 低於此值輸出 `STOP` |
| `min_growth_rate` | `0.04` | 確認 approaching 的最小成長率 |
| `min_area_ratio` | `0.002` | 套用 TTC 規則所需的最小人物框／畫面面積比 |
| `critical_area_ratio` | `0.08` | 人物中心在 ROI 且框占畫面 8% 時直接輸出 `STOP` |
| `evaluation_period_s` | `2.0` | VLM 分析間隔 |
| `model_name` | `""` | 空值使用 controller 第一個模型；否則須匹配 `/list_models` |
| `semantic_response_mode` | `compact` | `compact` 單一短風險或 `detailed` 多人物長回答 |
| `semantic_question` | How could... | VLM 要回答的 person-semantic 問題 |
| `work_context` | Robot manipulation... | 當前 robot task 與 workspace 描述 |
| `max_new_tokens` | `128` | 影片 VLM 單次最多生成 tokens |
| `use_cache` | `true` | 啟用 LLM KV cache，加速 autoregressive decoding |
| `semantic_timeout_s` | `10.0` | Fusion 接受 VLM action 的有效時間 |

圖片／相機設定檔：[config/risk_vqa.yaml](config/risk_vqa.yaml)

| Parameter | Default | 功能 |
|---|---:|---|
| `image_topic` | `/d435i/color/image_raw/compressed` | 訂閱影像 topic |
| `use_compressed_image` | `true` | `CompressedImage` 或一般 `Image` |
| `controller_url` | `http://127.0.0.1:11000` | Controller URL |
| `model_name` | `""` | 空字串時使用 controller 第一個模型 |
| `evaluation_period_s` | `2.0` | 分析間隔 |
| `prediction_horizon_s` | `10.0` | 圖片模式預測範圍 |
| `request_timeout_s` | `60.0` | Worker HTTP timeout |
| `max_new_tokens` | `384` | 單次最多生成 tokens |
| `use_cache` | `false` | 是否啟用 LLM KV cache |
| `work_context` | Robot manipulation... | 工作內容 |
| `allowed_actions` | 六種 actions | VLM action 白名單 |

## 10. 模型與權重

| 層級 | 模型／演算法 | 用途 |
|---|---|---|
| 即時 | YOLOv8n COCO class 0 | Person detection |
| 即時 | ByteTrack | Person tracking / Track ID |
| 即時 | Scale-growth formula | Pure-RGB optical TTC |
| 語意 | LLaVA v1.5 7B 或 13B | Person activity、impact、severity、advisory action |
| 語意 | CLIP ViT-L/14-336 | LLaVA image encoder |

目前 LLaVA checkpoint 的主要 image encoder 是 CLIP。Worker 預設另外啟用
`mm_use_sam3_conditioning:=true`：SAM3 先產生 visual saliency，將影像做 focus weighting
後再送入 CLIP；SAM3 並沒有取代 CLIP。

Risk VQA request 的 `use_sam3_candidates=false` 只關閉 SAM3 button candidate/keypoint
流程，不會關閉上述 image conditioning。如果只需要影片 person-semantic VQA，而且即時
人員偵測已由 YOLO 負責，可在啟動 worker 時停用 SAM3 conditioning：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ros2 launch vlpoint worker.launch.py \
  model_path:=liuhaotian/llava-v1.5-13b \
  quantization:=4bit \
  mm_use_sam3_conditioning:=false
```

此時 VQA 影像路徑是 `RGB image -> CLIP -> multimodal projector -> LLaVA`。

確認 Ultralytics：

```bash
python3 -c "import ultralytics; print(ultralytics.__version__)"
```

YOLO 權重第一次需取得一次，建議存到持久掛載的 `/workspace`：

```bash
cd /workspace
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
ls -lh /workspace/yolov8n.pt
```

影片 Risk VQA request 預設使用（`max_new_tokens` 可由參數調整）：

```json
{
  "temperature": 0.0,
  "max_new_tokens": 128,
  "use_cache": true,
  "task_type": "risk_assessment",
  "use_sam3_candidates": false
}
```

因此不執行 keypoint/candidate selection，也不在回答中輸出座標。

## 11. Troubleshooting

### `Cannot open video`

確認使用容器路徑並檢查檔案：

```bash
ls -lh /workspace/example/video/0610x9_v0629.mp4
```

主機的 `/home/lester/VLMotion/...` 在容器內應改成 `/workspace/...`。

### `malformed launch argument`

路徑含空白時要引用完整 argument：

```bash
"video_path:=/workspace/example/Synthetic Videos/demo.mp4"
```

### Controller 找不到 worker

```bash
curl -s -X POST http://127.0.0.1:11000/list_models
```

若 models 為空，確認 controller 先啟動，而且 worker 已完成模型載入與註冊。

### Controller 顯示 `address already in use`

`[Errno 98] ... 11000` 表示 port 11000 已有 process 監聽，不是 ROS log 或暫存檔
殘留。先確認現有 controller 是否可用：

```bash
curl -s -X POST http://127.0.0.1:11000/list_models
```

如果回傳 JSON（即使是 `{"models":[]}`），controller 已正常運行，直接沿用並啟動
worker 即可。不要再開第二個 controller。

若沒有正常回應，找出占用者：

```bash
ss -ltnp 'sport = :11000'
```

記下輸出中的 PID，確認後先正常終止該 process：

```bash
ps -fp <PID>
kill <PID>
```

再確認 port 已釋放後重新啟動 controller。因為 Docker 設為 `network_mode: host`，如果
container 內看不到 PID，也要在 Docker host 執行相同的 `ss` 指令。Worker 的 22000
port 發生相同錯誤時也使用相同步驟檢查。

### Hugging Face 又開始下載

```bash
echo "$HF_HOME"
du -sh /root/.cache/huggingface
find /root/.cache/huggingface/hub -name "*.incomplete"
```

`.incomplete` 表示該模型 snapshot 尚未下載完成。完整 cache 才能使用 offline mode。

### GUI 出現 Qt `xcb` plugin 錯誤

目前 `risk_video_gui` 直接使用 Qt 解碼 JPEG，不會在 GUI process 載入 OpenCV。修改後
請務必重新 build 並 source：

```bash
cd /workspace/ros2_ws
colcon build --packages-select risk_vqa --symlink-install
source install/setup.bash
```

如果仍無法開啟 GUI，再檢查 Docker 是否有正確傳入 `DISPLAY` 與 X11 socket。

### 從 SSH 顯示影片 GUI

這個 GUI 是 Qt/X11 application。建議從 client 使用 trusted X11 forwarding 連線：

```bash
ssh -Y -C <user>@<server-ip>
```

- Linux client 通常已有 X server。
- macOS 需先安裝並啟動 XQuartz，再建立 SSH 連線。
- Windows 可使用內建 X server 的 MobaXterm，或先啟動 VcXsrv。

SSH 登入後先在 host 檢查：

```bash
echo "$DISPLAY"
xauth list "$DISPLAY"
```

`DISPLAY` 應類似 `localhost:10.0`，不能是空字串。不要手動改成 `:0`；`:0` 代表
server 本機的實體桌面，不是 SSH client。若 `DISPLAY` 為空，確認 server 的
`/etc/ssh/sshd_config` 包含：

```text
X11Forwarding yes
X11UseLocalhost yes
```

Ubuntu server 還需要 `xauth`。修改 sshd 設定後，在 server 重新啟動 SSH service，
斷線並重新使用 `ssh -Y -C` 登入。

本專案的 Docker 使用 host network。從 SSH terminal 執行：

```bash
cd /home/lester/VLMotion
./docker/run.sh
```

`docker/run.sh` 會把目前 SSH 的動態 `DISPLAY` 和 MIT-MAGIC-COOKIE 傳進 container；
不再假設 display 固定為 `:1`。進入 container 後確認並啟動 GUI：

```bash
echo "$DISPLAY"
cd /workspace/ros2_ws
source install/setup.bash
ros2 launch risk_vqa risk_video_gui_demo.launch.py \
  video_path:=/workspace/example/video/0610x9_v0629.mp4 \
  yolo_model:=/workspace/yolov8n.pt \
  enable_vlm:=true
```

X11 forwarding 傳輸影片畫面可能延遲明顯。如果 GUI 更新率太低，建議使用 VNC、
NoMachine 或遠端桌面；只做運算時則使用 `enable_gui:=false show_window:=false`。

### Worker 啟動時顯示 GPU RAM 不足

這和 SSH DISPLAY 無關。先使用 `nvidia-smi` 確認沒有舊 worker 或其他模型占用 GPU。
約 12 GB GPU 建議先關閉 SAM3 conditioning，並將 concurrency 設為 1：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ros2 launch vlpoint worker.launch.py \
  model_path:=liuhaotian/llava-v1.5-7b \
  quantization:=4bit \
  mm_use_sam3_conditioning:=false \
  limit_model_concurrency:=1
```

### 修改程式後仍執行舊版本

```bash
cd /workspace/ros2_ws
colcon build --packages-select risk_vqa --symlink-install
source install/setup.bash
```

每個已開啟的 terminal 都要重新 `source install/setup.bash`。

## 12. 檔案結構

```text
risk_vqa/
├── config/
│   ├── risk_vqa.yaml                 # 圖片／相機 VQA 參數
│   └── video_risk.yaml               # 影片雙層模式參數
├── launch/
│   ├── risk_vqa.launch.py            # 自動相機 VQA
│   ├── risk_vqa_gui_standalone.launch.py
│   ├── risk_video_demo.launch.py      # 影片模式／OpenCV window／headless
│   ├── risk_video_gui_demo.launch.py  # 推薦：compact 影片雙層模式 + Qt GUI
│   └── risk_video_gui_detailed_demo.launch.py # summary/person/multi-factor GUI
├── risk_vqa/
│   ├── prompt.py                      # 圖片模式 prompt
│   ├── schema.py                      # 圖片模式 JSON 驗證與 action selection
│   ├── person_semantic.py             # 影片 person-semantic prompt/schema
│   ├── risk_vqa_node.py               # 非同步 VLM ROS node
│   ├── risk_vqa_gui.py                # Standalone 圖片 GUI
│   ├── ttc_tracker.py                 # Track state 與 optical TTC
│   ├── video_ttc_node.py              # YOLO + ByteTrack + TTC
│   ├── risk_fusion_node.py             # Real-time + VLM advisory fusion
│   └── risk_video_gui.py               # 影片監控與播放控制 GUI
└── test/
    ├── test_schema.py
    ├── test_person_semantic.py
    └── test_ttc_tracker.py
```

## 13. 已知限制與下一步

- YOLOv8n 在遮擋、遠距離、特殊姿態與 domain shift 下可能漏檢。
- ByteTrack 在長時間遮擋或人物交錯後可能更換 ID。
- Bbox 變形、橫向移動、camera motion 或 zoom 都可能造成錯誤 TTC。
- Pure RGB 無法直接得到公尺距離與真實 closing velocity。
- YOLO 與 LLaVA 共用 GPU 時可能互相競爭算力，應監看 `processing_ms`。
- ROS 2 Python、影片解碼與 YOLO 並非 hard real-time。
- 單張圖片 VQA 的 `time_to_impact_s` 是粗略語意估計，不是物理量測。

若要進入機器人實際安全控制，建議下一階段架構為：

```text
YOLO person detector
  -> multi-object tracking
  -> depth camera / calibrated 3D localization
  -> ego-motion compensation
  -> distance + relative closing velocity
  -> deterministic TTC
  -> validated safety state machine
  -> robot SLOW_DOWN / STOP interface
```

VLM 保留在非同步語意層，用於解釋情境、協助 replan 或通知人員，不直接觸發馬達控制。

## 14. Latency benchmark

目前架構的 stage-level benchmark 位於：

```text
/workspace/sam3_benchmark/risk_vqa_benchmark
```

它分開統計 real-time path 與非同步 VLM path，可輸出每個階段的 CSV、JSON、Markdown
以及 mean、P50、P95、P99、max。

先依照第 1 節分別啟動 controller、worker 與影片 GUI。進行 benchmark 時，worker 要加上
`enable_latency_timing:=true`，影片建議加上 `loop_video:=true`，避免影片先播完：

```bash
# Terminal 1：controller
ros2 launch vlpoint controller.launch.py

# Terminal 2：worker（開啟 stage timing）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ros2 launch vlpoint worker.launch.py \
  model_path:=liuhaotian/llava-v1.5-7b \
  quantization:=4bit \
  mm_use_sam3_conditioning:=false \
  limit_model_concurrency:=1 \
  enable_latency_timing:=true

# Terminal 3：循環播放待測影片；12 GB GPU 建議讓 YOLO 使用 CPU
ros2 launch risk_vqa risk_video_gui_demo.launch.py \
  video_path:=/workspace/example/video/0610x9_v0629.mp4 \
  loop_video:=true \
  yolo_model:=/workspace/yolov8n.pt \
  yolo_device:=cpu \
  enable_vlm:=true \
  evaluation_period_s:=2.0 \
  semantic_response_mode:=compact \
  max_new_tokens:=128 \
  use_cache:=true

# Terminal 4：錄製 latency
cd /workspace/ros2_ws
source install/setup.bash
python3 /workspace/sam3_benchmark/risk_vqa_benchmark/benchmark_node.py \
  --realtime-samples 300 \
  --vlm-samples 10 \
  --warmup-realtime 30 \
  --warmup-vlm 1 \
  --timeout-s 300 \
  --sample-gpu \
  --gpu-period-s 1.0 \
  --output-root /workspace/sam3_benchmark/results/risk_vqa \
  --run-name llava7b_yolo_cpu_sam3off
```

只錄即時層可設定 `--vlm-samples 0`；只錄 VLM path 可設定
`--realtime-samples 0`。程式收滿指定樣本或達到 `--timeout-s` 後會自動寫檔。
每次執行的結果位於：

```text
/workspace/sam3_benchmark/results/risk_vqa/<timestamp>_<run-name>/
├── realtime_samples.csv
├── vlm_samples.csv
├── fusion_samples.jsonl
├── gpu_samples.csv
├── summary.json
└── report.md
```

更多欄位與 SAM3 on/off 比較方式請見
[`sam3_benchmark/risk_vqa_benchmark/README.md`](../../../sam3_benchmark/risk_vqa_benchmark/README.md)。
