# risk_vqa_sim

> 2026-10-14 定位：本套件保留作雙相機 temporal VQA + zone 決策基準。現行人物時序研究入口是 [future_risk_vqa](../future_risk_vqa/README.md)，全專案操作以[使用指南](../../../docs/USAGE.md)為準。下列模型設定與測試結果屬本套件情境。

`risk_vqa_sim` 是給 Isaac Sim 雙側視角使用的 ROS 2 研究原型。它回答兩個不同問題：

1. VQA 觀察：左右視角是否清楚、誰在畫面中、他正在如何移動、是否可能進入危險區。
2. 規則決策：根據已驗證的 VQA 結果輸出 `GO`、`WATCH`、`HOLD` 或 `STOP`。

> 安全邊界：這個 package 只發布 advisory topic，不是 safety-rated controller。
> 不可直接取代安全光柵、雷射掃描器、安全 PLC、急停或正式風險評估。

## 1. 系統邏輯

```text
Isaac Sim left RGB  ─> 時間緩衝 ─┐
                                  ├─> 2 x N storyboard
Isaac Sim right RGB ─> 時間緩衝 ─┘       │
                                          v
                              LLaVA constrained VQA
                                          │
                                          v
                                schema 驗證與正規化
                                          │
                                          v
                             deterministic decision rules
                                          │
                         ┌────────────────┴───────────────┐
                         v                                v
                /risk_vqa_sim/decision          /risk_vqa_sim/command
```

系統刻意將 VLM 和最後決策拆開。VLM 只負責描述視覺證據；`risk_decision_node`
使用固定規則做決策，模型不能自由產生機器命令。

### 1.1 VQA 內部層次

一次 VQA request 會依序完成三層，避免為每一層各跑一次大型模型：

| 層 | 問題 | 固定輸出 |
|---|---|---|
| 可視性 | 左右危險區入口是否真的看得清楚 | `CLEAR / OCCLUDED / OUT_OF_VIEW / UNCERTAIN` |
| 人員觀察 | 每個人位在哪裡、觀察到何種移動 | `position`、`motion` |
| 時序意圖 | 比較 OLD→MID→NOW，下一步是否可能進入 | `intent`、`entry_time`、`confidence` |

預設從每支相機最近 2 秒取 3 張圖，組成一張 storyboard：上排是左相機、下排是右相機，
每排由左至右為 `OLDEST → MID → NOW`。這讓只能接收影像的 LLaVA worker 仍能比較時間變化。

### 1.2 使用的模型

預設沿用本 repository 的 VLMotion controller/worker：

- checkpoint：`liuhaotian/llava-v1.5-7b`
- 建議權重：4-bit，較適合 12 GB 級 GPU
- vision-language 推論：一張左右雙視角 temporal storyboard
- decoding：temperature 0，輸出固定 JSON
- SAM3 conditioning：此任務不使用，建議設為 `false`

模型名稱沒有硬寫死；`model_name` 留空時，節點會使用 controller 回傳的第一個 worker。
因此也能換成與目前 VLMotion worker 相容的 LLaVA checkpoint，但更換後必須重新驗證 prompt
遵循率、視角辨識率和意圖分類結果。

LLaVA 在這裡不是安全控制器，也不是精確 tracker。`L1`、`R1` 只是單次回答內的局部 ID，
不表示跨 request 或跨相機的永久身分。

### 1.3 一輪 VQA 如何執行

`dual_camera_vqa_node` 的每一輪不是收到一張圖就立即詢問模型，而是依序執行：

```text
1. 左右 image callback 持續寫入各自的 FrameBuffer
2. 每 evaluation_period_s（預設 2 秒）嘗試啟動一次 evaluation
3. 確認左右都有影像，而且最新影像沒有 stale
4. 確認左右最新 timestamp 相差不超過 max_camera_skew_s
5. 從最近 sample_window_s 內等距選出 sample_count 張圖
6. 確認實際 temporal span 至少達 minimum_temporal_span_s
7. 建立 2 x N storyboard 並發布給 UI
8. 將同一張 storyboard 與 constrained prompt 送給 LLaVA
9. 擷取模型回答中的 JSON object
10. schema 驗證、enum 正規化、補上 latency/evidence-age metadata
11. 發布 observation
12. risk_decision_node 依固定規則產生 decision 與 command
```

如果上一個模型 request 尚未完成，下一個 timer tick 不會再建立第二個 request，而會發布
`evaluation_already_in_flight`。因此實際 VQA 頻率為模型吞吐量所限制，不會累積無上限的 request queue。

啟動或恢復播放時，如果還沒有累積足夠時間證據，狀態會是 `warming_temporal_buffer`；這時不會
拿重複的單一影格假裝成運動資訊。

### 1.4 Prompt 的整體結構

目前 worker 使用 Vicuna/LLaVA conversation 格式。送進模型的完整文字是：

```text
<SYSTEM_PROMPT>###USER: <USER_PROMPT_WITH_IMAGE_TOKEN>###ASSISTANT:
```

其中只有一個 `<image>` token，對應一張 2 x N storyboard。沒有把六張圖各自送成 image token，
是為了避免 LLaVA v1.5 的 vision tokens 超過 context window，也避免六次獨立回答難以融合。

#### System prompt

目前實際 system prompt 定義在 `risk_vqa_sim/prompt.py`：

```text
A chat between a curious user and an artificial intelligence assistant.
The assistant follows the requested machine-readable schema exactly.
```

System prompt 刻意保持短，責任只有兩個：

1. 建立 LLaVA/Vicuna worker 所預期的 assistant 對話角色；
2. 強調回答必須遵守 machine-readable schema。

危險區、左右相機、時序和安全限制沒有塞進 system prompt，而是放在每次可配置的 user prompt。
這樣 `work_context`、取樣數和時間窗改變時，可以由 ROS parameters 產生正確描述，不需要修改
底層 conversation template。

#### User prompt

User prompt 由以下六個區塊組成：

```text
<image>

角色與場景
  You are observing a temporal storyboard from two fixed safety cameras.
  Current work: <work_context>

Storyboard 契約
  top row = LEFT CAMERA
  bottom row = RIGHT CAMERA
  each row = OLDEST ... NOW
  span = <sample_window_s>

推理順序
  1. visibility
  2. people observation
  3. temporal motion / intent / entry-time

Closed vocabularies
  visibility / position / motion / intent / entry_time

證據與禁止事項
  only visible evidence
  no identity / age / role / permission / exact metric distance
  no motion claim from one pose
  blocked view cannot become a clear empty view

輸出要求
  one JSON object only
  exact keys and example structure
```

`work_context` 預設為：

```text
A machine or robot is about to enter a shared hazard zone.
People may approach from either side.
```

它告訴模型目前要判斷的是「左右來向的人是否會進入共享危險區」，而不是一般圖片描述。
Prompt 在 node 初始化時建立一次，所以修改 `work_context`、`sample_count` 或 `sample_window_s`
後需要重新啟動 node。

### 1.5 為什麼問題要分成三層

模型雖然只做一次 generation，prompt 仍要求它依序回答三層：

| 層 | 實際問題 | 為什麼不能省略 |
|---|---|---|
| Visibility | 危險區來向是否真的看得清楚？ | 「沒看到人」可能只是遮擋，不能直接等同安全 |
| Observation | 有哪些人？目前在哪個 zone？觀察到什麼 motion？ | 將直接影像事實與未來預測拆開 |
| Temporal interpretation | 他較可能 ENTER、WAIT、PASS_BY 還是 MOVE_AWAY？ | 只有這層才允許根據 OLD→NOW 推測下一步 |

這個順序避免直接問模糊的「現在安全嗎？」。模型不負責回答 `GO/HOLD/STOP`，所以即使 VLM
對語意有不同表達，也必須先落到固定 observation labels，最後再由程式規則決策。

### 1.6 Closed vocabulary 與定義

每個 camera observation 必須包含：

```text
visibility:
  CLEAR | OCCLUDED | OUT_OF_VIEW | UNCERTAIN

position:
  OUTSIDE | APPROACH_ZONE | WAITING_ZONE | INSIDE_HAZARD_ZONE | UNKNOWN

motion:
  APPROACHING | CROSSING | STATIONARY | LEAVING | PASSING_BY | UNKNOWN

intent:
  ENTER | WAIT | PASS_BY | MOVE_AWAY | UNCERTAIN

entry_time:
  NOW | 0_TO_2_SECONDS | 2_TO_5_SECONDS | MORE_THAN_5_SECONDS |
  NOT_APPROACHING | UNKNOWN
```

重要語意邊界：

- `CLEAR` 表示視角清楚，不表示沒有人；是否有人由 `people` array 表達。
- `motion` 是 storyboard 中已觀察到的變化；`intent` 是對下一步的分類，兩者不可混用。
- `ENTER` 必須有路徑可能穿越危險區的證據；畫面中出現人不等於 `ENTER`。
- `PASS_BY` 必須能看到路徑平行於入口或遠離入口。
- `entry_time` 是粗略時間帶，不是幾何 TTC 或精確秒數。
- `confidence` 是該人物整組分類的信心，不是 detector probability。
- `L1/L2`、`R1/R2` 只在單次回答內有效，沒有跨 request 或跨 camera identity。

模型回答範例：

```json
{
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
        "evidence": "person moves toward the marked entrance across frames"
      }
    ]
  },
  "right": {
    "visibility": "CLEAR",
    "visibility_confidence": 0.96,
    "people": []
  },
  "scene_summary": "One person approaches from the left."
}
```

### 1.7 Model request 設定

目前送給 VLMotion worker 的 generation parameters：

| 欄位 | 值 | 目的 |
|---|---|---|
| `temperature` | `0.0` | 使用 deterministic decoding，降低同一畫面答案漂移 |
| `top_p` | `1.0` | temperature 0 時不額外截斷 token 分布 |
| `max_new_tokens` | `512` | 容納雙 camera JSON，同時限制無限生成 |
| `stop` | `###` | 配合 Vicuna conversation separator |
| `images` | 一張 base64 JPEG storyboard | 與 prompt 中唯一 `<image>` token 對應 |
| `task_type` | `risk_assessment` | 避免 worker 附加一般 VQA 自然語言要求 |
| `use_sam3_candidates` | `false` | 此任務不做按鈕/點位候選選擇 |
| `use_cache` | `false` | 預設每張新 storyboard 都重新推論 |

模型 response 可能在 JSON 前後加入文字，或輸出 Python dict 的單引號格式。Parser 會先尋找合法 JSON
object；找不到時才使用安全的 `ast.literal_eval` data-literal fallback，不會執行模型輸出的程式碼。

### 1.8 Schema 驗證與 fail-conservative normalization

VLM 回答不能直接進 decision。`schema.py` 會逐層檢查：

1. `left`、`right` 必須都是 object；
2. `visibility`、`position`、`motion`、`intent`、`entry_time` 必須屬於 enum；
3. `people` 必須存在且為 list；
4. 單一 camera 超過 12 個人物時視為無法可靠處理；
5. confidence 必須是有限數字並限制在 0–1；
6. 缺少 person ID 時只補局部 ID，同時記錄 validation error；
7. `evidence` 和 `scene_summary` 會限制長度。

正規化採取保守方向：

```text
invalid camera       → visibility=UNCERTAIN, confidence=0, people=[]
invalid position     → UNKNOWN
invalid motion       → UNKNOWN
invalid intent       → UNCERTAIN
invalid entry_time   → UNKNOWN
invalid confidence   → 0.0
```

特別是 `people` 缺少或型別錯誤時，不會當成「空畫面」，而是把整個 camera view 轉成
`UNCERTAIN`，防止 schema 錯誤誤導系統進入 `GO`。

### 1.9 決策規則的真正優先順序

模型不輸出 command。`RiskDecisionEngine` 依下列順序對每一份已驗證 observation 執行，第一個
符合的條件立即回傳：

| 優先序 | 條件 | state | command |
|---:|---|---|---|
| 1 | 任一 camera 不是 `CLEAR`，或 visibility confidence < 0.55 | `UNKNOWN` | `HOLD` |
| 2a | 左右皆 clear 且沒有人，但 clear streak 尚未達 2 | `HOLD` | `HOLD` |
| 2b | 左右皆 clear、沒有人且 clear streak ≥ 2 | `GO` | `GO` |
| 3 | 高信心人物已在區內、`CROSSING`、`NOW` 或 0–2 秒內進入 | `STOP` | `STOP` |
| 4 | 人物 confidence < 0.55，或 position/motion/intent 不確定 | `HOLD` | `HOLD` |
| 5 | `ENTER`、`APPROACHING`、位於 approach zone 或 2–5 秒內進入 | `HOLD` | `HOLD` |
| 6 | 有人，但只是在等待、路過或離開 | `WATCH` | `WATCH` |

立即衝突的 `STOP` 放在一般不確定性之前：如果模型以高信心看見人物已在危險區，即使 intent
仍是 `UNCERTAIN`，系統仍會 `STOP`。反過來，低信心的可疑人物至少會 `HOLD`，不會進入 `GO`。

### 1.10 系統如何跨輪迭代

目前 VQA 本身是 stateless：

- 每次 request 都是新的 storyboard 加同一套 prompt；
- 不把上一輪自然語言回答放回下一輪 context；
- 不讓模型根據自己前一次的結論反覆推理；
- 時序證據只來自當輪 storyboard 的 OLD→MID→NOW pixels。

跨輪保存的狀態只有 deterministic 層的 `clear_streak` 和資料時間：

```text
任一遮擋／有人／不確定／風險
  → clear_streak = 0

第 1 次左右 CLEAR 且 people=[]
  → clear_streak = 1
  → HOLD（等待第二次確認）

第 2 次連續左右 CLEAR 且 people=[]
  → clear_streak = 2
  → GO

之後仍持續 clear
  → clear_streak 繼續增加
  → GO
```

資料新鮮度另外由 `risk_decision_node` heartbeat 檢查：

- 超過 `input_timeout_s` 沒有新 observation：`UNKNOWN/HOLD`；
- 模型回答發布時，影像本身已超過 `max_evidence_age_s`：拒絕該回答並 `HOLD`；
- observation 發布後，其 evidence age 會隨 heartbeat 繼續增加，超過門檻後轉成 `HOLD`；
- invalid JSON 或 model pipeline error：清除 clear streak 並 `HOLD`。

因此「模型剛回答」不代表「影像很新」；系統同時追蹤 request receipt age 與真正視覺證據的 age。

### 1.11 邏輯的設計迭代

目前架構是由以下幾個設計階段收斂而來：

1. **單張圖直接問安全與否**：問題太籠統，無法知道錯在沒看到人、方向判斷或決策。
2. **拆成左右視角與原子問題**：先問 visibility 和 people，再分開 position、motion、intent。
3. **加入 temporal storyboard**：LLaVA v1.5 不是 video model，因此將 OLD/MID/NOW 放在同一張圖，
   讓 `APPROACHING/PASSING_BY` 至少有跨影格證據。
4. **改成 closed vocabulary JSON**：移除自由文字 command，使輸出能驗證、測試與統計 confusion matrix。
5. **VQA 與 command 分離**：模型只負責 observation；安全動作由 deterministic rules 產生。
6. **加入 fail-conservative schema**：缺欄位、錯誤 enum、遮擋或低信心都不能被解讀成安全。
7. **加入跨輪 clear confirmation**：一次空畫面不足以 `GO`，預設需要兩次連續 clear。
8. **加入 evidence age 與 timeout**：避免慢速 VLM 用過期畫面發布看似最新的安全結論。
9. **統一影片與 Isaac Sim 介面**：兩者都發布相同左右 image topics，VQA 邏輯不因來源改變。
10. **加入 UI 可觀察性**：同時顯示原始視角、模型 storyboard、raw answer、validated JSON 與最終規則，
    方便定位錯誤發生在哪一層。

這些迭代的核心原則是：每一層只回答自己能證明的事情，而且任何資料不足都不得自動變成安全。

## 2. Package 內容

```text
risk_vqa_sim/
├── config/risk_vqa_sim.yaml       所有 topic、模型、時序與決策門檻
├── launch/risk_vqa_sim.launch.py  同時啟動 VQA 與規則節點
├── launch/risk_vqa_video_demo.launch.py  雙影片來源＋VQA＋規則節點
├── risk_vqa_sim/
│   ├── dual_camera_vqa_node.py    雙相機訂閱、緩衝、storyboard、VLM request
│   ├── dual_video_source_node.py  以共同時間軸發布左右影片
│   ├── model_client.py            VLMotion controller/worker HTTP client
│   ├── prompt.py                  分層 VQA prompt 與固定答案集合
│   ├── risk_vqa_sim_gui.py        雙視角、storyboard、VQA 與 decision UI
│   ├── schema.py                  JSON parser、enum 與保守驗證
│   ├── storyboard.py              時間取樣與 2 x N 畫面組合
│   ├── decision.py                不依賴 ROS 的決策狀態機
│   └── risk_decision_node.py      ROS topic、timeout 與 heartbeat
├── docs/ISAAC_SIM.md              Isaac Sim 串接契約
├── docs/SCHEMA.md                 observation/decision 完整格式
├── docs/VIDEO_MODE.md             雙影片時間軸與同步方式
└── test/                          schema、storyboard、決策單元測試
```

## 3. Build

在主機 workspace：

```bash
cd /home/lester/VLMotion/ros2_ws
colcon build --packages-select risk_vqa_sim --symlink-install
source install/setup.bash
```

如果 ROS 2 在專案的 Docker 容器內，將路徑改成容器掛載位置，例如：

```bash
cd /workspace/ros2_ws
colcon build --packages-select risk_vqa_sim --symlink-install
source install/setup.bash
```

## 4. 啟動

### Terminal 1：VLMotion controller

```bash
ros2 launch vlpoint controller.launch.py
```

### Terminal 2：LLaVA worker

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ros2 launch vlpoint worker.launch.py \
  model_path:=liuhaotian/llava-v1.5-7b \
  quantization:=4bit \
  mm_use_sam3_conditioning:=false
```

模型 cache 尚未下載完整時，不要加 offline 環境變數。可用以下指令確認 worker 已註冊：

```bash
curl -s -X POST http://127.0.0.1:11000/list_models
```

### Terminal 3：目前先使用雙影片（推薦）

這個 launch 會同時啟動雙影片來源、VQA 和 decision node：

```bash
ros2 launch risk_vqa_sim risk_vqa_video_demo.launch.py
```

此指令現在預設同時開啟 PyQt5 UI，畫面包含：

- 左、右兩支影片的即時視角；
- LLaVA 實際收到的 `OLD → MID → NOW` storyboard；
- 左右 visibility、人員位置、motion、intent 與 entry-time；
- rule-based decision state、command、reason 與 observation age；
- 完整 observation JSON、decision JSON 與 raw VLM 回答；
- `Pause Videos`、`Continue Videos` 播放控制。

不需要 UI，例如透過 SSH 執行時：

```bash
ros2 launch risk_vqa_sim risk_vqa_video_demo.launch.py enable_gui:=false
```

關閉 GUI 視窗會一起關閉該 launch 中的影片、VQA 與 decision nodes。

預設對應已設定為：

```text
LEFT  = /workspace/example/video/0610_take3n4.mp4
RIGHT = /workspace/example/video/0610_take1n2.mp4
```

兩支原始影片都是 1920×1080、30 FPS。預設發布 960×540、10 FPS，並以較短的左側影片
約 18.37 秒為共同循環終點；到達終點時兩邊一起回到開頭，不會各自循環。

若 ROS 2 直接執行在主機而不是 `/workspace` 容器，覆寫成主機路徑：

```bash
ros2 launch risk_vqa_sim risk_vqa_video_demo.launch.py \
  left_video_path:=/home/lester/VLMotion/example/video/0610_take3n4.mp4 \
  right_video_path:=/home/lester/VLMotion/example/video/0610_take1n2.mp4
```

只播放一次或調整速度：

```bash
ros2 launch risk_vqa_sim risk_vqa_video_demo.launch.py \
  loop_videos:=false playback_rate:=0.5 output_fps:=10.0
```

如果左右影片事件開始時間不同，可以在 `config/risk_vqa_sim.yaml` 調整
`left_start_offset_s` 與 `right_start_offset_s`。這只裁掉各影片開頭，不會改變播放速度。

影片來源狀態：

```bash
ros2 topic echo /risk_vqa_sim/video_source_status
```

### 之後切換成 Isaac Sim

讓兩支 RGB camera 發布 `sensor_msgs/msg/Image`：

```text
/left_camera/rgb
/right_camera/rgb
```

兩個 message 應使用同一個 simulation clock，timestamp 差預設不可超過 0.2 秒。設定方式與
檢查項目見 [docs/ISAAC_SIM.md](docs/ISAAC_SIM.md)。

接著使用不包含影片來源的標準 launch：

```bash
ros2 launch risk_vqa_sim risk_vqa_sim.launch.py
```

如果 Isaac topic 名稱不同，可以直接覆寫：

```bash
ros2 launch risk_vqa_sim risk_vqa_sim.launch.py \
  left_image_topic:=/World/left_camera/image_raw \
  right_image_topic:=/World/right_camera/image_raw
```

使用 `sensor_msgs/msg/CompressedImage` 時：

```bash
ros2 launch risk_vqa_sim risk_vqa_sim.launch.py \
  left_image_topic:=/left_camera/rgb/compressed \
  right_image_topic:=/right_camera/rgb/compressed \
  use_compressed_image:=true
```

## 5. 觀察結果

```bash
ros2 topic echo /risk_vqa_sim/status
ros2 topic echo /risk_vqa_sim/observation
ros2 topic echo /risk_vqa_sim/decision
ros2 topic echo /risk_vqa_sim/command
```

查看實際送給模型的 storyboard：

```bash
ros2 run rqt_image_view rqt_image_view /risk_vqa_sim/storyboard/compressed
```

主要 topics：

| Topic | Type | 說明 |
|---|---|---|
| `/risk_vqa_sim/observation` | `std_msgs/String` JSON | 驗證後的左右視角 VQA 結果 |
| `/risk_vqa_sim/raw_model_response` | `std_msgs/String` | 除錯用的模型原始回答 |
| `/risk_vqa_sim/storyboard/compressed` | `sensor_msgs/CompressedImage` | 模型實際看到的 2 x N storyboard |
| `/risk_vqa_sim/status` | `std_msgs/String` | warming、evaluating、ok 或錯誤狀態 |
| `/risk_vqa_sim/video_source_status` | `std_msgs/String` JSON | 雙影片時間軸、循環與解碼狀態 |
| `/risk_vqa_sim/decision` | `std_msgs/String` JSON | state、原因、相關人物和資料年齡 |
| `/risk_vqa_sim/command` | `std_msgs/String` | `GO/WATCH/HOLD/STOP` advisory |

GUI 也可獨立啟動：

```bash
ros2 run risk_vqa_sim risk_vqa_sim_gui
```

完整 JSON 範例見 [docs/SCHEMA.md](docs/SCHEMA.md)。

## 6. 重要參數

編輯 `config/risk_vqa_sim.yaml`：

| 參數 | 預設 | 說明 |
|---|---:|---|
| `evaluation_period_s` | 2.0 | 多久執行一次 VQA；前一次未完成時不重疊 request |
| `sample_window_s` | 2.0 | storyboard 涵蓋的時間 |
| `sample_count` | 3 | 每支相機的取樣數，限制 2–6 |
| `minimum_temporal_span_s` | 0.8 | 啟動後至少累積多少時間才允許推論 |
| `max_camera_skew_s` | 0.20 | 左右最新影格允許的 timestamp 差 |
| `max_frame_age_s` | 2.0 | 任一 camera 停止更新後拒絕推論 |
| `input_timeout_s` | 5.0 | decision 節點多久沒收到 observation 就轉 `HOLD` |
| `max_evidence_age_s` | 5.0 | VLM 完成時影像證據太舊則拒絕該結果並 `HOLD` |
| `clear_confirmations_required` | 2 | 進入 `GO` 前需要幾次連續 clear |
| `person_confidence_threshold` | 0.55 | 低於此值將視為不確定並 `HOLD` |
| `output_fps` | 10.0 | 影片模式發布至 ROS 的頻率，不是原始檔案 FPS |
| `playback_rate` | 1.0 | 影片時間相對於 wall time 的倍率 |
| `left/right_start_offset_s` | 0.0 | 個別影片開始位置，用來手動對齊事件 |

`work_context` 必須依場景改寫，例如說明危險區是機械手臂工作區、AGV 交叉口或設備入口。
它是 VQA 判斷「入口與進入行為」的重要上下文，但不能用文字取代畫面中清楚可辨認的區域標示。

## 7. 測試

不啟動 ROS/GPU 即可測試純邏輯：

```bash
cd /home/lester/VLMotion/ros2_ws/src/risk_vqa_sim
python3 -m pytest -q
```

ROS 環境可再執行：

```bash
cd /home/lester/VLMotion/ros2_ws
colcon test --packages-select risk_vqa_sim
colcon test-result --verbose
```

## 8. 已知限制

- LLaVA v1.5 是通用 image VLM，不是真正 video model；storyboard 是兼容現有 worker 的折衷。
- 純 VQA 的人物 ID 不跨時間持久。若要精確回答「同一個人」的長時間軌跡，下一階段應加入
  detector/tracker，將 track crop 或 track metadata 交給 VQA。
- 兩相機若有重疊視野，目前不做跨相機 re-identification，可能把同一人列為 `L1` 與 `R1`。
- 目前兩個測試影片不是由同一個硬體 trigger 取得；系統只保證播放時間軸同步，不代表影片中的
  真實事件原本就在同一時刻發生。必要時要用 start offset 手動對齊。
- `entry_time` 是離散語意分類，不是幾何 TTC。若 Isaac Sim 可提供 ground truth position，應另建
  幾何安全層與 VQA 交叉驗證。
- prompt 遵循失敗、HTTP 錯誤、相機遮擋與資料 timeout 都會保守輸出 `HOLD`，不會假定安全。

在評估資料集上至少要分別量測：左右有人／無人、接近／路過、遮擋、多人、轉身、停止、
進入時間帶，以及最重要的 unsafe false negative。不要只看整體 accuracy。
