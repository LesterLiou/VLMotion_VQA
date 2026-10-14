# Research GUI 操作與調整指南

## 1. GUI 的用途

`future_risk_research_gui` 是研究監看器，用來確認每一層收到什麼資料，以及 VQA 最後為什麼
產生某個 action hint。它不是參數編輯器；畫面中的唯一控制按鈕是暫停／恢復。模型開關、
threshold 與取樣頻率都必須修改 YAML，然後重新啟動 launch。

目前 GUI 不會顯示完整 system/user prompt，也沒有直接修改 threshold 的控制元件。
`decision` tab 會保存 policy 使用的參數快照，完整 prompt 以 `future_risk_vqa/prompt.py` 為準。

影片版使用 `config/future_risk_video_demo.yaml`，Isaac Sim 使用
`config/future_risk_isaac.yaml`。

## 2. 畫面區域

### 左上與右上：VQA NOW SNAPSHOT

- 顯示該次 VQA inference 使用的 LEFT NOW 與 RIGHT NOW，不是持續更新的 live preview。
- 綠色 bbox、ID 與數字是該次 SAM3 detection/tracking 的框與 detection confidence。
- 黃色 footpoint 與文字是同一份 report 的 camera-local track ID。
- 上方 NOW、下方 storyboard 與右側 decision 會在模型完成後一起更新。
- `L*` 與 `R*` 是各相機自己的 ID，目前不代表兩個畫面中的同一個人已完成跨相機配對。

對某次 decision 而言，這兩張 NOW snapshot 與下方 storyboard 是唯一的 authoritative
evidence。SAM3 report timestamp 是 temporal sampling 的時間錨點，不應拿較新的 live frame
解釋較舊的 VQA decision。

### 左下：VLM INPUT — PER-ID OLD/MID/NOW STORYBOARD

這張圖是VLM request rows的完整總覽。預設`storyboard_mode: track_crops`且
`vqa_inference_mode: per_track`，每個目前可見的camera-local ID佔一列，而且每一列分別送出
一次request：

```text
L1 | LEFT  | T-old   L1 | LEFT  | T-mid   L1 | LEFT  | NOW
L3 | LEFT  | T-old   L3 | LEFT  | T-mid   L3 | LEFT  | NOW
R2 | RIGHT | T-old   R2 | RIGHT | T-mid   R2 | RIGHT | NOW
```

每格都用該timestamp的同一ID bbox裁切並放大，bbox外預設保留30%場景。某格顯示
`TRACK NOT FOUND`代表當時沒有符合時間容許值的同ID observation；系統不會用另一個人補上。
zone與預測軌跡預設不顯示。為了做ablation，可把`storyboard_mode`改成`global`，恢復原本
LEFT/RIGHT兩列的2x3全景。

### 右側：左右鏡頭的目前觀察與未來預測

UI 依 camera-local ID 分成左右兩張卡片。例如：

```text
LEFT CAMERA                              RIGHT CAMERA
L6 NOW: standing posture, walking, toward camera, facing camera
L6 NEXT: toward camera
```

`NOW`與`NEXT`的activity/orientation來自VLM，方向都相對該ID所屬相機。當
`motion_source=tracking_hint`時，顯示方向是tracking-fused值；VLM原方向保存在semantics的
`vlm_motion_trend`與`vlm_future_motion`。系統不再顯示或產生`MAP`；policy直接使用NEXT motion。

### 右上：Action banner

| 顯示 | 顏色 | 意義 |
|---|---|---|
| `STOP` | 紅 | 影片設定可由 image-space 近場區觸發，live／Isaac 目前關閉；校正後 zone policy 另可用 hazard/TTE |
| `DANGER_WARNING` | 橘 | 人員靠近、不確定、資料過期、語意不可用或仍在等確認 |
| `CONTINUE` | 綠 | 所有人皆為安全motion，且通過連續確認 |
| `NO_RESPONSE` | 灰 | 左右鏡頭都沒有目前可見的 person detection |
| `OBSERVE` | 藍 | action output 被關閉，只觀察 VQA 結果 |

banner 下方的 `WHY THIS DECISION?` 會用英文指出造成決策的 ID，例如
`L6 is moving toward the robot-mounted camera.`。其下仍保留 `reason_code`、`critical_tracks` 與
目前 policy mode，方便研究紀錄。這些是研究用提示；GUI 不會直接控制機器。

`VQA: schema_warning` 不是模型沒有執行，而是模型回傳的 JSON 未通過 contract。左右行為卡
會顯示第一條具體 validation error；請到 `raw_vlm` 比較模型原文，再到 `semantics` 查看清理
後的結果。新版 prompt 會動態填入當次 ID，並要求每個可見 ID 恰好回一筆 assessment。

`VQA: unavailable` 是另一種狀態，代表 controller/worker、網路、timeout 或 backend 無法完成
request。此時 per-camera card 顯示 `VQA UNAVAILABLE`、受影響 ID、`Confidence: N/A` 與原始
錯誤；它不是模型預測的 0%，也不會再被標成 schema warning。系統仍採 fail-safe
`DANGER_WARNING`。

### 狀態列

`SAM3 | VQA | VIDEO` 顯示三個主要元件的 state：

- `SAM3: ready`：person detection/tracking 有正常完成。
- `VQA: ready`：VLM 回覆通過 schema；`schema_warning` 表示 JSON 可讀但 contract 驗證有問題；
  `quality_warning`表示schema合法但偵測到答案坍縮；`unavailable`表示backend、request或解析失敗。
- `VIDEO`：影片來源的播放／暫停狀態；Isaac launch 沒有 video source，因此 VIDEO 可維持 `--`。

## 3. 五個資料頁籤

建議依照資料流由左至右檢查：`tracks → geometry → semantics → decision`，最後才看
`raw_vlm`。

| Tab | 內容 | 主要檢查欄位 |
|---|---|---|
| `tracks` | SAM3 與 tracker 的結構化結果 | `track_id`、`confidence`、`visible`、`missed_frames`、`age_frames` |
| `geometry` | footpoint、Kalman motion 與 future path | `tracking_reliable`、`position`、`velocity`、`future_path`、`motion_prediction_enabled` |
| `semantics` | 驗證後的兩層VQA結果 | `current_observation`、`future_prediction`、`validation_errors` |
| `decision` | deterministic policy 最後報告 | `action`、`reason_code`、`critical_tracks`、`policy_parameters` |
| `raw_vlm` | 模型尚未經 schema parser 處理的原始文字 | JSON 格式錯誤、漏 ID、多餘自然語言 |

如果`raw_vlm`看起來正確但`semantics.validation_errors`不為空，通常是current/future enum、
track ID或flat必填欄位不符合schema。null observable會正規化成`UNCERTAIN`並列在
`normalizations`，不產生schema warning。若NOW/NEXT正確但action不符預期，檢查
`decision`中的motion清單、品質門檻與連續確認次數。

per-track模式的`raw_vlm`是`responses`陣列，每筆包含`track_id`與該ID未解析的模型原文。
若`quality_warnings`含`semantic_output_collapse`，代表至少設定數量的ID得到完全相同的原始
類別、confidence與evidence；系統會顯示QUALITY WARNING並採fail-safe DANGER_WARNING。

## 4. Pause 的正確使用方式

按 `Pause after current inference` 後：

1. 已經開始的 SAM3/VQA inference 會先跑完。
2. 最後一次結果會更新到 GUI。
3. 影片停止，也不會排入新的 SAM3/VQA inference。
4. 畫面、ID、storyboard 與 JSON 保持不變，方便逐一檢查不同 ID。

暫停期間 geometry/semantics timeout 也會凍結，因此不會只因為等待檢查而變成
`tracking_report_stale`。Resume 後 timeout 才恢復正常判斷。

按鈕變成 `Resume video + inference` 後，按下才會繼續。關閉 GUI 視窗會結束整個 launch。

## 5. 依症狀調整 YAML

每次只改一組參數，並用固定影片、固定片段比較修改前後結果。

### 人常常漏偵測

```yaml
confidence_threshold: 0.30  # 預設 0.35，可每次下降 0.05
mask_threshold: 0.45        # 預設 0.50
inference_hz: 3.0           # 預設 2.0，GPU 足夠再提高
```

先降 `confidence_threshold`；只有 mask 明顯缺角時再調 `mask_threshold`。下降太多會增加誤偵測。

### 把非人物當成人

```yaml
confidence_threshold: 0.40  # 可每次上升 0.05
```

同時在 `tracks` tab 確認假目標的 confidence 分布，再決定門檻，不要只看單一 frame。

### 同一個人的 ID 經常改變

```yaml
minimum_match_score: 0.15   # 預設 0.20，降低會比較容易延續舊 ID
max_missed_frames: 5        # 預設 5，允許短暫遮擋
match_max_foot_distance_ratio: 0.20
match_velocity_smoothing: 0.60
```

`minimum_match_score` 太低可能把兩個靠近的人錯接成同一 ID；`max_missed_frames` 太高則會讓
已離開的人保留過久。新tracker會依timestamp估計pixel velocity，以預測footpoint與bbox做
下一次Hungarian matching；`match_velocity_smoothing`越高越信任歷史速度，快速變向時不宜過高。
兩人交叉仍可能ID switch，因為目前未加入appearance Re-ID。

### VQA 對動作變化判斷不足

```yaml
temporal_frame_count: 3
temporal_window_s: 2.0
vqa_hz: 0.5
storyboard_mode: track_crops
track_crop_padding_ratio: 0.30
```

- 動作較慢：先把 `temporal_window_s` 調到 `3.0` 或 `4.0`。
- 動作很快：可縮短 window，並在 GPU/worker 能負荷時提高 `vqa_hz`。
- 增加 `temporal_frame_count` 會提供更多時間點，但 storyboard 變寬、模型輸入更複雜；建議
  先比較 3 與 4，不要一次大幅增加。
- 人物太小時先用`track_crops`；若手與互動物件被裁掉，把`track_crop_padding_ratio`依序提高
  到`0.40`、`0.50`。
- 到`semantics.metadata.track_crop_coverage`檢查每個ID是否三格都有bbox；例如`L1: 2`表示
  三個時間點只找到兩個，不應把該列當作完整motion evidence。

### 想比較「有無預測軌跡」

```yaml
enable_motion_prediction: true
trajectory_context_mode: none     # baseline
# trajectory_context_mode: overlay
# trajectory_context_mode: numeric
# trajectory_context_mode: both
```

`none` 不把軌跡給 VLM；`overlay` 畫進 storyboard；`numeric` 放進 prompt context；`both`
同時使用。做 ablation 時，其餘設定與影片片段必須保持一致。

### 系統太常顯示 DANGER_WARNING

先看 `decision.reason_code`，不要直接降低所有 threshold：

- `future_behavior_unavailable`：檢查 controller/worker 狀態、request timeout 與 retry 設定。
- `future_behavior_schema_invalid`：檢查 `raw_vlm` 與 prompt/schema。
- `future_behavior_assessment_missing`：VLM 漏掉某個可見 ID。
- `vqa_future_uncertain_warning`：檢查 `semantics` 的NEXT motion是否為`UNCERTAIN`。

### STOP 或 DANGER_WARNING 太少／太多

- 也可以修改 `behavior_stop_motions`、`behavior_warning_motions`，但 motion mapping 改動應視為
  另一組實驗條件並寫進 run metadata。

## 6. 建議的檢查流程

1. 先確認左右 SAM3 mask、bbox 與 ID 正確。
2. 在 `tracks` 確認每個可見人物都有穩定 ID。
3. 在 storyboard 確認 OLD/MID/NOW 時序與 VLM 實際看到的畫面。
4. 等 action 更新後按 Pause，讓當次 inference 完成並凍結。
5. 在 `semantics` 逐 ID 核對 future event、confidence 與 evidence。
6. 在 `decision` 核對 event mapping、threshold、reason code 與 critical IDs。
7. 只有 schema 異常時查看 `raw_vlm`；只有 tracking 異常時回到 `tracks`。
8. 記錄問題片段與 run name，每次只改一個變因再重跑。

目前研究階段建議保持 `enable_zone_logic: false` 與 `trajectory_context_mode: none`，先建立純
temporal VQA baseline。調整順序固定為：person detection → ID tracking → VQA event/schema →
action threshold。否則前一層的錯誤容易被誤判成 threshold 問題。

## 7. 修改後重啟

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select future_risk_vqa
source install/setup.bash
ros2 launch future_risk_vqa future_risk_video_demo.launch.py
```

單純修改已安裝為 symlink 的 Python/YAML 通常仍建議完整重啟，因為 ROS parameters 是節點啟動
時讀取，不會由 GUI 即時更新。
