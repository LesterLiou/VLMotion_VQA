# Future Risk VQA 研究延續與交接紀錄

交接背景日期：2026-08-06；2026-10-14 整理補註

> 9 月後設定已有變動：目前 source YAML 為 CPU SAM3、track_crops/per_track，且 live／Isaac 已關閉 image-space 近場 STOP（影片仍啟用），延遲與失敗實測見 [SAM3_CPU_TRIAL](../../../../robopoint_runtime/SAM3_CPU_TRIAL.md)。本頁保留研究假設、風險與實驗設計；操作與預設值以 [CONFIGURATION](CONFIGURATION.md) 和[使用指南](../../../../docs/USAGE.md)為準，以下速度建議不是本次重測結果。

這份文件描述目前實際運作的系統，而不是最初設計。後續研究、論文實驗、重構與除錯都應先
以本文件、`config/future_risk_isaac.yaml`和程式碼互相核對。README負責快速使用；本文件負責
保存研究假設、決策來源、限制與下一步。

## 1. 現在研究的問題

輸入是安裝在機器人左右兩側、向外看的RGB相機。系統需要：

1. 找到每台相機中的人並維持camera-local track ID。
2. 描述每個人的NOW姿勢、活動與身體朝向。
3. 根據OLD/MID/NOW預測camera-relative NEXT motion。
4. 在沒有真實公尺標定前，以一個可調的image-space近場區先做STOP實驗。
5. 讓每個action都能追溯到current tracking或某一輪timestamped VQA evidence。

目前不宣稱：

- 單張RGB可以量出可靠的1公尺距離；
- SAM3 floor mask等於可通行走道；
- 左右相機的`L*`與`R*`是同一套跨相機身分；
- VLM的身體朝向等於精確頭部pose或gaze；
- VLM的future motion是具有安全認證的機器人控制命令。

## 2. 現行資料流與決策優先序

```text
LEFT / RIGHT RGB
        │
        ├─ SAM3 "person" → mask、bbox、footpoint、camera-local tracking
        │
        ├─ SAM3 "floor"（低頻）→ 藍色floor/走道候選，只供顯示
        │
        ├─ normalized紅色梯形 + person腳部mask → current image-space STOP
        │
        └─ OLD/MID/NOW storyboard + tracking hint
                              │
                              ▼
                    LLaVA compact JSON
                              │
                              ▼
                    deterministic policy
```

`future_behavior_policy`的有效優先序：

1. 沒有geometry report：`NO_RESPONSE`。
2. 最新geometry沒有任何visible person：`NO_RESPONSE`。
3. 最新visible person進入image-space紅區：立即`STOP`。
4. tracking report過期：`DANGER_WARNING`。
5. VQA不可用、schema錯誤、缺少assessment：`DANGER_WARNING`。
6. 任一NEXT motion為`TOWARD_CAMERA`或`UNCERTAIN`：`DANGER_WARNING`。
7. 所有assessment均為`AWAY_FROM_CAMERA`、`LATERAL_ACROSS_VIEW`或`STATIONARY`：
   目前一次確認後`CONTINUE`。

總優先序是：

```text
STOP > DANGER_WARNING > CONTINUE
NO_RESPONSE只代表目前沒有visible person
```

## 3. VLM compact contract

模型每個ID只允許輸出：

```json
{
  "assessments": [
    {
      "track_id": "L1",
      "body_orientation": "FACING_CAMERA",
      "body_posture": "STANDING",
      "current_activity": "WALKING",
      "future_motion": "TOWARD_CAMERA"
    }
  ]
}
```

封閉詞彙：

- `body_orientation`：`FACING_CAMERA`、`BACK_TO_CAMERA`、`SIDE_ON`、`UNCERTAIN`
- `body_posture`：`STANDING`、`SQUATTING`、`SITTING`、`KNEELING`、`LYING_DOWN`、
  `BENDING_OR_CROUCHING`、`UNCERTAIN`
- `current_activity`：`WALKING`、`CHANGING_POSTURE`、
  `REACHING_OR_HANDLING_OBJECT`、`WORKING`、`OTHER`、`UNCERTAIN`
- `future_motion`：`TOWARD_CAMERA`、`AWAY_FROM_CAMERA`、`STATIONARY`、`UNCERTAIN`

目前prompt沒有要求`LATERAL_ACROSS_VIEW`，但內部tracking/policy vocabulary仍接受它。

### 已從VLM輸出移除

- `scene_visibility`
- `future_activity`
- `observation_confidence`、`prediction_confidence`
- 長篇`evidence`
- VLM current `motion_trend`
- workspace `MAP` / `future_event`
- `STOPPING / WAITING`

Schema仍可讀部分legacy欄位以便舊log和測試相容，但新prompt不再要求它們。不要因為schema內
仍有legacy normalization就誤以為它們是現行模型contract。

## 4. NOW、NEXT與deterministic motion

NOW顯示：

- body orientation：目前只由VLM判斷；SAM3沒有orientation能力。
- body posture：由VLM判斷。
- current activity：由VLM判斷。
- current motion：預設由timestamp-matched bbox change的tracking hint提供。

NEXT顯示：

- VLM先回答`future_motion`。
- `motion_source: tracking_hint`時，policy採用的motion會由OLD/MID/NOW bbox高度與水平位移支持。
- tracking hint不可靠時採用`UNCERTAIN`。
- `SITTING`且activity不是`WALKING / CHANGING_POSTURE`時，現在與未來motion強制為
  `STATIONARY`，避免坐著工作被bbox抖動判為接近。

這些motion都相對相機：

- `TOWARD_CAMERA`：bbox尺度增加，朝機器人相機方向接近。
- `AWAY_FROM_CAMERA`：bbox尺度縮小。
- `LATERAL_ACROSS_VIEW`：主要是橫向位移。
- `STATIONARY`：尺度與水平位移都在門檻內。
- `UNCERTAIN`：時間coverage或變化證據不足。

它們不是pixel速度的直接robot command，也不是workspace座標。

## 5. 走道／地板候選與近場STOP區

### 藍色區域

SAM3每隔幾秒額外使用`floor`prompt，合併mask、做morphological close，保留最大contour並畫成
藍色半透明區域。

```yaml
enable_walkway_detection: true
walkway_prompt: floor
walkway_confidence_threshold: 0.35
walkway_inference_interval_s: 5.0
show_walkway_overlay: true
```

這一層目前只用於研究顯示，不參與action。floor可能包含桌下地面，也可能漏掉被人或家具
遮住的區域，因此不能直接叫做可通行free space。

### 紅色區域

目前左右相機使用相同初始梯形：

```yaml
enable_approximate_stop_zone: true
approximate_stop_foot_overlap_ratio: 0.05
left_approximate_stop_polygon_normalized:  [0.20, 1.00, 0.80, 1.00, 0.635, 0.76, 0.365, 0.76]
right_approximate_stop_polygon_normalized: [0.20, 1.00, 0.80, 1.00, 0.635, 0.76, 0.365, 0.76]
```

座標原點在影像左上，右下接近`(1,1)`。頂點順序是bottom-left、bottom-right、top-right、
top-left。

進入條件採OR：

1. mask腳底點位於紅色polygon內；或
2. person mask最底部15%與紅區重疊比例大於等於
   `approximate_stop_foot_overlap_ratio`。

第二項是為了處理單腳跨界、footpoint median剛好落在邊界外、ID切換造成腳底中心跳動等問題。
輸出可在`tracks`tab檢查：

```text
inside_approximate_stop_zone
stop_zone_foot_overlap_ratio
footpoint_uv
```

紅區不依賴floor mask，避免floor segmentation暫時失敗時失去STOP能力。

### 重要限制

`APPROX 1m STOP`只是影像中的近似名稱，`approximate_stop_zone_metric=false`。要得到真實1公尺：

1. 在左右相機前實際量出多個地面點；
2. 建立每台相機的image-footpoint到共同ground frame homography，或使用depth；
3. 評估不同距離、人物高度、遮擋與相機振動下的距離誤差；
4. 通過測試後才可啟用metric zone/TTE policy。

## 6. 時間語意：不要把VQA evidence當live畫面

GUI上方NOW與下方storyboard是某一輪已完成推論的輸入證據。NOW/NEXT文字與它們屬於同一輪。
右側Action則使用最新tracking report，兩者可以相差很多秒。

主要時間欄位：

- `pipeline_ms`：VQA node從組圖、prompt、worker round trip、parse到結果完成的總時間。
- `round_trip_ms`：HTTP送到worker直到收到文字回答的時間。
- `semantics_age_s`：VQA結果抵達policy後，到現在已經過多久；不是推論時間。
- `displayed evidence age ≈ pipeline_ms / 1000 + semantics_age_s`。
- `geometry_age_s`：最新tracking/geometry結果抵達policy後的年齡。
- benchmark的`camera_to_vlm_ms`：最接近camera capture到VLM結果的端到端latency。

實際觀察過的例子：

```text
pipeline = 13.38 s
VQA result age = 6.72 s
displayed evidence age ≈ 20.10 s
```

這時不能用左側20秒前的紅區畫面質問「目前為何不是STOP」；應查看同一decision時刻之前最新的
`person_tracks`與`geometry_prediction`。反過來，如果最新geometry的
`inside_approximate_stop_zone=true`但action仍不是STOP，才是policy bug。

近期實測一個人進區的時間序列：

```text
1786024184.87  舊geometry仍為CONTINUE
1786024185.52  新SAM3 report：L14 inside_stop=true
1786024185.67  policy輸出STOP
```

約0.8秒反應時間主要受SAM3 cycle限制。設定`inference_hz=2.0`不代表一定有2 FPS；若一次模型
推論約1秒，實際更新率仍約1 Hz。floor prompt更新輪還會額外增加工作。

## 7. Storyboard與inference mode

合法組合：

| storyboard | inference | 用途 | 主要代價 |
|---|---|---|---|
| `hybrid` | `joint` | 比較模式；全景NOW加per-ID crops，一次request | 多人容易混ID或複製答案 |
| `track_crops` | `per_track` | 每個ID獨立request，適合精度對照 | latency約隨人數增加 |
| `global` | `joint` | 純全景時間序列ablation | 遠處人物細節不足 |
| `track_crops` | `joint` | crop-only多人joint ablation | 仍可能collapse |

`storyboard_mode: joint`是無效設定，會讓VQA node在啟動時退出。`joint`只能放在
`vqa_inference_mode`。`per_track`必須搭配`storyboard_mode: track_crops`。

目前推薦速度設定：

```yaml
storyboard_mode: hybrid
vqa_inference_mode: joint
max_new_tokens: 512
```

目前推薦精度診斷設定：

```yaml
storyboard_mode: track_crops
vqa_inference_mode: per_track
per_track_max_new_tokens: 128
```

## 8. Joint多人輸出collapse

LLaVA 7B在一張圖同時處理多人時，曾把多個不同姿勢的人全部回答為同一組
`standing / working / uncertain`。這不是parser把答案複製，而是raw model output已經相同。

目前：

```yaml
enable_collapse_detection: false
collapse_min_identical_tracks: 3
```

開啟collapse detection只產生`quality_warnings`診斷，不改變confidence或action。辦公室多人本來
就可能同為坐著工作，因此「相同答案」不一定是錯誤；研究時應和人工標註、crop清晰度與raw VLM
回答一起判讀。

建議實驗：

1. 固定同一批影像與ID。
2. 比較`hybrid+joint`與`track_crops+per_track`。
3. 記錄JSON validity、per-ID macro F1、collapse rate、latency與GPU memory。
4. 分開報告「真實多人同狀態」和「不同姿勢卻被複製」兩種情況。

## 9. Confidence的現況

VLM不再輸出自評confidence，因為小模型的百分比沒有可靠校正基礎。

內部仍可能看到外層`confidence`或`confidence_control`，它只表示：

- OLD/MID/NOW crop coverage；
- tracking是否可靠；
- legacy schema相容用neutral base。

在目前`future_behavior_policy`中，這個品質值是diagnostic，不改變action。不要在論文中把它稱為
「模型預測正確率」或「VLM信心」。若要研究可校正confidence，應以held-out validation set做
reliability diagram、ECE、Brier score或conformal prediction，而不是要求模型自己猜百分比。

## 10. 已修正問題清單

- 移除`scene_visibility`、`future_activity`、MAP、STOPPING/WAITING。
- 修正`STANDING`、`CONTINUE_WORKING`、`TOWARD_CAMERA`等值放錯欄位造成schema warning；
  新prompt使用嚴格封閉詞彙，schema保留部分legacy normalization。
- 無person時明確`NO_RESPONSE`，不因沒有fresh VQA誤判warning。
- 坐著且沒有姿勢轉換時固定`STATIONARY`。
- quality/collapse warning不再直接蓋掉安全motion action。
- STOP紅區優先於VQA semantics。
- 紅區判斷由footpoint單點擴充為footpoint或腳部mask overlap。
- UI明確標示VQA evidence不是live，顯示pipeline/result/evidence timing。
- `storyboard_mode`與`vqa_inference_mode`啟動時驗證，避免把`joint`填錯欄位後靜默運作。
- Transformers同時設定`max_new_tokens`與checkpoint legacy `max_length`的warning已在worker側處理；
  response長度以`max_new_tokens`為準。
- 曾發生多組重複policy node導致action閃爍；啟動新launch前必須先停止舊launch，並確認topic只有
  一個正式publisher。

## 11. 目前已知風險

### P0：安全與時間

1. **SAM3不是高頻安全sensor**：目前約1秒一輪，進區到STOP可能延遲約1秒以上。
2. **沒有STOP hysteresis/latch**：短暫漏偵測或ID切換仍可能讓STOP解除；foot overlap只改善
   邊界漏判，沒有完全解決時序抖動。
3. **semantics freshness依receipt time**：VQA完成時輸入影像已經老了`pipeline_ms`；目前
   `semantics_timeout_s`沒有直接限制capture-to-now evidence age。
4. **current ID與舊semantics ID可能不同**：policy motion branch可使用VQA metadata中的舊
   `input_tracks`；紅區STOP使用current geometry，但CONTINUE/DANGER的語意仍可能落後ID切換。
5. **紅區不是metric**：不能宣稱符合ISO安全距離或真實1公尺。

### P1：感知

1. SAM3可能漏掉極近、被裁切、遮擋或低confidence人物。
2. camera-local tracker沒有appearance Re-ID，交叉、遮擋與離場重入會換ID。
3. `floor`prompt不是free-space估計，也沒有障礙物扣除。
4. orientation完全由VLM判斷，沒有pose/keypoint/head-pose hint。
5. bbox尺度會受到坐下、彎腰、手臂動作影響，不能等同深度變化。

### P2：VLM

1. 7B joint多人grounding不穩，可能複製答案。
2. VLM future prediction缺少校正概率與ground-truth horizon定義。
3. `WORKING`在辦公室容易成為模型預設答案。
4. compact輸出降低token，但不保證語意準確。

## 12. 建議下一步研究順序

### 第一階段：把STOP做成可量測的current perception

1. 新增獨立高頻person detector或輕量pose detector；SAM3留作mask refinement。
2. 新增STOP enter/exit hysteresis，例如進區1幀立刻STOP、連續3個fresh reports確定離區才解除。
3. 記錄`camera stamp → person report → geometry → action`四個時間，報P50/P95/P99。
4. 建立腳跨界資料集，標註first-foot-entry frame，量STOP recall與反應時間。
5. 將GUI增加真正live preview，與timestamped VQA evidence分欄顯示。

### 第二階段：真實地面與1公尺區

1. 每台相機收集至少4個不共線地面控制點，實務上建議更多點做RANSAC與held-out error。
2. 將image footpoint投影到robot base frame。
3. 以機器人外形做inflation，建立0–1 m hazard polygon與外層warning polygon。
4. 驗證左右相機投影到共同座標的一致性。
5. 完成後才設定`calibration_valid=true`、`enable_zone_logic=true`並研究
   `zone_safety_policy`。

若可使用RealSense depth，應比較RGB homography、depth footpoint與兩者fusion的距離誤差。

### 第三階段：姿勢與面向

1. 加入2D pose/keypoint模型，使用肩、髖、膝關節估計standing/sitting/crouching。
2. 使用肩線、臉部可見性或head-pose模型產生orientation hint。
3. VLM只處理activity與模糊情境，deterministic pose layer處理可直接量測的姿勢。
4. 分別報告body orientation與head/gaze，不混為同一標籤。

### 第四階段：VLM與temporal ablation

至少比較：

- NOW-only vs OLD/MID/NOW；
- global vs hybrid vs track crops；
- joint vs per track；
- tracking hint on/off；
- LLaVA 7B vs候選較強VLM；
- compact JSON vs較完整語意輸出；
- sitting stationary rule on/off。

## 13. 評估指標

### Detection與tracking

- person precision/recall、mask IoU
- footpoint pixel error
- IDF1、ID switches、fragmentation
- current red-zone entry precision/recall
- STOP response latency P50/P95/P99
- STOP flicker count與unsafe false negative duration

### VLM

- posture/activity/orientation/future-motion macro F1
- 各類precision/recall與confusion matrix
- JSON validity、missing/duplicate ID rate
- `UNCERTAIN` precision/recall
- output collapse rate
- per-ID consistency across adjacent cycles

### 系統

- SAM3 `inference_ms`
- VQA `pipeline_ms`、worker `round_trip_ms`
- `camera_to_tracks_ms`、`camera_to_vlm_ms`、`vlm_to_policy_ms`
- GPU utilization、VRAM、OOM rate
- action confusion matrix，尤其unsafe CONTINUE false negatives

## 14. Benchmark與實驗記錄

Live pipeline預設logger輸出：

```text
/workspace/results/future_risk_vqa/YYYYmmdd_HHMMSS_<run_name>/
├── manifest.json
├── events.jsonl
└── storyboards/
```

完整latency benchmark：

```bash
cd /workspace/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 /workspace/sam3_benchmark/future_risk_benchmark/benchmark_node.py \
  --track-samples 100 \
  --vlm-samples 10 \
  --warmup-tracks 5 \
  --warmup-vlm 1 \
  --timeout-s 600 \
  --sample-gpu \
  --output-root /workspace/sam3_benchmark/results/future_risk \
  --run-name EXPERIMENT_NAME
```

要拆worker內部GPU stage，重啟worker時加入：

```text
enable_latency_timing:=true
```

它會插入CUDA synchronize，只在benchmark使用，完成後改回`false`。

每個實驗至少記錄：

- Git commit或working-tree patch識別
- config完整副本
- model ID/revision/quantization
- camera topic、解析度、FPS
- 場景、人數、照明與遮擋
- warmup與sample count
- output directory與人工標註版本

## 15. 常用重現與診斷命令

建置：

```bash
cd /workspace/ros2_ws
colcon build --packages-select future_risk_vqa --symlink-install
source install/setup.bash
```

啟動live UI：

```bash
ros2 launch future_risk_vqa future_risk_live_ui.launch.py
```

確認模式：

```bash
ros2 param get /future_vqa storyboard_mode
ros2 param get /future_vqa vqa_inference_mode
ros2 param get /future_vqa motion_source
```

只取一次current report：

```bash
ros2 topic echo /future_risk/person_tracks std_msgs/msg/String --once --field data
ros2 topic echo /future_risk/geometry_prediction std_msgs/msg/String --once --field data
ros2 topic echo /future_risk/decision std_msgs/msg/String --once --field data
```

確認publisher數量，避免重複節點造成閃爍：

```bash
ros2 topic info /future_risk/decision --verbose
ros2 node list
```

修改YAML後要重新啟動既有launch。不要在舊launch仍執行時再疊加一組測試節點。

## 16. 程式位置

| 功能 | 檔案 |
|---|---|
| compact prompt | `future_risk_vqa/prompt.py` |
| enum、legacy normalization與schema validation | `future_risk_vqa/schema.py` |
| OLD/MID/NOW與hybrid storyboard | `future_risk_vqa/storyboard.py` |
| SAM3 person/floor、tracking、紅區判斷 | `future_risk_vqa/sam3_person_tracker_node.py` |
| footpoint、polygon與腳部mask overlap | `future_risk_vqa/geometry.py` |
| ground projection與Kalman | `future_risk_vqa/ground_trajectory_node.py` |
| tracking motion與sitting rule | `future_risk_vqa/semantic_quality.py` |
| VQA scheduling、client與metadata | `future_risk_vqa/future_vqa_node.py` |
| motion decision mapping | `future_risk_vqa/policy.py` |
| current紅區優先序與policy node | `future_risk_vqa/future_policy_node.py` |
| UI evidence/action timing | `future_risk_vqa/research_gui.py` |
| UI文字解釋 | `future_risk_vqa/ui_presenter.py` |
| 主要Isaac/live參數 | `config/future_risk_isaac.yaml` |
| 影片demo參數 | `config/future_risk_video_demo.yaml` |
| latency benchmark | `/workspace/sam3_benchmark/future_risk_benchmark/` |

## 17. 接手時先回答的檢查問題

1. 目前畫面是live preview還是timestamped VQA evidence？
2. Action使用的geometry stamp與VQA evidence stamp相差多少？
3. `inside_approximate_stop_zone`和`stop_zone_foot_overlap_ratio`是多少？
4. 當輪SAM3是否漏人或發生ID switch？
5. raw VLM是否已經複製答案，還是parser/schema造成？
6. 實驗要最佳速度還是最佳per-ID準確度？
7. 現在使用的是image-space近似區還是真實metric zone？
8. 是否只有一組正式pipeline與一個decision publisher？

只有把上述時間、資料來源與實驗條件固定下來，後續結果才可以比較與重現。
