# 新 checkpoint 測試與研究建議

實際 checkpoint：`robopoint_runtime/runs/robopoint_vqa_only_317677`。
使用者提供的 `robopoint_vqa_only_3176775k` 不存在。config 是 Vicuna/LLaVA 13B + CLIP ViT-L/14-336，完整分片權重；trainer_state 記錄 global_step=3824、epoch=1.0，不能把目錄名稱當成實際訓練步數。

## 2026-09-17 實測

- 六個 safetensors 分片的 header/data 長度一致；這是結構檢查，不是來源 checksum 比對。
- 現有 vlpoint-inference 容器：Torch 2.11.0+cu128、Transformers 5.13.1、RTX 5070 12GB。
- 4-bit、SAM3 conditioning 關閉，載入 19.27 秒。測試期間整張 GPU 用量一次快照為 9142 MiB（包含桌面）。
- `example/image/image (14).png` 是便利商店櫃檯場景。
- 描述請求 1.152 秒：`A man is standing in front of a store counter with a laptop.` 櫃檯設備被說成 laptop，屬於需標註核對的視覺錯誤。
- 簡單 JSON 請求 1.796 秒：輸出文字清單，strict JSON 不通過。
- 使用既有 risk_vqa prompt + future_risk_vqa controller client：4.281 秒，HTTP 成功，但輸出 `time\_to\_impact\_s`，包含 JSON 不合法跳脫，strict JSON 不通過。已用既有 risk_vqa parser 修復並通過 validate_report（validation_errors=[]）；詳見 smoke_317677/parser_result.json。必須區分 raw format success 與 repaired success。
- 以上是單張圖片 smoke test；不是完整 SAM3/ROS 相機/Isaac pipeline 測試，也不能用來推估準確率或 p95 latency。
- 測試後已停止本次 worker/controller 並釋放 GPU；沒有修改原有 launch/YAML 預設。
- 原始結果：`smoke_317677/results.json`、`smoke_317677/system_prompt_result.json`。

## 啟動方式

每個 terminal 先進入容器並載入環境：

```bash
docker exec -it vlpoint-inference bash
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

Terminal 1：

```bash
ros2 launch vlpoint controller.launch.py
```

Terminal 2：

```bash
bash /workspace/robopoint_runtime/test_trained_worker.sh
```

上述腳本相當於：

```bash
ros2 launch vlpoint worker.launch.py \
  model_path:=/workspace/robopoint_runtime/runs/robopoint_vqa_only_317677 \
  quantization:=4bit mm_use_sam3_conditioning:=false \
  limit_model_concurrency:=1 enable_latency_timing:=true
```

model_path 是 checkpoint 路徑；model_name 是 controller 的註冊名稱，不能填檔案路徑。worker.launch.py 與 vlpoint.launch.py 的預設模型不同，建議明確傳入 model_path。
CLIP 是這份 checkpoint 記錄的視覺編碼器；先關閉額外 SAM3 conditioning 建立 baseline，外部 SAM3 人物追蹤是另一個功能。

Terminal 3 擇一：

```bash
# 既有雙影片 demo
ros2 launch future_risk_vqa future_risk_video_demo.launch.py \
  model_name:=robopoint_vqa_only_317677

# Isaac Sim 必須已開始發布相機影像與 /clock
ros2 launch future_risk_vqa future_risk_isaac.launch.py \
  model_name:=robopoint_vqa_only_317677 \
  left_image_topic:=/left_camera/rgb right_image_topic:=/right_camera/rgb

# 現場雙相機，topic 需符合現有相機實際輸出
ros2 launch future_risk_vqa future_risk_live_ui.launch.py \
  model_name:=robopoint_vqa_only_317677
```

檢查註冊：`curl -s -X POST http://127.0.0.1:11000/list_models`。
若重新執行 smoke scripts，worker 需在 22001，controller 需在 11000；正常啟動腳本預設 worker 是 22000。

## 調參與研究設計

- Isaac／live UI 基礎設定：`ros2_ws/src/future_risk_vqa/config/future_risk_isaac.yaml`。
- 影片設定：`ros2_ws/src/future_risk_vqa/config/future_risk_video_demo.yaml`。
- 單相機風險系統：`ros2_ws/src/risk_vqa/config/risk_vqa.yaml`。
- 模型評估必須分開 `motion_source: vlm` 與 `tracking_hint`；預設 tracking_hint 使用 bbox 變化提供方向，不能把系統方向分數全歸功於 VLM。
- 比較 global、hybrid+joint、track_crops+per_track，固定影片與 sampling；記錄姿態/activity/motion macro-F1、每個 ID 的正確率、JSON 原始/修復成功率、p50/p95 latency、峰值 VRAM。
- 12GB 上 13B 與 SAM3/Isaac 同時執行尚未驗證；先逐個量測，再決定拆 GPU 或比較較小模型。

優先方向：

1. 場域 SFT：便利商店排隊、結帳、拿取/放回商品、補貨、蹲下、遮擋與多人交叉。用部署時相同 storyboard 和 JSON schema 做 instruction tuning；加入 UNCERTAIN 與無事件反例。依門市/人物/錄影 session 切 train/test，避免相鄰影格洩漏。
2. 時序預測：輸入 t-2/t-1/t 影格，標註未來 1/2/3 秒實際事件；資料生成時不能把未來影格或未來 GT 放進輸入 context。比較單張與時序、VLM 與 tracking、兩者融合。
3. Isaac sim-to-real：用 Replicator 隨機化光照、相機、人物/遮擋與貨架配置；記錄 ID、位置與未來軌跡建立標籤。比較 real-only、sim-only、sim+real；最後在未見過的真實門市測試。
4. 幾何與語意融合：深度/校正投影估計距離與碰撞時間，VLM 判斷活動；測漏報、每小時誤報、預警提前量。現有 calibration_valid=false 和 identity homography 是 placeholder，不能當公尺 GT。
5. 防止遺忘：若還要 RoboPoint 點選/操作能力，混合空間點位與便利商店 VQA 訓練，分別量測 point accuracy 與 VQA；資料夾叫 vqa_only 不足以證明能力有無退化。

建議主題：「跨模擬與真實便利商店的目標時序 VQA 與幾何融合風險預測」。先用原模型 vs 新模型建立 baseline，再做時序/融合/sim-to-real ablation。

參考：
- RoboPoint：https://arxiv.org/abs/2406.10721
- RoboSpatial：https://arxiv.org/abs/2411.16537
- NVIDIA Isaac Sim：https://developer.nvidia.com/isaac/sim/
- Replicator 合成資料：https://developer.nvidia.com/blog/generating-synthetic-datasets-isaac-sim-data-replicator/
