# 使用指南

## 選擇執行模式

| 要做的事 | 入口 | 額外需求 |
|---|---|---|
| 單張圖片問答 | `robopoint_runtime/run_single_image.sh` | 本地 checkpoint、CUDA 與 CLIP 快取 |
| 雙影片研究 | `future_risk_video_demo.launch.py` | 兩個本地影片、SAM3、controller／worker |
| 即時雙相機 | `future_risk_live_ui.launch.py` | 已發布的影像 topics，或另開 RealSense driver |
| Isaac Sim | `future_risk_isaac.launch.py` | 外部 Isaac 場景正在發布影像與 `/clock` |
| 純 CPU 邏輯測試 | `bash scripts/check.sh` | 開發依賴；不需權重或 ROS |

本機權重、影片、USD 場景與實驗輸出已被 Git 忽略，新 clone 不會包含它們。`isaac/X9_scene_with_products_nova_carters.usd` 和 `robopoint_runtime/runs/robopoint_vqa_only_317677` 是本機資產，需另外備份／轉移。HF SAM3 使用受權限控制的模型時，另行完成存取與下載；設定 `local_files_only: true` 需要預先有快取。

## 容器與建置

主機在專案根目錄操作。預設服務／容器名稱均為 `vlpoint-inference`：

```bash
# 建置並啟動；已有可用容器時可先略過重建
bash docker/build.sh
# 進入容器；傳入 ROS_DOMAIN_ID，預設 0
bash docker/run.sh 0
```

`build.sh` 實際執行 compose `up -d --build`。`run.sh` 也會確保服務已啟動，並帶入目前 terminal 的 X11 設定。遠端 GUI 需先有可用 DISPLAY／X11 forwarding；headless 模式可加 `enable_gui:=false`。容器採 host network、GPU 與 USB 掛載，供研究環境使用。

每個容器 terminal 都要載入 ROS 與 workspace。首次建置：

```bash
cd /workspace
source environment.sh 0
cd ros2_ws
# 主研究路徑只需要這兩個套件
colcon build --symlink-install --packages-select vlpoint future_risk_vqa
source install/setup.bash
```

比較舊系統時，再將 `risk_vqa risk_vqa_sim` 加進 `--packages-select`。音訊／機器人套件需另備各自依賴；`audio_common` 含歷史 ROS1 元件，避免直接對整個工作區執行無選擇的 colcon build。

Dockerfile 現在只建立環境與複製來源，不再用 `colcon build ... || true` 隱藏失敗。每次建置都應確認真正成功。本機安裝可用同樣的選擇性建置流程，`environment.sh` 會以自身位置找到 workspace；影片、輸出等 `/workspace` 路徑需改成當前機器的絕對路徑。

## 最小模型測試

已有本地 checkpoint 與容器時，從主機執行：

```bash
bash robopoint_runtime/run_single_image.sh \
  --image 'example/image/image (14).png' \
  --question 'What is the person doing?'
# 保持同一個模型載入，連續問問題
bash robopoint_runtime/run_single_image.sh \
  --image 'example/image/image (14).png' --interactive
```

這個模式自行載入 4-bit 模型，不需 controller／ROS／SAM3。先停止會佔用同張 GPU 的 worker 或 pipeline。預設 checkpoint 是 13B、CLIP ViT-L/14-336，不是已完成 SAM3 encoder 替換的模型。CLI 參數見 `--help`；[單圖指南](../robopoint_runtime/SINGLE_IMAGE_VQA.md) 與[日期化測試紀錄](../robopoint_runtime/TEST_TRAINED_MODEL.md) 有更詳細說明。

## 雙影片／相機／Isaac 完整流程

先檢查 source YAML：

- 影片：`ros2_ws/src/future_risk_vqa/config/future_risk_video_demo.yaml`，其中 `future_risk_video_source` 決定左右影片路徑與播放設定。
- Live／Isaac：`ros2_ws/src/future_risk_vqa/config/future_risk_isaac.yaml`，名稱雖有 Isaac，live launch 也使用此設定。

目前 SAM3 在 CPU，逐 ID VQA；這是保留顯存的實驗設定，並未達成即時。9 月測試曾有 57 秒的 CPU 首輪與模型 JSON 解析失敗。要改回 GPU，先分別量測 SAM3、13B worker 與 Isaac 的顯存，再在 YAML 改 `sam3_person_tracker.ros__parameters.device`，不能只提高 `inference_hz`。

以下每個 terminal 先進入容器，執行：

```bash
cd /workspace
source environment.sh 0
source ros2_ws/install/setup.bash
```

Terminal 1，啟動模型路由：

```bash
ros2 launch vlpoint controller.launch.py
```

Terminal 2，選一個 worker。測試本地 checkpoint：

```bash
bash robopoint_runtime/test_trained_worker.sh
```

或明確指定 baseline 模型；權重需已下載或可連網取得：

```bash
ros2 launch vlpoint worker.launch.py \
  model_path:=liuhaotian/llava-v1.5-7b \
  quantization:=4bit mm_use_sam3_conditioning:=false \
  limit_model_concurrency:=1
```

Terminal 3，選一個資料來源。使用本地 checkpoint 時 `model_name` 為 `robopoint_vqa_only_317677`；若換模型，從 controller 查註冊名稱再傳入，名稱不是檔案路徑：

```bash
curl -s -X POST http://127.0.0.1:11000/list_models
```

雙影片（修改 source YAML 後，直接明確傳入，避免讀到 install 裡的舊副本）：

```bash
ros2 launch future_risk_vqa future_risk_video_demo.launch.py \
  config:=/workspace/ros2_ws/src/future_risk_vqa/config/future_risk_video_demo.yaml \
  model_name:=robopoint_vqa_only_317677
```

已經有 live 相機 topics：

```bash
ros2 launch future_risk_vqa future_risk_live_ui.launch.py \
  config:=/workspace/ros2_ws/src/future_risk_vqa/config/future_risk_isaac.yaml \
  model_name:=robopoint_vqa_only_317677 \
  left_image_topic:=/left_camera/color/image_raw/compressed \
  right_image_topic:=/right_camera/color/image_raw/compressed \
  use_compressed_image:=true
```

Live launch 不會自己開相機。雙 RealSense driver 可另開 `ros2 launch future_risk_vqa future_risk_realsense_cameras.launch.py --show-args` 查看序號／名稱參數，再依實際硬體啟動。X9 場景的 topic 可能是 `/camera/left/color/image_raw/compressed`，請先 `ros2 topic list -t`，跨機連線看 [Camera LAN](../ros2_ws/src/future_risk_vqa/image/README_Camera_LAN.md)。

Isaac 必須另外開啟場景與 ROS bridge，再執行：

```bash
ros2 launch future_risk_vqa future_risk_isaac.launch.py \
  config:=/workspace/ros2_ws/src/future_risk_vqa/config/future_risk_isaac.yaml \
  model_name:=robopoint_vqa_only_317677 \
  left_image_topic:=/left_camera/rgb right_image_topic:=/right_camera/rgb
```

Isaac launch 固定 simulation time；live／影片使用 wall time。topic 與 message type 應一起確認，raw 是 `sensor_msgs/msg/Image`，compressed 是 `sensor_msgs/msg/CompressedImage`。

## 觀察、保存與排錯

```bash
ros2 topic echo /future_risk/decision --once
ros2 topic hz /future_risk/person_tracks
ros2 param get /sam3_person_tracker device
ros2 param get /future_vqa vqa_inference_mode
ros2 param get /future_risk_policy decision_mode
```

logger 預設輸出至 `/workspace/results/future_risk_vqa/`，可用 launch 的 `run_name`、`output_root` 改位置。GUI 上方影像是該輪 VQA 使用的 NOW snapshot；不要用較舊畫面判讀新 decision。完整解釋見 [GUI_GUIDE](../ros2_ws/src/future_risk_vqa/docs/GUI_GUIDE.md)。

- `ModuleNotFoundError: cv_bridge`：先 source ROS 並保留既有 PYTHONPATH，再檢查 `python3 -c 'import rclpy, cv_bridge'`。本次容器已有此套件；Dockerfile 也明列 `ros-humble-cv-bridge`。
- 找不到 worker：查 `/list_models`、controller／worker URL 與 model_name。
- 缺 SAM3／CLIP：檢查 HF 快取和 `local_files_only`，權重不由 Git 提供。
- `schema_warning`／`UNAVAILABLE`：保留 raw VLM 回覆與 validation errors，HTTP 成功不代表語意或 JSON 正確。
- 改 YAML 沒生效：用 `config:=` 指向 source，重啟並用 `ros2 param get` 確認。
- OOM／延遲：分開量測各模型；歷史單圖結果不能保證完整 pipeline 可同時運作。

關閉實驗以各 terminal 的 Ctrl+C 停止；離開 shell不會停止容器。要停止 compose 服務可執行 `bash docker/stop.sh`。
