# future_risk_vqa

現行雙相機人物時序風險研究套件：SAM3 person／floor → camera-local tracking → OLD/MID/NOW crops → LLaVA／RoboPoint → deterministic policy → 研究 GUI 與 logger。風險提示未直接接入機器人控制。

## 目前實作

- 左右相機各自保留 `L*`／`R*` 人物 ID，不做跨相機 re-ID。
- VLM 描述 body orientation、posture、current activity 與 future camera-relative motion；精確輸出 contract 在 `future_risk_vqa/prompt.py` 和 `schema.py`。
- source YAML 目前使用 SAM3 **CPU**、`track_crops + per_track`、`motion_source: tracking_hint`；VLM 原始方向保存在 audit 欄位。`hybrid + joint` 可作比較。
- policy 預設 `future_behavior_policy`；影片設定開啟近場 image-space STOP，live／Isaac 設定關閉此區。兩者未校正的 metric zone／TTE 均關閉。
- 紅區是近似影像區域，不代表已知一公尺。`normalized_debug` 也不是公尺座標。
- 本地 13B checkpoint 已有單圖測試；9 月 CPU SAM3 首輪約 57 秒且 VQA 解析失敗，完整即時 pipeline 尚未通過驗證。

## 使用與閱讀

| 需求 | 文件 |
|---|---|
| 建置、模型與影片／live／Isaac 啟動 | [專案使用指南](../../../docs/USAGE.md) |
| 讀程式與資料流 | [架構導覽](../../../docs/ARCHITECTURE.md) |
| 調參、功能開關與 launch 覆蓋 | [CONFIGURATION](docs/CONFIGURATION.md) |
| ROS topics／JSON | [SCHEMA](docs/SCHEMA.md) |
| GUI、schema warning 與 evidence 時間 | [GUI_GUIDE](docs/GUI_GUIDE.md) |
| 幾何標定 | [CALIBRATION](docs/CALIBRATION.md) |
| 研究限制與後續實驗 | [RESEARCH_CONTINUATION](docs/RESEARCH_CONTINUATION.md) |
| X9 跨機相機與 RViz | [Camera LAN](image/README_Camera_LAN.md) |
| 開發、測試與 Git 更新 | [workflow](../../../docs/WORKFLOW.md) |

## 啟動入口

| Launch | 功能 |
|---|---|
| `future_risk_video_demo.launch.py` | 雙影片、pipeline、可選 GUI／logger；影片路徑由 YAML 決定 |
| `future_risk_live_ui.launch.py` | 訂閱既有 live topics；預設 compressed，使用 wall time |
| `future_risk_isaac.launch.py` | 訂閱 Isaac topics；simulation time，需要 `/clock` |
| `future_risk_pipeline.launch.py` | 四個核心節點，不開來源、GUI 或 logger |
| `future_risk_realsense_cameras.launch.py` | 啟動雙 RealSense driver；需依序號指定相機 |

不要只啟動 pipeline 就期待模型與相機出現；controller、worker 與影像來源要另外啟動。`model_path` 是 checkpoint 位置，`model_name` 是 controller 的註冊名稱。

## 程式入口

`setup.py` 列出 console scripts；`launch/` 組合節點；`config/` 是 source 設定。核心實作在 `future_risk_vqa/`：`tracking.py`、`geometry.py`、`motion.py`、`storyboard.py`、`prompt.py`、`schema.py`、`model_client.py`、`policy.py` 及對應 `*_node.py`。從專案根目錄執行 `bash scripts/check.sh`；完整 ROS node 測試用 `bash scripts/check.sh --ros`。
