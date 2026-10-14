# VLMotion / Risk VQA

這是一個 ROS 2 視覺語言研究專案：用相機辨識人、分析動作，再產生可追溯的風險提示；也保留原始的語言選點與 Stretch 機器人操作功能。

**目前已實作的研究重點**是 `future_risk_vqa`：左右相機 → SAM3 人物偵測與追蹤 → OLD/MID/NOW 時序影像 → LLaVA／RoboPoint VQA → 確定性規則 → GUI 與實驗紀錄。`STOP` 等輸出是研究提示，風險 pipeline 本身沒有直接連接機器人控制器。

最近保留的模型實測是 2026-09-17 的本地 RoboPoint 13B checkpoint。完整即時 pipeline 尚有延遲、顯存與 JSON 格式問題；不能把單圖測試成功當作整套系統驗證完成。SAM3-native foundation VLM 另有[研究提案](docs/research/SAM3_FOUNDATION_VLM_PLAN.md)，目前沒有對應的 `foundation_vlm/` 實作。

## 從哪裡開始

| 目的 | 閱讀入口 |
|---|---|
| 安裝、單圖測試、影片／實機／Isaac 啟動 | [使用指南](docs/USAGE.md) |
| 理解整體架構、讀程式的順序 | [架構與程式導覽](docs/ARCHITECTURE.md) |
| 修改設定、模型、prompt、測試與更新 Git | [開發與更新 workflow](docs/WORKFLOW.md) |
| 查 Git 遠端、同步狀態、整理決策與限制 | [專案盤點紀錄](docs/MAINTENANCE.md) |
| 選擇詳細文件 | [文件索引](docs/README.md) |

## 目錄與用途

```text
VLMotion/
├── ros2_ws/src/
│   ├── future_risk_vqa/       # 現行雙相機、人物時序與未來風險研究
│   ├── vlpoint/               # LLaVA/RoboPoint controller、worker 與模型實作
│   ├── risk_vqa/              # 單影片 YOLO/TTC + VLM，比較基準
│   ├── risk_vqa_sim/          # 雙相機 temporal VQA + zone 規則，比較基準
│   ├── white_point_pipeline/  # 語言選點、3D 轉換與機器人運動
│   ├── vlservo/               # Stretch 視覺伺服與操作 GUI
│   ├── audio_listener_worker/ # 遠端語音輸入（選用）
│   └── audio_common/          # 音訊與訊息來源碼（選用，含 ROS1 歷史元件）
├── robopoint_runtime/         # 本地新 checkpoint 的單圖／worker 測試入口
├── sam3/                     # SAM3 Git 子模組；風險 tracker 使用的是 HF loader
├── sam3_benchmark/           # 系統與邊緣裝置 benchmark Git 子模組
├── docker/                  # ROS 2 Humble / CUDA 開發環境
├── scripts/                 # 靜態檢查、CPU/ROS 測試與唯讀 Git 稽核
├── docs/                    # 現行指南、研究提案、封存參考
├── example/                 # 範例圖片；影片需另備
└── isaac/                   # 本機 USD 場景，未納入 Git
```

## 快速使用

Ubuntu 22.04／ROS 2 Humble／Python 3.10 是部署基礎；模型推論需要 CUDA 環境與另外準備的權重。已有容器與本地 checkpoint 時，可先做最小單圖測試：

```bash
bash robopoint_runtime/run_single_image.sh \
  --image 'example/image/image (14).png' \
  --question 'What is the person doing?'
```

完整建置與多 terminal 啟動見[使用指南](docs/USAGE.md)。CPU 開發檢查：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
bash scripts/check.sh
```

Git 連線與最新 commit 查詢（不做 fetch、pull 或 checkout）：

```bash
python3 scripts/audit_git.py --remote
```

主程式 origin：`git@github.com:LesterLiou/VLMotion_VQA.git`；原始 upstream：`git@github.com:tsun-chuan123/VLMotion.git`。目前開發分支為 `LesterVQA`；子模組需各自提交，再更新主專案的 commit 指標，詳見 [workflow](docs/WORKFLOW.md)。
