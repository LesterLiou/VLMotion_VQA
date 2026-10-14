# RoboPoint runtime

這裡保留本地新 checkpoint 的可重複操作入口。模型在 `runs/robopoint_vqa_only_317677/`，是 Vicuna／LLaVA 13B + CLIP；權重、虛擬環境、快取與 smoke 輸出均不提交到 Git。

| 檔案 | 用途 |
|---|---|
| `run_single_image.sh`／`single_image_vqa.py` | 單圖問答與 `--interactive` 多輪模式；從主機轉入既有容器，或直接於容器執行 |
| `test_trained_worker.sh` | 將 checkpoint 作為現有 VLPoint worker 啟動；`ROBOPOINT_CHECKPOINT` 可換位置 |
| [SINGLE_IMAGE_VQA](SINGLE_IMAGE_VQA.md) | CLI 操作詳解 |
| [TEST_TRAINED_MODEL](TEST_TRAINED_MODEL.md) | 2026-09-17 權重結構檢查、單圖實測與研究方向，非最新重新測速 |
| [SAM3_CPU_TRIAL](SAM3_CPU_TRIAL.md) | 2026-09-17 CPU SAM3 試驗的實際延遲與失敗結果 |

先看[專案使用指南](../docs/USAGE.md)選擇單圖或完整 pipeline；更新方式見 [workflow](../docs/WORKFLOW.md)。舊 smoke scripts 與原始 JSON 留在本機 `smoke_317677/` 作實驗證據，未列為維護入口。不要把目錄名稱當訓練步數，也不要把單圖測試視為完整 ROS／相機成功。
