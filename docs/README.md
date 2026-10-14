# 文件索引

現行入口依 2026-10-14 checkout 整理；操作設定以 source YAML、launch 與程式為準。單篇文件寫得較晚，並不代表其中每個舊的測試結果都重新驗證過。

## 建議閱讀順序

1. [專案首頁](../README.md)：知道系統目的與各套件角色。
2. [使用指南](USAGE.md)：選擇單圖、雙影片、live 相機或 Isaac 模式。
3. [架構與程式導覽](ARCHITECTURE.md)：沿 topic 與函式閱讀資料流。
4. [開發與更新 workflow](WORKFLOW.md)：修改、測試、提交與子模組更新。
5. [盤點與整理紀錄](MAINTENANCE.md)：確認遠端狀態、歷史限制和本次驗證範圍。

## 各主題的唯一入口

| 主題 | 維護文件 |
|---|---|
| Future risk 套件 | [README](../ros2_ws/src/future_risk_vqa/README.md) |
| 功能開關、參數與 launch 覆蓋 | [CONFIGURATION](../ros2_ws/src/future_risk_vqa/docs/CONFIGURATION.md) |
| ROS topic／JSON contract | [SCHEMA](../ros2_ws/src/future_risk_vqa/docs/SCHEMA.md)，精確驗證邏輯見 `schema.py` |
| GUI 與時間語意 | [GUI_GUIDE](../ros2_ws/src/future_risk_vqa/docs/GUI_GUIDE.md) |
| 公尺座標標定 | [CALIBRATION](../ros2_ws/src/future_risk_vqa/docs/CALIBRATION.md) |
| 研究假設、已知風險與待辦 | [RESEARCH_CONTINUATION](../ros2_ws/src/future_risk_vqa/docs/RESEARCH_CONTINUATION.md)，含 8 月交接背景 |
| X9 跨機相機與 RViz | [Camera LAN](../ros2_ws/src/future_risk_vqa/image/README_Camera_LAN.md) |
| 新 checkpoint 單圖與 worker | [Runtime README](../robopoint_runtime/README.md) |
| 9 月單圖測試證據 | [TEST_TRAINED_MODEL](../robopoint_runtime/TEST_TRAINED_MODEL.md) |
| 9 月 CPU SAM3 試驗 | [SAM3_CPU_TRIAL](../robopoint_runtime/SAM3_CPU_TRIAL.md) |
| Future risk 系統 benchmark | [benchmark 指南](../sam3_benchmark/future_risk_benchmark/README.md)，目前仍有本地未提交內容 |
| 單影片基準 | [risk_vqa](../ros2_ws/src/risk_vqa/README.md) |
| 雙相機 zone 基準 | [risk_vqa_sim](../ros2_ws/src/risk_vqa_sim/README.md) |
| 選用語音 | [audio README](../ros2_ws/src/audio_listener_worker/README.md) 與 [整合測試](../ros2_ws/src/audio_listener_worker/INTEGRATION_TEST.md) |

## 研究與封存

[Foundation VLM 計畫](research/SAM3_FOUNDATION_VLM_PLAN.md) 是仍有參考價值的未實作提案，與現行部署分開閱讀。

[2026-08-07 SAM3 encoder 設計稽核](archive/SAM3_VLM_MODEL_DESIGN_20260807.md) 已被上述規劃取代，僅作歷史參考；外部模型規格、測速與當時硬體結論沒有在本次整理中重測。上游 `sam3/`、音訊套件的授權、貢獻指南與 changelog 原地保留。

更新文件時，改對應主題的維護文件並從此處連結；實驗結果保存日期、環境與產物路徑，避免再建立 `SUMMARY`、`FINAL`、`NEW` 等互相重複的操作指南。
