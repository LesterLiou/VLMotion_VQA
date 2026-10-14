# 專案盤點與整理紀錄

盤點日期：2026-10-14（Asia/Taipei）。這是目前本機 checkout 的紀錄，不代表尚未提交的內容已發布到 GitHub。

## Git 遠端與版本

使用 `git ls-remote --heads` 即時查詢下列遠端；沒有 pull、checkout、reset、commit 或 push。

| Repository／分支 | 遠端 | 本地／遠端狀態 |
|---|---|---|
| root `LesterVQA` | `git@github.com:LesterLiou/VLMotion_VQA.git` | `ead552a` 與遠端 `LesterVQA` 一致，工作目錄含既有未提交／已暫存內容 |
| root `main` | 追蹤 `git@github.com:tsun-chuan123/VLMotion.git` 的 `main` | `752bd45` 與 upstream/main 一致；你的 origin/main 是另一個 `6accebb`，兩者不可混用 |
| root `sam3.1` | 同一 upstream 的 `sam3.1` | `8a78399` 與遠端一致；不是本次 checkout 的開發分支 |
| `sam3/` 的 `main` | `git@github.com:LesterLiou/sam3.git` | `5dd401d` 與遠端 main 一致，盤點前乾淨；遠端另有 `lester` 分支 `970db62` |
| `sam3_benchmark/` 的 `VQA` | `git@github.com:LesterLiou/sam3_benchmark.git` | `b376eec` 與遠端 main 一致；本地 VQA 沒 upstream，遠端已無 VQA；另有 thor `8cab575` |
| `sam3_benchmark/sam3` | 同一 SAM3 repository | 宣告的 nested submodule，尚未初始化；鎖定 `5dd401d`，沒有獨立可檢查的 working tree |

「分支 SHA 一致」不表示本機所有工作已上傳，也不表示其他研究分支沒有變化。root 的 `.gitmodules` 已把 benchmark 的 remote-update branch 改成 `main`；未改本地 VQA 分支、upstream 或 checkout。SAM3 子模組的加入本來就在 staged 狀態，原樣保留。

需重查時執行 `python3 scripts/audit_git.py --remote`。腳本唯讀列出各 repository 的 remotes、本地分支、dirty state、即時遠端 SHA，以及未初始化子模組；不連網時不會宣稱版本最新。

## 本次整理內容

- 重寫首頁，新增文件索引、架構、使用與 workflow，將現行風險研究、比較基準、機器人選點／伺服和未實作模型計畫分清楚。
- Foundation VLM 主計畫移到 `docs/research/`；被取代的 encoder 設計稽核移到 `docs/archive/`，保留有用的歷史推理與限制，明確標記非部署指南。
- `future_risk_vqa` README 改成清楚入口；重複的 `FUTURE_RISK_ISAAC_PARAMETERS.md` 整併到 `docs/CONFIGURATION.md`，以目前兩份 YAML 列參數。刪除重複的 audio `INTEGRATION_SUMMARY.md`，保留 README 與整合測試。
- 更正 baseline 定位、CPU／逐 ID 目前設定、GUI STOP 解釋、單圖互動模式與相機文件中缺失產物的描述。
- root／benchmark ignore 補上環境、結果、權重、影片與 ROS 輸出；取消全域 `*.jpg`，改為忽略具體 worker 產圖。Docker build context 同步排除資產、快取、虛擬環境與私密設定。
- Docker 去除重複 USB／udev／DRI 掛載、補 `cv_bridge`、修正未引用的 PyQt5 版本條件，取消隱藏失敗的自動全 workspace build。
- `environment.sh` 依專案位置載入 workspace；docker/run 不再依賴呼叫時 cwd，並驗證 domain；stop 的 X11 撤銷設定對應 run 的授權。
- 重複的根目錄相機連線腳本改為轉呼叫 `image/` 內的唯一實作，維持 source 入口相容。
- 新增 Git 唯讀稽核、靜態／CPU／ROS 測試入口，以及 GitHub Actions CPU workflow。

## 為何保留部分舊程式

`risk_vqa` 與 `risk_vqa_sim` 有各自場景、規則、schema 與測試，仍有 baseline／ablation 價值。VLServo 的舊 demo 被其他模組以動態 import/subprocess 呼叫，不能只憑檔名或沒有 console entry 就判為無用。audio_common、SAM3 的授權、changelog 與上游指南也保留。

本次沒有刪除權重、影片、USD、實驗證據或既有 build/install，亦沒有移動 ROS package/import 路徑。大容量本機產物改由 ignore 隔離；Git ignore 不等於備份。清理前被取代的本機文件另留在 `/tmp/vlmotion-cleanup-20261014/` 供本次核對，`/tmp` 不是持久備份。

## 驗證與尚存限制

本次驗證結果：

| 項目 | 結果 |
|---|---|
| 靜態檢查 | 191 個 Python、10 個 shell、12 個 YAML 與 59 個維護文件連結通過 |
| CPU 測試 | 116 passed，含本地 benchmark 統計測試；不含需 ROS 的 inference-mode 測試 |
| 完整 ROS 測試 | 118 passed，含 inference-mode 測試；使用既有 ROS 容器，沒有啟動模型或硬體 |
| 選擇性 colcon 建置 | `vlpoint`、`future_risk_vqa` 均成功，build/install/log 放容器 `/tmp/vlmotion-maintenance/`，未覆寫既有 install |
| Launch 定義 | video、live_ui、isaac 三個 `--show-args` 通過；僅驗證載入與參數，不代表 node 實際推論 |
| Docker Compose | `docker compose -f docker/docker-compose.yml config --quiet` 通過 |
| 無子模組 CI 模擬 | `/tmp` 來源快照靜態檢查通過；未初始化的私有子模組連結明確提示 |
| Shell／ignore | 用假的 Docker 命令核對跨 cwd、合法／非法 domain 與相機 wrapper 的 source 行為；實際 Git ignore 範例通過，沒有已追蹤檔案被新規則誤忽略 |
| 差異格式 | root 與 benchmark 的 `git diff --check` 通過 |

本次未重建完整 Docker image，也未啟動模型、相機、GUI 或機器人；9 月實測數字均保留原日期。

現有執行容器曾在測試收集階段因 pytest／anyio 外掛版本不合而失敗；測試入口已停用非必要自動外掛。首次手動測試另因覆寫 PYTHONPATH 丟失 ROS 路徑，使 `cv_bridge` import 失敗；核對後確認套件已安裝，source ROS 並保留 PYTHONPATH 後完整測試通過。Dockerfile 仍明列 cv_bridge 依賴，避免依賴其他套件間接安裝。

依賴仍有未完全鎖定的 Torch／Transformers／SAM3 版本，Docker 全新建置與權重存取尚需實際環境驗證。先記錄已驗證 runtime，再逐步建立可重現的 image／依賴版本；不要任意升級現有模型環境後才嘗試回推差異。

最新研究套件、runtime 工具、docs 與 benchmark 新目錄原本即有不少未追蹤內容；不提交它們，遠端 clone 就仍是舊版。具體提交與子模組發布順序見 [workflow](WORKFLOW.md)。
