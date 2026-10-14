# 開發與更新 workflow

## 開始修改之前

```bash
git status --short
git diff
git diff --cached
python3 scripts/audit_git.py --remote
```

先辨識本次修改與既有未提交研究內容。主 repository 的 `git diff` 看不到子模組裡的完整檔案差異，還要進去查看 `git -C sam3_benchmark status --short`／`diff`。目前 root 有 staged 的 SAM3 新子模組與既有程式改動；整理不會自動 commit、stash、reset 或 pull。

## 改一個功能時如何更新

| 變更 | 必須一起核對的內容 |
|---|---|
| 相機／播放路徑、裝置、threshold | source YAML、launch 覆蓋、CONFIGURATION、重啟後 `ros2 param get` |
| storyboard／prompt | `storyboard.py`、`prompt.py`、`schema.py`、模式測試、raw JSON success 指標 |
| schema／新欄位 | producer、parser、policy、UI、logger、SCHEMA 與相容性 |
| action 優先序／freshness | `policy.py` 和 `future_policy_node.py`，連同空偵測、過期與重複 heartbeat |
| 模型／checkpoint | worker `model_path`、controller 註冊 `model_name`、量化／SAM3 設定與實驗紀錄 |
| Python／CUDA／ROS 依賴 | Dockerfile、package.xml、對應 runtime 與實際建置驗證 |
| Git 子模組 | 在子 repository 提交／推送，再更新 root 的 gitlink |

YAML 大多只在節點啟動時讀取。開發中直接 `config:=/workspace/ros2_ws/src/...yaml` 最容易確認；使用 install/share 預設時應重新 colcon build，再 source。`--symlink-install` 也不保證所有資料檔都是 symlink。

文件採主題單一入口：總使用方式改 `docs/USAGE.md`，參數改套件 `docs/CONFIGURATION.md`，研究結果保存具體日期與環境。計畫放 `docs/research/`，被取代但有推理價值的資料放 `docs/archive/`，無獨立價值的摘要移除。

## 檢查與測試

建立獨立 CPU 開發環境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
bash scripts/check.sh
```

`check.sh` 先檢查 first-party Python／shell／YAML 語法及維護文件連結，再執行三套風險套件的 CPU 測試。它停用環境內自動載入的 pytest 外掛，並使用 importlib mode 避免不同套件的 `test_schema.py` 同名衝突。CPU 模式明確排除需要 ROS node import 的 `test_future_vqa_modes.py`；有 ROS 時執行完整模式：

```bash
# 在已安裝 cv_bridge 的 ROS 容器內
source /opt/ros/humble/setup.bash
cd /workspace
bash scripts/check.sh --ros
```

只做靜態檢查：`bash scripts/check.sh --static`。變更 ROS 包裝／launch 時另外建置受影響套件並用 `ros2 launch ... --show-args` 驗證；變更模型流程需真實 checkpoint smoke test。純函式測試通過不代表 GPU、相機、GUI 或實際機器人已驗證。

`.github/workflows/checks.yml` 對 push、pull request 和手動執行跑同一套 CPU 檢查，不下載模型，也不要求私有 SSH 子模組。benchmark 測試在本地套件存在時會額外執行；CI 若沒有該子模組會清楚顯示略過。`sam3_benchmark/future_risk_benchmark` 目前尚未提交到遠端，須先發布該子 repository 的程式，才能建立完整 checkout 的跨 repo CI。要更新 CI，就修改 `requirements-dev.txt`、`scripts/check*.py/sh` 與此 workflow；若要新增 ROS／GPU job，應使用有相應依賴的 runner，不能把 CPU 綠燈當成整體整合綠燈。

## 更新主 repository

目前 root 使用 `LesterVQA`，追蹤 `origin/LesterVQA`；`main` 追蹤原作者 `upstream/main`，並不是你的 `origin/main`。先完成可回復的提交或自行保存未提交內容，再更新：

```bash
git fetch origin --prune
git log --oneline --left-right HEAD...origin/LesterVQA
# 僅在目前分支為 LesterVQA、工作目錄乾淨且確定可快轉時
git merge --ff-only origin/LesterVQA
```

原作者更新另外查看 `git fetch upstream --prune` 與 `git log --oneline HEAD..upstream/main`，逐項選擇要整合的變更。SHA 不同不一定表示落後；看提交祖先關係再決定 merge／rebase，避免把不同研究分支混成「全部最新」。

## 更新子模組與提交順序

一般 clone／重現以 root 記錄的 commit 為準：

```bash
git clone --recurse-submodules git@github.com:LesterLiou/VLMotion_VQA.git
cd VLMotion_VQA
git switch LesterVQA
git submodule update --init --recursive
```

目前本地 root `.gitmodules` 已將 `sam3_benchmark` 的 `branch` 從不存在的遠端 `VQA` 改成 `main`，但這不會替本地 `VQA` 分支設定 upstream，也不會修改目前 checkout。若將來需要把本地 `VQA` 發布為研究分支，可明確 `git -C sam3_benchmark push -u origin VQA`；或在乾淨工作樹切回 `main`，依團隊分支策略整合研究提交。兩者擇一，勿直接覆蓋現有分支。

在子模組內修改後：

1. 到該 repository 檢查 diff，執行其測試，提交實際來源碼／文件。
2. 推送子 repository 的分支，確認 commit 在遠端可取得。
3. 回 root 檢查 `git diff --submodule=log`，stage 子模組路徑與 `.gitmodules`（若有修改）。
4. 提交 root，讓 gitlink 指向已發布 commit，最後才推送 root。

刻意升級 dependency 時，先在乾淨 checkout fetch／review，選定 commit，再測試並更新指標；不要對目前有未提交功能的子模組盲目 `git submodule update --remote`。

## 實驗、資產與交付

每次實驗記錄 root／子模組 SHA、dirty diff、模型 revision、Python／Torch／Transformers／CUDA、GPU、source YAML、影片與取樣方式。分開報告 raw JSON 通過率、修復後通過率、準確率、p50／p95、VRAM，以及 `motion_source: vlm` 或 `tracking_hint`。結果與權重另行備份，Git 只追蹤必要的小型、去識別化摘要；若要保存結果摘要，放在文件目錄並記錄外部資產位置，不要直接 force-add 整個 `runs/`。

提交前逐檔 stage 並檢查：

```bash
bash scripts/check.sh
git diff --check
git diff --stat
git diff --cached --stat
git diff --cached --submodule=log
```

`.gitignore` 不會移除已追蹤檔案；新 clone 能重現的前提是新套件與文件真的被提交，子模組也已先發布。推送後再執行 `python3 scripts/audit_git.py --remote`，確認對應分支 SHA。
