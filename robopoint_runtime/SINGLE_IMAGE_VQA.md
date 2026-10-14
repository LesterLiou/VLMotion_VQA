# 單張圖片測試新模型

在主機執行；既有 `vlpoint-inference` 容器必須正在運作。也可在容器內執行同一個 shell 腳本。不需要 source ROS、controller、worker 或 SAM3。

先在原 worker 與 pipeline terminal 按 Ctrl+C，避免重複載入模型造成 OOM。腳本不會自動停止其他程序；可用 GPU 記憶體不足 9 GiB 時會提示先釋放資源。此門檻是保守預檢，不保證所有輸入都不會 OOM。

```bash
cd /home/lester/VLMotion
bash robopoint_runtime/run_single_image.sh \
  --image 'example/image/image (14).png' \
  --question 'What is the person doing?'
```

換圖片與問題即可。支援中文問題，但回答語言及正確性取決於模型。相對圖片路徑一律相對 VLMotion 專案根目錄；圖片應放在專案內，主機專案內絕對路徑也會轉換到容器 `/workspace`。專案外的圖片需先自行放進專案。

選項：

- `--max-new-tokens 128`：限制輸出長度，預設 256。
- `--output robopoint_runtime/smoke_317677/my_answer.json`：保存問題、回答、模型路徑、載入與推論時間；會覆寫指定檔案。
- `--model-path robopoint_runtime/runs/robopoint_vqa_only_317677`：更換相同架構 checkpoint，預設即為這份新模型。
- `--interactive`：同一張圖保留模型載入並連續問答；輸入 exit 或 quit 結束。
- `--help`：查看參數。

實作：`single_image_vqa.py`，直接使用專案現有 model builder 與 llava_v1 prompt。4-bit、單 GPU、關閉 SAM3 conditioning、離線載入本機模型和 CLIP 快取。一般單次命令重新載入模型並在回答後退出；`--interactive` 保留載入至對話結束。JSON 結果檔是程式封裝，並不代表模型回答本身會遵守 JSON schema。
