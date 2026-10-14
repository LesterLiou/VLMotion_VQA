# SAM3-Native Foundation VLM：單一模型完整研究與執行計畫

> 2026-10-14 整理註記：這是保留的研究提案，並非已實作系統。目前 checkout 沒有 `foundation_vlm/`；部署以 [使用指南](../USAGE.md) 為準，模型建議與外部資料仍須研究時重新查證。

> 文件日期：2026-08-07\
> 文件狀態：新的主計畫；取代先前以「VLM + SAM3 外部工具」為產品主線的規劃\
> 目標：以 SAM 3 的高解析、小物件與 dense localization 表示為視覺核心，訓練一個可泛化、單次 forward、單一 checkpoint、本地部署的 foundation VLM

## 0. 修正後的結論

新的目標不是讓 VLM 先理解文字、再呼叫另一個 SAM3 模型，也不是只替換 LLaVA 的 CLIP model name。

建議正式模型定義為：

    SAM3-Native-VLM-8B
    = SAM 3 vision trunk + neck
    + multi-scale visual tokenizer
    + DeepStack-style visual injection
    + Qwen3-8B-Base language decoder
    + language / grounding / mask auxiliary heads
    + joint multimodal pretraining

部署時是一份權重、一個 process、一個模型類別與一次端到端 forward。模型內仍然會有 vision、fusion、language 等模組；任何使用 SAM3 image encoder 的 VLM 都不可能在結構上完全沒有模組邊界。「單一模型」在這份計畫中的精確定義是：

- 不依賴外部 detector 或另一個 VLM。
- 不需要先產生 noun phrase，再進行第二次模型推論。
- 視覺與語言參數由共同 loss 聯合最佳化。
- 同一份 checkpoint 保存 vision backbone、fusion、LLM 與可選 dense heads。
- 推論入口是一個 generate/forward 呼叫，可同時回傳文字、box、point 或 mask。
- 正式部署完全本地，不依賴雲端 API。

### 最終建議規格

| 元件 | 主版本建議 |
|---|---|
| Vision backbone | SAM 3 ViT trunk，保留 multi-level features |
| Dense vision | 保留 SAM 3 neck/FPN 作為模型內部特徵，不是外部工具 |
| Language backbone | Qwen3-8B-Base |
| 架構 | multi-scale tokenizer + DeepStack-style injection |
| 預設 visual tokens | 576；高解析模式 1,296；不可再以 144 作為 foundation 主設定 |
| 輸入 | 1008 square global view；高解析圖片可加入 overlapping tiles |
| 輸出 | text + structured coordinates；選配 mask |
| 模型大小 | 約 8.7B–9B，依 fusion 與 heads 而定 |
| 訓練 | alignment → multimodal mid-training → dense co-training → SFT → optional preference/RL |
| 部署 | 單一 Hugging Face compatible checkpoint，本地 PyTorch/TensorRT/vLLM custom backend |

### Foundation model 不等於一定從零開始

這份計畫是「以兩個 pretrained foundation components 初始化，再進行大規模 joint multimodal continual pretraining」。訓練完成後得到的是新的、可跨任務遷移的 foundation VLM，而不是只有一個 task adapter。

不建議把 SAM3 和 Qwen3 權重全部丟掉從 random initialization 開始。真正從零同時重建語言、視覺與跨模態能力，資料與算力會比本計畫高一到兩個數量級，而且無助於驗證 SAM3 representation 的價值。只有在 clean-room 權重、完全不同 tokenizer 或授權策略是硬需求時，才另立 from-scratch program。

## 1. 為什麼上一版計畫需要改

上一版的「完整 SAM3 當工具」仍然是很好的低成本 baseline，但不符合現在的 foundation-model 目標。

另外，上一版 hard-swap 的 144-token spatial pooling 適合確認程式能否跑通，卻不應成為最終架構。SAM 3 在 1008×1008 輸入下產生 72×72，也就是 5,184 個 1024-D trunk tokens。直接 average-pool 到 12×12，只留下 144 tokens，可能在模型看見語言之前就丟失小物件、細部輪廓和局部相對位置。

更關鍵的是：SAM 3 的小物件能力不一定只存在於最後一層 image-encoder output。它同時來自：

- 1008 高解析輸入。
- 32-layer ViT trunk 與多個 global-attention blocks。
- multi-scale neck/FPN。
- detector queries、presence prediction 與 hard negatives。
- detection/segmentation/concept losses。
- SAM 3 的大規模 concept 與 mask 資料。

因此只抽最後的 trunk tokens、丟掉 neck 和 dense supervision，不能假設仍會保留完整的小物件優勢。新版把 trunk、multi-level feature、FPN 與 dense auxiliary losses 都納入同一個 end-to-end model。

SAM 3 官方模型與資料引擎說明：[SAM 3 paper](https://arxiv.org/abs/2511.16719)、[official repository](https://github.com/facebookresearch/sam3)。

## 2. 重新 survey 後可借鑑的設計

### 2.1 SAM 3

SAM 3 是 promptable concept segmentation foundation model，將 detection、segmentation 與 tracking 建立在共享 vision backbone 上，並使用 presence head 區分相近概念。它證明 vision representation 適合 dense、open-vocabulary 與 hard-negative 場景，但官方沒有提供「移植到任意 LLM 並訓練通用 VLM」的完整 recipe。

對本計畫的啟示：

- 使用預訓練 vision trunk 起步，不從零重訓。
- 保留 FPN/dense loss，避免只剩 caption-friendly 的全域特徵。
- 將 no-target、相近類別與小物件負樣本放入 foundation pretraining。
- 對 final layer 與 global-attention intermediate layers 做 ablation。

### 2.2 SAM3-I

SAM3-I 將複雜 instruction 直接整合進 SAM3，使用分階段 alignment 與 joint losses，而不是永遠依靠外部 agent 把指令轉為 noun phrase。它不是通用聊天 VLM，但證明 SAM3 的 representation 能在單一模型內接受更複雜語言 supervision。

可借鑑：

- simple query → complex query → joint training 的 curriculum。
- 保留原 SAM3 concept capability 的 preservation losses。
- instruction-to-mask、one-to-one、one-to-many、no-target 資料設計。

來源：[SAM3-I paper](https://arxiv.org/abs/2512.04585)、[official code](https://github.com/debby-0527/SAM3-I)。

### 2.3 Qwen3-VL 與 DeepStack

Qwen3-VL 使用 multi-level ViT features 和 DeepStack，將不同視覺層的特徵注入 language decoder 的不同層，而不是只在 LLM input 放一次 final vision tokens。這對小物件尤其重要：低/中層保留局部細節，後層包含語意。

可借鑑：

- 不只使用 SAM3 final hidden state。
- 從 SAM3 global-attention blocks 7、15、23、31 取 features。
- 將不同層 feature 經各自的 merger/projector，注入 LLM 的多個深度。
- 讓視覺資訊持續參與 reasoning，而不是在第一層後逐漸被文字表示淹沒。

來源：[Qwen3-VL technical report](https://arxiv.org/abs/2511.21631)、[DeepStack paper](https://arxiv.org/abs/2406.04334)。

### 2.4 LLaVA-OneVision-1.5 / 2

LLaVA-OneVision-1.5 保留清楚的 ViT–MLP–LLM 模組界面，公開 alignment、mid-training、instruction tuning 資料與訓練框架。其報告使用約 85M mid-training samples、22M instruction samples，合計約 64B compressed multimodal tokens。

2026 年的 LLaVA-OneVision-2 進一步公開 Qwen3-based 4B/8B 訓練、Megatron pipeline、native-resolution/spatial data 與完整 logs。它的 codec-aligned video encoder不是本計畫要採用的 vision backbone，但其 single-node 4B quickstart、checkpoint merge、packed dataset 與分散式訓練骨架比舊 LLaVA repository 更適合作為工程起點。

可借鑑：

- codebase、data packing、stage recipe 與可重現的訓練基礎設施。
- region/OCR-aware pretraining。
- 使用大量 mid-training 建立 foundation capability，不把幾十萬筆 connector tuning 稱為 foundation training。

來源：[LLaVA-OneVision-1.5 paper](https://arxiv.org/abs/2509.23661)、[LLaVA-OneVision-2 official code](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2)。

### 2.5 OMG-LLaVA 與 Sa2VA

OMG-LLaVA 和 Sa2VA 都示範了 segmentation foundation model、LLM、pixel output 可以構成一個 end-to-end checkpoint，而不是兩個外部服務。它們的重點不是照抄實作，而是證明「一個 vision encoder、一個 LLM、共同 token space、可選 pixel head」是合理模型邊界。

來源：[OMG-LLaVA](https://arxiv.org/abs/2406.19389)、[Sa2VA](https://arxiv.org/abs/2501.04001)。

## 3. 建議的單一模型架構

### 3.1 整體資料流

    RGB / multi-image
          │
          ▼
    SAM 3 ViT trunk
      blocks 7 / 15 / 23 / 31
          │
          ├──────────────► SAM 3 neck/FPN ─► dense auxiliary head
          │                                      │
          ▼                                      │
    Multi-scale visual tokenizer                 │
      2D RoPE + spatial merge + token router     │
          │                                      │
          ▼                                      │
    visual embeddings at LLM input               │
          │                                      │
          ├── DeepStack injection at LLM layers ◄┘
          ▼
    Qwen3 decoder
          │
          ├── LM head: text / JSON / coordinate tokens
          ├── grounding head: point / box / presence
          └── SEG hidden-state adapter ─► mask head

    所有分支在同一 computation graph、同一 checkpoint 內共同訓練

### 3.2 Vision backbone

初始化方式：

- 由完整 Sam3Model checkpoint 正確取出 vision encoder。
- 不使用目前會產生 missing/unexpected keys 的 direct Sam3VisionModel 載入路徑。
- 第一版保留原始 1008 preprocessing 與 SAM normalization。
- 取出 block 7、15、23、31 的 72×72×1024 features。
- 保留 neck/FPN 的 multi-scale 256-D features，供 dense loss 與 token router 使用。

訓練策略：

- alignment 初期凍結 SAM3。
- mid-training 解凍 neck、最後 8 blocks。
- 最終再決定是否用 layer-wise learning-rate decay 解凍全部 trunk。
- 持續加入 SAM-style mask/detection/presence batches，防止 dense 能力退化。
- 可保留一份 frozen SAM3 teacher，對部分 batches 做 feature/mask distillation；teacher 只存在於訓練，不進正式 checkpoint。

### 3.3 Multi-scale visual tokenizer

不建議只用 average pooling 或無空間結構的 Q-Former。

主版本採用：

1. Space-to-depth spatial merger\
   將相鄰 patch features concatenate 後投影，而非直接平均，保留小物件訊號。

2. 2D position\
   每個 token 保存 global image coordinate、tile coordinate、scale id 與 view id。

3. Multi-level projection\
   blocks 7/15/23/31 各有獨立 LayerNorm + merger + projector。

4. Learned token router\
   使用 FPN feature 和 query-independent saliency 選擇額外局部 tokens；不可只靠已知類別 detector，否則會漏掉 novel concepts。

5. DeepStack injection\
   final visual tokens 放在 LLM input；中間層 features 以 gated residual 方式注入數個 LLM layers。

### 3.4 Visual token budget

| 模式 | 方法 | Tokens | 用途 |
|---|---|---:|---|
| Architecture smoke test | 6×6 pooling | 36–144 | 只驗證介面，不作主模型 |
| Foundation default | 3×3 space-to-depth merge | 24×24 = 576 | 一般 VQA、grounding、小物件平衡 |
| High-resolution | 2×2 space-to-depth merge | 36×36 = 1,296 | 小物件、OCR、dense scenes |
| Dynamic | 576 global + 256–768 routed local | 832–1,344 | 建議最終模式 |
| Raw | 不壓縮 72×72 | 5,184 | 只做研究上限，不直接進所有 LLM layers |

預設應從 576 開始，而不是 144。

### 3.5 任意解析度與小物件

SAM3 原生 1008 square 會扭曲非正方形圖片，也可能讓超高解析影像中的小物件縮得過小。建議：

- global view：整張影像 resize/pad 到 1008。
- local views：依原始 aspect ratio 產生 overlapping 1008 tiles。
- 每個 tile 帶有原圖 normalized offset 與 scale。
- 訓練時混合 1-view、2-view、4-view，而不是只在推論突然開 tiling。
- token router 保留 global context，並挑選局部高解析 tokens。
- tile selection 在模型內完成；不呼叫外部 detector。
- 設定硬 token cap，避免一張大圖無限制增加 LLM context。

### 3.6 Language decoder

主版本使用 Qwen3-8B-Base，而不是 instruction checkpoint。

原因：

- base checkpoint 適合進行大規模 multimodal continual pretraining。
- 中文、多語與一般 language capability 較適合目標情境。
- Apache 2.0，權重與 Transformers 支援完整。
- decoder architecture 比直接拆 Qwen3-VL/Qwen3.5 的既有 vision-language coupling 更乾淨。

來源：[Qwen3-8B-Base model card](https://huggingface.co/Qwen/Qwen3-8B-Base)。

建議尺寸：

| 階段 | LLM | 用途 |
|---|---|---|
| 架構搜尋 | Qwen3-4B-Base | 快速比較 fusion/token/loss |
| 主 foundation model | Qwen3-8B-Base | 品質、成本、部署平衡 |
| 大型版本 | Qwen3-14B-Base | 有明確 8B scaling gain 後再做 |
| Teacher/baseline | Qwen3.5-9B/27B native VLM | 產生或過濾資料、比較能力；不作第一版 hard-swap backbone |

不建議：

- 從 Vicuna 13B 開始：語言與中文能力、訓練生態已落後。
- 直接把 SAM3 插入 Qwen3-VL/Qwen3.5 checkpoint：其 position、DeepStack 與原 vision tower 已共同對齊，會同時破壞多個 pretrained contracts。
- 用 instruct model 做大規模 mid-training 起點：更容易造成 instruction alignment 漂移；base model較乾淨。

### 3.7 輸出設計

加入 special tokens：

- image / image_end
- tile / tile_end
- point
- box
- seg
- no_target
- 0–999 coordinate bins

文字、point 與 box 可由 LM head 自回歸輸出。mask 建議使用 seg token 的 hidden state 經 prompt adapter 餵給 checkpoint 內的 mask head。

這不是外部兩階段系統：seg hidden state、mask head 與 vision feature 都屬於同一模型並接受 joint gradient。

## 4. Training objective

主要 loss：

    L_total
      = λ_lm       × next-token language loss
      + λ_region   × region-caption / region-text alignment
      + λ_point    × point or coordinate loss
      + λ_box      × L1 + GIoU
      + λ_mask     × focal + dice
      + λ_presence × no-target / presence BCE
      + λ_contrast × optional global image-text contrastive loss
      + λ_preserve × SAM feature or mask preservation loss

原則：

- 不是每筆資料都有所有 labels；以 task mask 啟用對應 loss。
- 不要一開始固定所有 λ；先量測各分支 gradient norm，再用 GradNorm、uncertainty weighting 或人工校準。
- 語言 batch 與 dense batch 需要交錯，避免某一種 supervision 長時間壟斷 optimizer。
- object size、語言、資料來源與 task 都要有 batch-level sampler。
- final model 必須接受全模型 joint gradient；只訓 connector 不足以稱為本計畫的最終 foundation model。

## 5. 資料策略

### 5.1 Foundation mixture

建議 main mid-training 起始比例：

| 資料能力 | 比例 |
|---|---:|
| 一般 image-text、caption、world knowledge | 25% |
| interleaved image-text、多圖理解 | 10% |
| OCR、document、chart、UI | 15% |
| detection、region caption、referring、grounding | 20% |
| instance/part segmentation、小物件 | 15% |
| spatial reasoning、counting、no-target、hard negatives | 10% |
| text-only replay | 5% |

比例是 sampling ratio，不是原始資料量。需要依 validation loss 與 benchmark Pareto front 調整。

### 5.2 規模

| 版本 | Mid-training | Instruction SFT | 定位 |
|---|---:|---:|---|
| Architecture pilot | 1M–5M samples | 0.2M–1M | 找出正確架構 |
| 4B foundation MVP | 20M–40M | 3M–8M | 第一個可發布版本 |
| 8B main | 50M–100M | 5M–20M | 正式 foundation model |
| Frontier attempt | 100M+ | 20M+ | 需要更大資料治理與算力 |

LLaVA-OneVision-1.5 公開 recipe 的量級約為 85M mid-training 與 22M instruction samples，這是規劃 8B foundation model 的合理參照，不代表必須完全複製。

### 5.3 小物件資料治理

每個 detection/mask target 保存：

- 原圖寬高。
- object area ratio。
- bbox、mask、point、part/instance id。
- occlusion、blur、contrast、crowding。
- prompt concept 與 hard-negative concepts。
- crop/tile 到原圖的可逆 transform。

Sampling 按面積分桶：

- tiny：小於 0.1% image area。
- small：0.1%–1%。
- medium：1%–10%。
- large：大於 10%。

每個 dense batch 應強制包含 tiny/small targets。對 crop augmentation 必須同時保留 global view，避免模型只會辨識裁切後的大物件。

### 5.4 統一 sample schema

每筆訓練資料至少保存以下欄位，不把所有 supervision 塞入單一 conversation 字串：

    sample_id
    media: image path / original width / original height
    views: global and tile transforms
    conversations: role / text / referenced region ids
    regions: bbox / point / mask / class / attributes / area ratio
    targets: text / coordinate tokens / presence / no_target
    task_tags: caption / OCR / grounding / mask / multi-image
    provenance: dataset / license / split / content hash

所有 box、point、mask 都以原圖座標作 canonical ground truth；processor 只在 runtime 產生可逆的 model-space 座標。

### 5.5 必要資料類型

- 一般 caption 與知識型 image-text。
- object/part/attribute region captions。
- crowded scenes、相似實例與遮擋。
- negative/no-target queries。
- OCR/UI 中的小字、小 icon、按鈕與標誌。
- RefCOCO family、gRefCOCO 類 referring。
- COCO/LVIS/SA-Co 類 detection/segmentation。
- 自建繁中 instruction、region QA 與 spatial reasoning。
- 純文字 multilingual replay。
- 真實部署域只是一個 evaluation/domain slice，不可主導 foundation corpus。

### 5.6 品質與授權

- 先建立 dataset registry：來源、license、允許用途、版本、hash、去重狀態。
- image-level 與 near-duplicate 去重後再 split。
- 所有 benchmark test/dev image 做 leakage blacklist。
- 合成資料需保存 teacher、prompt、sampling settings 與 verifier。
- 可以使用本地大型 VLM teacher，不依賴雲端 API。
- 低品質 alt-text、OCR mismatch、錯誤 box/mask 應以規則和模型雙重過濾。
- 發布前需另外審查 SAM License 與所有 dataset licenses。

## 6. 分階段訓練

### Stage 0：Architecture contract 與 baseline，1–2 週

完成：

- 正確載入 SAM3 pretrained vision weights。
- 建立 blocks 7/15/23/31 feature hooks。
- 實作 144、576、1,296 token modes。
- 建立 prefix-only 與 DeepStack 兩條路徑。
- 建立單一 config/model/checkpoint class。
- 確認 text-only、image-text、grounding、mask forward 都能 save/reload。
- 建立原始 SAM3、Qwen native VLM、舊 VLM baseline。

Gate：

- 0 missing vision keys。
- coordinate transform round-trip 正確。
- 所有 heads 有預期 gradient。
- 單 checkpoint reload 後輸出一致。
- 500-step throughput、VRAM、NaN smoke test 通過。

### Stage 1：Vision-language alignment，1M–5M samples

凍結：

- SAM3 trunk/neck。
- Qwen3 LLM。

訓練：

- spatial mergers。
- projectors。
- token router。
- DeepStack gates。
- special token embeddings。

建議起始參數：

| Group | LR |
|---|---:|
| merger/projector/router | 1e-4–5e-4 |
| LLM | 0 |
| SAM3 | 0 |

其他：

- BF16。
- AdamW，beta2 約 0.95。
- weight decay 0.05–0.1。
- warmup 1%–3%，cosine decay。
- gradient clip 1.0。
- 以 packed multimodal tokens，而不是固定 image count 衡量 batch。
- LLM parameters 雖凍結，autograd 仍需穿過 LLM 回到 fusion；不可把整個 LLM forward 放入 no_grad。

Gate：

- caption/region retrieval 明顯高於 random connector。
- 576 tokens 在 small-object proxy 顯著優於 144。
- DeepStack 相對 prefix-only 有穩定增益，否則先保留簡單架構。

### Stage 2：Multimodal mid-training，50M–100M main run

第一段約 70% steps：

- 全量訓練 LLM。
- 全量訓練 fusion。
- 解凍 SAM3 neck 與最後 8 vision blocks。
- SAM3 前 24 blocks 保持 frozen。

第二段約 20% steps：

- 視 validation 決定解凍全部 SAM3。
- 使用 layer-wise LR decay；越靠近輸入 LR 越低。
- 加入更多 tiny/small、OCR、region 與 hard-negative batches。

最後約 10% steps：

- 降低所有 LR 5–10 倍。
- 增加高品質資料比例。
- 強化 text-only 與 original-SAM replay，做 capability consolidation。

建議起始 LR：

| Parameter group | LR |
|---|---:|
| fusion/router/new heads | 5e-5–1e-4 |
| Qwen3 LLM | 1e-5–2e-5 |
| SAM3 neck / last 8 blocks | 1e-6–5e-6 |
| SAM3 lower blocks，若解凍 | 1e-7–1e-6 |

Sequence curriculum：

- 主要 single-image batches 使用 4K context。
- OCR、multi-image 與多 tile batches 使用 8K。
- 16K 以上只佔少量長序列資料，先確認 4K/8K 能力與 throughput。
- 1,296-token high-resolution mode 先控制在 20%–30% image batches，再依 small-object gain 調整。

Gate：

- general VLM、OCR、grounding、small-object 同時形成可接受 Pareto front。
- standalone SAM dense metrics 不可出現不可逆的大幅退化。
- text-only capability 保留。
- 沒有單一資料來源控制 validation improvement。

### Stage 3：Dense co-training

Dense task 可在 Stage 2 交錯進行，不必等 language mid-training 結束。curriculum：

1. concept / simple referring。
2. attribute / spatial relation。
3. complex instruction。
4. one-to-many、no-target、hard negatives。
5. reasoning segmentation、parts、tiny objects。

訓練：

- LM head。
- grounding/coordinate heads。
- seg hidden-state adapter。
- SAM3 neck/mask/detector heads。
- vision upper blocks。
- LLM 全參數或高 rank LoRA；主 foundation run 建議全參數。

加入 original SAM3 preservation batch 或 frozen-teacher distillation，避免 VLM loss 將 vision backbone推向只有 caption semantics。

### Stage 4：Multimodal instruction SFT，5M–20M samples

目標：

- 多輪對話。
- 中文/多語 instruction。
- image、multi-image、region 與 dense outputs。
- structured JSON/coordinate。
- 不確定性、no-target 與拒絕亂猜。
- general assistant 能力。

訓練範圍：

- fusion、LLM 全量。
- vision neck/upper blocks 使用更低 LR。
- lower vision blocks 可重新凍結。
- 混入 10%–20% mid-training replay，避免 narrow SFT collapse。

### Stage 5：Preference / reasoning post-training

只在 SFT model 穩定後做：

- rejection sampling 或 DPO：回答品質、grounding consistency。
- verifiable RL：coordinate、box、mask IoU、counting、OCR exact match。
- 不以不可驗證的自由文字 reward 單獨驅動 dense capability。
- 保留 KL/reference 約束，避免破壞一般與小物件能力。

### Stage 6：Distillation 與部署版本

保留一個 BF16 master checkpoint，再衍生：

- 8B INT8/FP8 host version。
- 8B 4-bit LLM + BF16/INT8 vision edge version。
- 4B distilled edge version。
- 可選移除訓練專用 teacher/preservation modules；不可移除正式需要的 tokenizer/heads。

量化前後都要跑 small-object、OCR、coordinate 與 mask regression；不能只跑語言 perplexity。

## 7. 算力與基礎設施

以下是規劃區間，正式預算必須由 500-step benchmark、實際 sequence length、tile ratio 與 cluster MFU 推回。

| 目標 | 建議資源起點 | 說明 |
|---|---|---|
| 4B architecture search | 8×80GB | 1M–5M subset、比較 token/fusion |
| 4B foundation MVP | 16–32×80GB | 20M–40M mid-training |
| 8B main foundation | 32–64×80GB | 50M–100M、full LLM + partial/full vision |
| 14B scale-up | 64–128×80GB | 先證明 8B 不再受資料/架構限制 |
| Dense full unfreeze | 在上述基礎增加記憶體/節點 | 1008 vision activations 很昂貴 |

必要工程：

- FSDP2 或 DeepSpeed ZeRO-3。
- Flash Attention。
- activation checkpointing。
- sequence/data packing。
- distributed sampler 依 task 與 object-size 分桶。
- dataloader cache 與高速 shared storage。
- optimizer/checkpoint sharding。
- 每個 run 保存 data manifest、git commit、config、seed 與 tokenizer hash。
- 以 multimodal tokens/sec、MFU、peak memory、I/O wait 監控，而不只看 steps/sec。

容量規劃：

- 8B + SAM3 + fusion 約 8.7B–9B parameters。
- BF16 weights 本身約 18GB；full-training optimizer、gradient、master weights 與 activations 遠高於此。
- 50M–100M image samples、tiles、masks、OCR 與中間 metadata 通常需要數十 TB 級 storage。
- 不建議 cache 全部 5,184×1024 raw SAM features；會迅速進入數百 TB。
- checkpoint、optimizer state 與 evaluation artifacts 應使用獨立 object storage/model volume。

## 8. 實驗矩陣

架構搜尋時一次只改一個變因：

| 編號 | 變因 |
|---:|---|
| A1 | final SAM block vs blocks 7/15/23/31 |
| A2 | trunk only vs trunk + FPN |
| A3 | 144 vs 576 vs 1,296 tokens |
| A4 | average pool vs space-to-depth merger |
| A5 | prefix-only vs DeepStack |
| A6 | fixed tokens vs global + routed local tokens |
| A7 | frozen SAM vs last-8 unfreeze vs full unfreeze |
| A8 | 無 dense auxiliary loss vs 有 dense loss |
| A9 | 無 SAM preservation vs mask loss vs feature distillation |
| A10 | global-only vs global + tiles |
| A11 | Qwen3-4B proxy vs Qwen3-8B main |
| A12 | LM coordinate tokens vs dedicated coordinate head |
| A13 | text/box only vs text/box/mask joint output |

最重要的四個實驗：

1. SAM3 trunk final-only 是否真的優於通用 VLM encoder。
2. 小物件增益來自 trunk、FPN、high resolution 還是 dense loss。
3. 576/1,296 tokens 是否值得其 latency。
4. full joint training 是否保留 SAM3 原本的 localization。

## 9. 評估

### General foundation VLM

- general VQA。
- knowledge/reasoning。
- multi-image。
- multilingual/繁體中文。
- hallucination與 no-target。
- text-only non-regression。

### 小物件與 dense vision

- COCO/LVIS AP-small。
- SA-Co concept grounding。
- RefCOCO/+/g、gRefCOCO。
- point-in-mask、PCK、box IoU、mask IoU。
- tiny/small/medium/large 分桶結果。
- crowded scene、occlusion、near-duplicate instance。
- novel concept 與 no-target false positive。

### OCR/UI

- TextVQA、DocVQA、ChartQA、OCRBench。
- 自建繁中小字、icon、button、label benchmark。
- 不同原始解析度、縮放倍率與 tile 數。

### Efficiency

- vision encoder latency。
- time-to-first-token。
- end-to-end P50/P95。
- peak VRAM/RAM。
- 576 vs 1,296 vs dynamic token cost。
- 1/2/4 views scaling。
- 量化前後 regression。
- 長時間 memory leak。

### Foundation model gate

不能只用單一實驗資料證明成功。至少要求：

- general VLM 不低於同尺寸 native VLM baseline 的可接受範圍。
- small-object/grounding 有統計顯著提升。
- OCR 與大物件能力沒有以不可接受幅度換取小物件提升。
- original SAM3 dense benchmark 保留。
- 中文與 text-only 能力保留。
- 所有能力來自單 checkpoint，沒有 inference-time external model。

## 10. Codebase 重構

不建議把這個 foundation training 全部塞進目前舊 LLaVA/Vicuna class。現有 repository 可保留作 baseline 與 ROS deployment，但主訓練應建立獨立 package：

    foundation_vlm/
    ├── configs/
    │   ├── sam3_qwen3_4b_proxy.yaml
    │   └── sam3_qwen3_8b.yaml
    ├── modeling/
    │   ├── configuration_sam3_vlm.py
    │   ├── modeling_sam3_vlm.py
    │   ├── sam3_backbone.py
    │   ├── multiscale_tokenizer.py
    │   ├── deepstack_adapter.py
    │   └── dense_heads.py
    ├── data/
    │   ├── registry.py
    │   ├── packing.py
    │   ├── transforms.py
    │   └── task_sampler.py
    ├── train/
    │   ├── stage1_alignment.py
    │   ├── stage2_midtrain.py
    │   ├── stage4_sft.py
    │   └── distributed.py
    ├── eval/
    │   ├── general_vlm.py
    │   ├── small_object.py
    │   ├── grounding.py
    │   └── efficiency.py
    └── tests/

模型類別應支援：

- Hugging Face PreTrainedConfig/PreTrainedModel。
- from_pretrained/save_pretrained。
- 單 checkpoint sharding。
- text-only 與 image-text。
- variable views/tiles。
- generate + structured outputs。
- FSDP/ZeRO-compatible state dict。
- vision、fusion、LLM 分組 LR。
- optional dense heads 但共同封裝。
- export/quantization hooks。

可用 LLaVA-OneVision-2 的 Qwen3/Megatron distributed training framework，以及 OneVision-1.5 的 85M mid-training data recipe 作起點；模型本體則實作 SAM3 multi-level features、space-to-depth merger 與 DeepStack。

## 11. 16 週執行安排

| 週 | 工作 | 交付物 |
|---:|---|---|
| 1–2 | 正確 vision load、feature hooks、single model config/checkpoint | model contract + tests |
| 2–4 | 4B proxy，144/576/1296、prefix/DeepStack、trunk/FPN ablation | architecture decision |
| 1–5 | dataset registry、dedup、license、small-object sampler | versioned data manifest |
| 5–6 | 1M–5M alignment | aligned 4B checkpoint |
| 7–10 | 4B 20M–40M mid-training + dense co-training | foundation MVP |
| 9–11 | 8B infrastructure benchmark、scale validation | 8B launch gate |
| 11–14 | 8B main mid-training | 8B base checkpoint |
| 14–15 | 5M–20M SFT、grounding/mask consolidation | 8B instruct checkpoint |
| 15–16 | evaluation、quantization、model card、local deployment | release candidate |

data pipeline 與 model engineering 必須並行；不要等模型程式完成後才開始處理資料治理。

## 12. 前兩週立即執行清單

1. 將正式目標命名為 SAM3-Native-VLM，而不是現有 RoboPoint 的 encoder replacement。
2. 決定第一個 proxy 使用 Qwen3-4B-Base，主版本使用 Qwen3-8B-Base。
3. 建立新的 foundation_vlm package，不把舊 Vicuna class 當長期基礎。
4. 修正 SAM3 checkpoint loading，驗證 vision weights 0 missing keys。
5. 實作 block 7/15/23/31 hooks 與 FPN output contract。
6. 實作 3×3 space-to-depth merger，輸出 576 tokens。
7. 實作 prefix-only baseline 與 DeepStack variant。
8. 將 LM、box、point、presence、mask loss 放入統一 model output。
9. 準備 1M–5M architecture subset，強制包含 tiny/small objects、OCR 與 no-target。
10. 完成 500-step benchmark，據此估算 4B/8B full run。
11. 先做 A1–A8 ablation，再啟動 50M–100M main run。
12. 從第一天就保存 data manifest、license、benchmark blacklist 與 experiment lineage。

## 13. 主要風險

### 風險 1：誤把 detector 能力當成 encoder 能力

SAM3 的小物件表現可能主要來自 neck、queries、presence head 與 dense losses。解法是 trunk-only、trunk+FPN、with/without dense loss 的正式 ablation。

### 風險 2：高解析 token 被過度壓縮

144 tokens 可能讓 foundation model失去主要差異化。解法是以 576 作預設，1,296 作高解析上限，再研究 internal token routing。

### 風險 3：語言 loss 破壞 localization

大規模 caption loss 容易讓 vision features 偏向全域語意。解法是 dense batch、SAM replay、feature/mask preservation 與 vision LR 分層。

### 風險 4：資料量大但 supervision 不平衡

數千萬 caption 不能替代 region、part、mask、hard negative。必須用 task/object-size sampler 控制實際看到的 distribution。

### 風險 5：單模型不等於單一 objective

一個 checkpoint 仍需要 language、grounding、presence、mask 等多任務 supervision。強迫所有能力只用 next-token loss，通常會犧牲 dense precision。

### 風險 6：授權與發布

SAM 3 使用 Meta 自訂 SAM License，不是 Apache/MIT。即使 Qwen3 為 Apache 2.0，整合 checkpoint 的發布條件仍需法律與授權審查：[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)。

## 14. 決策摘要

正式推薦：

    SAM 3 blocks 7/15/23/31 + FPN
    → multi-scale 2D-aware space-to-depth tokenizer
    → 576 default / 1,296 high-res visual tokens
    → DeepStack-style injection
    → Qwen3-8B-Base
    → LM + grounding + presence + optional mask heads
    → joint multimodal mid-training
    → single local checkpoint

研究流程：

    4B proxy 找架構
    → 1M–5M alignment
    → 20M–40M 4B foundation MVP
    → 通過 gates
    → 50M–100M 8B main mid-training
    → 5M–20M instruction SFT
    → preference/RL
    → quantized local release

這條路線會保留「一個模型」的產品與研究定義，也真正把 SAM3 的小物件能力當成 foundation representation，而不是把它當成 LLM 外面的工具。
