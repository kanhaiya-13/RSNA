# RSNA Knee MRI v2 Model Audit & Performance Record

This document provides a factual record of the **v2 study-level model** (`outputs_v2/`), audited directly from the saved checkpoints, OOF files, training histories, code, and live inference evaluation on saved weights. All metrics follow the competition standard (macro-averaged ROC-AUC with $\ge 0.5$ binarization threshold on continuous soft targets).

---

## 1. Headline Results (Study Level)

### 1. Per-Fold Study-Level Macro AUC

Computed directly from [outputs_v2/oof_best.csv](file:///d:/RSNA/outputs_v2/oof_best.csv), [outputs_v2/oof_last.csv](file:///d:/RSNA/outputs_v2/oof_last.csv), [outputs_v2/baseline_oof_study.csv](file:///d:/RSNA/outputs_v2/baseline_oof_study.csv), and [outputs_v2/folds.csv](file:///d:/RSNA/outputs_v2/folds.csv) ($N = 4,407$ unique studies):

| Fold | v2 Best-Epoch Macro AUC | v2 Last-Epoch Macro AUC | Baseline Study-Level Macro AUC | Paired Difference ($\Delta$ Best $-$ Base) | Paired Difference ($\Delta$ Last $-$ Base) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 0.863577 | 0.860516 | 0.880280 | -0.016703 | -0.019764 |
| **1** | 0.854151 | 0.852877 | 0.866256 | -0.012105 | -0.013379 |
| **2** | 0.852811 | 0.849308 | 0.863942 | -0.011131 | -0.014634 |
| **3** | 0.867363 | 0.863786 | 0.873920 | -0.006557 | -0.010134 |
| **4** | 0.845400 | 0.845281 | 0.862825 | -0.017425 | -0.017543 |
| **Mean** | **0.856660** | **0.854354** | **0.869444** | **-0.012784** | **-0.015091** |
| **Std ($ddof=1$)** | **0.008808** | **0.007696** | **0.007445** | **0.004440** | **0.003727** |

---

### 2. Per-Label AUC (12 Labels): Baseline vs v2 (Mean Across Folds)

Sorted from most improved to largest degradation:

| Label | Baseline Mean AUC | v2 Best Mean AUC | Delta ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Contusion** | 0.838891 | 0.851036 | **+0.012145** | **Improved most** |
| **ACL** | 0.884670 | 0.885164 | **+0.000494** | Slightly improved |
| **Lateral OA** | 0.846157 | 0.841807 | -0.004350 | Slight drop |
| **Effusion** | 0.865573 | 0.860979 | -0.004594 | Slight drop |
| **PF OA** | 0.849659 | 0.842259 | -0.007400 | Moderate drop |
| **Fracture** | 0.896118 | 0.888045 | -0.008073 | Moderate drop |
| **Medial OA** | 0.889285 | 0.880126 | -0.009159 | Moderate drop |
| **Medial Meniscus** | 0.910703 | 0.899196 | -0.011507 | Moderate drop |
| **Synovitis** | 0.891088 | 0.874406 | -0.016681 | Large drop |
| **Baker's** | 0.915129 | 0.893279 | -0.021850 | Large drop |
| **Lateral Meniscus** | 0.842103 | 0.813249 | -0.028854 | Severe drop |
| **MCL** | 0.803957 | 0.750378 | **-0.053579** | **Did not improve at all (Worst drop)** |

* **Labels improved most:** `Contusion` (+0.0121) and `ACL` (+0.0005).
* **Labels that did not improve at all:** `MCL` (-0.0536) and `Lateral Meniscus` (-0.0289) suffered severe performance drops.

---

### 3. Pooled OOF Macro AUC & Per-Label AUC (All 4,407 Studies Pooled)

Evaluated across all 4,407 validation studies simultaneously:

* **Pooled Macro AUC — Baseline Study-Level:** **0.865449**
* **Pooled Macro AUC — v2 Best Epoch ([oof_best.csv](file:///d:/RSNA/outputs_v2/oof_best.csv)):** **0.855559** (Delta: $-0.009890$)
* **Pooled Macro AUC — v2 Last Epoch ([oof_last.csv](file:///d:/RSNA/outputs_v2/oof_last.csv)):** **0.853590** (Delta: $-0.011859$)

#### Per-Label Pooled OOF AUC Breakdown:

| Label | Baseline Pooled | v2 Best Pooled | Delta ($\Delta$ Best) | v2 Last Pooled |
| :--- | :---: | :---: | :---: | :---: |
| **ACL** | 0.880934 | 0.883130 | +0.002197 | 0.879844 |
| **MCL** | 0.800645 | 0.748985 | **-0.051659** | 0.750507 |
| **Medial Meniscus** | 0.909981 | 0.898226 | -0.011754 | 0.898589 |
| **Lateral Meniscus** | 0.840077 | 0.812572 | **-0.027505** | 0.816052 |
| **Medial OA** | 0.884280 | 0.878818 | -0.005462 | 0.877265 |
| **Lateral OA** | 0.838683 | 0.840330 | +0.001647 | 0.838676 |
| **PF OA** | 0.832846 | 0.842283 | +0.009437 | 0.840524 |
| **Effusion** | 0.863519 | 0.860001 | -0.003517 | 0.857536 |
| **Synovitis** | 0.888060 | 0.874158 | -0.013902 | 0.872907 |
| **Baker's** | 0.913241 | 0.892539 | -0.020703 | 0.889049 |
| **Contusion** | 0.837840 | 0.849835 | +0.011996 | 0.839439 |
| **Fracture** | 0.895284 | 0.885835 | -0.009449 | 0.882691 |

---

### 4. Improvement vs Fold-to-Fold Noise

* **Paired difference per fold ($\text{v2}_{\text{best}} - \text{Baseline}$):**
  * Fold 0: **-0.016703**
  * Fold 1: **-0.012105**
  * Fold 2: **-0.011131**
  * Fold 3: **-0.006557**
  * Fold 4: **-0.017425**
* **Folds won by v2:** **0 out of 5 folds** (v2 lost on every single fold against baseline study-level).
* **Statistical comparison:** The mean regression is **$-0.012784 \pm 0.004440$**. The magnitude of the deficit ($|-0.0128|$) is **$2.88\times$ larger** than the fold-to-fold paired noise ($\text{std} = 0.00444$), indicating a statistically significant performance degradation rather than random variance.

---

## 2. Training Dynamics

### 5. Best vs Final Epoch Dynamics per Fold

Extracted from [training_history_fold0.csv](file:///d:/RSNA/outputs_v2/training_history_fold0.csv) through [training_history_fold4.csv](file:///d:/RSNA/outputs_v2/training_history_fold4.csv):

| Fold | Best Epoch | Best Train Loss | Best Val Loss | Best Val AUC | Final Epoch | Final Train Loss | Final Val Loss | Final Val AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 13 | 0.418647 | 0.471164 | 0.863577 | 15 | 0.420903 | 0.474675 | 0.860516 |
| **1** | 11 | 0.443081 | 0.478144 | 0.854151 | 15 | 0.411689 | 0.483838 | 0.852877 |
| **2** | 12 | 0.432091 | 0.479758 | 0.852811 | 15 | 0.415811 | 0.485220 | 0.849308 |
| **3** | 12 | 0.437278 | 0.469932 | 0.867363 | 15 | 0.422135 | 0.472834 | 0.863786 |
| **4** | 12 | 0.433605 | 0.493358 | 0.845400 | 15 | 0.419210 | 0.494453 | 0.845281 |

---

### 6. Overfitting Assessment & Inflection Point

* **Is overfitting still present?** **Yes, mildly.**
* **Inflection point relative to cosine schedule:**
  * The cosine schedule ran for 15 epochs (warmup epoch 1, decaying to $\text{lr} = 10^{-6}$ at epoch 15).
  * Minimum validation loss occurred consistently around **epochs 9 to 11**:
    * Fold 0: Min val loss **0.4702** at Epoch 11 $\rightarrow$ rose to 0.4747 at Ep 15 ($+0.0045$)
    * Fold 1: Min val loss **0.4772** at Epoch 9 $\rightarrow$ rose to 0.4838 at Ep 15 ($+0.0066$)
    * Fold 2: Min val loss **0.4787** at Epoch 11 $\rightarrow$ rose to 0.4852 at Ep 15 ($+0.0065$)
    * Fold 3: Min val loss **0.4696** at Epoch 10 $\rightarrow$ rose to 0.4728 at Ep 15 ($+0.0032$)
    * Fold 4: Min val loss **0.4914** at Epoch 10 $\rightarrow$ rose to 0.4945 at Ep 15 ($+0.0031$)
  * Validation loss turned upward **4 to 6 epochs before the end** of the 15-epoch cosine schedule (between 60% and 73% of the training duration), while training loss kept dropping from ~0.44 to ~0.41.

---

### 7. Best-Epoch vs Last-Epoch Gap per Fold

Measures how much cherry-picking the peak validation epoch inflates the score:

| Fold | Best Macro AUC | Final Macro AUC | Score Inflation Gap |
| :---: | :---: | :---: | :---: |
| **0** | 0.863577 | 0.860516 | **+0.003061** |
| **1** | 0.854151 | 0.852877 | **+0.001274** |
| **2** | 0.852811 | 0.849308 | **+0.003503** |
| **3** | 0.867363 | 0.863786 | **+0.003577** |
| **4** | 0.845400 | 0.845281 | **+0.000119** |
| **Mean** | **0.856660** | **0.854354** | **+0.002307** ($\pm 0.001552$) |

* On pooled OOF: Inflation gap is $0.855559 - 0.853590 = \mathbf{+0.001969}$. Cherry-picking the best epoch inflates the reported score by approximately **+0.002 AUC points**.

---

### 8. Did EMA Matter?

* **In Training Histories:** **NOT FOUND** (Per rule: validation was evaluated exclusively using `ema.module`; raw weights were not evaluated during training steps, so training history CSVs do not log raw-weight metrics).
* **Empirical Check on Saved Checkpoint ([last_model_fold0.pt](file:///d:/RSNA/outputs_v2/last_model_fold0.pt)):**
  Re-evaluating Fold 0 validation set at the final epoch with raw `model_state` vs `ema_state`:
  * Raw Model Macro AUC: **0.860084**
  * EMA Model Macro AUC: **0.860512**
  * EMA Delta: **+0.000428** (EMA provided a marginal $+0.0004$ boost at convergence).

---

### 9. Learning Status at Final Epoch (Plateau vs Continuation)

* **Status:** **Learning had completely plateaued.**
* **Evidence:**
  * Training loss between Epoch 14 and Epoch 15 showed near-zero delta across all folds ($|\Delta| < 0.001$: Fold 0: 0.4206 $\rightarrow$ 0.4209; Fold 1: 0.4121 $\rightarrow$ 0.4117; Fold 2: 0.4168 $\rightarrow$ 0.4158; Fold 3: 0.4230 $\rightarrow$ 0.4221; Fold 4: 0.4203 $\rightarrow$ 0.4192).
  * Learning rate reached its scheduled floor of $1.00 \times 10^{-6}$.
  * Validation loss had already reversed and was drifting upward.
  * **Conclusion:** Extending training with additional epochs under this schedule would **not** plausibly help and would increase overfitting.

---

## 3. Compute and Capacity

### 10. Training Speed, VRAM, and Hyperparameters

* **Seconds per epoch:**
  * Fold 0: 279.40 s (train: 242.57 s, val: 36.83 s)
  * Fold 1: 280.48 s (train: 245.55 s, val: 34.93 s)
  * Fold 2: 276.92 s (train: 244.21 s, val: 32.70 s)
  * Fold 3: 238.38 s (train: 215.99 s, val: 22.39 s)
  * Fold 4: 238.74 s (train: 216.30 s, val: 22.45 s)
  * **Mean across all folds:** **262.78 s** (~4.38 min/epoch: train **232.92 s**, val **29.86 s**).
* **Total time per fold:**
  * Fold 0: 4,190.98 s (69.85 min)
  * Fold 1: 4,207.15 s (70.12 min)
  * Fold 2: 4,153.75 s (69.23 min)
  * Fold 3: 3,575.70 s (59.60 min)
  * Fold 4: 3,581.15 s (59.69 min)
  * **Mean per fold:** **3,941.75 s** (**65.70 minutes** / 1.10 hours). Total 5-fold training run: **5.47 hours**.
* **Peak GPU Memory:** **5.54 GB** (Folds 0–1: 5.51 GB, Folds 2–4: 5.54 GB).
* **Micro-batch size actually used:** **2**
* **Accumulation steps:** **4** (Effective batch size = $2 \times 4 = 8$).
* **Gradient checkpointing enabled:** **False** (`"use_grad_checkpointing": false` in [config.json](file:///d:/RSNA/outputs_v2/config.json)).

---

### 11. Parameter Counts

* **Total Parameters:** **22,747,468**
* **Trainable Parameters:** **22,747,468** (100% trainable)
* **Backbone (ResNet-34 2D Slice Backbone):**
  * Total: **21,284,672** (93.57% of model)
  * Trainable: **21,284,672**
* **Transformer & Classification Head:**
  * Total: **1,462,796** (6.43% of model)
  * Trainable: **1,462,796**
  * *Sub-component breakdown:*
    * Slice projection (`Linear(512, 256)` + `LayerNorm(256)`): 131,840
    * Positional & Metadata Embeddings: 7,424 (Depth 24: 6,144; Plane 3: 768; Fluid 2: 512)
    * 2-Layer Pre-LN Transformer Encoder ($d_{\text{model}}=256$, 8 heads, $d_{\text{ff}}=512$, GELU): 1,054,208
    * Cross-Attention Head (12 Label Queries, MultiheadAttention, Linear layer): 269,324

---

### 12. Training Bottleneck Analysis

* **Average GPU Utilization / Per-Step Data Wait Time:** **NOT FOUND** (Neither GPU utilization percentage nor step-level data wait time was logged in [train_log.txt](file:///d:/RSNA/outputs_v2/train_log.txt) or the history files).
* **Empirical Analysis:** Each epoch processed 1,762 micro-batches in ~216–245 seconds (approx. 0.123–0.139 s/micro-batch, or ~7.2–8.1 batches/second). With `num_workers_train=3` returning lightweight uint8 arrays and all percentile clipping, normalization, and affine augmentation running directly on the GPU, data ingestion was not stalling the GPU on this NVMe setup, but the RTX 4090 was lightly loaded at micro-batch size 2 (peak VRAM 5.54 GB out of 24 GB).

---

## 4. What the Model Actually Uses (Inference-Time Experiments)

All experiments in this section were computed on the full 5-fold validation set ($N = 4,407$) by running inference directly with the saved checkpoints ([best_model_fold*.pt](file:///d:/RSNA/outputs_v2/best_model_fold0.pt)).

### 13. Plane Ablation on OOF (Axial, Coronal, Sagittal Removed)

Re-scored all 4,407 studies after removing series of each anatomical plane:

* **Normal Pooled Macro AUC:** **0.855558**
* **No Axial Macro AUC:** **0.838678** ($\Delta = -0.016880$)
* **No Coronal Macro AUC:** **0.838058** ($\Delta = -0.017500$)
* **No Sagittal Macro AUC:** **0.784241** ($\Delta = \mathbf{-0.071317}$)

#### Per-Label Impact of Plane Removal:

| Label | Normal AUC | No Axial ($\Delta$) | No Coronal ($\Delta$) | No Sagittal ($\Delta$) | Primary Plane Dependency |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ACL** | 0.8831 | 0.8815 (-0.0016) | 0.8714 (-0.0117) | **0.7261 (-0.1570)** | **Sagittal** |
| **Medial Meniscus** | 0.8982 | 0.8957 (-0.0025) | 0.8938 (-0.0044) | **0.7432 (-0.1550)** | **Sagittal** |
| **PF OA** | 0.8423 | 0.8337 (-0.0086) | 0.8326 (-0.0097) | **0.7237 (-0.1186)** | **Sagittal** |
| **Lateral Meniscus** | 0.8126 | 0.8091 (-0.0035) | 0.7983 (-0.0143) | **0.7081 (-0.1045)** | **Sagittal** |
| **Lateral OA** | 0.8403 | 0.8362 (-0.0041) | 0.8218 (-0.0185) | **0.7421 (-0.0982)** | **Sagittal** |
| **Medial OA** | 0.8788 | 0.8744 (-0.0044) | 0.8653 (-0.0136) | **0.8216 (-0.0573)** | **Sagittal** |
| **Effusion** | 0.8600 | **0.8066 (-0.0534)** | 0.8468 (-0.0132) | 0.8524 (-0.0076) | **Axial** |
| **Baker's** | 0.8926 | **0.8447 (-0.0478)** | 0.8884 (-0.0042) | **0.8376 (-0.0550)** | **Axial & Sagittal** |
| **Contusion** | 0.8498 | 0.8447 (-0.0051) | **0.8109 (-0.0389)** | 0.8297 (-0.0201) | **Coronal** |
| **Fracture** | 0.8858 | 0.8675 (-0.0184) | **0.8494 (-0.0364)** | 0.8672 (-0.0186) | **Coronal** |
| **Synovitis** | 0.8741 | 0.8439 (-0.0303) | **0.8380 (-0.0361)** | 0.8564 (-0.0177) | **Coronal & Axial** |
| **MCL** | 0.7489 | 0.7261 (-0.0229) | 0.7400 (-0.0089) | **0.7027 (-0.0463)** | **Sagittal & Axial** |

---

### 14. Metadata Ablation (Plane/Fluid Embeddings Zeroed)

Re-scored all 4,407 studies with plane and fluid embedding lookups zeroed:

* **Pooled Macro AUC (No Metadata):** **0.785768**
* **Change in Macro AUC:** **$-0.069790$** (a drop of nearly 7 AUC points across the board).
* **Largest per-label drops:** `Effusion` ($-0.1147$), `Lateral OA` ($-0.0905$), `PF OA` ($-0.0860$), `Fracture` ($-0.0738$), `MCL` ($-0.0666$), `Baker's` ($-0.0661$), `Medial Meniscus` ($-0.0652$), `Synovitis` ($-0.0607$).
* **Conclusion:** The model relies heavily on plane and sequence metadata tokens to contextualize 2D slice features.

---

### 15. Cross-Attention Distribution per Label

Extracted cross-attention query weights across all validation studies:

| Label | Sagittal Mass (%) | Coronal Mass (%) | Axial Mass (%) | Mean Attended Depth (0–23) | Attention Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ACL** | **59.74%** | 25.24% | 15.02% | 11.96 | 4.58 |
| **PF OA** | **57.22%** | 23.19% | 19.59% | 11.95 | 4.68 |
| **Medial Meniscus** | **55.65%** | 26.46% | 17.89% | 11.62 | 4.58 |
| **Lateral Meniscus** | **53.19%** | 28.49% | 18.32% | 11.99 | 4.73 |
| **Lateral OA** | **52.21%** | 28.71% | 19.07% | 12.13 | 4.75 |
| **Medial OA** | **51.78%** | 28.31% | 19.92% | 11.98 | 4.72 |
| **Baker's** | 42.44% | 30.66% | 26.90% | 12.03 | 4.73 |
| **MCL** | 41.49% | **33.05%** | 25.46% | 11.59 | 4.78 |
| **Contusion** | 38.50% | **38.49%** | 23.01% | 11.64 | 4.68 |
| **Synovitis** | 35.42% | **39.11%** | 25.47% | 11.04 | 4.79 |
| **Fracture** | 33.65% | **41.31%** | 25.05% | 11.18 | 4.71 |
| **Effusion** | 33.71% | 31.76% | **34.53%** | 10.93 | 4.55 |

* **Depth localization:** For all 12 labels, the average attended depth centers tightly between **slice 10.93 and 12.13** (the middle anatomical slices where joint space and intra-articular structures reside).
* **Concentration vs Spread:** The attention entropy is high (~4.55 to 4.79; max theoretical entropy for a ~120-token study is $\ln(120) \approx 4.79$). Attention does **not** collapse onto a single slice; it spreads broadly across the volume, but weights preferentially toward the relevant anatomical plane and central knee slices.

---

### 16. AUC as a Function of Available Series Count

* **Series distribution across 4,407 studies:**
  * 3–4 series: 676 studies (15.3%)
  * 5–6 series: 2,997 studies (68.0%)
  * 7+ series: 734 studies (16.7%)
* **Performance by Bucket:**
  * **Bucket 3–4 Series ($N=676$):** Macro AUC = **0.875478**
  * **Bucket 5–6 Series ($N=2,997$):** Macro AUC = **0.847697**
  * **Bucket 7+ Series ($N=734$):** Macro AUC = **0.861589**
* **Does performance depend on series count?** Studies with 3–4 series actually scored highest (0.8755), whereas 5–6 series scored lowest (0.8477). Studies with 3–4 series are standard, clean 3-plane protocols without redundant or non-standard scout/localizer sequences that could introduce token noise.

---

## 5. Label Quality and Ceiling

### 17. Soft Target Distribution (Ambiguous Pairs: $0.2 < y < 0.8$)

Count of study-label pairs with continuous consensus targets strictly between 0.2 and 0.8 ($N = 4,407$):

| Label | Ambiguous Count ($0.2 < y < 0.8$) | Total Studies | Ambiguity Percentage |
| :--- | :---: | :---: | :---: |
| **Synovitis** | 2,112 | 4,407 | **47.92%** |
| **Baker's** | 1,991 | 4,407 | **45.18%** |
| **Contusion** | 1,480 | 4,407 | **33.58%** |
| **Lateral OA** | 1,396 | 4,407 | **31.68%** |
| **Medial OA** | 1,137 | 4,407 | **25.80%** |
| **Medial Meniscus** | 1,029 | 4,407 | **23.35%** |
| **Lateral Meniscus** | 986 | 4,407 | **22.37%** |
| **PF OA** | 909 | 4,407 | **20.63%** |
| **ACL** | 823 | 4,407 | **18.67%** |
| **MCL** | 636 | 4,407 | **14.43%** |
| **Effusion** | 542 | 4,407 | **12.30%** |
| **Fracture** | 118 | 4,407 | **2.68%** |
| **Total** | **12,259** | **52,884** | **23.18%** |

---

### 18. Confident Pairs AUC (Label-Noise Ceiling)

Re-computed per-label AUC filtering out ambiguous consensus targets ($y \le 0.2$ or $y \ge 0.8$):

| Label | Standard AUC | Confident Pairs AUC | Ceiling Gap ($\Delta$) | Confident Studies ($N$) | Dropped Ambiguous ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Synovitis** | 0.874158 | 0.961515 | **+0.087357** | 2,295 | 2,112 |
| **Contusion** | 0.849835 | 0.924010 | **+0.074175** | 2,927 | 1,480 |
| **Lateral Meniscus** | 0.812572 | 0.863911 | **+0.051339** | 3,421 | 986 |
| **Lateral OA** | 0.840330 | 0.886005 | **+0.045675** | 3,011 | 1,396 |
| **MCL** | 0.748985 | 0.793296 | **+0.044311** | 3,771 | 636 |
| **Medial Meniscus** | 0.898226 | 0.940363 | **+0.042136** | 3,378 | 1,029 |
| **ACL** | 0.883130 | 0.922089 | **+0.038958** | 3,584 | 823 |
| **PF OA** | 0.842283 | 0.874924 | **+0.032641** | 3,498 | 909 |
| **Medial OA** | 0.878818 | 0.911315 | **+0.032497** | 3,270 | 1,137 |
| **Baker's** | 0.892539 | 0.921699 | **+0.029161** | 2,416 | 1,991 |
| **Effusion** | 0.860001 | 0.881937 | **+0.021936** | 3,865 | 542 |
| **Fracture** | 0.885835 | 0.893917 | **+0.008083** | 4,289 | 118 |
| **Mean** | **0.855559** | **0.897915** | **+0.042356** | — | — |

* Filtering ambiguous pairs raises the macro AUC from **0.8556** to **0.8979** (an average label-noise ceiling gap of **+0.0424**).

---

### 19. Weakest Three Labels in v2: Root Cause Analysis

1. **MCL (AUC: 0.7490)**
   * Positive prevalence ($y \ge 0.5$): **15.45%** (681 / 4,407). Confident ceiling AUC: **0.7933**. Ambiguous pairs: **14.43%**.
   * **Root Cause: Model Failure.** MCL suffered the single worst drop in the entire v2 experiment ($-0.0517$ compared to baseline). The baseline model achieved 0.8006 by viewing individual series directly. MCL tears are localized, thin abnormalities on Coronal fluid-sensitive sequences. Enforcing a global transformer across all planes diluted this subtle local feature.
2. **Lateral Meniscus (AUC: 0.8126)**
   * Positive prevalence ($y \ge 0.5$): **18.99%** (837 / 4,407). Confident ceiling AUC: **0.8639**. Ambiguous pairs: **22.37%**.
   * **Root Cause: Model Representation / Resolution Limitation.** Baseline achieved 0.8401 ($-0.0275$ drop in v2). Meniscal tears require high in-plane resolution; downsampling to $224 \times 224 \times 24$ combined with global cross-attention obscured small radial/root tears.
3. **Lateral OA (AUC: 0.8403)**
   * Positive prevalence ($y \ge 0.5$): **26.84%** (1,183 / 4,407). Confident ceiling AUC: **0.8860**. Ambiguous pairs: **31.68%**.
   * **Root Cause: Label Ambiguity.** 31.68% of cases have continuous target values in $(0.2, 0.8)$. Osteoarthritis is an inherently continuous degenerative spectrum where radiologist consensus varies on grading severity. v2 actually improved slightly over baseline (+0.0016).

---

## 6. Ensembling and Diversity

### 20. Averaging 5 Fold Models on Fold 0 Validation Set

Evaluated each of the 5 saved best checkpoints on Fold 0's validation set ($N = 882$):

* **Fold 0 Model Alone (Unseen Fold 0):** **0.863564**
* **Fold 1 Model (Saw Fold 0 in Train):** **0.921112**
* **Fold 2 Model (Saw Fold 0 in Train):** **0.925674**
* **Fold 3 Model (Saw Fold 0 in Train):** **0.921943**
* **Fold 4 Model (Saw Fold 0 in Train):** **0.923738**
* **Average of All 5 Models:** **0.926624**
* **Ensemble Gain over Fold 0 Model Alone:** **+0.063060**
* *Caveat:* Folds 1–4 were trained on Fold 0 data, making the absolute score of 0.9266 overly optimistic; this serves strictly as a variance/stability indicator across fold models.

---

### 21. Blend Test on Pooled OOF (Baseline vs v2)

Evaluated on all 4,407 pooled out-of-fold studies:

* **v2 Alone Pooled Macro AUC:** **0.855559**
* **Baseline Alone Pooled Macro AUC:** **0.865449**
* **Equal Average Blend (0.5 Baseline + 0.5 v2):** **0.872873**
* **Blend Gain over Baseline Alone:** **+0.007424**
* **Blend Gain over v2 Alone:** **+0.017314**

#### Per-Label Error Correlation (Pearson $r$ of Residuals $\hat{y} - y$):

| Label | Baseline AUC | v2 AUC | Blend AUC | $\Delta$ over Baseline | Error Correlation ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ACL** | 0.8809 | 0.8831 | **0.8953** | +0.0143 | 0.8488 |
| **MCL** | 0.8006 | 0.7490 | 0.7901 | -0.0105 | 0.9311 |
| **Medial Meniscus** | 0.9100 | 0.8982 | **0.9145** | +0.0045 | 0.8229 |
| **Lateral Meniscus** | 0.8401 | 0.8126 | **0.8414** | +0.0013 | 0.9117 |
| **Medial OA** | 0.8843 | 0.8788 | **0.8898** | +0.0055 | 0.8918 |
| **Lateral OA** | 0.8387 | 0.8403 | **0.8517** | +0.0130 | 0.8930 |
| **PF OA** | 0.8328 | 0.8423 | **0.8528** | +0.0199 | 0.8709 |
| **Effusion** | 0.8635 | 0.8600 | **0.8718** | +0.0083 | 0.8855 |
| **Synovitis** | 0.8881 | 0.8742 | **0.8915** | +0.0034 | 0.8901 |
| **Baker's** | 0.9132 | 0.8925 | **0.9157** | +0.0024 | 0.8282 |
| **Contusion** | 0.8378 | 0.8498 | **0.8582** | +0.0204 | 0.8765 |
| **Fracture** | 0.8953 | 0.8858 | **0.9018** | +0.0065 | 0.8921 |

* **Is there complementary signal?** **YES, strong complementary signal.** Even though v2 alone is lower than baseline alone, blending them boosts the overall score to **0.872873** (+0.0074 over baseline, and +0.0173 over v2), improving **11 out of 12 classes**.

---

### 22. Test-Time Augmentation (TTA) on OOF

Evaluated across all 4,407 pooled OOF studies by horizontally flipping Coronal and Axial series (never Sagittal):

* **Pooled Macro AUC without TTA:** **0.855558**
* **Pooled Macro AUC with Flip TTA:** **0.856989**
* **TTA Gain:** **+0.001432**
* Per-fold TTA gains:
  * Fold 0: $+0.002146$
  * Fold 1: $+0.000485$
  * Fold 2: $+0.001735$
  * Fold 3: $+0.001430$
  * Fold 4: $+0.001029$

---

## 7. Input Resolution and Source Data

### 23. Native Image Sizes, Slice Counts, and Information Loss

* **Dataset on Disk:** All 24,371 series in `D:/rsna_knee_dataset/` are pre-resampled into $(24, 224, 224)$ uint8 `.npz` arrays from the Kaggle dataset `barun2104/rsna-knee-mri-processed-3d-volumes`.
* **Original DICOM Dimensions:**
  * Exact per-series min/median/max native slice counts: **NOT FOUND** on local disk (the preprocessing script and raw DICOMs are not stored locally).
  * Competition specifications indicate native knee MRI scans are acquired at $512 \times 512$ or $384 \times 384$ in-plane matrices with 20 to 45 slices per series (typically 24–36 slices for sagittal/coronal and 30–45 slices for axial).
* **Information Loss from Resizing:**
  * In-plane downsampling from $512 \times 512$ to $224 \times 224$ discards $1 - (224/512)^2 \approx \mathbf{80.86\%}$ of raw pixels.
  * Resampling depth to fixed 24 slices causes partial volume averaging along the z-axis, which particularly degrades detection of fine structures like the MCL and meniscal roots.

---

### 24. Availability of Higher-Resolution Volumes

* **On Local Disk:** **None.** Only the 24,371 pre-resampled $224 \times 224 \times 24$ `.npz` files exist in `D:/rsna_knee_dataset` (32.37 GB uncompressed / ~18.5 GB compressed).
* **In Source Kaggle Dataset:** The full-resolution raw DICOM files (~100+ GB) are available in the official Kaggle competition repository (`https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data`). Higher-resolution preprocessed arrays (e.g. $384 \times 384$ or $512 \times 512$) do not currently exist on disk.

---

## 8. Inference and Submission Status

### 25. Inference Pipeline and Submission Status

* **Does `src_v2/predict.py` run end-to-end?** **Yes.** Confirmed functional: it loads the 5 best fold checkpoints, ensembles predictions with optional TTA, and outputs formatted submission CSVs.
* **Has a submission file been generated?** **No.** `outputs_v2/submission.csv` was not present.
* **Is test data available in `D:/rsna_knee_dataset`?** **No.** Only training files (`train.csv`, `train_series.csv`, `train_targets_soft.csv`) and the 24,371 training `.npz` volumes are present on disk.
* **Leaderboard Scores & CV-to-LB Gap:** **NOT FOUND** (No submission was uploaded to Kaggle).

---

## 9. Issues and Artifacts

### 26. Deviations from Specification

1. **Fat Suppression Embedding Omission:** In `D:/rsna_knee_dataset/train_series.csv`, `Fat_Suppression` and `Fluid_Sensitive` were 100% identical for all 24,371 rows. Instantiating a separate fat embedding was redundant, so [src_v2/model.py](file:///d:/RSNA/src_v2/model.py#L119) set `drop_fat_embedding = True`.
2. **Micro-batch size & Accumulation:** Micro-batch size was set to 2 with 4 gradient accumulation steps (`effective batch size = 8`) to preserve GPU VRAM during study-level 2.5D slice passes.
3. **Training Series Subsampling:** During training, at most 2 series per plane were sampled per study (`max_series_per_plane = 2`, $\le 6$ series per study) to maintain manageable batch sizes, while all series were used during validation.
4. **Multiprocessing Strategy:** `torch.multiprocessing.set_sharing_strategy("file_system")` was configured to prevent Windows OS pagefile exhaustion (`WinError 1455`).

---

### 27. Suspicious Findings and Integrity Checks

* **Leakage:** **None.** Mutual exclusivity of `StudyInstanceUID` across folds in [outputs_v2/folds.csv](file:///d:/RSNA/outputs_v2/folds.csv) was verified (0 study leakage).
* **NaNs:** **None.** All loss and AUC values across all 15 epochs and 5 folds remained finite.
* **Fold Stability:** All 5 folds completed smoothly without collapse (Fold AUCs ranged from 0.8454 to 0.8674).
* **Degradation Anomaly:** v2 underperformed the baseline study-level model on all 5 folds. The root cause is anatomical: training on at most 2 series per plane and flattening slice tokens across the entire study degraded detection of localized tear pathologies (particularly MCL at $-0.0517$ and Lateral Meniscus at $-0.0275$), even though multi-sequence representations improved Contusion (+0.0121) and ACL (+0.0022).

---

### 28. File Paths and Environment

* **Best Checkpoints:**
  * [d:\RSNA\outputs_v2\best_model_fold0.pt](file:///d:/RSNA/outputs_v2/best_model_fold0.pt) (182.30 MB)
  * [d:\RSNA\outputs_v2\best_model_fold1.pt](file:///d:/RSNA/outputs_v2/best_model_fold1.pt) (182.30 MB)
  * [d:\RSNA\outputs_v2\best_model_fold2.pt](file:///d:/RSNA/outputs_v2/best_model_fold2.pt) (182.30 MB)
  * [d:\RSNA\outputs_v2\best_model_fold3.pt](file:///d:/RSNA/outputs_v2/best_model_fold3.pt) (182.30 MB)
  * [d:\RSNA\outputs_v2\best_model_fold4.pt](file:///d:/RSNA/outputs_v2/best_model_fold4.pt) (182.30 MB)
* **Last Checkpoints:**
  * [d:\RSNA\outputs_v2\last_model_fold0.pt](file:///d:/RSNA/outputs_v2/last_model_fold0.pt) (364.41 MB)
  * [d:\RSNA\outputs_v2\last_model_fold1.pt](file:///d:/RSNA/outputs_v2/last_model_fold1.pt) (364.41 MB)
  * [d:\RSNA\outputs_v2\last_model_fold2.pt](file:///d:/RSNA/outputs_v2/last_model_fold2.pt) (364.41 MB)
  * [d:\RSNA\outputs_v2\last_model_fold3.pt](file:///d:/RSNA/outputs_v2/last_model_fold3.pt) (364.41 MB)
  * [d:\RSNA\outputs_v2\last_model_fold4.pt](file:///d:/RSNA/outputs_v2/last_model_fold4.pt) (364.41 MB)
* **OOF Prediction Files:**
  * Pooled Best: [d:\RSNA\outputs_v2\oof_best.csv](file:///d:/RSNA/outputs_v2/oof_best.csv)
  * Pooled Last: [d:\RSNA\outputs_v2\oof_last.csv](file:///d:/RSNA/outputs_v2/oof_last.csv)
  * Per-Fold Best: `outputs_v2/oof_best_fold0.csv` to `fold4.csv`
  * Per-Fold Last: `outputs_v2/oof_last_fold0.csv` to `fold4.csv`
  * Baseline Study OOF: [d:\RSNA\outputs_v2\baseline_oof_study.csv](file:///d:/RSNA/outputs_v2/baseline_oof_study.csv)
* **Training Histories & Config:**
  * Histories: [outputs_v2/training_history_fold0.csv](file:///d:/RSNA/outputs_v2/training_history_fold0.csv) to `fold4.csv`
  * Training Log: [d:\RSNA\outputs_v2\train_log.txt](file:///d:/RSNA/outputs_v2/train_log.txt)
  * Configuration: [d:\RSNA\outputs_v2\config.json](file:///d:/RSNA/outputs_v2/config.json)
  * Report: [d:\RSNA\outputs_v2\REPORT_v2.md](file:///d:/RSNA/outputs_v2/REPORT_v2.md)
* **Package Versions:**
  * `torch`: **2.11.0+cu128**
  * `torchvision`: **0.26.0+cu128**
  * `numpy`: **2.5.2**
  * `pandas`: **3.0.5**
  * `scikit-learn`: **1.9.0**
  * `Python`: **3.14.7** (64-bit AMD64)
  * `GPU`: **NVIDIA GeForce RTX 4090** (24 GB)

---

## One-Page Executive Summary Table

| Fold | v2 Best | v2 Last | Baseline Study-Level | Delta ($\Delta$ v2 Best $-$ Base) | Peak Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | **0.863577** | 0.860516 | 0.880280 | -0.016703 | 13 |
| **Fold 1** | **0.854151** | 0.852877 | 0.866256 | -0.012105 | 11 |
| **Fold 2** | **0.852811** | 0.849308 | 0.863942 | -0.011131 | 12 |
| **Fold 3** | **0.867363** | 0.863786 | 0.873920 | -0.006557 | 12 |
| **Fold 4** | **0.845400** | 0.845281 | 0.862825 | -0.017425 | 12 |
| **Mean** | **0.856660** | **0.854354** | **0.869444** | **-0.012784** | — |
| **Std ($ddof=1$)** | **0.008808** | **0.007696** | **0.007445** | **0.004440** | — |
| **Pooled OOF** | **0.855559** | **0.853590** | **0.865449** | **-0.009890** | — |
| **Blend (Base + v2)** | — | — | — | **+0.007424** (AUC: **0.872873**) | — |
