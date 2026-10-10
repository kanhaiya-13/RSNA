Listed directory RSNA
Listed directory src
Listed directory outputs
Viewed requirements.txt:1-14
Viewed cv_summary.csv:1-8
Viewed README.md:1-136
Viewed model.py:1-176
Viewed dataset.py:1-188
Viewed losses.py:1-59
Viewed metrics.py:1-77
Viewed train_baseline.py:1-642
Viewed analyze_training.py:1-456
Viewed training_history.csv:1-32
Viewed training_history_fold0.csv:1-32
Ran command: `python -c "
import os, glob, pandas as pd
print('Outputs dir files:', os.listdir('outputs'))
for f in sorted(glob.glob('outputs/training_history_fold*.csv')):
    df = pd.read_csv(f)
    print(f, 'Shape:', df.shape, 'Max AUC epoch:', df.loc[df['macro_auc'].idxmax(), ['epoch', 'macro_auc', 'train_loss', 'val_loss']].to_dict())
"`
Ran command: `python -c "
import os
path = 'D:/rsna_knee_dataset'
if os.path.exists(path):
    print('Dataset dir exists. Contents sample:', os.listdir(path)[:20])
    print('Total files in dataset dir:', len(os.listdir(path)))
else:
    print('Dataset dir does not exist.')
"`
Ran command: `python -c "
import os, glob
print(glob.glob('D:/rsna_knee_dataset/*.csv'))
"`
Ran command: `python -c "
import glob
print('All files in RSNA repo:')
for p in sorted(glob.glob('**/*', recursive=True)):
    if not p.startswith('.git') and not p.startswith('.venv'):
        print(p)
"`
Ran command: `python -c "
import pandas as pd
import numpy as np
import torch
from src.model import KneeClassifier

# 1. Dataset stats
train_soft = pd.read_csv('D:/rsna_knee_dataset/train_targets_soft.csv')
train_series = pd.read_csv('D:/rsna_knee_dataset/train_series.csv')
print('--- Dataset Info ---')
print('train_soft shape:', train_soft.shape)
print('train_soft columns:', list(train_soft.columns))
print('train_series shape:', train_series.shape)
print('train_series columns:', list(train_series.columns))

# Merged info
merged = train_soft.merge(train_series[['StudyInstanceUID', 'SeriesInstanceUID']], on='StudyInstanceUID', how='inner')
print('merged shape:', merged.shape)
print('unique studies in train_soft:', train_soft['StudyInstanceUID'].nunique())
print('unique studies in merged:', merged['StudyInstanceUID'].nunique())
print('unique series in merged:', merged['SeriesInstanceUID'].nunique())

labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', \"Baker's\", 'Contusion', 'Fracture']
print('\n--- Class Distribution (in train_soft) ---')
for col in labels:
    mean_val = train_soft[col].mean()
    pos_05 = (train_soft[col] >= 0.5).sum()
    pos_gt0 = (train_soft[col] > 0).sum()
    total = len(train_soft)
    print(f'{col:<20}: mean={mean_val:.4f}, count>=0.5={pos_05}/{total} ({pos_05/total*100:.2f}%), count>0={pos_gt0}/{total} ({pos_gt0/total*100:.2f}%)')

print('\n--- Class Distribution (in merged study-series rows) ---')
for col in labels:
    mean_val = merged[col].mean()
    pos_05 = (merged[col] >= 0.5).sum()
    pos_gt0 = (merged[col] > 0).sum()
    total = len(merged)
    print(f'{col:<20}: mean={mean_val:.4f}, count>=0.5={pos_05}/{total} ({pos_05/total*100:.2f}%), count>0={pos_gt0}/{total} ({pos_gt0/total*100:.2f}%)')

# Check train.csv if available
if os.path.exists('D:/rsna_knee_dataset/train.csv'):
    train_orig = pd.read_csv('D:/rsna_knee_dataset/train.csv')
    print('\n--- train.csv info ---')
    print('train.csv shape:', train_orig.shape)
    print('train.csv columns:', list(train_orig.columns))
    print(train_orig.head(2))
"`
Created inspect_all.py
Ran command: `python "C:\Users\23101A0034\.gemini\antigravity-ide\brain\95edc411-73ff-4bce-a74c-2658e5705dcb\scratch\inspect_all.py"`
Edited inspect_all.py
Ran command: `python "C:\Users\23101A0034\.gemini\antigravity-ide\brain\95edc411-73ff-4bce-a74c-2658e5705dcb\scratch\inspect_all.py"`
Ran command: `.\.venv\Scripts\python.exe "C:\Users\23101A0034\.gemini\antigravity-ide\brain\95edc411-73ff-4bce-a74c-2658e5705dcb\scratch\inspect_all.py"`

I have started the inspection script using the project's virtual environment and am awaiting the output to collect the exact parameters, data statistics, checkpoint information, and performance metrics across all 5 folds.

Used tool: manage_task

I am monitoring the execution of the inspection task.

Viewed task-56.log

I am waiting for the execution script to complete so I can review the output.

Viewed task-56.log

I'll wait for the background script to complete.

Viewed task-56.log

I am pausing tool calls to allow the background script to complete.

Used tool: manage_task

I will wait for the script to finish and return the full inspection data.

Edited inspect_all.py
Ran command: `.\.venv\Scripts\python.exe "C:\Users\23101A0034\.gemini\antigravity-ide\brain\95edc411-73ff-4bce-a74c-2658e5705dcb\scratch\inspect_all.py"`
Ran command: `.\.venv\Scripts\python.exe -c "
import torch, torchvision, sklearn, numpy, pandas, sys
print('Python:', sys.version)
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('numpy:', numpy.__version__)
print('pandas:', pandas.__version__)
print('scikit-learn:', sklearn.__version__)

from src.model import KneeClassifier
model = KneeClassifier(num_classes=12, backbone_name='resnet34', pretrained=False, pool_mode='mean', dropout=0.3)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total parameters:', total_params)
print('Trainable parameters:', trainable_params)
"`
Ran command: `.\.venv\Scripts\python.exe -m pip list`
Searched for "metric"
Viewed knee-ai-frontend-spec.md:1-60
Ran command: `.\.venv\Scripts\python.exe -c "
import pandas as pd
series_df = pd.read_csv('D:/rsna_knee_dataset/train_series.csv')
print('Unique studies in train_series:', series_df['StudyInstanceUID'].nunique())
print('Total series in train_series:', len(series_df))
print('Anatomical planes distribution:\n', series_df['Anatomical_Plane'].value_counts())
print('Fluid Sensitive distribution:\n', series_df['Fluid_Sensitive'].value_counts(dropna=False))
print('Fat Suppression distribution:\n', series_df['Fat_Suppression'].value_counts(dropna=False))
series_per_study = series_df.groupby('StudyInstanceUID').size()
print('Series per study: min={}, max={}, mean={:.2f}, median={}'.format(series_per_study.min(), series_per_study.max(), series_per_study.mean(), series_per_study.median()))
"`
Ran command: `.\.venv\Scripts\python.exe -c "
import os, time
for fold in range(5):
    f_hist = f'outputs/training_history_fold{fold}.csv'
    f_ckpt = f'outputs/best_model_fold{fold}.pt'
    print(f'Fold {fold} hist mtime:', time.ctime(os.path.getmtime(f_hist)), '| ckpt mtime:', time.ctime(os.path.getmtime(f_ckpt)))
"`
Ran command: `.\.venv\Scripts\python.exe -c "
import numpy as np
best_aucs = [0.8403263565566279, 0.8277046374408367, 0.829611254838459, 0.8352994526403111, 0.8186947163515329]
print('Mean:', np.mean(best_aucs))
print('Std (ddof=0, pop):', np.std(best_aucs, ddof=0))
print('Std (ddof=1, sample):', np.std(best_aucs, ddof=1))
"`
Ran command: `.\.venv\Scripts\python.exe -c "
import pandas as pd
for fold in range(5):
    df = pd.read_csv(f'outputs/training_history_fold{fold}.csv')
    best_ep = int(df.loc[df['macro_auc'].idxmax(), 'epoch'])
    t_loss_start = df.loc[0, 'train_loss']
    v_loss_start = df.loc[0, 'val_loss']
    t_loss_best = df.loc[best_ep-1, 'train_loss']
    v_loss_best = df.loc[best_ep-1, 'val_loss']
    t_loss_end = df.loc[29, 'train_loss']
    v_loss_end = df.loc[29, 'val_loss']
    auc_best = df.loc[best_ep-1, 'macro_auc']
    auc_end = df.loc[29, 'macro_auc']
    print(f'Fold {fold}: Best epoch {best_ep}')
    print(f'   Epoch 1 : train_loss={t_loss_start:.4f}, val_loss={v_loss_start:.4f}, auc={df.loc[0, \"macro_auc\"]:.4f}')
    print(f'   Epoch {best_ep:<2}: train_loss={t_loss_best:.4f}, val_loss={v_loss_best:.4f}, auc={auc_best:.4f}')
    print(f'   Epoch 30: train_loss={t_loss_end:.4f}, val_loss={v_loss_end:.4f}, auc={auc_end:.4f}')
"`
Ran command: `.\.venv\Scripts\python.exe -c "
import pandas as pd
for fold in range(5):
    df = pd.read_csv(f'outputs/training_history_fold{fold}.csv')
    best_ep = int(df.loc[df['macro_auc'].idxmax(), 'epoch'])
    t_loss_start = df.loc[0, 'train_loss']
    v_loss_start = df.loc[0, 'val_loss']
    t_loss_best = df.loc[best_ep-1, 'train_loss']
    v_loss_best = df.loc[best_ep-1, 'val_loss']
    t_loss_end = df.loc[29, 'train_loss']
    v_loss_end = df.loc[29, 'val_loss']
    auc_best = df.loc[best_ep-1, 'macro_auc']
    auc_end = df.loc[29, 'macro_auc']
    auc_start = df.loc[0, 'macro_auc']
    print(f'Fold {fold}: Best epoch {best_ep}')
    print(f'   Epoch 1 : train_loss={t_loss_start:.4f}, val_loss={v_loss_start:.4f}, auc={auc_start:.4f}')
    print(f'   Epoch {best_ep:<2}: train_loss={t_loss_best:.4f}, val_loss={v_loss_best:.4f}, auc={auc_best:.4f}')
    print(f'   Epoch 30: train_loss={t_loss_end:.4f}, val_loss={v_loss_end:.4f}, auc={auc_end:.4f}')
"`
Ran command: `.\.venv\Scripts\python.exe -c "
import glob
print('OOF / prediction files found:')
for p in glob.glob('outputs/**/*.*', recursive=True):
    if 'oof' in p.lower() or 'pred' in p.lower():
        print(' -', p)
for p in glob.glob('*.*'):
    if 'oof' in p.lower() or 'pred' in p.lower():
        print(' -', p)
"`

# RSNA Knee Abnormality Detection — 5-Fold Baseline Audit & Performance Record

---

## 1. Task and Metric

### 1. Prediction Targets
* **Exact Label Names (12 labels in order):**
  1. `ACL` (Anterior Cruciate Ligament tear)
  2. `MCL` (Medial Collateral Ligament tear)
  3. `Medial Meniscus` (Medial meniscus tear)
  4. `Lateral Meniscus` (Lateral meniscus tear)
  5. `Medial OA` (Medial compartment osteoarthritis)
  6. `Lateral OA` (Lateral compartment osteoarthritis)
  7. `PF OA` (Patellofemoral osteoarthritis)
  8. `Effusion` (Joint effusion)
  9. `Synovitis`
  10. `Baker's` (Baker's cyst / popliteal cyst)
  11. `Contusion` (Bone contusion)
  12. `Fracture`
* **Count:** 12
* **Type:** **Multi-label classification** (multiple abnormalities can occur simultaneously in the same knee). The targets in [`train_targets_soft.csv`](file:///D:/rsna_knee_dataset/train_targets_soft.csv) are soft continuous probability labels in $[0, 1]$ reflecting consensus radiologist annotations.

---

### 2. Official Competition Metric vs Validation Metric
* **Official Competition Metric:** Macro-averaged ROC-AUC (mean of the Area Under the Receiver Operating Characteristic Curve across all 12 abnormality classes).
* **Validation Metric Used:** Macro-averaged ROC-AUC computed via [`compute_macro_auc`](file:///d:/RSNA/src/metrics.py#L16-L76).
* **Are they the same?** Yes. Both evaluate macro-averaged ROC-AUC across the 12 abnormality categories. The implementation in [`src/metrics.py`](file:///d:/RSNA/src/metrics.py) explicitly binarizes continuous soft targets at threshold $\ge 0.5$ (`col_true >= 0.5`) before passing to `roc_auc_score` so that standard ROC-AUC can be calculated.

---

### 3. Metric Implementation
The metric is a custom wrapper in [`src/metrics.py`](file:///d:/RSNA/src/metrics.py) that calls `scikit-learn`'s `roc_auc_score` per class and averages valid classes:

```python
def compute_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute per-class and macro-averaged ROC-AUC.
    """
    if labels is None:
        labels = [f"class_{i}" for i in range(y_true.shape[1])]

    per_class:  Dict[str, float] = {}
    valid_aucs: List[float]      = []

    for i, lbl in enumerate(labels):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        # For continuous soft pseudo-labels (floats in [0, 1]), threshold at >= 0.5 for ROC-AUC
        if col_true.dtype.kind == 'f' and not np.all(np.isin(col_true, [0, 1])):
            col_true_eval = (col_true >= 0.5).astype(int)
        else:
            col_true_eval = col_true.astype(int)

        if len(np.unique(col_true_eval)) < 2:
            # AUC undefined when only one class is present in split
            per_class[lbl] = float("nan")
            continue

        try:
            auc = roc_auc_score(col_true_eval, col_pred)
            per_class[lbl] = float(auc)
            valid_aucs.append(float(auc))
        except Exception:
            per_class[lbl] = float("nan")

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return macro_auc, per_class
```

---

## 2. Data

### 4. Modalities, Sequences, Planes, Studies, Slices, Size, and Format
* **Modality:** Knee MRI (Magnetic Resonance Imaging).
* **Planes & Sequences ([`train_series.csv`](file:///D:/rsna_knee_dataset/train_series.csv)):**
  * **Anatomical Planes:** Sagittal (9,864 series), Coronal (8,609 series), Axial (5,898 series).
  * **Weighting / Suppression:** Fluid Sensitive (`1`: 14,010 series, `0`: 10,361 series), Fat Suppression (`1`: 14,010 series, `0`: 10,361 series).
* **Number of Studies:** **4,407** unique studies in [`train_targets_soft.csv`](file:///D:/rsna_knee_dataset/train_targets_soft.csv).
* **Number of Series:** **24,371** total series (an average of 5.53 series per study; range: 3 to 14 series per study).
* **Slices per Study / Volume:** Each single series volume has **24 slices** (`FIXED_DEPTH = 24`). Across a whole study (mean 5.53 series), there are ~72 to 336 slices total.
* **Image Size:** $224 \times 224$ pixels (`IMG_HEIGHT = 224`, `IMG_WIDTH = 224`).
* **Format:** `.npz` (NumPy compressed archive format containing key `'data'`, stored as `uint8`, pre-processed from original competition DICOMs from Kaggle dataset `barun2104/rsna-knee-mri-processed-3d-volumes`).

---

### 5. Class Distribution per Label & Imbalance Severity
Computed on both the 4,407 unique studies and the 24,371 merged study-series training rows (binarized at threshold $\ge 0.5$):

| Label | Positive Studies ($N=4407$) | Prevalence (%) | Negative Studies | Series-level Pos ($N=24371$) | Series Prev (%) | Imbalance Ratio (Neg:Pos) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fracture** | 313 | 7.10% | 4,094 | 1,733 | 7.11% | **13.1 : 1** |
| **Synovitis** | 569 | 12.91% | 3,838 | 3,272 | 13.43% | **6.4 : 1** |
| **MCL** | 681 | 15.45% | 3,726 | 3,935 | 16.15% | **5.2 : 1** |
| **Lateral Meniscus** | 837 | 18.99% | 3,570 | 4,815 | 19.76% | **4.1 : 1** |
| **Contusion** | 959 | 21.76% | 3,448 | 5,430 | 22.28% | **3.5 : 1** |
| **ACL** | 951 | 21.58% | 3,456 | 5,674 | 23.28% | **3.3 : 1** |
| **Baker's** | 1,091 | 24.76% | 3,316 | 5,790 | 23.76% | **3.2 : 1** |
| **Lateral OA** | 1,183 | 26.84% | 3,224 | 6,642 | 27.25% | **2.7 : 1** |
| **Medial OA** | 1,634 | 37.08% | 2,773 | 9,030 | 37.05% | **1.7 : 1** |
| **Medial Meniscus** | 1,955 | 44.36% | 2,452 | 10,914 | 44.78% | **1.2 : 1** |
| **PF OA** | 2,009 | 45.59% | 2,398 | 11,065 | 45.40% | **1.2 : 1** |
| **Effusion** | 2,625 | 59.56% | 1,782 | 14,507 | 59.53% | **0.68 : 1** (majority pos) |

* **Imbalance Severity:** High for `Fracture` (7.1%), `Synovitis` (12.9%), and `MCL` (15.5%); moderate for `Lateral Meniscus`, `Contusion`, `ACL`, and `Baker's` (19–25%); relatively balanced for `Medial OA`, `Medial Meniscus`, `PF OA`, and `Effusion` (37–60%).

---

### 6. Volume to Model Input Pipeline
* **Input Representation:** 2.5D CNN architecture ([`src/model.py`](file:///d:/RSNA/src/model.py)).
* **Flow:**
  1. Each individual series is loaded as a 3D volume `[D, H, W]` = `[24, 224, 224]`, shaped into `[1, 24, 224, 224]`.
  2. Batch shape: `[B, 1, 24, 224, 224]`.
  3. Depth folding: The depth dimension is folded into the batch dimension via `x.permute(0, 2, 1, 3, 4).reshape(B * 24, 1, 224, 224)`.
  4. 2D Slice Encoding: All $B \times 24$ slices pass through the 2D `SliceEncoder` (ResNet-34) in a single GPU forward pass $\rightarrow [B \times 24, 512]$.
  5. Unfolding & Pooling: Reshaped to `[B, 24, 512]` and aggregated along depth via **Mean Pooling** $\rightarrow [B, 512]$.
  6. Multi-label Head: `Dropout(0.3) -> Linear(512, 12)` $\rightarrow [B, 12]$ raw logits.
* **Plane Handling:** Evaluated **per series independently**. Planes are not concatenated or fused prior to classification; each series volume is treated as an independent sample with the study's soft labels.

---

### 7. Preprocessing
1. **Intensity Normalization:** Min-max scaling per volume to $[0, 1]$ in [`src/dataset.py`](file:///d:/RSNA/src/dataset.py#L93-L98) via `(vol - vol.min()) / (vol.max() - vol.min())`. If $(v_{max} - v_{min}) < 10^{-6}$, returns zeros. No Hounsfield windowing or z-score normalization.
2. **Depth Resampling:** [`_sample_depth`](file:///d:/RSNA/src/dataset.py#L101-L127) uses 1D linear interpolation across the depth axis to target `fixed_depth = 24` (fast-path skipped if $D=24$).
3. **Spatial Resizing:** [`_resize_slices`](file:///d:/RSNA/src/dataset.py#L129-L155) uses trilinear interpolation via `torch.nn.functional.interpolate` to $(224, 224)$ (fast-path skipped if already $224 \times 224$).
4. **Cropping:** None.
5. **Train vs Inference Inconsistencies:** None. All normalization, depth sampling, and resizing steps are identical between training and validation/inference. The only difference is data augmentation (`augment=True` vs `augment=False`).

---

### 8. Augmentations Used
* **List of Augmentations:**
  * **Random Horizontal Flip:** `p = 0.5` ([`src/dataset.py:L176-L177`](file:///d:/RSNA/src/dataset.py#L176-L177)):
    ```python
    if self.augment and random.random() > 0.5:
        vol = vol.flip(dims=[-1])
    ```
* **Applied per slice or per volume?** Applied **per volume**. `vol.flip(dims=[-1])` flips the entire 3D tensor across the spatial width axis simultaneously, ensuring spatial consistency across all 24 slices.
* **Other Augmentations:** None (no rotation, scaling, brightness/contrast, affine, cutout, or Albumentations).

---

## 3. Cross-Validation

### 9. Fold Creation & Splitting Scheme
* **Method:** `sklearn.model_selection.GroupKFold(n_splits=5)`.
* **Group Column:** `StudyInstanceUID` (`groups = df["StudyInstanceUID"].values`).
* **Patient/Study Leakage Prevention:** Yes. Because all series from the same study share the same `StudyInstanceUID`, all series of any given study are strictly grouped together and placed exclusively in either the train split or the validation split. No study appears in both train and validation.
* **Stratification:** **None.** Standard `GroupKFold` was used, which does not stratify by multi-label targets.
* **Seed:** `seed = 42` is passed in code, though `GroupKFold` in scikit-learn is deterministic based on group order.

---

### 10. Possible Leakage
* **Between Folds / Train & Val:**
  * **Study Leakage:** None. GroupKFold isolates `StudyInstanceUID`.
  * **Feature / Normalization Leakage:** None. Min-max normalization is computed strictly on-the-fly per individual volume.
  * **Label Leakage:** None.
  * *Note on Patient ID:* The competition dataset identifies cases by `StudyInstanceUID`. If the same patient had multiple separate studies under different `StudyInstanceUID`s without a linked patient ID, that cannot be isolated, but at the study level isolation is complete.

---

## 4. Model

### 11. Architecture Specifications
* **Backbone:** `resnet34` ([`torchvision.models.resnet34`](file:///d:/RSNA/src/model.py#L27-L109)).
* **Pretrained Weights & Source:** `weights='DEFAULT'` (`ResNet34_Weights.IMAGENET1K_V1`) from torchvision (ImageNet-1k pre-training).
* **Input Channels:** 1 channel (greyscale MRI).
* **Adaptation for 1 Channel:** The first convolution layer `backbone.conv1` ($64 \times 3 \times 7 \times 7$) was replaced with a 1-channel `nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)`. Pretrained weights were initialized by **averaging the 3 RGB channel filters**:
  ```python
  self.conv1.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
  ```
  *(stated reason in code: preserves ImageNet feature statistics).*
* **Spatial Pooling:** Global Average Pooling (`backbone.avgpool` $\rightarrow [B \times 24, 512]$).
* **Depth Pooling:** 1D Mean Pooling over depth dimension $D=24$ (`feats.mean(dim=1)` $\rightarrow [B, 512]$).
* **Head Design:** `nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, 12))`.
* **Number of Outputs:** 12 raw logits (independent binary classifiers).

---

### 12. Parameter Counts
* **Total Parameters:** **21,284,556** (~21.28 M)
* **Trainable Parameters:** **21,284,556** (100% trainable; no layers frozen)

---

### 13. Slice-Level Aggregation
* **Mechanism:** **1D Mean Pooling** across the 24 depth slices (`feats.mean(dim=1)` in [`src/model.py:L173`](file:///d:/RSNA/src/model.py#L173)).
* **Sequence / Attention Model:** None (no multi-head attention, MIL, LSTM, or Transformer).

---

## 5. Training Setup

### 14. Loss, Optimizer, LR, and Scheduling
* **Loss Function:** `nn.BCEWithLogitsLoss()` via [`compute_bce_loss`](file:///d:/RSNA/src/losses.py#L17-L59). Continuous soft targets are supported natively.
  * **Class weights / pos_weight / focal parameters:** None.
* **Optimizer:** `torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)`.
* **Initial Learning Rate:** `1e-4` ($0.0001$).
* **Weight Decay:** `1e-2` ($0.01$).
* **Scheduler:** `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)`.
* **Warmup:** None (0 warmup epochs).

---

### 15. Training Hyperparameters
* **Batch Size:** 8 per GPU (`CFG.BATCH_SIZE = 8`).
* **Gradient Accumulation:** 1 step (`CFG.ACCUM_STEPS = 1`, effective batch size = 8).
* **Epochs:** 30 epochs per fold.
* **Early Stopping:** None configured (all 30 epochs ran; the best checkpoint was tracked and saved dynamically).
* **Mixed Precision:** Enabled (`CFG.MIXED_PREC = True`) using `torch.cuda.amp.autocast(dtype=torch.float16)` and `torch.cuda.amp.GradScaler()`.
* **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

---

### 16. Checkpoint Strategy
* **Checkpoint Kept:** Best validation checkpoint based on **`macro_auc`** (mean ROC-AUC across all 12 classes).
* **Saved To:** [`outputs/best_model_fold{fold}.pt`](file:///d:/RSNA/outputs/).

---

### 17. Hardware & Training Time per Fold
* **Hardware:**
  * CPU: Intel Core i9-13900K (24 cores / 32 threads)
  * RAM: 64 GB DDR5-4000
  * GPU: NVIDIA GPU with 24 GB VRAM (CUDA 12.8, TF32 enabled, cuDNN benchmark enabled)
  * OS: Windows 11 Pro 64-bit
* **Training Time per Fold:** ~1.8 to 2.2 hours (~110–130 minutes per 30-epoch fold, ~3.6–4.4 minutes per epoch across 2,437 training steps).

---

### 18. Reproducibility & Seeds
* **Random Seed:** `42` set across Python `random`, NumPy `np.random`, and PyTorch `torch.manual_seed` / `torch.cuda.manual_seed_all`.
* **Reproducibility:** **Partially reproducible.** In [`train_baseline.py:L157-L158`](file:///d:/RSNA/train_baseline.py#L157-L158), `torch.backends.cudnn.benchmark = True` and `torch.backends.cudnn.deterministic = False` are explicitly set to optimize fixed-shape convolution kernel throughput, which introduces slight floating-point non-determinism across runs.

---

## 6. Results

### 19. Best Epoch & Losses per Fold
Values extracted directly from [`outputs/training_history_fold*.csv`](file:///d:/RSNA/outputs/):

| Fold | Best Macro ROC-AUC | Best Epoch | Train BCE Loss at Best Epoch | Val BCE Loss at Best Epoch |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | **0.840326** | 10 | 0.434507 | 0.497184 |
| **Fold 1** | **0.827705** | 8 | 0.455531 | 0.500056 |
| **Fold 2** | **0.829611** | 7 | 0.468263 | 0.501343 |
| **Fold 3** | **0.835299** | 8 | 0.457138 | 0.495529 |
| **Fold 4** | **0.818695** | 11 | 0.418035 | 0.525210 |

---

### 20. Mean and Standard Deviation Across Folds
* **Mean Validation Macro ROC-AUC:** **0.830327**
* **Standard Deviation Across Folds:**
  * Population Std ($\text{ddof}=0$): **0.007313**
  * Sample Std ($\text{ddof}=1$): **0.008176**

---

### 21. Per-Label ROC-AUC Breakdown (at Best Epoch per Fold)
From [`outputs/cv_summary.csv`](file:///d:/RSNA/outputs/cv_summary.csv):

| Label | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Mean AUC** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Medial Meniscus** | 0.868068 | 0.865508 | 0.871074 | 0.865663 | 0.860516 | **0.866166** |
| **Medial OA** | 0.885558 | 0.846082 | 0.866430 | 0.868279 | 0.851799 | **0.863630** |
| **Synovitis** | 0.886260 | 0.855403 | 0.852974 | 0.853134 | 0.863117 | **0.862178** |
| **Baker's** | 0.855059 | 0.841805 | 0.891123 | 0.870076 | 0.846063 | **0.860825** |
| **Fracture** | 0.878259 | 0.854095 | 0.838001 | 0.868180 | 0.809378 | **0.849583** |
| **ACL** | 0.863529 | 0.855907 | 0.833037 | 0.832305 | 0.852032 | **0.847362** |
| **Effusion** | 0.829289 | 0.832413 | 0.834061 | 0.852778 | 0.827722 | **0.835252** |
| **PF OA** | 0.818701 | 0.808571 | 0.822283 | 0.840677 | 0.790519 | **0.816150** |
| **Lateral OA** | 0.813708 | 0.813332 | 0.820758 | 0.804047 | 0.808423 | **0.812054** |
| **Contusion** | 0.830419 | 0.805673 | 0.768960 | 0.799154 | 0.787762 | **0.798394** |
| **Lateral Meniscus** | 0.791909 | 0.798267 | 0.786991 | 0.790573 | 0.782648 | **0.790078** |
| **MCL** | 0.763158 | 0.755399 | 0.769641 | 0.778727 | 0.744358 | **0.762257** |
| **Macro Average** | **0.840326** | **0.827705** | **0.829611** | **0.835299** | **0.818695** | **0.830327** |

---

### 22. Training vs Validation Dynamics
* **Overfitting / Underfitting Diagnosis:** Distinct **overfitting** occurs starting around epochs 7–11 across all 5 folds.
  * **Epochs 1 to 8–10:** Train loss drops from ~0.560 to ~0.440; validation loss drops from ~0.535 to ~0.495; validation Macro-AUC increases from ~0.760 to ~0.830.
  * **Epochs 11 to 30:** Train loss continues to fall steeply down to **0.324–0.326**, while validation loss reverses direction and climbs to **0.504–0.531**. Validation Macro-AUC degrades by ~0.015–0.020 (dropping to ~0.807–0.824).
* **Validation Stability:** Validation curves are stable and smooth (no numerical instabilities or NaN losses), but the generalization gap widens continuously after epoch 10 due to weak regularization and lack of heavy spatial augmentations.

---

### 23. Out-Of-Fold (OOF) Prediction File & Scores
* **OOF Prediction Array File:** **NOT FOUND** (the training script does not dump raw out-of-fold probability arrays to disk; it only saves per-epoch metric CSVs and model checkpoints).
* **OOF Summary Path:** [`outputs/cv_summary.csv`](file:///d:/RSNA/outputs/cv_summary.csv) and [`outputs/plots/05_oof_summary.png`](file:///d:/RSNA/outputs/plots/05_oof_summary.png).
* **OOF Metrics (Averaged across folds from summary):**
  * **Overall Macro AUC:** **0.830327**
  * Per-label AUCs: Medial Meniscus (0.8662), Medial OA (0.8636), Synovitis (0.8622), Baker's (0.8608), Fracture (0.8496), ACL (0.8474), Effusion (0.8353), PF OA (0.8162), Lateral OA (0.8121), Contusion (0.7984), Lateral Meniscus (0.7901), MCL (0.7623).

---

### 24. Leaderboard Scores & CV-to-LB Gap
* **Public Leaderboard Score:** **NOT FOUND** (no submission log or leaderboard submission file recorded in repository).
* **Private Leaderboard Score:** **NOT FOUND**.
* **CV vs Leaderboard Gap:** **NOT FOUND**.

---

## 7. Inference

### 25. Fold Model Combination
* **Ensembling Strategy:** **NOT FOUND** in codebase (no dedicated standalone inference or submission script has been written yet).
* *Note:* The clinical spec ([`knee-ai-frontend-spec.md:L6-L12`](file:///d:/RSNA/knee-ai-frontend-spec.md#L6-L12)) specifies an ensemble averaging sigmoid probabilities across the 5 folds.

---

### 26. Test-Time Augmentation (TTA)
* **TTA Used:** **None / NOT FOUND** (no inference script exists, and `validate_one_epoch` runs with `augment=False`).

---

### 27. Thresholds and Post-Processing
* **Post-Processing / Thresholds:** **None.**
* Raw sigmoid probabilities are passed directly to `compute_macro_auc`. No temperature calibration, per-class threshold tuning, or multi-series aggregation functions are implemented in the code.

---

## 8. Diagnosis

### 28. Lowest Performing Labels & Evidence-Based Explanations
1. **`MCL` (Mean AUC = 0.7623 — lowest of all 12 labels):**
   * *Anatomical / Representational cause:* The Medial Collateral Ligament is a narrow, superficial coronal structure. Because the model uniformly downsamples volumes to 24 slices and applies depth-wise **mean pooling**, the small localized signal of an MCL tear is averaged out by the remaining ~22 normal slices.
   * *Data imbalance:* Only 15.45% positive cases in the dataset.
2. **`Lateral Meniscus` (Mean AUC = 0.7901 vs Medial Meniscus at 0.8662):**
   * *Anatomical cause:* Lateral meniscus tears are focal, subtle abnormalities occurring in 1–2 slices on sagittal views. Mean pooling dilutes this local signal.
   * *Data imbalance:* Lateral meniscus tears are less than half as frequent (18.99%) as medial meniscus tears (44.36%).
3. **`Contusion` (Mean AUC = 0.7984):**
   * *Sequence dependence:* Bone contusions (marrow edema) are prominently visible only on fluid-sensitive fat-suppressed sequences (`Fluid_Sensitive=1`, `Fat_Suppression=1`). The current model receives individual single series without sequence metadata conditioning, forcing it to look for contusions on non-fluid-sensitive series where they cannot be seen.

---

### 29. Primary Bottleneck
* **Primary Limitation:** **Input Representation and Spatial/Sequence Aggregation (Architectural/Data Pipeline).**
* **Justification:**
  1. *Series Independence & Study Label Noise:* A patient study contains 3–14 series across 3 planes (Sagittal, Coronal, Axial). Assigning the whole-study target to every individual series creates label noise (e.g. an ACL tear visible only on Sagittal is assigned as positive to an Axial series, forcing the model to learn spurious features).
  2. *Mean Pooling Dilution:* Focal tears (MCL, meniscus, subtle fractures) are diluted when averaged across 24 slices.
  3. *Premature Overfitting from Weak Regularization:* Model capacity (21.3M parameters) is adequate, but with only horizontal flips for augmentation, the model begins overfitting by epoch 8–10 (train loss drops to 0.32 while validation loss climbs to 0.53).

---

### 30. Top 3 Highest-Impact Improvements (Ranked)
1. **Multi-Plane / Multi-Sequence Joint Study Aggregation (Highest Impact):**
   Group the 3–6 series belonging to a study (Sagittal, Coronal, Axial) into a multi-input architecture so the model evaluates the full knee examination simultaneously instead of predicting on isolated single series.
2. **Attention-Based Slice Pooling / Multiple Instance Learning (MIL):**
   Replace uniform depth mean-pooling with gated attention pooling or a Transformer slice aggregator. This allows the network to automatically weight and localize the 1–3 slices containing the tear/abnormality without drowning in normal tissue slices.
3. **Advanced Medical Augmentation & Class Imbalance Handling:**
   Introduce 3D affine transforms, rotations ($\pm 15^\circ$), elastic deformations, slice dropout, Mixup/CutMix, and incorporate focal loss or positive class weighting (`pos_weight`) to boost rare classes like `Fracture` (7.1%), `Synovitis` (12.9%), and `MCL` (15.5%).

---

## 9. Artifacts

### 31. Paths of Checkpoints, Configs, Logs, and Scripts
* **Checkpoints:**
  * [`outputs/best_model_fold0.pt`](file:///d:/RSNA/outputs/best_model_fold0.pt) (85.28 MB)
  * [`outputs/best_model_fold1.pt`](file:///d:/RSNA/outputs/best_model_fold1.pt) (85.28 MB)
  * [`outputs/best_model_fold2.pt`](file:///d:/RSNA/outputs/best_model_fold2.pt) (85.28 MB)
  * [`outputs/best_model_fold3.pt`](file:///d:/RSNA/outputs/best_model_fold3.pt) (85.28 MB)
  * [`outputs/best_model_fold4.pt`](file:///d:/RSNA/outputs/best_model_fold4.pt) (85.28 MB)
* **Configuration:**
  * Defined in [`train_baseline.py:L81-L134`](file:///d:/RSNA/train_baseline.py#L81-L134) (`Config` class) and serialized into the `config` key of each checkpoint dictionary.
* **Training Logs & Evaluation CSVs:**
  * [`outputs/training_history_fold0.csv`](file:///d:/RSNA/outputs/training_history_fold0.csv)
  * [`outputs/training_history_fold1.csv`](file:///d:/RSNA/outputs/training_history_fold1.csv)
  * [`outputs/training_history_fold2.csv`](file:///d:/RSNA/outputs/training_history_fold2.csv)
  * [`outputs/training_history_fold3.csv`](file:///d:/RSNA/outputs/training_history_fold3.csv)
  * [`outputs/training_history_fold4.csv`](file:///d:/RSNA/outputs/training_history_fold4.csv)
  * [`outputs/cv_summary.csv`](file:///d:/RSNA/outputs/cv_summary.csv)
  * Plot visualizations: [`outputs/plots/01_loss_curves.png`](file:///d:/RSNA/outputs/plots/01_loss_curves.png), [`02_auc_curves.png`](file:///d:/RSNA/outputs/plots/02_auc_curves.png), [`03_best_auc_per_fold.png`](file:///d:/RSNA/outputs/plots/03_best_auc_per_fold.png), [`04_per_class_auc.png`](file:///d:/RSNA/outputs/plots/04_per_class_auc.png), [`05_oof_summary.png`](file:///d:/RSNA/outputs/plots/05_oof_summary.png)
* **OOF Predictions File:** **NOT FOUND** (summary metrics stored in [`outputs/cv_summary.csv`](file:///d:/RSNA/outputs/cv_summary.csv)).
* **Submission Files:** **NOT FOUND**.
* **Scripts:**
  * Entry-point training script: [`train_baseline.py`](file:///d:/RSNA/train_baseline.py)
  * Modules: [`src/dataset.py`](file:///d:/RSNA/src/dataset.py), [`src/model.py`](file:///d:/RSNA/src/model.py), [`src/losses.py`](file:///d:/RSNA/src/losses.py), [`src/metrics.py`](file:///d:/RSNA/src/metrics.py)
  * Analysis & plotting script: [`analyze_training.py`](file:///d:/RSNA/analyze_training.py)

---

### 32. Relevant Package Versions
* **`torch`:** `2.11.0+cu128`
* **`torchvision`:** `0.26.0+cu128`
* **`scikit-learn`:** `1.9.0`
* **`numpy`:** `2.5.2`
* **`pandas`:** `3.0.5`
* **`tqdm`:** `4.70.0`
* **`matplotlib`:** `3.11.1`
* **`Python`:** `3.14.7`
* **`albumentations`:** NOT INSTALLED
* **`timm`:** NOT INSTALLED
* **`pydicom`:** NOT INSTALLED (pre-converted `.npz` volumes were used)

---

## 5-Fold Performance Summary Table

| Fold | Best Macro ROC-AUC | Epoch | Train Loss | Val Loss | Key Observations & Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Fold 0** | **0.840326** | 10 | 0.4345 | 0.4972 | Highest overall fold score. Top on `Medial OA` (0.8856) and `Synovitis` (0.8863). Overfitting began after epoch 11. |
| **Fold 1** | **0.827705** | 8 | 0.4555 | 0.5001 | Quick peak at epoch 8. Strong on `Medial Meniscus` (0.8655) and `ACL` (0.8559); low on `MCL` (0.7554). |
| **Fold 2** | **0.829611** | 7 | 0.4683 | 0.5013 | Earliest peak (epoch 7). Strongest score on `Baker's cyst` (0.8911); weakest on `Contusion` (0.7690). |
| **Fold 3** | **0.835299** | 8 | 0.4571 | 0.4955 | Lowest validation loss (0.4955). High on `Medial OA` (0.8683) and `Fracture` (0.8682); top `MCL` score (0.7787). |
| **Fold 4** | **0.818695** | 11 | 0.4180 | 0.5252 | Lowest fold score. Highest validation loss (0.5252). Weakest scores on `MCL` (0.7444), `PF OA` (0.7905), and `Fracture` (0.8094). |
| **OOF / Mean** | **0.830327** <br> ($\pm 0.0073$) | **—** | **—** | **—** | **5-Fold Mean:** Strongest classes are `Medial Meniscus` (0.8662), `Medial OA` (0.8636), `Synovitis` (0.8622), `Baker's` (0.8608). Weakest classes are `MCL` (0.7623), `Lateral Meniscus` (0.7901), and `Contusion` (0.7984). |