# RSNA Knee MRI — Multi-Label Abnormality Classification

PyTorch baseline for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition.

## System Requirements

| Component | Spec |
|-----------|------|
| CPU | Intel i9-13900K (8P+16E cores, 32 threads) |
| RAM | 64 GB DDR5-4000 |
| GPU | 24 GB VRAM (multi-GPU supported) |
| OS | Windows 11 Pro 64-bit |
| Python | 3.10+ |

## Project Structure

```
rsna/
├── train_baseline.py      ← Entry-point: training loop, CLI args, Config
├── src/
│   ├── __init__.py
│   ├── dataset.py         ← KneeMRIDataset (NPZ loading, augmentation)
│   ├── model.py           ← SliceEncoder + KneeClassifier (2.5D CNN)
│   ├── losses.py          ← compute_bce_loss (soft-label + sample-weight)
│   └── metrics.py         ← compute_macro_auc (ROC-AUC, soft-label safe)
├── outputs/               ← Checkpoints + training histories (runtime)
│   ├── best_model_fold0.pt
│   ├── best_model_fold1.pt
│   └── training_history_fold*.csv
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data
#    3D volumes  → https://www.kaggle.com/datasets/barun2104/rsna-knee-mri-processed-3d-volumes
#    Competition CSVs → https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
#    Place ALL files in: D:/rsna_knee_dataset/
```

## Training

```bash
# Train fold 0 (hardware-tuned defaults: batch=8, workers=8, fp16, torch.compile)
python train_baseline.py

# Train a specific fold
python train_baseline.py --fold 2

# Full 5-fold cross-validation (run each in its own terminal or sequentially)
for fold in 0 1 2 3 4; do python train_baseline.py --fold $fold; done

# Custom batch size + gradient accumulation (effective batch = 16)
python train_baseline.py --batch_size 8 --accum_steps 2

# All options
python train_baseline.py --help
```

## Architecture — 2.5D CNN

```
Input:  [B, 1, 24, 224, 224]       (batch × channel × depth × H × W)
          ↓  fold depth into batch
       [B×24, 1, 224, 224]
          ↓  ResNet-34 (patched conv1: 3ch→1ch)
       [B×24, 512]
          ↓  reshape + mean-pool over depth
       [B, 512]
          ↓  Dropout(0.3) → Linear(512, 12)
Output: [B, 12]  raw logits → sigmoid → soft probabilities
```

## Hardware Optimizations

| Feature | Setting | Benefit |
|---------|---------|---------|
| `NUM_WORKERS` | 8 | Saturates NPZ I/O; i9-13900K has 32 threads |
| `BATCH_SIZE` | 8 | Safe for 24 GB VRAM in fp16 |
| `prefetch_factor` | 2 | Hides D: disk latency with DDR5 buffer |
| `persistent_workers` | True | Avoids per-epoch worker respawn cost |
| `multiprocessing_context` | spawn | Required on Windows (CUDA + multiprocessing) |
| AMP fp16 | `autocast` + `GradScaler` | ~2× memory reduction, ~1.5× throughput |
| `cuDNN benchmark` | True | Auto-selects fastest kernel for fixed 224×224 shape |
| TF32 matmul | `"high"` | Free throughput on Ampere/Ada GPUs |
| `torch.compile` | `reduce-overhead` | Eliminates Python overhead via CUDA graphs |
| `nn.DataParallel` | Auto | Splits batch across all detected GPUs |
| Gradient accumulation | `--accum_steps` | Scale effective batch without extra VRAM |

## Labels (12 classes)

| Index | Label |
|-------|-------|
| 0 | ACL |
| 1 | MCL |
| 2 | Medial Meniscus |
| 3 | Lateral Meniscus |
| 4 | Medial OA |
| 5 | Lateral OA |
| 6 | PF OA |
| 7 | Effusion |
| 8 | Synovitis |
| 9 | Baker's |
| 10 | Contusion |
| 11 | Fracture |

## Outputs

After training each fold:

| File | Description |
|------|-------------|
| `outputs/best_model_fold{N}.pt` | Best checkpoint (by macro AUC), saved immediately on improvement |
| `outputs/training_history_fold{N}.csv` | Per-epoch loss, macro-AUC, and per-class AUC |

### Loading a Checkpoint

```python
from src.model import KneeClassifier

ckpt  = torch.load("outputs/best_model_fold0.pt", map_location="cpu")
model = KneeClassifier(**{k: ckpt["config"][k] for k in ["pool_mode", "num_classes"]})
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Restored fold {ckpt['fold']}  epoch {ckpt['epoch']}  AUC={ckpt['best_auc']:.4f}")
```
