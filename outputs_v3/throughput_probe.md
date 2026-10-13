# Throughput Probe — V3

GPU: NVIDIA RTX 4090  |  Total VRAM: 25.8 GB  |  VRAM limit (90%): 23.2 GB
Probe steps: 15 per (backbone, micro_batch)

| Backbone | micro_batch | imgs/s | s/step | peak_VRAM_GB | OOM |
|----------|-------------|--------|--------|--------------|-----|
| densenet121          |           8 |   50.9 |  0.157 |        13.09 | no  |
| densenet121          |          16 |    2.9 |  5.558 |        26.00 | no (Shared RAM) |
| inception_v3         |           8 |   94.4 |  0.085 |         5.75 | no  |
| inception_v3         |          16 |  100.1 |  0.160 |        11.15 | no  |
| inception_v3         |          24 |   99.1 |  0.242 |        16.51 | no  |
| inception_v3         |          32 |   28.2 |  1.135 |        21.89 | no  |
| efficientnet_b3      |           8 |   46.6 |  0.172 |        16.93 | no  |

## Chosen configuration per backbone

| backbone | best_micro_batch | peak_VRAM_GB |
|----------|------------------|--------------|
| densenet121          |                8 |        13.09 |
| inception_v3         |               24 |        16.51 |
| efficientnet_b3      |                8 |        16.93 |

## CHOSEN for training
| densenet121 | 8 |
