Fashion baseline training run
============================

Timestamp: 2025-11-20T15:31:06.208198
Data root: ../torch_data/
Model architecture: ResNet
Num classes: 50

Hyperparameters:
- train_count: 45000
- val_count: 5000
- epochs: 20
- batch_size: 64
- lr_head: 0.001
- lr_backbone: 0.0001
- weight_decay: 0.05
- from_scratch flag passed: True

Optimizer & scheduler:
- optimizer: AdamW
- scheduler: CosineAnnealingLR
