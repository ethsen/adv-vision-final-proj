Fashion baseline training run
============================

Timestamp: 2025-11-17T22:14:37.187606
Data root: ../torch_data/
Model architecture: ResNet 18
Num classes: 50

Hyperparameters:
- train_count: 10000
- val_count: 5000
- epochs: 8
- batch_size: 64
- lr_head: 0.001
- lr_backbone: 0.0001
- weight_decay: 0.05
- from_scratch flag passed: True

Optimizer & scheduler:
- optimizer: AdamW
- scheduler: CosineAnnealingLR
