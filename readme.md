# UXFormer

## Project Structure

```
.
├── model.py          # UXFormer model definition
├── train.py          # Training script
├── test.py           # Evaluation and inference script
├── Needle_data.py    # Dataset class
├── utils.py          # Loss functions and evaluation utilities
└── dataset/
    ├── train/
    │   ├── original/
    │   ├── filtered/
    │   └── label/
    ├── val/
    └── test/
```

## Dataset

Each split expects three subdirectories with matching filenames:

- `original/` — raw grayscale images
- `filtered/` — preprocessed/filtered versions of the images
- `label/` — binary segmentation masks

## Training

```bash
python train.py
```

Key settings in `train.py`:

| Parameter | Default |
|-----------|---------|
| Input size | 224 × 224 |
| Batch size | 1 |
| Optimizer | AdamW (lr=1e-4) |
| Scheduler | ReduceLROnPlateau |
| Epochs | 150 |
| Loss | Dice (×0.9) + BCE (×0.1) |

Checkpoints and metrics are saved to `./logs/`.

## Evaluation

```bash
python test.py
```

Reported metrics: Dice, Jaccard (IoU), F1, Recall, Precision, Accuracy.

With `save_segmentation=True` and `save_attention=True`, results are saved under `./result/`:

```
result/
├── original/
├── filtered/
├── target/
├── prediction/
├── overlay/
└── attention_maps/
    ├── d3/         # grayscale + heatmap per decoder level
    ├── d3_overlay/
    └── ...
```