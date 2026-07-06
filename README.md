# CNN Architectures from Scratch: Sports Ball Classification

Implementations of classic CNN architectures built from scratch in PyTorch (no pretrained weights, no `torchvision.models` shortcuts) and benchmarked on a 15-class sports ball image classification dataset.

## Overview

This project re-implements three widely used CNN architectures component-by-component — starting from basic building blocks (activations, convolution blocks, squeeze-excite modules) up to the full network — then trains and evaluates each one on the same dataset for direct comparison.

| Architecture   | Status        | Notes |
|----------------|---------------|-------|
| DenseNet121    | ✅ Complete    | Blocks (6, 12, 24, 16), growth rate k=32, compression θ=0.5 |
| MobileNetV2    | ✅ Complete    | Built per Table 2 of the [original paper](https://arxiv.org/abs/1801.04381); ~3.5M params, verified against `torchvision` |
| EfficientNet-B0| 🔄 In progress | MBConv blocks with squeeze-excite, dynamic stage construction via `cfg` list; training underway |

A pretrained (transfer learning) DenseNet121 baseline achieves 80%+ accuracy and serves as a reference point for what these from-scratch implementations are working toward.

## Dataset

- **Task**: 15-class sports ball image classification
- **Size**: ~9,000 images
- **Source**: [Kaggle](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification)
- **Split**: Pre-split train/test folders from the source dataset, further divided into train/val/test using `random_split` with a seeded generator (separate `ImageFolder` instances are used per split to keep transforms isolated)

### Data augmentation notes
- `RandomResizedCrop` is used with `scale=(0.6, 1.0)` rather than the PyTorch default (`0.08–1.0`), since aggressive cropping risks cutting the ball subject out of frame entirely in this object-centric dataset.

## Project Structure

```
project_root/
├── configs/           # Training/model configs (hyperparameters, cfg lists for stage construction, etc.)
├── data/              # Dataset loading, ImageFolder splits, transforms
├── models/            # DenseNet121, MobileNetV2, EfficientNet-B0 implementations
├── utils/             # Helper functions (metrics, checkpointing, diagnostics, etc.)
├── engine.py          # Training/eval loop logic
├── train.py           # Entry point for training a model
├── inference.py       # Entry point for running inference / evaluation
└── requirements.txt
```

## Methodology

Each architecture was built and validated using a consistent process:

1. **Conceptual grounding** — understand the architecture from the original paper before writing code.
2. **Component-first implementation** — smallest building block → block group → full network.
3. **Sanity checks** — output shape verification and parameter count comparison against `torchvision`'s implementation (the gold-standard correctness check used throughout).
4. **Training** — Adam optimizer with `CosineAnnealingLR` scheduling, kept consistent across architectures for fair comparison.

### Debugging approach

When training appeared stuck or accuracy stayed near-random, the following diagnostics were used, roughly in order:
- Gradient norm checks
- Logit standard deviation checks
- Class index alignment verification
- **Tiny-batch overfit test** — fit a fixed ~32-sample batch for ~30 steps to confirm the architecture can memorize; separates real bugs from genuinely slow learning

This was particularly useful for EfficientNet-B0, where slow early-epoch accuracy turned out to be expected behavior for a from-scratch, non-pretrained implementation rather than a bug.

## Training Configuration

| Setting        | Value |
|----------------|-------|
| Optimizer      | Adam |
| Learning rate  | 0.001 |
| Weight decay   | 1e-4 |
| Scheduler      | CosineAnnealingLR |
| Epochs (DenseNet121) | 60 |

## Known Limitations

- **EfficientNet-B0**: Stochastic depth (used in the original paper for regularization) is not yet implemented — deliberately deferred for this iteration.
- From-scratch models are trained without pretrained weights, so an accuracy gap versus transfer-learning baselines is expected and attributable to initialization rather than architectural errors (confirmed via parameter-count matching against `torchvision`).

## Results

| Architecture    | Test Accuracy | Notes |
|-----------------|---------------|-------|
| DenseNet121 (from scratch) | — | |
| MobileNetV2 (from scratch) | — | |
| EfficientNet-B0 (from scratch) | — | training in progress |
| DenseNet121 (pretrained, transfer learning) | 80%+ | reference baseline |

> Fill in your actual test accuracy numbers once training/eval is complete.

## References

- Huang et al., [*Densely Connected Convolutional Networks*](https://arxiv.org/abs/1608.06993) (DenseNet)
- Sandler et al., [*MobileNetV2: Inverted Residuals and Linear Bottlenecks*](https://arxiv.org/abs/1801.04381)
- Tan & Le, [*EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*](https://arxiv.org/abs/1905.11946)
- Kaggle "EfficientNet from scratch" notebook (structural reference template)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train a model (architecture/hyperparameters set via configs/)
python train.py --config configs/<config_name>.yaml

# Run inference / evaluation
python inference.py --config configs/<config_name>.yaml --checkpoint <path_to_checkpoint>
```

> Adjust flags/args above to match your actual `train.py` / `inference.py` argument parsing.

---

*Built as a learning exercise in implementing CNN architectures from first principles, with parameter-count matching against `torchvision` used throughout as the architectural correctness benchmark.*

## 📬 Contact

For questions, collaboration, or feedback: **Tran Bao Minh "James" Nguyen**  
Data Scientist & Dashboard Designer  

- [LinkedIn](https://www.linkedin.com/in/tran-bao-minh-nguyen-296b01333/)  
- [Portfolio & More Projects](https://baominhnt.github.io/tbmnguyen.com/index.html)
