# ResNet-18 — PyTorch Implementation

Reproduction of the ResNet architecture from the paper:
> **Deep Residual Learning for Image Recognition**
> He, Zhang, Ren, Sun — Microsoft Research, CVPR 2016

---

## Results

| Split      | Loss   | Accuracy |
|------------|--------|----------|
| Validation | 0.4744 | 87.26%   |

Trained for 90 epochs on CIFAR-10. Notable jump at epoch 38 when LR scheduler reduced learning rate — validation accuracy jumped from ~84% to ~87% in a single epoch, demonstrating the effectiveness of LR decay.

---

## Key Contributions (vs VGGNet)

- **Residual (skip) connections** — `F(x) + x` allows gradients to flow directly through the network, solving the degradation problem
- Networks of 100+ layers now trainable — previously impossible
- **Projection shortcuts** — 1×1 Conv + BN when spatial dimensions or channel counts change
- BatchNorm after every Conv — training much more stable
- No Dropout — BatchNorm + weight decay sufficient for regularization
- Achieved superhuman performance on ImageNet (3.57% top-5 error)

---

## Architecture — ResNet-18

| Stage   | Layer / Block               | Output Size   | Details                          |
|---------|-----------------------------|---------------|----------------------------------|
| Stem    | Conv(7×7, 64, stride=1)     | 32 × 32       | BN → ReLU → MaxPool              |
| Layer 1 | BasicBlock × 2              | 32 × 32       | 64 channels, stride=1            |
| Layer 2 | BasicBlock × 2              | 16 × 16       | 128 channels, stride=2 (first)   |
| Layer 3 | BasicBlock × 2              | 8 × 8         | 256 channels, stride=2 (first)   |
| Layer 4 | BasicBlock × 2              | 4 × 4         | 512 channels, stride=2 (first)   |
| Head    | AdaptiveAvgPool → FC(10)    | 10            | Global average pooling           |

**BasicBlock structure:**
```
Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+shortcut) → ReLU
```

Projection shortcut (1×1 Conv + BN) applied when stride ≠ 1 or channels change.

---

## Implementation Details

### Reproduced from paper
- BasicBlock with identity and projection shortcuts
- 4 stages with channel progression: 64 → 128 → 256 → 512
- BatchNorm after every Conv
- AdaptiveAvgPool instead of fixed-size pooling
- SGD with momentum=0.9, weight_decay=0.0001
- LR decay factor=0.1 on plateau
- Random crop (padding=4) + Random horizontal flip

### Deviations from paper
- Dataset: CIFAR-10 instead of ImageNet
- Stem: stride=1 instead of stride=2 (preserves spatial resolution for 32×32 input)
- MaxPool: stride=1 instead of stride=2 (same reason)
- LR scheduler: ReduceLROnPlateau instead of fixed step decay at epoch 30/60

---

## Training

| Parameter     | Value  |
|---------------|--------|
| Epochs        | 90     |
| Batch size    | 256    |
| Learning rate | 0.1    |
| Momentum      | 0.9    |
| Weight decay  | 0.0005 |

---

## Comparison with AlexNet

| Model    | Val Accuracy | Epochs | Parameters |
|----------|-------------|--------|------------|
| AlexNet  | 81.76%      | 90     | ~61M       |
| ResNet-18| 87.26%      | 90     | ~11M       |

ResNet-18 achieves +5.5% accuracy with **5× fewer parameters**.

---

## Project Structure

```
resnet/
├── model.py      # BasicBlock + ResNet18 architecture
├── dataset.py    # CIFAR-10 loading, transforms, DataLoaders
├── train.py      # Training loop with checkpoint save/resume
├── eval.py       # Evaluation function (loss + accuracy)
├── test.py       # Final test set evaluation
├── utils.py      # WrapperDataset, checkpoint utilities
└── config.py     # All hyperparameters and constants
```

---

## Usage

**Train:**
```bash
python resnet/train.py
```

**Test:**
```bash
python resnet/test.py
```
