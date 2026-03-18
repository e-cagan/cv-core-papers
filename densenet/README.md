# DenseNet-121 — PyTorch Implementation

Reproduction of the DenseNet architecture from the paper:
> **Densely Connected Convolutional Networks**
> Huang, Liu, van der Maaten, Weinberger — CVPR 2017

---

## Results

| Split      | Loss   | Accuracy |
|------------|--------|----------|
| Validation | 0.2276 | 93.20%   |

Trained for 60 epochs on CIFAR-10. Notable jump at epoch 45 when LR scheduler reduced learning rate — validation accuracy jumped from ~88% to ~92.8% in a single epoch.

---

## Progression vs Previous Implementations

| Model       | Val Accuracy | Epochs |
|-------------|-------------|--------|
| AlexNet     | 81.76%      | 90     |
| ResNet-18   | 87.26%      | 90     |
| DenseNet-121| 93.20%      | 60     |

DenseNet achieves the highest accuracy in the fewest epochs — dense connectivity and feature reuse at work.

---

## Key Contributions (vs ResNet)

- **Dense connectivity** — each layer receives feature maps from *all* preceding layers via concatenation (not addition)
- **Feature reuse** — low-level and high-level features available at every layer, no redundant relearning
- **Gradient flow** — direct connections to all previous layers, vanishing gradient virtually eliminated
- **Parameter efficiency** — each layer only needs to produce `k` new feature maps (growth rate)
- **Pre-activation** — BN → ReLU → Conv ordering (vs ResNet's Conv → BN → ReLU)

---

## Architecture — DenseNet-121

| Stage         | Layers        | Output Channels | Details                         |
|---------------|---------------|-----------------|---------------------------------|
| Stem          | Conv(3×3, 24) | 24              | BN → ReLU, no MaxPool (CIFAR-10)|
| Dense Block 1 | 6 layers      | 96              | k=12 per layer                  |
| Transition 1  | Conv(1×1) + AvgPool | 48        | θ=0.5 compression               |
| Dense Block 2 | 12 layers     | 192             | k=12 per layer                  |
| Transition 2  | Conv(1×1) + AvgPool | 96        | θ=0.5 compression               |
| Dense Block 3 | 24 layers     | 384             | k=12 per layer                  |
| Transition 3  | Conv(1×1) + AvgPool | 192       | θ=0.5 compression               |
| Dense Block 4 | 16 layers     | 384             | k=12 per layer                  |
| Head          | BN → ReLU → AvgPool → FC(10) | 10 | Global average pooling      |

**DenseLayer structure (pre-activation):**
```
BN → ReLU → Conv(3×3, k) → concat([x_0, x_1, ..., x_{l-1}, out])
```

**TransitionLayer structure:**
```
BN → ReLU → Conv(1×1, θ·C) → AvgPool(2×2, stride=2)
```

---

## Implementation Details

### Reproduced from paper
- Dense connectivity with `torch.cat` along channel dimension
- Growth rate k=12 (CIFAR-10 configuration)
- Compression factor θ=0.5 in transition layers
- Pre-activation ordering: BN → ReLU → Conv
- `nn.ModuleList` for dynamic DenseBlock construction
- SGD with momentum=0.9, weight_decay=0.0001
- LR decay factor=0.1 on plateau
- Random crop (padding=4) + Random horizontal flip

### Deviations from paper
- Dataset: CIFAR-10 instead of ImageNet
- Stem: 3×3 Conv (paper uses 7×7 for ImageNet)
- No MaxPool after stem (preserves 32×32 spatial resolution)
- LR scheduler: ReduceLROnPlateau instead of fixed step decay at epoch 50/75/

---

## Training

| Parameter     | Value  |
|---------------|--------|
| Epochs        | 60     |
| Batch size    | 64     |
| Learning rate | 0.1    |
| Momentum      | 0.9    |
| Weight decay  | 0.0001 |
| Growth rate k | 12     |
| Compression θ | 0.5    |

---

## Project Structure

```
densenet/
├── model.py      # DenseLayer + DenseBlock + TransitionLayer + DenseNet121
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
python densenet/train.py
```

**Test:**
```bash
python densenet/test.py
```