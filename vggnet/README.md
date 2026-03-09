# VGGNet-11 / VGGNet-16 — PyTorch Implementation

Reproduction of the VGGNet architecture from the paper:
> **Very Deep Convolutional Networks for Large-Scale Image Recognition**
> Simonyan & Zisserman, Oxford — ICLR 2015

---

## Hardware Limitation Note

VGG-16 (138M parameters) and VGG-11 both require more VRAM than available on the development GPU (3.68 GB) when training with 224×224 inputs. Forward pass was verified as correct; training could not be completed due to CUDA OOM errors during backpropagation.

The architecture is implemented correctly and can be trained on hardware with ≥8 GB VRAM.

---

## Key Contributions (vs AlexNet)

- All convolutions use 3×3 kernels with stride=1, padding=1 — spatial size preserved within blocks
- Depth is the primary driver of performance, not kernel size
- Local Response Normalization (LRN) removed — paper explicitly shows it provides no improvement
- MaxPool uses kernel=2, stride=2 — non-overlapping (vs AlexNet's overlapping pool)
- Two variants implemented: VGG-11 (8 conv layers) and VGG-16 (13 conv layers)

---

## Architecture — VGG-16

| Block  | Layers                              | Output Size    |
|--------|-------------------------------------|----------------|
| Block 1 | Conv(64) → Conv(64) → MaxPool      | 112 × 112      |
| Block 2 | Conv(128) → Conv(128) → MaxPool    | 56 × 56        |
| Block 3 | Conv(256) × 3 → MaxPool            | 28 × 28        |
| Block 4 | Conv(512) × 3 → MaxPool            | 14 × 14        |
| Block 5 | Conv(512) × 3 → MaxPool            | 7 × 7          |
| FC1    | 25088 → 4096 → ReLU → Dropout(0.5) | —              |
| FC2    | 4096 → 4096 → ReLU → Dropout(0.5)  | —              |
| FC3    | 4096 → 10                           | —              |

All Conv layers: kernel=3, stride=1, padding=1, followed by ReLU.

---

## Architecture — VGG-11

| Block  | Layers                              | Output Size    |
|--------|-------------------------------------|----------------|
| Block 1 | Conv(64) → MaxPool                 | 112 × 112      |
| Block 2 | Conv(128) → MaxPool                | 56 × 56        |
| Block 3 | Conv(256) → Conv(256) → MaxPool    | 28 × 28        |
| Block 4 | Conv(512) → Conv(512) → MaxPool    | 14 × 14        |
| Block 5 | Conv(512) → Conv(512) → MaxPool    | 7 × 7          |
| FC1    | 25088 → 4096 → ReLU → Dropout(0.5) | —              |
| FC2    | 4096 → 4096 → ReLU → Dropout(0.5)  | —              |
| FC3    | 4096 → 10                           | —              |

---

## Implementation Details

### Reproduced from paper
- All 3×3 convolutions, stride=1, padding=1
- Non-overlapping MaxPool (kernel=2, stride=2)
- No LRN — explicitly removed per paper findings
- Dropout=0.5 on FC layers
- SGD with momentum=0.9, weight_decay=0.0005
- LR decay: factor=0.1 on plateau
- PCA color augmentation, random crop, random horizontal flip

### Deviations from paper
- Dataset: CIFAR-10 instead of ImageNet
- Training incomplete due to VRAM constraint (3.68 GB available, ~8 GB required)

---

## Training Configuration

| Parameter     | Value  |
|---------------|--------|
| Batch size    | 64     |
| Learning rate | 0.01   |
| Momentum      | 0.9    |
| Weight decay  | 0.0005 |
| Dropout       | 0.5    |

---

## Project Structure

```
vggnet/
├── model.py      # VGG-11 and VGG-16 architecture
├── dataset.py    # CIFAR-10 loading, transforms, PCA, DataLoaders
├── train.py      # Training loop with checkpoint save/resume
├── eval.py       # Evaluation function (loss + accuracy)
├── test.py       # Final test set evaluation
├── utils.py      # PCA augmentation, WrapperDataset, checkpoint utilities
└── config.py     # All hyperparameters and constants
```

---

## Usage

**Train:**
```bash
python vggnet/train.py
```

**Test:**
```bash
python vggnet/test.py
```

> Requires GPU with ≥8 GB VRAM for 224×224 input.