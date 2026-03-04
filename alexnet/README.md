# AlexNet — PyTorch Implementation

Reproduction of the AlexNet architecture from the paper:
> **ImageNet Classification with Deep Convolutional Neural Networks**
> Krizhevsky, Sutskever, Hinton — NeurIPS 2012

---

## Results

| Split      | Loss   | Accuracy |
|------------|--------|----------|
| Validation | 0.5681 | 81.76%   |
| Test       | 0.6076 | 80.93%   |

Trained for 90 epochs on CIFAR-10. Note: CIFAR-10 images (32×32) are upsampled to 227×227 to match the original architecture, which introduces minor blur and accounts for the accuracy gap compared to ImageNet results.

---

## Architecture

5 convolutional layers followed by 3 fully-connected layers.

| Layer  | Type    | Output Size       | Details                              |
|--------|---------|-------------------|--------------------------------------|
| Conv1  | Conv2d  | 96 × 55 × 55      | kernel=11, stride=4 → ReLU → LRN → MaxPool |
| Conv2  | Conv2d  | 256 × 27 × 27     | kernel=5, pad=2 → ReLU → LRN → MaxPool     |
| Conv3  | Conv2d  | 384 × 13 × 13     | kernel=3, pad=1 → ReLU                     |
| Conv4  | Conv2d  | 384 × 13 × 13     | kernel=3, pad=1 → ReLU                     |
| Conv5  | Conv2d  | 256 × 6 × 6       | kernel=3, pad=1 → ReLU → MaxPool           |
| FC1    | Linear  | 4096              | ReLU → Dropout(0.5)                        |
| FC2    | Linear  | 4096              | ReLU → Dropout(0.5)                        |
| FC3    | Linear  | 10                | Output (CIFAR-10 classes)                  |

MaxPool: kernel=3, stride=2 (overlapping pooling as described in paper)

---

## Implementation Details

### Reproduced from paper
- Local Response Normalization (LRN) after Conv1 and Conv2
- Overlapping max pooling
- PCA color augmentation — computed from training set eigenvectors/eigenvalues
- Random crop + Random horizontal flip (train only)
- SGD with momentum=0.9, weight_decay=0.0005
- Learning rate decay: factor=0.1 when validation loss plateaus (ReduceLROnPlateau)
- Dropout=0.5 on FC layers

### Deviations from paper
- Dataset: CIFAR-10 instead of ImageNet (32×32 → upsampled to 227×227)
- Single GPU (original paper uses 2 GPUs with split architecture)

---

## Training

**Hyperparameters:**

| Parameter     | Value  |
|---------------|--------|
| Epochs        | 90     |
| Batch size    | 128    |
| Learning rate | 0.001  |
| Momentum      | 0.9    |
| Weight decay  | 0.0005 |
| Dropout       | 0.5    |

---

## Project Structure

```
alexnet/
├── model.py      # AlexNet architecture
├── dataset.py    # CIFAR-10 loading, transforms, PCA computation, DataLoaders
├── train.py      # Training loop with checkpoint save/resume
├── eval.py       # Evaluation function (loss + accuracy)
├── test.py       # Final test set evaluation
├── utils.py      # LRN, PCA augmentation, WrapperDataset, checkpoint utilities
└── config.py     # All hyperparameters and constants
```

---

## Usage

**Train:**
```bash
python alexnet/train.py
```

**Test:**
```bash
python alexnet/test.py
```