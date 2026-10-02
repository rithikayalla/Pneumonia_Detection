# Pneumonia Detection from Chest X-Rays (CNN)

A convolutional neural network built from scratch in PyTorch that classifies chest X-rays as **NORMAL** or **PNEUMONIA**. Images are preprocessed with OpenCV and NumPy (CLAHE contrast enhancement, resizing, normalization, and augmentation).

## Results

Evaluated on the held-out test set (624 images):

| Metric | Score |
|---|---|
| Accuracy | 87.0% |
| Pneumonia recall | 98.2% |
| Pneumonia precision | 83.8% |
| Normal recall | 68.4% |
| F1 (macro) | 0.85 |

**Confusion matrix**

| | Predicted Normal | Predicted Pneumonia |
|---|---|---|
| **Actual Normal** | 160 | 74 |
| **Actual Pneumonia** | 7 | 383 |

Recall on the pneumonia class is reported alongside accuracy because the dataset is imbalanced (~3:1 pneumonia:normal) and a missed pneumonia case is costlier than a false alarm. The model missed only 7 of 390 pneumonia cases.

Its main weakness is over-calling pneumonia on healthy X-rays (74 of 234 normal images misclassified). This is a known pattern on this dataset, where the test set differs somewhat from the training distribution. Transfer learning from a pretrained backbone (e.g. ResNet18) is a natural next step to reduce these false positives.

The checkpoint was selected by validation accuracy, and the test set was evaluated once, with no tuning against it.

## Approach

**Preprocessing (OpenCV + NumPy)**
- Load as grayscale and apply CLAHE to enhance local contrast in lung regions
- Resize to 224×224 and standardize pixel intensities
- Training-time augmentation: small rotations, zoom, shifts, and brightness/contrast jitter (no horizontal flips, since chest anatomy is asymmetric)

**Model**
- 4 convolutional blocks (Conv → BatchNorm → ReLU ×2 → MaxPool), 32 → 256 channels
- Global average pooling, dropout (0.5), and a linear classifier

**Training**
- Weighted cross-entropy loss to handle class imbalance
- AdamW optimizer with cosine learning-rate schedule
- The original validation split has only 16 images, so train + val are merged and re-split 85/15 (stratified)
- Best checkpoint selected by validation accuracy

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle, about 5,800 pediatric chest X-rays. The data is not included in this repo. Download it and place it at:

```
data/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

## Usage

```bash
pip install -r requirements.txt

# Train
python pneumonia_cnn.py --mode train --data_dir data/chest_xray --epochs 25

# Evaluate on the test set
python pneumonia_cnn.py --mode test --data_dir data/chest_xray

# Predict a single image
python pneumonia_cnn.py --mode predict --image path/to/xray.jpeg
```

Runs on CUDA, Apple Silicon (MPS), or CPU. A GPU is strongly recommended for training.

