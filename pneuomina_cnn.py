"""
Pneumonia detection from chest X-rays with a CNN.

Shows how a small CNN trained from scratch can differentiate healthy chest X-rays
apart from ones showing pneumonia. The model is written in PyTorch, and
all the image preprocessing is done with OpenCV and NumPy.

Data comes from the Kaggle "Chest X-Ray Images (Pneumonia)" dataset by
Paul Mooney. The script expects the folders in this format:
    data/chest_xray/train/{NORMAL,PNEUMONIA}/*.jpeg
    data/chest_xray/val/{NORMAL,PNEUMONIA}/*.jpeg
    data/chest_xray/test/{NORMAL,PNEUMONIA}/*.jpeg

How to run it:
    python pneumonia_cnn.py --mode train --data_dir data/chest_xray
    python pneumonia_cnn.py --mode test  --data_dir data/chest_xray
    python pneumonia_cnn.py --mode predict --image some_xray.jpeg
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

CLASSES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224
# Rough mean/std for the grayscale images.
MEAN, STD = 0.5, 0.25


# --- Preprocessing ---

# CLAHE boosts contrast locally instead of across the whole image at once, which helps the cloudy patches pneumonia leaves in the lungs stand out.
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(path: str) -> np.ndarray:
    """Read an X-ray in grayscale, sharpen its contrast, and shrink it to 224x224.

    The original scans are all different sizes (some are huge), so resizing
    them to one fixed size is what lets them go through the CNN in batches.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = _clahe.apply(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def augment(img: np.ndarray) -> np.ndarray:
    """Randomly but slightly changes a training image,
    so the model doesn't memorize it.

    Each time an image is loaded it gets a slightly different rotation, zoom,
    position, and brightness, a bit like X-rays taken on different machines
    or with the patient standing slightly differently. I deliberately left out
    horizontal flips because chest anatomy isn't symmetric (the heart sits on the
    left), so a mirrored X-ray isn't a realistic example.
    """
    h, w = img.shape
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += np.random.uniform(-0.05, 0.05, size=2) * (w, h)  # move it up to 5% in each direction
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Small random changes to contrast (alpha) and brightness (beta)
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-15, 15)
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255)
    return img.astype(np.uint8)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """Turn a 0-255 image into a normalized tensor with a channel dimension.

    PyTorch expects images shaped (channels, height, width), and since these
    are grayscale there's just one channel.
    """
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.expand_dims(x, 0))


# --- Dataset ---

def list_images(split_dir: str):
    """Collect (image path, label) pairs from a split folder.

    The label is just the class's position in CLASSES: 0 for NORMAL,
    1 for PNEUMONIA.
    """
    items = []
    for label, cls in enumerate(CLASSES):
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for p in glob.glob(os.path.join(split_dir, cls, ext)):
                items.append((p, label))
    return items


class XrayDataset(Dataset):
    """Loads X-rays from disk one at a time, only augmenting the training set."""

    def __init__(self, items, train: bool):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = preprocess(path)
        if self.train:
            img = augment(img)
        return to_tensor(img), label


# --- Model ---

def conv_block(c_in, c_out):
    """Two 3x3 convolutions followed by a pooling step that halves the image size.

    Batch norm after each convolution keeps training stable, and stacking two
    convolutions before pooling lets each block pick up slightly more complex
    patterns compared to a single layer.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class PneumoniaCNN(nn.Module):
    """A small CNN with four conv blocks and a simple classifier on top.

    Early blocks pick up basic things like edges and textures, and later ones
    combine those into larger patterns. The image keeps shrinking while the
    number of feature channels grows, which is the standard CNN shape.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),     # 224x224 -> 112x112
            conv_block(32, 64),    # 112x112 -> 56x56
            conv_block(64, 128),   # 56x56   -> 28x28
            conv_block(128, 256),  # 28x28   -> 14x14
        )
        # Averaging each feature map down to one number keeps the classifier
        # small, which means fewer parameters and less room to overfit.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),  # randomly drop half the features while training to fight overfitting
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# --- Training and evaluation ---

def run_epoch(model, loader, criterion, device, optimizer=None):
    """Make one full pass over a dataset.

    Pass an optimizer and it trains. If you leave it out, it just evaluates.
    Sharing one loop for both keeps training and validation consistent.
    Returns the average loss, the accuracy, and the raw labels and predictions.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(labels, preds), labels, preds


def train(args, device):
    # The dataset's official validation folder only has 16 images, which is
    # far too few to judge the model by. So I pool it with the training data
    # and carve out my own 15% validation split, keeping the class ratio the
    # same in both halves (that's what stratify does).
    items = list_images(os.path.join(args.data_dir, "train")) + \
        list_images(os.path.join(args.data_dir, "val"))
    train_items, val_items = train_test_split(
        items, test_size=0.15, stratify=[l for _, l in items], random_state=42
    )

    # There are about 3 pneumonia images for every normal one. Without
    # correcting for that, the model can score well by mostly guessing
    # pneumonia, so the loss weights mistakes on the rarer class more heavily.
    counts = np.bincount([l for _, l in train_items], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(device)
    print(f"train={len(train_items)} val={len(val_items)} class counts={counts.tolist()}")

    train_loader = DataLoader(XrayDataset(train_items, train=True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(XrayDataset(val_items, train=False), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    model = PneumoniaCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Gradually lower the learning rate over training so the later epochs
    # make smaller, more careful adjustments.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")
        # Only keep the version that does best on validation. The test set
        # stays untouched until the very end so the final score is honest.
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), args.ckpt)
            print(f"  saved best model ({best_acc:.4f})")


def test(args, device):
    """Score the saved model on the held-out test set.

    Besides accuracy, this prints a confusion matrix and per-class
    precision/recall, since accuracy alone hides a lot on imbalanced data.
    """
    items = list_images(os.path.join(args.data_dir, "test"))
    loader = DataLoader(XrayDataset(items, train=False), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    _, acc, labels, preds = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    print(f"Test accuracy: {acc:.4f}")
    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))


def predict(args, device):
    """Run the model on a single X-ray and print how likely each class is."""
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    x = to_tensor(preprocess(args.image)).unsqueeze(0).to(device)  # add a batch dimension of 1
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    print({c: round(float(p), 4) for c, p in zip(CLASSES, probs)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test", "predict"], default="train")
    ap.add_argument("--data_dir", default="data/chest_xray")
    ap.add_argument("--image")
    ap.add_argument("--ckpt", default="pneumonia_cnn.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    # Fixed seeds so runs are repeatable
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    # Use a GPU if there is one: CUDA on NVIDIA cards, MPS on Apple Silicon
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    {"train": train, "test": test, "predict": predict}[args.mode](args, device)