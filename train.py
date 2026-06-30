"""Train a CNN directly on image pixels.

Usage:
    python augment.py   # optional
    python train.py
"""

import os
import re
import time

import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from cnn import (
    CLASS_NAMES,
    build_model,
    get_eval_transform,
    get_train_transform,
    save_model,
)
from utils import AUG_PREFIX, list_images

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def source_group(path):
    name = os.path.basename(path)
    if name.startswith(AUG_PREFIX):
        name = name[len(AUG_PREFIX) :]
        name = re.sub(
            r"_(flip|bright_up|bright_down|contrast_up)\.jpg$",
            ".jpg",
            name,
        )
    return os.path.splitext(name)[0]


class PhotoDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def load_dataset():
    paths, labels = [], []
    for path in list_images("real"):
        paths.append(path)
        labels.append(0)
    for path in list_images("screen"):
        paths.append(path)
        labels.append(1)
    return paths, torch.tensor(labels, dtype=torch.long)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if is_train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def main():
    paths, labels = load_dataset()
    if len(paths) < 20:
        raise SystemExit("Need images in real/ and screen/.")

    groups = [source_group(p) for p in paths]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(paths, labels, groups))

    train_paths = [paths[i] for i in train_idx]
    train_labels = labels[train_idx]
    test_paths = [paths[i] for i in test_idx]
    test_labels = labels[test_idx]

    print(f"Device: {DEVICE}")
    print(
        f"Images: {len(paths)} total | "
        f"train={len(train_paths)} test={len(test_paths)} "
        f"(grouped by source photo)"
    )

    model, weights = build_model()
    model.to(DEVICE)
    train_tf = get_train_transform(weights)
    eval_tf = get_eval_transform(weights)

    train_loader = DataLoader(
        PhotoDataset(train_paths, train_labels, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        PhotoDataset(test_paths, test_labels, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )
        test_loss, test_acc = run_epoch(model, test_loader, criterion)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={train_loss:.3f} train_acc={train_acc * 100:.1f}%  "
            f"test_loss={test_loss:.3f} test_acc={test_acc * 100:.1f}%"
        )

    print(f"\nBest grouped test accuracy: {best_acc * 100:.1f}%")
    print(f"Training time: {time.time() - t0:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Retraining best weights on all data...")
    full_loader = DataLoader(
        PhotoDataset(paths, labels, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, 6):
        train_loss, train_acc = run_epoch(
            model, full_loader, criterion, optimizer
        )
        print(
            f"Full fit {epoch}/5  loss={train_loss:.3f} acc={train_acc * 100:.1f}%"
        )

    save_model(model, weights)
    print(f"Saved CNN model to cnn_model.pt ({CLASS_NAMES[0]}=0, {CLASS_NAMES[1]}=1)")


if __name__ == "__main__":
    main()
