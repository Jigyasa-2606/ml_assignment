"""Shared CNN model helpers for train.py and predict.py."""

import os

import torch
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.transforms import v2 as T

IMAGE_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_model.pt")
CLASS_NAMES = ("real", "screen")  # 0 = real, 1 = screen


def build_model(num_classes=2):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    return model, weights


def get_eval_transform(weights):
    transforms = weights.transforms()
    return transforms


def get_train_transform(weights):
    base = weights.transforms()
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            base,
        ]
    )


def save_model(model, weights_meta):
    torch.save(
        {
            "arch": "mobilenet_v3_small",
            "state_dict": model.state_dict(),
            "image_size": IMAGE_SIZE,
            "class_names": CLASS_NAMES,
        },
        MODEL_PATH,
    )


def load_model(map_location=None):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run: python augment.py && python train.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=map_location or "cpu")
    model, weights = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = get_eval_transform(weights)
    return model, transform
