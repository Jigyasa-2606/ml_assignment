"""Spot the fake photo: CNN trained on raw pixels.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
"""

import sys

import torch
from PIL import Image

from cnn import load_model

_cache = None


def predict(image_path: str) -> float:
    global _cache
    if _cache is None:
        _cache = load_model()

    model, transform = _cache
    x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        return float(model(x).softmax(dim=1)[0, 1].item())


if __name__ == "__main__":
    print(predict(sys.argv[1]))
