
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
