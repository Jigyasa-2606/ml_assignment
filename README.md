# Spot the Fake Photo

Given one image, decide if it is a **real photo** or a **photo of a screen**
(someone re-photographing a phone/laptop instead of the real thing).

```bash
python predict.py some_image.jpg
# → prints one number 0–1  (0 = real, 1 = screen)
```

---

## Setup

```bash
pip install -r requirements.txt
```

`cnn_model.pt` is included — you can run `predict.py` immediately.

---

## Approach

1. **Data** — Collected 50 real and 51 screen photos captured using phone. Used `augment.py`
   to add mild copies (flip, brightness, contrast) → 505 training images total.

2. **Model** — Fine-tuned **MobileNetV3-Small** (pretrained on ImageNet) directly on
   raw pixels. Chosen because it is small (~6 MB), fast on CPU, and works well with
   transfer learning on limited data.

3. **Training** — 20 epochs with Adam (lr=1e-4), grouped train/test split so all
   augmented copies of the same source photo stay in the same split. Saved best
   checkpoint, then fine-tuned on all data for the submission model.

4. **Inference** — `predict.py` loads the CNN, preprocesses the image, and returns
   the softmax probability of class "screen".

---

## Accuracy

| Evaluation | Result |
|------------|--------|
| test set (20% of source photos) | **83%** 

All data came from one phone and similar lighting/scenes. I expect accuracy to
drop somewhat on completely unseen devices and screen types — the held-out grouped
split is the more meaningful local metric.

---

## Latency *(required)*

**~57 ms per image** on laptop CPU (Apple Silicon Mac, no GPU).

Measured over 101 images (median run, includes model load amortized
over repeated calls). Feels instant for a single photo check.

---

## Cost per image *(required)*

**On-device (phone):** ~**$0** per image — inference runs locally after the model
is bundled in the app (~6 MB).

On-device is preferred: free at scale and lower latency (no network round-trip).

---

## What I would improve with more time

- Collect **more diverse data** (different phones, screens, outdoor lighting, printouts)
- Use **early stopping** instead of training for avoiding overfitting.
- Add a **two-threshold policy** (auto-accept low scores, auto-reject high, review middle)
- Monitor production false positives/negatives and retrain periodically as cheaters adapt
---

## Files

| File | Purpose |
|------|---------|
| `predict.py` | Run inference (required interface) |
| `train.py` | Train the CNN |
| `cnn.py` | Model build / load / save |
| `utils.py` | Image folder helpers |
| `cnn_model.pt` | Trained weights |
| `requirements.txt` | Python dependencies |

---

## How we would keep it accurate as cheaters adapt

- Continuously collect new fraud examples from production
- Retrain or recalibrate the model on fresh data
- Combine image score with metadata (capture timestamp, device attestation) where available
- Adversarial testing against new cheat methods (projectors, edited photos, different screens)
