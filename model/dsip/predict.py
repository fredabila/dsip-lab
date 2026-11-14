"""Run a trained model on new images, applying the same preprocessing used in training."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dsip.config import Config, weights_path
from dsip.preprocess import IMAGE_EXTENSIONS, load_image, save_image


class Predictor:
    def __init__(self, cfg: Config, run_dir: Path, conf: float = 0.25, threshold: float = 0.5):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(str(weights_path(run_dir)))
        self.names = [self.model.names[i] for i in range(len(self.model.names))]
        self.pipeline = cfg.pipeline
        self.conf = conf
        self.threshold = threshold
        self.pos = self.names.index(cfg.positive_class) if cfg.positive_class in self.names else None

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        return self.pipeline(img)

    def classify(self, img: np.ndarray) -> tuple[str, dict[str, float]]:
        r = self.model.predict(img, imgsz=self.cfg.imgsz, verbose=False)[0]
        probs = r.probs.data.cpu().numpy()
        scores = {n: float(p) for n, p in zip(self.names, probs)}
        if self.pos is not None and len(self.names) == 2:
            label = self.names[self.pos] if probs[self.pos] >= self.threshold else self.names[1 - self.pos]
        else:
            label = self.names[int(probs.argmax())]
        return label, scores

    def detect(self, img: np.ndarray):
        r = self.model.predict(img, imgsz=self.cfg.imgsz, conf=self.conf, verbose=False)[0]
        boxes = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append({"class": self.names[int(b.cls[0])], "confidence": float(b.conf[0]), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return boxes, r.plot()


def _gather(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def predict(cfg: Config, run_dir: Path, inputs: list[str], out_dir: str | None = None, conf: float = 0.25, threshold: float = 0.5) -> Path:
    files = _gather(inputs)
    if not files:
        raise ValueError("No images found")
    pred = Predictor(cfg, run_dir, conf=conf, threshold=threshold)
    out = Path(out_dir) if out_dir else run_dir / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Predicting {len(files)} image(s); preprocessing: {pred.pipeline}")

    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if cfg.task == "classify":
            w.writerow(["file", "prediction", *[f"p_{n}" for n in pred.names]])
        else:
            w.writerow(["file", "class", "confidence", "x1", "y1", "x2", "y2"])
        for fp in files:
            img = pred.preprocess(load_image(fp))
            if cfg.task == "classify":
                label, scores = pred.classify(img)
                w.writerow([str(fp), label, *[f"{scores[n]:.5f}" for n in pred.names]])
                print(f"{fp.name}: {label} ({scores[label]:.3f})")
            else:
                boxes, annotated = pred.detect(img)
                save_image(out / f"{fp.stem}_pred.png", annotated)
                for b in boxes:
                    w.writerow([str(fp), b["class"], f"{b['confidence']:.4f}", *(f"{b[k]:.1f}" for k in ("x1", "y1", "x2", "y2"))])
                counts: dict[str, int] = {}
                for b in boxes:
                    counts[b["class"]] = counts.get(b["class"], 0) + 1
                print(f"{fp.name}: {counts or 'no detections'}")
    print(f"\nResults saved to {out}")
    return out
