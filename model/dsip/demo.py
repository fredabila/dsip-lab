"""A small web UI (Gradio) to show a trained model live: upload an image, see the
preprocessed input and the prediction. Install with: pip install -e ".[demo]"
"""

from __future__ import annotations

from pathlib import Path

import cv2

from dsip.config import Config
from dsip.predict import Predictor
from dsip.preprocess import to_bgr_uint8


def demo(cfg: Config, run_dir: Path, share: bool = False, port: int = 7860) -> None:
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError('The demo needs gradio: pip install -e ".[demo]"') from e

    pred = Predictor(cfg, run_dir)

    def run(image):
        if image is None:
            return None, None
        bgr = to_bgr_uint8(cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image)
        x = pred.preprocess(bgr)
        if cfg.task == "classify":
            _, scores = pred.classify(x)
            return cv2.cvtColor(x, cv2.COLOR_BGR2RGB), scores
        boxes, annotated = pred.detect(x)
        counts: dict[str, float] = {}
        for b in boxes:
            counts[b["class"]] = counts.get(b["class"], 0) + 1
        total = sum(counts.values()) or 1
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), {f"{k} ({int(v)})": v / total for k, v in counts.items()}

    gr.Interface(
        fn=run,
        inputs=gr.Image(label="Input image"),
        outputs=[gr.Image(label="Preprocessed / detections"), gr.Label(label="Prediction")],
        title=f"{cfg.name}",
        description=f"Task: {cfg.task} | Preprocessing: {pred.pipeline} | Weights: {run_dir}. "
        "Educational project: not for clinical use.",
    ).launch(share=share, server_port=port)
