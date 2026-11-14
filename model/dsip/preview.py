"""Visualise the preprocessing pipeline: one row per sample, one column per step.

Optionally adds the log-magnitude Fourier spectrum before/after, which is a nice
figure for the "signal processing" part of a report.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2

from dsip.config import Config
from dsip.preprocess import load_image, log_spectrum


def _samples(cfg: Config, n: int) -> list[tuple[Path, str]]:
    from dsip.prepare import _images, collect_classify

    rng = random.Random(cfg.seed)
    if cfg.task == "classify":
        by_label = defaultdict(list)
        for it in collect_classify(cfg):
            by_label[it.label].append(it.src)
        picks = []
        labels = sorted(by_label)
        for i in range(n):
            lab = labels[i % len(labels)]
            picks.append((rng.choice(by_label[lab]), lab))
        return picks
    files = _images(cfg.source, cfg.exclude)
    return [(p, "") for p in rng.sample(files, min(n, len(files)))]


def preview(cfg: Config, n: int = 4, out: str | None = None, spectrum: bool = False, images: list[str] | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = [(Path(p), "") for p in images] if images else _samples(cfg, n)
    pipeline = cfg.pipeline
    rows = []
    for path, label in samples:
        img = load_image(path)
        cols = [("original" + (f"\n[{label}]" if label else ""), img)]
        cols += list(pipeline.trace(img))
        if spectrum:
            cols.insert(1, ("spectrum (original)", log_spectrum(img)))
            cols.append(("spectrum (final)", log_spectrum(cols[-1][1])))
        rows.append(cols)

    ncol = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.6 * len(rows)), squeeze=False)
    for r, cols in enumerate(rows):
        for c in range(ncol):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(cols):
                title, im = cols[c]
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im, cmap="gray")
                if r == 0 or c == 0:
                    ax.set_title(title, fontsize=8)
    fig.suptitle(f"{cfg.name}: {pipeline}", fontsize=10)
    fig.tight_layout()
    out_path = Path(out) if out else Path("figures") / f"{cfg.name}_preprocess.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved preview to {out_path}")
    return out_path
