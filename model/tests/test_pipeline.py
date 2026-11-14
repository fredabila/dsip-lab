"""Tests: preprocessing ops on random images, plus an end-to-end smoke test on a tiny
synthetic dataset (no downloads; trains from scratch for 1 epoch on CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from dsip.config import from_dict, load_config
from dsip.preprocess import OPS, Pipeline, load_image, save_image

RNG = np.random.default_rng(0)


def _img(h=96, w=80):
    return RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", sorted(set(OPS) - {"reinhard_normalize"}))
def test_ops_shape_and_dtype(name):
    img = _img()
    out = OPS[name](img)
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
    if name != "resize":
        assert out.shape == img.shape


def test_reinhard_matches_target_stats():
    out = OPS["reinhard_normalize"](_img(), target_mean=[150, 140, 120], target_std=[20, 5, 5])
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    assert np.allclose(lab.mean(0), [150, 140, 120], atol=3)


def test_fft_lowpass_reduces_high_frequency_energy():
    img = _img(64, 64)
    out = OPS["fft_filter"](img, kind="lowpass", cutoff=0.1)
    assert np.abs(np.diff(out.astype(float), axis=1)).mean() < np.abs(np.diff(img.astype(float), axis=1)).mean()


def test_pipeline_validation():
    with pytest.raises(ValueError, match="Unknown preprocessing op"):
        Pipeline(["nope"])
    with pytest.raises(ValueError, match="unknown parameter"):
        Pipeline([{"clahe": {"clip": 2}}])
    p = Pipeline(["grayscale", {"resize": {"size": 32}}])
    assert p(_img()).shape == (32, 32, 3)
    assert [label for label, _ in p.trace(_img())] == ["grayscale", "resize(size=32)"]


def test_load_image_handles_gray_and_16bit(tmp_path):
    g16 = (RNG.integers(0, 4096, (20, 30))).astype(np.uint16)
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), g16)
    img = load_image(p)
    assert img.shape == (20, 30, 3) and img.dtype == np.uint8


def test_example_configs_are_valid():
    for path in Path(__file__).parent.parent.joinpath("configs").glob("*.yaml"):
        load_config(path)


# ----------------------------------------------------------- end-to-end


def _synthetic_classify(root: Path, n=24):
    """'sick' images contain small dark blobs (parasite-like); 'healthy' ones don't."""
    for label in ("healthy", "sick"):
        for i in range(n):
            img = np.full((48, 48, 3), (180, 160, 200), np.uint8)
            img = cv2.add(img, RNG.integers(0, 20, img.shape, dtype=np.uint8))
            if label == "sick":
                for _ in range(3):
                    c = tuple(int(v) for v in RNG.integers(8, 40, 2))
                    cv2.circle(img, c, 4, (90, 40, 120), -1)
            patient = f"p{i % 6}"
            save_image(root / patient / label / f"{label}_{i}.png", img)
    # a duplicate file that must be removed, and a mask that must be excluded
    save_image(root / "p0" / "sick" / "copy.png", load_image(root / "p0" / "sick" / "sick_0.png"))
    save_image(root / "p0" / "sick" / "sick_0_mask.png", np.zeros((48, 48, 3), np.uint8))


def test_classify_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _synthetic_classify(tmp_path / "raw")
    cfg = from_dict(
        {
            "name": "toy",
            "task": "classify",
            "data": {"source": "raw", "group_level": 1, "exclude": ["*_mask*"], "split": {"val": 0.2, "test": 0.2}, "balance": "oversample"},
            "positive_class": "sick",
            "preprocess": [{"clahe": {"clip_limit": 2.0}}, {"resize": {"size": 32}}],
            "model": "yolo11n-cls.yaml",  # from scratch: no download needed
            "train": {"epochs": 1, "imgsz": 32, "batch": 8, "device": "cpu", "workers": 0, "plots": False},
        }
    )
    from dsip.prepare import prepare

    summary = prepare(cfg, workers=2)
    manifest = (tmp_path / cfg.output / "manifest.csv").read_text()
    assert "mask" not in manifest
    assert ("copy.png" in manifest) != ("sick_0.png" in manifest)  # exactly one of the duplicates kept
    # patient-level split: no patient appears in two splits
    rows = [r.split(",") for r in manifest.strip().splitlines()[1:]]
    splits_per_patient = {}
    for split, _, group, *_ in rows:
        splits_per_patient.setdefault(group, set()).add(split)
    assert all(len(s) == 1 for s in splits_per_patient.values())
    assert set(summary["counts"]) == {"train", "val", "test"}

    from dsip.train import train

    snap_cfg_path = train(cfg, run_name="t") / "dsip_config.yaml"
    run_dir = snap_cfg_path.parent
    assert (run_dir / "weights" / "best.pt").exists()

    from dsip.evaluate import evaluate

    m = evaluate(load_config(snap_cfg_path), run_dir)
    assert {"accuracy", "roc_auc", "at_threshold"} <= set(m)
    assert (run_dir / "eval_test" / "confusion_matrix.png").exists()
    assert json.loads((run_dir / "eval_test" / "metrics.json").read_text())["positive_class"] == "sick"

    from dsip.predict import predict

    out = predict(load_config(snap_cfg_path), run_dir, ["raw/p1/sick"])
    assert len((out / "predictions.csv").read_text().strip().splitlines()) > 1

    from dsip.preview import preview

    assert preview(cfg, n=2, spectrum=True).exists()


def test_detect_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw"
    for i in range(16):
        img = np.full((64, 64, 3), 200, np.uint8)
        x, y = (int(v) for v in RNG.integers(8, 40, 2))
        cv2.rectangle(img, (x, y), (x + 14, y + 14), (60, 20, 90), -1)
        save_image(raw / "images" / f"s{i}.png", img)
        (raw / "labels").mkdir(parents=True, exist_ok=True)
        (raw / "labels" / f"s{i}.txt").write_text(f"0 {(x + 7) / 64:.4f} {(y + 7) / 64:.4f} {14 / 64:.4f} {14 / 64:.4f}\n")
    cfg = from_dict(
        {
            "name": "toydet",
            "task": "detect",
            "data": {"source": "raw", "classes": ["parasite"], "split": {"val": 0.25, "test": 0.25}},
            "preprocess": ["clahe"],
            "model": "yolo11n.yaml",
            "train": {"epochs": 1, "imgsz": 64, "batch": 4, "device": "cpu", "workers": 0, "plots": False},
        }
    )
    from dsip.prepare import prepare

    prepare(cfg, workers=2)
    data = yaml.safe_load((tmp_path / cfg.output / "data.yaml").read_text())
    assert data["names"] == {0: "parasite"} and "test" in data

    from dsip.evaluate import evaluate
    from dsip.train import train

    run_dir = train(cfg, run_name="t")
    m = evaluate(load_config(run_dir / "dsip_config.yaml"), run_dir)
    assert "mAP50" in m
