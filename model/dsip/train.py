"""Train a model on a prepared dataset with Ultralytics YOLO (classification or detection)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dsip.config import RUN_CONFIG_NAME, Config


def data_arg(cfg: Config) -> str:
    out = cfg.output
    if cfg.task == "classify":
        if not (out / "train").exists():
            raise FileNotFoundError(f"{out}/train not found. Run `dsip prepare` first.")
        return str(out.resolve())
    y = out / "data.yaml"
    if not y.exists():
        raise FileNotFoundError(f"{y} not found. Run `dsip prepare` first.")
    return str(y.resolve())


def train(cfg: Config, run_name: str | None = None, **overrides) -> Path:
    from ultralytics import YOLO

    params = {
        "epochs": 30,
        "imgsz": cfg.imgsz,
        "batch": 32 if cfg.task == "classify" else 16,
        "patience": 10,
        "seed": cfg.seed,
        "plots": True,
    }
    params.update(cfg.train)
    params.update({k: v for k, v in overrides.items() if v is not None})
    run_name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"Training '{cfg.name}' ({cfg.task}) from {cfg.model}")
    print(f"Params: {params}")
    model = YOLO(cfg.model)
    model.train(
        data=data_arg(cfg),
        project=str(cfg.runs_dir.resolve()),
        name=run_name,
        exist_ok=False,
        **params,
    )
    run_dir = Path(model.trainer.save_dir)
    cfg.to_yaml(run_dir / RUN_CONFIG_NAME)
    print(f"\nRun saved to {run_dir}")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")
    return run_dir
