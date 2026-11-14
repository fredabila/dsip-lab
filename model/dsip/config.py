"""Loading and validating project configs (configs/*.yaml)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dsip.preprocess import Pipeline

TASKS = {"classify", "detect"}

DEFAULT_MODELS = {"classify": "yolo11n-cls.pt", "detect": "yolo11n.pt"}

RUN_CONFIG_NAME = "dsip_config.yaml"


@dataclass
class Config:
    name: str
    task: str
    source: Path | None
    output: Path
    runs_dir: Path
    model: str
    preprocess: list[Any] = field(default_factory=list)
    class_map: dict[str, str] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    group_level: int | None = None
    split: dict[str, float] = field(default_factory=lambda: {"val": 0.15, "test": 0.15})
    balance: str = "none"
    max_per_class: int | None = None
    seed: int = 42
    dedupe: bool = True
    positive_class: str | None = None
    train: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def pipeline(self) -> Pipeline:
        return Pipeline(self.preprocess)

    @property
    def imgsz(self) -> int:
        return int(self.train.get("imgsz", 224 if self.task == "classify" else 640))

    def to_yaml(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.raw, sort_keys=False), encoding="utf-8")


def load_config(path: str | Path, source: str | None = None) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = copy.deepcopy(raw)
    if source is not None:
        raw.setdefault("data", {})["source"] = source
    cfg = from_dict(raw)
    cfg.path = path
    return cfg


def from_dict(raw: dict[str, Any]) -> Config:
    name = raw.get("name")
    if not name:
        raise ValueError("Config needs a `name`")
    task = raw.get("task", "classify")
    if task not in TASKS:
        raise ValueError(f"`task` must be one of {sorted(TASKS)}, got '{task}'")

    data = raw.get("data", {}) or {}
    balance = data.get("balance", "none") or "none"
    if balance not in {"none", "oversample", "undersample"}:
        raise ValueError("data.balance must be none | oversample | undersample")
    split = {"val": 0.15, "test": 0.15, **(data.get("split") or {})}
    if split["val"] + split["test"] >= 1:
        raise ValueError("data.split val + test must be < 1")

    cfg = Config(
        name=name,
        task=task,
        source=Path(data["source"]).expanduser() if data.get("source") else None,
        output=Path(data.get("output", f"data/prepared/{name}")).expanduser(),
        runs_dir=Path(raw.get("runs_dir", f"runs/{name}")).expanduser(),
        model=raw.get("model") or DEFAULT_MODELS[task],
        preprocess=raw.get("preprocess") or [],
        class_map={str(k): str(v) for k, v in (data.get("class_map") or {}).items()},
        classes=[str(c) for c in (data.get("classes") or [])],
        exclude=list(data.get("exclude") or []),
        group_level=data.get("group_level"),
        split=split,
        balance=balance,
        max_per_class=data.get("max_per_class"),
        seed=int(data.get("seed", 42)),
        dedupe=bool(data.get("dedupe", True)),
        positive_class=raw.get("positive_class"),
        train=dict(raw.get("train") or {}),
        raw=raw,
    )
    cfg.pipeline  # validate preprocessing steps early
    return cfg


def latest_run(cfg: Config) -> Path:
    """Most recently trained run directory for this config."""
    runs = [p for p in cfg.runs_dir.glob("*") if (p / "weights" / "best.pt").exists()]
    if not runs:
        raise FileNotFoundError(f"No trained runs found in {cfg.runs_dir}. Run `dsip train` first.")
    return max(runs, key=lambda p: (p / "weights" / "best.pt").stat().st_mtime)


def resolve_run(run: str | None, config: str | None) -> tuple[Path, Config]:
    """Find a run directory and the config it was trained with."""
    if run:
        run_dir = Path(run)
    elif config:
        run_dir = latest_run(load_config(config))
    else:
        raise ValueError("Pass --run <run dir> or --config <config.yaml>")
    snap = run_dir / RUN_CONFIG_NAME
    if snap.exists():
        cfg = load_config(snap)
    elif config:
        cfg = load_config(config)
    else:
        raise FileNotFoundError(f"{snap} not found; pass --config too")
    return run_dir, cfg


def weights_path(run_dir: Path) -> Path:
    w = run_dir / "weights" / "best.pt"
    if not w.exists():
        raise FileNotFoundError(f"No weights at {w}")
    return w
