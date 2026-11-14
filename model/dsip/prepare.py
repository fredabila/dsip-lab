"""Turn a raw downloaded dataset into a clean, split, preprocessed training set.

Classification (task: classify)
    Any folder tree where each image's *parent folder name* is its label, e.g.
        cell_images/Parasitized/*.png, cell_images/Uninfected/*.png
        IDC_regular/<patient_id>/0/*.png, IDC_regular/<patient_id>/1/*.png
    Output: <output>/{train,val,test}/<class>/*.png  (Ultralytics classify layout)

Detection (task: detect)
    A YOLO-format dataset: images/ + labels/ (one .txt per image, "cls cx cy w h"),
    either already split (train/valid/test) or a single pool that we split.
    Output: <output>/images/{split}/, <output>/labels/{split}/ and data.yaml
"""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

from dsip.config import Config
from dsip.preprocess import IMAGE_EXTENSIONS, Pipeline, load_image, save_image

SPLITS = ("train", "val", "test")


@dataclass
class Item:
    src: Path
    label: str  # class name (classify) or "" (detect)
    group: str  # patient / slide id used to keep related images in one split
    split: str = ""
    dst_name: str = ""


def _is_excluded(rel: str, patterns: list[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) for p in patterns)


def _images(root: Path, exclude: list[str]) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            rel = p.relative_to(root).as_posix()
            if not _is_excluded(rel, exclude):
                out.append(p)
    return out


def _unique_name(src: Path, root: Path) -> str:
    digest = hashlib.md5(src.relative_to(root).as_posix().encode()).hexdigest()[:8]
    return f"{src.stem}_{digest}.png"


def _dedupe(items: list[Item]) -> list[Item]:
    """Drop byte-identical files (some Kaggle datasets ship every image twice).

    Duplicates that land in both train and test silently inflate your test score.
    """
    seen: dict[str, Item] = {}
    conflicts = 0
    for it in items:
        h = hashlib.md5(it.src.read_bytes()).hexdigest()
        if h in seen:
            conflicts += seen[h].label != it.label
            continue
        seen[h] = it
    removed = len(items) - len(seen)
    if removed:
        print(f"Removed {removed} duplicate files" + (f" ({conflicts} had conflicting labels!)" if conflicts else ""))
    return list(seen.values())


def _split_items(items: list[Item], cfg: Config, rng: random.Random) -> None:
    """Assign train/val/test. With grouping, whole groups (patients) go to one split."""
    v, t = cfg.split["val"], cfg.split["test"]

    def assign(seq: list, weights: list[int]) -> dict:
        total = sum(weights) or 1
        result, acc = {}, 0
        for key, w in zip(seq, weights):
            frac = acc / total
            result[key] = "test" if frac < t else "val" if frac < t + v else "train"
            acc += w
        return result

    if cfg.group_level is not None:
        groups = defaultdict(list)
        for it in items:
            groups[it.group].append(it)
        keys = sorted(groups)
        rng.shuffle(keys)
        mapping = assign(keys, [len(groups[k]) for k in keys])
        for k, its in groups.items():
            for it in its:
                it.split = mapping[k]
    else:  # stratified per class
        by_label = defaultdict(list)
        for it in items:
            by_label[it.label].append(it)
        for its in by_label.values():
            rng.shuffle(its)
            mapping = assign(list(range(len(its))), [1] * len(its))
            for i, it in enumerate(its):
                it.split = mapping[i]


def _balance_train(items: list[Item], mode: str, rng: random.Random) -> list[Item]:
    if mode == "none":
        return items
    train = [it for it in items if it.split == "train"]
    rest = [it for it in items if it.split != "train"]
    by_label = defaultdict(list)
    for it in train:
        by_label[it.label].append(it)
    counts = {k: len(v) for k, v in by_label.items()}
    balanced = []
    if mode == "oversample":
        target = max(counts.values())
        for label, its in by_label.items():
            balanced += its
            for i in range(target - len(its)):
                src = rng.choice(its)
                balanced.append(Item(src.src, label, src.group, "train", src.dst_name.replace(".png", f"_dup{i}.png")))
    else:  # undersample
        target = min(counts.values())
        for its in by_label.values():
            balanced += rng.sample(its, target)
    return balanced + rest


def _process(item: Item, dst: Path, pipeline: Pipeline) -> str | None:
    try:
        save_image(dst, pipeline(load_image(item.src)))
        return None
    except Exception as e:  # keep going on a single corrupt file
        return f"{item.src}: {e}"


def _write_all(jobs: list[tuple[Item, Path]], pipeline: Pipeline, workers: int) -> list[str]:
    errors: list[str] = []
    total = len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, err in enumerate(pool.map(lambda j: _process(j[0], j[1], pipeline), jobs), 1):
            if err:
                errors.append(err)
            if i % 2000 == 0 or i == total:
                print(f"  processed {i}/{total} images", flush=True)
    return errors


def _check_source(cfg: Config) -> Path:
    if cfg.source is None:
        raise ValueError("No dataset source. Set data.source in the config or pass --source <dir>.")
    if not cfg.source.exists():
        raise FileNotFoundError(f"Dataset source not found: {cfg.source}")
    return cfg.source


def _reset_output(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------ classification


def collect_classify(cfg: Config) -> list[Item]:
    root = _check_source(cfg)
    items: list[Item] = []
    for p in _images(root, cfg.exclude):
        folder = p.parent.name
        if cfg.class_map:
            if folder not in cfg.class_map:
                continue
            label = cfg.class_map[folder]
        else:
            if p.parent == root:
                continue  # loose files at the root have no label
            label = folder
        if cfg.group_level is not None:
            parts = p.relative_to(root).parts[:-1]
            idx = len(parts) - 1 - int(cfg.group_level)
            group = parts[idx] if idx >= 0 else parts[0]
        else:
            group = p.relative_to(root).as_posix()
        items.append(Item(p, label, group, dst_name=_unique_name(p, root)))
    if not items:
        raise ValueError(
            f"No labelled images found under {root}. Images must sit in folders named after "
            "their class (or set data.class_map to map folder names to classes)."
        )
    return items


def prepare_classify(cfg: Config, overwrite: bool = False, workers: int = 8) -> dict:
    rng = random.Random(cfg.seed)
    items = collect_classify(cfg)
    if cfg.dedupe:
        items = _dedupe(items)
    counts = Counter(it.label for it in items)
    print(f"Found {len(items)} images in {len(counts)} classes: {dict(counts)}")
    small = [c for c, n in counts.items() if n < 10]
    if small:
        print(f"  WARNING: classes with < 10 images (wrong folder?): {small}")

    if cfg.max_per_class:
        by_label = defaultdict(list)
        for it in items:
            by_label[it.label].append(it)
        items = []
        for its in by_label.values():
            rng.shuffle(its)
            items += its[: cfg.max_per_class]
        print(f"Subsampled to at most {cfg.max_per_class} per class -> {len(items)} images")

    _split_items(items, cfg, rng)
    empty = [c for c in counts if not any(it.split == "train" and it.label == c for it in items)]
    if empty:
        raise ValueError(f"No training images for class(es) {empty}: too few images/patients for this split")
    items = _balance_train(items, cfg.balance, rng)

    out = cfg.output
    _reset_output(out, overwrite)
    pipeline = cfg.pipeline
    print(f"Preprocessing: {pipeline}")
    jobs = [(it, out / it.split / it.label / it.dst_name) for it in items]
    errors = _write_all(jobs, pipeline, workers)

    summary = _summarise(items, cfg, errors)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "group", "source", "file"])
        for it in items:
            w.writerow([it.split, it.label, it.group, str(it.src), f"{it.split}/{it.label}/{it.dst_name}"])
    return summary


# ----------------------------------------------------------------- detection


def _label_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def _existing_split(rel_parts: tuple[str, ...]) -> str | None:
    lowered = [p.lower() for p in rel_parts]
    for name, split in (("train", "train"), ("valid", "val"), ("val", "val"), ("validation", "val"), ("test", "test")):
        if name in lowered:
            return split
    return None


def _detect_names(cfg: Config, root: Path) -> list[str]:
    if cfg.classes:
        return cfg.classes
    for y in [root / "data.yaml", *root.glob("*.yaml")]:
        if y.exists():
            names = (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)]
            if names:
                return list(names)
    raise ValueError("Could not find class names: add data.classes to the config or a data.yaml with `names`")


def prepare_detect(cfg: Config, overwrite: bool = False, workers: int = 8) -> dict:
    root = _check_source(cfg)
    rng = random.Random(cfg.seed)
    names = _detect_names(cfg, root)

    items: list[Item] = []
    presplit = True
    for p in _images(root, cfg.exclude):
        it = Item(p, "", p.relative_to(root).as_posix(), dst_name=_unique_name(p, root))
        s = _existing_split(p.relative_to(root).parts[:-1])
        if s is None:
            presplit = False
        it.split = s or ""
        items.append(it)
    if not items:
        raise ValueError(f"No images found under {root}")
    if cfg.dedupe:
        items = _dedupe(items)

    if not presplit:
        print("No train/val/test folders found: splitting randomly")
        for it in items:
            it.label = "all"
        _split_items(items, cfg, rng)
    elif not any(it.split == "test" for it in items):
        print("Dataset has no test split: evaluation will use the validation split")

    out = cfg.output
    _reset_output(out, overwrite)
    pipeline = cfg.pipeline
    print(f"Found {len(items)} images, classes: {names}")
    print(f"Preprocessing: {pipeline}")

    jobs, missing = [], 0
    for it in items:
        lbl = _label_for(it.src)
        dst_lbl = out / "labels" / it.split / it.dst_name.replace(".png", ".txt")
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        if lbl.exists():
            shutil.copyfile(lbl, dst_lbl)
        else:
            missing += 1
            dst_lbl.write_text("")
        jobs.append((it, out / "images" / it.split / it.dst_name))
    if missing:
        print(f"  {missing} images had no label file (treated as background / no objects)")
    errors = _write_all(jobs, pipeline, workers)

    splits = {s for s in SPLITS if any(it.split == s for it in items)}
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val" if "val" in splits else "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    if "test" in splits:
        data_yaml["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    for it in items:
        it.label = "image"
    return _summarise(items, cfg, errors)


# ------------------------------------------------------------------- common


def _summarise(items: list[Item], cfg: Config, errors: list[str]) -> dict:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for it in items:
        table[it.split][it.label] += 1
    summary = {
        "name": cfg.name,
        "task": cfg.task,
        "source": str(cfg.source),
        "output": str(cfg.output),
        "preprocess": cfg.preprocess,
        "balance": cfg.balance,
        "group_level": cfg.group_level,
        "counts": {s: dict(table[s]) for s in SPLITS if s in table},
        "errors": errors[:50],
        "n_errors": len(errors),
    }
    (cfg.output / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSplit counts:")
    for s in SPLITS:
        if s in table:
            print(f"  {s:5s} " + "  ".join(f"{k}={v}" for k, v in sorted(table[s].items())))
    if errors:
        print(f"\n{len(errors)} files could not be read (see prepare_summary.json)")
    print(f"\nPrepared dataset written to {cfg.output}")
    return summary


def prepare(cfg: Config, overwrite: bool = False, workers: int = 8) -> dict:
    if cfg.task == "classify":
        return prepare_classify(cfg, overwrite, workers)
    return prepare_detect(cfg, overwrite, workers)
