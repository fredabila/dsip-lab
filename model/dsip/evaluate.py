"""Evaluate a trained model on the held-out test split with clinically meaningful metrics.

Classification: accuracy, balanced accuracy, per-class precision / recall (sensitivity) / F1,
confusion matrix, ROC-AUC; for binary tasks also sensitivity, specificity, PPV, NPV, PR-AUC
and the decision threshold that reaches a target sensitivity.
Detection: Ultralytics mAP@0.5, mAP@0.5:0.95, precision, recall.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from dsip.config import Config, weights_path
from dsip.preprocess import IMAGE_EXTENSIONS


def _plot_confusion(cm: np.ndarray, names: list[str], path: Path, normalize: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 3, 1.2 * len(names) + 2.5))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{data[i, j]:.2f}" if normalize else str(int(data[i, j]))
            ax.text(j, i, txt, ha="center", va="center", color="white" if data[i, j] > data.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_curves(y_true: np.ndarray, probs: np.ndarray, names: list[str], pos: int | None, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fig, ax = plt.subplots(figsize=(5.5, 5))
    targets = [pos] if pos is not None else range(len(names))
    for k in targets:
        fpr, tpr, _ = roc_curve(y_true == k, probs[:, k])
        ax.plot(fpr, tpr, label=f"{names[k]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=150)
    plt.close(fig)

    if pos is not None:
        prec, rec, _ = precision_recall_curve(y_true == pos, probs[:, pos])
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(rec, prec)
        ax.set_xlabel("Recall (sensitivity)")
        ax.set_ylabel("Precision (PPV)")
        ax.set_title(f"Precision-recall curve: {names[pos]}")
        fig.tight_layout()
        fig.savefig(out / "pr_curve.png", dpi=150)
        plt.close(fig)


def _binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(np.sum(pred & y))
    tn = int(np.sum(~pred & ~y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))

    def div(a, b):
        return float(a / b) if b else float("nan")

    return {
        "threshold": threshold,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
    }


def evaluate_classify(
    cfg: Config, run_dir: Path, split: str = "test", threshold: float = 0.5, target_sensitivity: float = 0.95
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from ultralytics import YOLO

    model = YOLO(str(weights_path(run_dir)))
    names = [model.names[i] for i in range(len(model.names))]
    index = {n: i for i, n in enumerate(names)}

    split_dir = cfg.output / split
    if not split_dir.exists():
        raise FileNotFoundError(f"{split_dir} not found (did you run `dsip prepare`?)")
    files, labels = [], []
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if cls_dir.name not in index:
            print(f"  skipping folder '{cls_dir.name}': model was not trained on it")
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(f)
                labels.append(index[cls_dir.name])
    if not files:
        raise ValueError(f"No images in {split_dir}")
    print(f"Evaluating {len(files)} {split} images with {weights_path(run_dir)}")

    probs = []
    batch = 64
    for i in range(0, len(files), batch):
        results = model.predict([str(f) for f in files[i : i + batch]], imgsz=cfg.imgsz, verbose=False)
        probs += [r.probs.data.cpu().numpy() for r in results]
    probs = np.stack(probs)
    y = np.array(labels)

    pos = index.get(cfg.positive_class) if cfg.positive_class else None
    if pos is None and len(names) == 2:
        pos = 1
    if pos is not None and len(names) == 2:
        pred = np.where(probs[:, pos] >= threshold, pos, 1 - pos)
    else:
        pred = probs.argmax(1)

    out = run_dir / f"eval_{split}"
    out.mkdir(exist_ok=True)
    present = sorted(set(y) | set(pred))
    report = classification_report(
        y, pred, labels=list(range(len(names))), target_names=names, output_dict=True, zero_division=0
    )
    metrics: dict = {
        "split": split,
        "n_images": len(files),
        "classes": names,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class": {n: report[n] for n in names},
        "macro_f1": report["macro avg"]["f1-score"],
    }

    try:
        if len(names) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y == pos, probs[:, pos]))
        elif len(present) == len(names):
            metrics["roc_auc_macro_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        pass  # only one class present in y

    if pos is not None and len(names) == 2:
        yb = y == pos
        metrics["positive_class"] = names[pos]
        metrics["at_threshold"] = _binary_metrics(yb, probs[:, pos], threshold)
        metrics["pr_auc"] = float(average_precision_score(yb, probs[:, pos]))
        fpr, tpr, thr = roc_curve(yb, probs[:, pos])
        ok = np.where(tpr >= target_sensitivity)[0]
        if len(ok):
            t = float(min(thr[ok[0]], 1.0))
            metrics[f"at_sensitivity_{target_sensitivity:g}"] = _binary_metrics(yb, probs[:, pos], t)

    cm = confusion_matrix(y, pred, labels=list(range(len(names))))
    metrics["confusion_matrix"] = cm.tolist()
    _plot_confusion(cm, names, out / "confusion_matrix.png", normalize=False)
    _plot_confusion(cm, names, out / "confusion_matrix_normalized.png", normalize=True)
    _plot_curves(y, probs, names, pos if len(names) == 2 else None, out)

    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "pred", "correct", *[f"p_{n}" for n in names]])
        for fp, t, p, pr in zip(files, y, pred, probs):
            w.writerow([str(fp), names[t], names[p], int(t == p), *[f"{x:.5f}" for x in pr]])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _print_classify(metrics)
    print(f"\nPlots, predictions.csv and metrics.json saved to {out}")
    return metrics


def _print_classify(m: dict) -> None:
    print(f"\nAccuracy           {m['accuracy']:.4f}")
    print(f"Balanced accuracy  {m['balanced_accuracy']:.4f}")
    print(f"Macro F1           {m['macro_f1']:.4f}")
    for k in ("roc_auc", "roc_auc_macro_ovr", "pr_auc"):
        if k in m:
            print(f"{k:<18} {m[k]:.4f}")
    print(f"\n{'class':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'n':>8}")
    for n, r in m["per_class"].items():
        print(f"{n:<20}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1-score']:>10.3f}{int(r['support']):>8}")
    for key in [k for k in m if k.startswith("at_")]:
        b = m[key]
        print(
            f"\n[{m['positive_class']} @ threshold {b['threshold']:.3f}] "
            f"sensitivity={b['sensitivity']:.3f} specificity={b['specificity']:.3f} "
            f"PPV={b['ppv']:.3f} NPV={b['npv']:.3f}  (TP={b['TP']} FP={b['FP']} TN={b['TN']} FN={b['FN']})"
        )


def evaluate_detect(cfg: Config, run_dir: Path, split: str = "test") -> dict:
    import yaml
    from ultralytics import YOLO

    data_yaml = cfg.output / "data.yaml"
    if split == "test" and "test" not in yaml.safe_load(data_yaml.read_text(encoding="utf-8")):
        print("No test split in data.yaml: evaluating on val")
        split = "val"
    model = YOLO(str(weights_path(run_dir)))
    res = model.val(
        data=str(data_yaml.resolve()),
        split=split,
        imgsz=cfg.imgsz,
        project=str(run_dir.resolve()),
        name=f"eval_{split}",
        exist_ok=True,
        plots=True,
    )
    metrics = {
        "split": split,
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "per_class_mAP50_95": {model.names[int(c)]: float(v) for c, v in zip(res.box.ap_class_index, res.box.maps[res.box.ap_class_index])},
    }
    out = run_dir / f"eval_{split}"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def evaluate(cfg: Config, run_dir: Path, split: str = "test", threshold: float = 0.5, target_sensitivity: float = 0.95) -> dict:
    if cfg.task == "classify":
        return evaluate_classify(cfg, run_dir, split, threshold, target_sensitivity)
    return evaluate_detect(cfg, run_dir, split)
