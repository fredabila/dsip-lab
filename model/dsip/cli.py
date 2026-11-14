"""Command line interface. Run `python -m dsip --help` (or `dsip --help` once installed)."""

from __future__ import annotations

import argparse
import sys


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", help="Run directory (e.g. runs/malaria_cells/20261114-101500)")
    p.add_argument("--config", "-c", help="Config file; uses its most recent run if --run is not given")


def compare(runs_dir: str, out: str | None = None) -> list[dict]:
    """Collect eval_test/metrics.json from every run into one table."""
    import csv
    import json
    from pathlib import Path

    rows = []
    for f in sorted(Path(runs_dir).glob("*/*/eval_*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        at = m.get("at_threshold", {})
        rows.append(
            {
                "run": f"{f.parts[-4]}/{f.parts[-3]}",
                "split": m.get("split", ""),
                "accuracy": m.get("accuracy"),
                "balanced_acc": m.get("balanced_accuracy"),
                "auc": m.get("roc_auc", m.get("roc_auc_macro_ovr")),
                "sensitivity": at.get("sensitivity"),
                "specificity": at.get("specificity"),
                "mAP50": m.get("mAP50"),
            }
        )
    if not rows:
        print(f"No evaluated runs found in {runs_dir}/")
        return rows
    cols = [c for c in rows[0] if any(r[c] is not None for r in rows)]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("-" if v is None else str(v))

    width = max(len(r["run"]) for r in rows) + 2
    print("".join(c.ljust(width if c == "run" else 14) for c in cols))
    for r in rows:
        print("".join(fmt(r[c]).ljust(width if c == "run" else 14) for c in cols))
    if out:
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {out}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dsip", description="Medical image model training toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ops", help="List available preprocessing operations")

    p = sub.add_parser("preview", help="Save a figure of the preprocessing pipeline on sample images")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--source", help="Override data.source")
    p.add_argument("-n", type=int, default=4, help="Number of sample images")
    p.add_argument("--images", nargs="*", help="Specific images to use instead of random samples")
    p.add_argument("--spectrum", action="store_true", help="Also show Fourier magnitude spectra")
    p.add_argument("--out", help="Output PNG path")

    p = sub.add_parser("prepare", help="Split + preprocess the raw dataset")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--source", help="Override data.source (where you downloaded the dataset)")
    p.add_argument("--overwrite", action="store_true", help="Rebuild if output already exists")
    p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("train", help="Train a model")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--name", help="Run name (default: timestamp)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device", help="e.g. 0 for first GPU, cpu, mps")
    p.add_argument("--model", help="Override model weights, e.g. yolo11s-cls.pt")
    p.add_argument("--no-eval", action="store_true", help="Skip test-set evaluation after training")

    p = sub.add_parser("evaluate", help="Evaluate on the test split (metrics + plots)")
    _add_run_args(p)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for the positive class")
    p.add_argument("--target-sensitivity", type=float, default=0.95)

    p = sub.add_parser("predict", help="Predict on new images / folders")
    _add_run_args(p)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for positive class")

    p = sub.add_parser("compare", help="Table of test metrics for all trained runs")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", help="Also save the table as CSV")

    p = sub.add_parser("export", help="Export weights (onnx, tflite, torchscript, ...)")
    _add_run_args(p)
    p.add_argument("--format", default="onnx")

    p = sub.add_parser("demo", help="Launch a Gradio web demo")
    _add_run_args(p)
    p.add_argument("--share", action="store_true", help="Create a public link (e.g. from Colab)")
    p.add_argument("--port", type=int, default=7860)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from dsip.config import load_config, resolve_run, weights_path

    if args.cmd == "ops":
        from dsip.preprocess import describe_ops

        print(describe_ops())
    elif args.cmd == "preview":
        from dsip.preview import preview

        preview(load_config(args.config, args.source), n=args.n, out=args.out, spectrum=args.spectrum, images=args.images)
    elif args.cmd == "prepare":
        from dsip.prepare import prepare

        prepare(load_config(args.config, args.source), overwrite=args.overwrite, workers=args.workers)
    elif args.cmd == "train":
        from dsip.train import train

        cfg = load_config(args.config)
        if args.model:
            cfg.model = cfg.raw["model"] = args.model
        run_dir = train(cfg, run_name=args.name, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device)
        if not args.no_eval:
            from dsip.config import load_config as _lc
            from dsip.evaluate import evaluate

            evaluate(_lc(run_dir / "dsip_config.yaml"), run_dir)
    elif args.cmd == "evaluate":
        from dsip.evaluate import evaluate

        run_dir, cfg = resolve_run(args.run, args.config)
        evaluate(cfg, run_dir, split=args.split, threshold=args.threshold, target_sensitivity=args.target_sensitivity)
    elif args.cmd == "predict":
        from dsip.predict import predict

        run_dir, cfg = resolve_run(args.run, args.config)
        predict(cfg, run_dir, args.inputs, out_dir=args.out, conf=args.conf, threshold=args.threshold)
    elif args.cmd == "compare":
        compare(args.runs_dir, args.out)
    elif args.cmd == "export":
        from ultralytics import YOLO

        run_dir, cfg = resolve_run(args.run, args.config)
        path = YOLO(str(weights_path(run_dir))).export(format=args.format, imgsz=cfg.imgsz)
        print(f"Exported to {path}")
    elif args.cmd == "demo":
        from dsip.demo import demo

        run_dir, cfg = resolve_run(args.run, args.config)
        demo(cfg, run_dir, share=args.share, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
