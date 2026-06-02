"""Train an RF-DETR sweep (Nano / Small) with resume-aware orchestration.

Mirrors 02_yolo26_models/train_yolo26.py so the two architectures share the same
workflow: unique per-run output folders, two-level checkpointing, and a metrics
table on the SAME val/test split (built from the shared split.json).

Reads:
  data/dataset_coco/{train,valid,test}/...            (built by prepare_dataset.py)

Writes (one self-contained folder per fresh invocation):
  03_rf_detr_models/outputs/<run_id>/rfdetr_<nano|small>/...  (RF-DETR output_dir:
        checkpoint.pth, checkpoint_best_ema.pth, checkpoint_best_total.pth, metrics.csv)
  03_rf_detr_models/outputs/<run_id>/progress.json
  03_rf_detr_models/outputs/<run_id>/results_summary.csv

Checkpointing (two levels)
--------------------------
- Level 1 (RF-DETR, automatic): writes `checkpoint.pth` each epoch plus
  best-EMA/best-total. A killed run resumes from `checkpoint.pth` via the
  `resume=` argument, restoring weights, optimizer, and epoch.
- Level 2 (this script): each model is tracked pending / in_progress / done in
  the run's progress.json. Continue an interrupted sweep with `--resume`:
  finished models are SKIPPED, in-progress ones RESUMED from their checkpoint,
  the rest started fresh. See AGENTS.md for the cross-machine note.

Usage:
  python train_rfdetr.py                       # fresh nano+small sweep -> new folder
  python train_rfdetr.py --models nano
  python train_rfdetr.py --epochs 30 --batch 4 --grad-accum 4
  python train_rfdetr.py --resume latest       # continue most recent run folder
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COCO_DIR = ROOT / "data" / "dataset_coco"
OUTPUTS_DIR = HERE / "outputs"

# Lazily imported so dataset prep / --help work without rfdetr installed.
def _model_classes() -> dict:
    from rfdetr import RFDETRNano, RFDETRSmall
    return {"nano": RFDETRNano, "small": RFDETRSmall}


MODEL_TAGS = ("nano", "small")

# Identical hyperparameters across both scales for a fair comparison. RF-DETR
# uses AdamW internally; the reportable knobs are set explicitly here.
HYPERPARAMS = dict(
    epochs=50,
    batch_size=4,         # small GPUs: keep low and lean on grad accumulation
    grad_accum_steps=4,   # effective batch = batch_size * grad_accum_steps = 16
    lr=1e-4,
    lr_encoder=1.5e-4,
    early_stopping=True,
    early_stopping_patience=10,
)

# Confidence threshold for precision/recall reporting (mAP uses all detections).
PR_THRESHOLD = 0.5


@dataclass
class RunContext:
    """Everything tied to a single run folder; paths are relative to `root`."""
    root: Path

    @property
    def progress_file(self) -> Path:
        return self.root / "progress.json"

    @property
    def summary_csv(self) -> Path:
        return self.root / "results_summary.csv"

    def load_progress(self) -> dict:
        if self.progress_file.exists():
            return json.loads(self.progress_file.read_text())
        return {}

    def save_progress(self, progress: dict) -> None:
        self.progress_file.write_text(json.dumps(progress, indent=2))


def new_run_id() -> str:
    rid = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not (OUTPUTS_DIR / rid).exists():
        return rid
    i = 2
    while (OUTPUTS_DIR / f"{rid}_{i}").exists():
        i += 1
    return f"{rid}_{i}"


def latest_run_id() -> str | None:
    runs = sorted(p.name for p in OUTPUTS_DIR.glob("run_*") if p.is_dir())
    return runs[-1] if runs else None


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def param_count(model) -> int | None:
    """Best-effort parameter count across RF-DETR's wrapper attributes."""
    for attr in ("model.model", "model"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return sum(p.numel() for p in obj.parameters())
        except (AttributeError, TypeError):
            continue
    return None


def best_checkpoint(out_dir: Path) -> Path | None:
    for name in ("checkpoint_best_ema.pth", "checkpoint_best_total.pth", "checkpoint.pth"):
        p = out_dir / name
        if p.exists():
            return p
    return None


def evaluate(model, split_folder: str) -> dict:
    """Evaluate a loaded RF-DETR model on a COCO split via supervision metrics.

    mAP uses all detections (threshold 0); precision/recall are reported at
    PR_THRESHOLD. Any metric that can't be computed is recorded as None so a
    single failure never sinks the whole run.
    """
    import supervision as sv
    from supervision.metrics import MeanAveragePrecision
    from PIL import Image

    split_dir = COCO_DIR / split_folder
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=str(split_dir),
        annotations_path=str(split_dir / "_annotations.coco.json"),
    )

    targets, predictions = [], []
    for path, _image, annotations in ds:
        det = model.predict(Image.open(path), threshold=0.0)
        targets.append(annotations)
        predictions.append(det)

    out: dict = {"mAP50": None, "mAP50_95": None, "precision": None, "recall": None}
    try:
        m = MeanAveragePrecision().update(predictions, targets).compute()
        out["mAP50"] = round(float(m.map50), 4)
        out["mAP50_95"] = round(float(m.map50_95), 4)
    except Exception as exc:
        print(f"    [warn] mAP failed on {split_folder}: {exc}")

    try:
        from supervision.metrics import Precision, Recall
        pr_pred = [d[d.confidence >= PR_THRESHOLD] for d in predictions]
        out["precision"] = round(float(Precision().update(pr_pred, targets).compute().precision_at_50), 4)
        out["recall"] = round(float(Recall().update(pr_pred, targets).compute().recall_at_50), 4)
    except Exception as exc:
        print(f"    [warn] precision/recall unavailable on {split_folder}: {exc}")

    return out


def train_one(ctx: RunContext, tag: str, model_cls, entry: dict, overrides: dict) -> dict:
    """Train (or resume) a single RF-DETR model and record val + test metrics."""
    run_name = f"rfdetr_{tag}"
    out_dir = ctx.root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.pth"

    resuming = entry.get("status") == "in_progress" and ckpt.exists()
    hp = {**HYPERPARAMS, **overrides}

    model = model_cls()
    if resuming:
        print(f"\n>>> RESUMING {run_name} from {ckpt.relative_to(ctx.root)}")
        model.train(dataset_dir=str(COCO_DIR), output_dir=str(out_dir),
                    resume=str(ckpt), **hp)
    else:
        print(f"\n>>> TRAINING {run_name} fresh")
        model.train(dataset_dir=str(COCO_DIR), output_dir=str(out_dir), **hp)

    # Reload the best checkpoint for evaluation.
    best = best_checkpoint(out_dir)
    eval_model = model_cls(pretrain_weights=str(best)) if best else model
    try:
        eval_model.optimize_for_inference()
    except Exception:
        pass

    val_metrics = evaluate(eval_model, "valid")
    test_metrics = evaluate(eval_model, "test")

    return {
        "status": "done",
        "run_dir": run_name,
        "weights": str(best.relative_to(ctx.root)) if best else None,
        "params": param_count(eval_model),
        "val": val_metrics,
        "test": test_metrics,
    }


def write_summary(ctx: RunContext, progress: dict) -> None:
    rows = []
    for tag in MODEL_TAGS:
        name = f"rfdetr_{tag}"
        e = progress.get(name)
        if not e or e.get("status") != "done":
            continue
        rows.append(
            {
                "model": name,
                "params_M": round(e["params"] / 1e6, 2) if e.get("params") else None,
                **{f"val_{k}": v for k, v in e["val"].items()},
                **{f"test_{k}": v for k, v in e["test"].items()},
            }
        )
    if not rows:
        return
    with ctx.summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== Summary ({ctx.summary_csv.relative_to(HERE)}) ===")
    hdr = list(rows[0])
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(r[h]) for h in hdr))


def resolve_context(resume: str | None) -> RunContext:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if resume:
        run_id = latest_run_id() if resume == "latest" else resume
        if run_id is None:
            raise SystemExit("--resume latest: no existing run_* folders in outputs/")
        root = OUTPUTS_DIR / run_id
        if not root.exists():
            raise SystemExit(f"--resume {resume}: {root.relative_to(HERE)} does not exist")
        print(f"Resuming run folder: outputs/{run_id}")
    else:
        run_id = new_run_id()
        root = OUTPUTS_DIR / run_id
        root.mkdir(parents=True)
        print(f"New run folder: outputs/{run_id}")
    return RunContext(root=root)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODEL_TAGS), choices=list(MODEL_TAGS))
    ap.add_argument("--epochs", type=int, help="override epochs")
    ap.add_argument("--batch", type=int, dest="batch_size", help="override batch size")
    ap.add_argument("--grad-accum", type=int, dest="grad_accum_steps", help="override grad accumulation steps")
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="continue an existing run folder ('latest' or e.g. run_20260602-144300)")
    args = ap.parse_args()

    if not (COCO_DIR / "train" / "_annotations.coco.json").exists():
        raise SystemExit(f"{COCO_DIR} not ready — run prepare_dataset.py first.")

    overrides = {k: v for k, v in
                 (("epochs", args.epochs), ("batch_size", args.batch_size),
                  ("grad_accum_steps", args.grad_accum_steps))
                 if v is not None}

    device = device_str()
    print(f"Device: {device}  (torch {torch.__version__}, cuda={torch.cuda.is_available()})")
    if device == "cpu":
        print("WARNING: RF-DETR on CPU is impractically slow — use the GPU desktop.")

    model_classes = _model_classes()
    ctx = resolve_context(args.resume)
    progress = ctx.load_progress()

    for tag in args.models:
        name = f"rfdetr_{tag}"
        entry = progress.get(name, {"status": "pending"})

        if entry.get("status") == "done" and entry.get("weights") and (ctx.root / entry["weights"]).exists():
            print(f"\n>>> SKIP {name} (already done: {entry['weights']})")
            continue

        # Mark in_progress BEFORE training so a crash is recoverable as a resume.
        progress[name] = {**entry, "status": "in_progress"}
        ctx.save_progress(progress)

        result = train_one(ctx, tag, model_classes[tag], entry, overrides)
        progress[name] = result
        ctx.save_progress(progress)
        params = f"{result['params']/1e6:.2f}M" if result.get("params") else "n/a"
        print(f"    {name} done — params={params} "
              f"val mAP50-95={result['val']['mAP50_95']} test mAP50-95={result['test']['mAP50_95']}")

    write_summary(ctx, progress)
    print(f"\nAll artifacts: {ctx.root.relative_to(HERE)}")


if __name__ == "__main__":
    main()
