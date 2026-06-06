"""Train a YOLO26 scaling sweep (n / s / m) with resume-aware orchestration.

Reads:
  data/dataset/data.yaml                              (built by prepare_dataset.py)

Writes (one self-contained folder per fresh invocation):
  02_yolo26_models/outputs/<run_id>/yolo26<n|s|m>/... (Ultralytics dirs, weights, plots)
  02_yolo26_models/outputs/<run_id>/progress.json     (sweep state for THIS run)
  02_yolo26_models/outputs/<run_id>/results_summary.csv

Run folders
-----------
Every fresh invocation creates a unique, timestamped run folder
`outputs/run_YYYYMMDD-HHMMSS/` (a `_2`, `_3`, ... suffix is appended on the rare
same-second collision). All artifacts for that sweep live inside it, and paths
recorded in progress.json are relative to that folder.

Checkpointing (two levels)
--------------------------
- Level 1 (Ultralytics, automatic): every epoch writes `last.pt` (+ `best.pt`).
  A killed run resumes mid-model from `last.pt`, restoring weights, optimizer,
  epoch counter and LR scheduler.
- Level 2 (this script): each model is tracked pending / in_progress / done in
  the run's progress.json. To continue an interrupted sweep, point at its folder
  with `--resume`: finished models are SKIPPED, in-progress ones RESUMED, the
  rest started fresh. Without `--resume` a new folder is created from scratch.
  Cross-machine: commit a run folder's progress.json + best.pt; done models
  never re-train. See AGENTS.md for the cross-machine note.

Usage:
  python train_yolo26.py                      # fresh full n/s/m sweep -> new folder
  python train_yolo26.py --models n s         # subset
  python train_yolo26.py --epochs 50 --batch 8
  python train_yolo26.py --resume latest      # continue most recent run folder
  python train_yolo26.py --resume run_20260602-144300   # continue a specific one
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Must be set before torch is imported so MPS falls back to CPU for any op
# that produces NaN/Inf on Apple Silicon (affects larger YOLO26 variants).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_YAML = ROOT / "data" / "dataset" / "data.yaml"
OUTPUTS_DIR = HERE / "outputs"
# Cache pretrained weights here instead of letting Ultralytics drop them in the
# cwd (repo root). Git-ignored.
PRETRAINED_DIR = HERE / ".pretrained"

# Identical hyperparameters across all scales for a fair comparison. Every value
# the assignment requires us to report is set explicitly here (not left to
# Ultralytics "auto") so the report can quote them directly.
# Exception: `batch` varies per scale (see BATCH_BY_SCALE) because the larger
# variants don't fit Apple-Silicon unified memory at batch 16. Ultralytics
# normalises weight decay + LR to a nominal batch size (nbs=64) via gradient
# accumulation, so the effective optimization stays comparable across scales.
HYPERPARAMS = dict(
    epochs=100,
    patience=20,        # early stop if val mAP plateaus for 20 epochs
    imgsz=640,          # upscales the ~350 px tiles — helps small-pool recall
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,           # final LR = lr0 * lrf
    cos_lr=True,        # cosine LR scheduler
    weight_decay=0.0005,
    warmup_epochs=3.0,
    seed=0,
    # Augmentations (reported): Ultralytics defaults + mosaic, closed near the end.
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5, flipud=0.0,
    degrees=0.0, translate=0.1, scale=0.5,
    mosaic=1.0, close_mosaic=10,
    save_period=5,      # periodic epoch snapshots (epoch5.pt, epoch10.pt, ...)
    plots=True,
    val=True,
)

MODEL_FILES = {"n": "yolo26n.pt", "s": "yolo26s.pt", "m": "yolo26m.pt", "l": "yolo26l.pt", "x": "yolo26x.pt"}

# Per-scale batch size: the larger variants overflow Apple-Silicon unified memory
# at batch 16, so l/x train smaller. A CLI --batch still overrides everything.
# nbs=64 gradient accumulation keeps the effective optimization comparable.
BATCH_BY_SCALE = {"n": 16, "s": 16, "m": 16, "l": 8, "x": 8}


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
    """Unique, human-sortable run id; disambiguate same-second collisions."""
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


def pretrained_path(weights: str) -> str:
    """Resolve a pretrained weight file inside PRETRAINED_DIR, downloading once.

    Passing the full cache path to Ultralytics keeps the auto-download out of the
    repo root (cwd), where it would otherwise land.
    """
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PRETRAINED_DIR / weights
    if not dest.exists():
        attempt_download_asset(str(dest))
    return str(dest)


def device_str() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def param_count(model: YOLO) -> int:
    return sum(p.numel() for p in model.model.parameters())


def extract_metrics(metrics) -> dict:
    """Pull the assignment's required metrics from an Ultralytics results object."""
    box = metrics.box
    speed = getattr(metrics, "speed", {}) or {}
    return {
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "inference_ms": round(float(speed.get("inference", 0.0)), 3),
    }


def train_one(ctx: RunContext, tag: str, weights: str, entry: dict, device: str, overrides: dict) -> dict:
    """Train (or resume) a single model and record val + test metrics.

    Ultralytics writes into `<run folder>/yolo26<tag>/`; recorded paths are
    relative to the run folder so progress.json stays portable.
    """
    run_name = f"yolo26{tag}"
    run_dir = ctx.root / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"

    resuming = entry.get("status") == "in_progress" and last_pt.exists()
    if resuming:
        print(f"\n>>> RESUMING {run_name} from {last_pt.relative_to(ctx.root)}")
        model = YOLO(str(last_pt))
        try:
            model.train(resume=True)
        except Exception as exc:  # already finished, or args mismatch → fall back
            print(f"    resume not applicable ({exc}); using existing best.pt")
            model = YOLO(str(best_pt))
    else:
        print(f"\n>>> TRAINING {run_name} fresh ({weights})")
        model = YOLO(pretrained_path(weights))
        # MPS (Apple Silicon) AMP produces NaN/Inf in the EMA → disable it.
        mps_overrides = {"amp": False} if device == "mps" else {}
        # Per-scale batch; a CLI --batch (in overrides) still wins.
        scale_batch = {"batch": BATCH_BY_SCALE[tag]}
        train_kwargs = dict(
            data=str(DATA_YAML),
            project=str(ctx.root),
            name=run_name,
            exist_ok=True,
            device=device,
            **{**HYPERPARAMS, **scale_batch, **mps_overrides, **overrides},
        )
        try:
            model.train(**train_kwargs)
        except FileNotFoundError:
            if device == "mps":
                # MPS silently fails to serialise checkpoints for larger models;
                # wipe the broken run dir and retry on CPU.
                import shutil
                print(f"\n    MPS checkpoint save failed for {run_name} — retrying on CPU.")
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                model = YOLO(pretrained_path(weights))
                model.train(**{**train_kwargs, "device": "cpu"})
            else:
                raise

    # Always evaluate the best checkpoint on val and the held-out test split.
    # Route Ultralytics' eval output into THIS run folder (otherwise it defaults
    # to a stray runs/detect/ in the cwd).
    best = YOLO(str(best_pt))

    def evaluate(split: str) -> dict:
        return extract_metrics(best.val(
            data=str(DATA_YAML), split=split,
            project=str(ctx.root), name=f"{run_name}_eval_{split}",
            exist_ok=True, plots=True, verbose=False,
        ))

    val_metrics = evaluate("val")
    test_metrics = evaluate("test")

    return {
        "status": "done",
        "run_dir": run_name,
        "weights": str(best_pt.relative_to(ctx.root)),
        "params": param_count(best),
        "val": val_metrics,
        "test": test_metrics,
    }


def write_summary(ctx: RunContext, progress: dict) -> None:
    rows = []
    for tag in MODEL_FILES:
        name = f"yolo26{tag}"
        e = progress.get(name)
        if not e or e.get("status") != "done":
            continue
        rows.append(
            {
                "model": name,
                "params_M": round(e["params"] / 1e6, 2),
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
    """Pick the run folder: resume an existing one, or mint a fresh unique id."""
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
    ap.add_argument("--models", nargs="+", default=list(MODEL_FILES), choices=list(MODEL_FILES))
    ap.add_argument("--epochs", type=int, help="override epochs")
    ap.add_argument("--batch", type=int, help="override batch size")
    ap.add_argument("--imgsz", type=int, help="override image size")
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="continue an existing run folder ('latest' or e.g. run_20260602-144300)")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found — run prepare_dataset.py first.")

    overrides = {k: v for k, v in
                 (("epochs", args.epochs), ("batch", args.batch), ("imgsz", args.imgsz))
                 if v is not None}

    device = device_str()
    print(f"Device: {device}  (torch {torch.__version__}, cuda={torch.cuda.is_available()})")
    if device == "cpu":
        print("WARNING: training on CPU — fine for an n-only smoke test, slow for the full sweep.")
    elif device == "mps":
        print("INFO: training on Apple MPS (Metal) — 3-6× faster than CPU on Apple Silicon.")

    ctx = resolve_context(args.resume)
    progress = ctx.load_progress()

    for tag in args.models:
        name = f"yolo26{tag}"
        entry = progress.get(name, {"status": "pending"})

        if entry.get("status") == "done" and (ctx.root / entry.get("weights", "")).exists():
            print(f"\n>>> SKIP {name} (already done: {entry['weights']})")
            continue

        # Mark in_progress BEFORE training so a crash is recoverable as a resume.
        progress[name] = {**entry, "status": "in_progress"}
        ctx.save_progress(progress)

        result = train_one(ctx, tag, MODEL_FILES[tag], entry, device, overrides)
        progress[name] = result
        ctx.save_progress(progress)
        print(f"    {name} done — params={result['params']/1e6:.2f}M "
              f"val mAP50-95={result['val']['mAP50_95']} test mAP50-95={result['test']['mAP50_95']}")

    write_summary(ctx, progress)
    print(f"\nAll artifacts: {ctx.root.relative_to(HERE)}")


if __name__ == "__main__":
    main()
