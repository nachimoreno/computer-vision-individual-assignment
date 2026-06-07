"""Generate the single final-report notebook (05_report/report.ipynb).

This is an AUTHORING tool, not part of the report. It assembles a notebook from
markdown narrative + lightweight code cells that ONLY load pre-computed artifacts
(no training, no model inference). After generation, execute it once to bake the
outputs in:

    python 05_report/build_report.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 05_report/report.ipynb

Design rules (see C:/Users/nachi/.claude/plans/...):
- Every visual is produced as a CELL OUTPUT (IPython.display.Image or a matplotlib
  figure) so the executed .ipynb embeds the bytes and is self-contained.
- Paths resolve from a robustly-discovered repo ROOT, so execution works under
  nbconvert regardless of CWD.
- Helpers fail soft (warn, never raise) so one missing artifact can't abort the run.
- OBB is OUT OF SCOPE: no OBB section; Q4 is answered conceptually + flagged.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent / "report.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(src: str) -> None:
    cells.append(new_code_cell(src.strip("\n")))


# --------------------------------------------------------------------------- #
# 1. Title
# --------------------------------------------------------------------------- #
md(r"""
# Detecting Swimming Pools from Aerial Imagery — YOLO26 vs RF-DETR

**Author:** Ignacio Agustín Moreno · `nachi@student.ie.edu`
**Date:** 2026-06-07
**Hardware:** NVIDIA GeForce RTX 5060 Ti (16 GB, Blackwell), CUDA build of PyTorch.

This notebook is the consolidated report for the individual assignment. It covers
the full pipeline: **(1)** semi-automatic annotation with GroundingDINO, **(2)**
the YOLO26 `n/s/m/l/x` training sweep, **(3)** the RF-DETR
`nano/small/medium/base/large` sweep, **(4)** a unified head-to-head evaluation,
and the required experimental + failure analysis.

> **Reproducibility note.** Every figure and table below is loaded from
> artifacts already produced by the training and evaluation scripts
> (`02_yolo26_models/`, `03_rf_detr_models/`, `04_evaluation/`). **The notebook
> does not re-train or re-run inference** — it only reads and displays saved
> outputs, so it executes in seconds and the saved `.ipynb` is self-contained.

> **Scope.** Oriented Bounding Boxes (OBB, assignment Step 4) were **not
> attempted** in this submission. The OBB analysis question (Q4) is therefore
> answered conceptually and explicitly marked out of scope.
""")

# --------------------------------------------------------------------------- #
# 2. Setup
# --------------------------------------------------------------------------- #
code(r"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

%matplotlib inline
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def find_root(start: Path) -> Path:
    # Walk up until we find a known anchor artifact; robust to any CWD.
    for cand in [start.resolve(), *start.resolve().parents]:
        if (cand / "04_evaluation" / "outputs" / "comparison.csv").exists():
            return cand
    return start.resolve()


def latest_run(outputs_dir: Path) -> Path:
    runs = sorted(p for p in outputs_dir.glob("run_*") if p.is_dir())
    return runs[-1] if runs else outputs_dir


ROOT = find_root(Path.cwd())
YOLO_RUN = latest_run(ROOT / "02_yolo26_models" / "outputs")
RFDETR_RUN = latest_run(ROOT / "03_rf_detr_models" / "outputs")
EVAL = ROOT / "04_evaluation" / "outputs"

print("Repo ROOT :", ROOT)
print("YOLO run  :", YOLO_RUN.name)
print("RF-DETR run:", RFDETR_RUN.name)
""")

# --------------------------------------------------------------------------- #
# 3. Helpers
# --------------------------------------------------------------------------- #
code(r"""
def show_image(path, caption=None, width=900):
    p = Path(path)
    if not p.exists():
        print(f"⚠ MISSING: {p}")
        return
    if caption:
        display(Markdown(f"*{caption}*"))
    display(Image(filename=str(p), width=width))


def show_grid(paths, ncols=3, titles=None, figsize_per=(4.6, 4.6), suptitle=None):
    paths = [Path(p) for p in paths]
    if not paths:
        print("⚠ no images to show")
        return
    n = len(paths)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, paths):
        idx = paths.index(p)
        title = titles[idx] if titles else p.name
        if p.exists():
            ax.imshow(plt.imread(str(p)))
            ax.set_title(title, fontsize=9)
        else:
            ax.text(0.5, 0.5, f"MISSING\n{p.name}", ha="center", va="center", fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    plt.show()


def show_table(csv_path, round_=3, cols=None, rename=None, highlight=None):
    p = Path(csv_path)
    if not p.exists():
        print(f"⚠ MISSING: {p}")
        return None
    df = pd.read_csv(p)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if rename:
        df = df.rename(columns=rename)
    if round_ is not None:
        df = df.round(round_)
    sty = df.style.hide(axis="index")
    if highlight:
        sty = sty.highlight_max(subset=[c for c in highlight if c in df.columns], color="#cdebc5")
    display(sty)
    return df


def load_rfdetr_metrics(model_dir):
    # RF-DETR metrics.csv logs sparse train & val rows per epoch -- merge them.
    p = Path(model_dir) / "metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "epoch" not in df.columns:
        return None
    return df.groupby("epoch").mean(numeric_only=True).reset_index()


def plot_rfdetr_curves(run_dir, tags):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for tag in tags:
        d = load_rfdetr_metrics(Path(run_dir) / f"rfdetr_{tag}")
        if d is None or d.empty:
            continue
        ycol = "val/ema_mAP_50_95" if "val/ema_mAP_50_95" in d.columns else "val/mAP_50_95"
        if ycol in d.columns:
            ax1.plot(d["epoch"], d[ycol], marker="o", ms=3, label=tag)
        if "val/loss" in d.columns:
            ax2.plot(d["epoch"], d["val/loss"], marker="o", ms=3, label=tag)
    ax1.set(title="RF-DETR — val mAP50-95 (EMA) vs epoch", xlabel="epoch", ylabel="mAP50-95")
    ax2.set(title="RF-DETR — val loss vs epoch", xlabel="epoch", ylabel="loss")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


print("helpers ready")
""")

# --------------------------------------------------------------------------- #
# 4. Artifact audit
# --------------------------------------------------------------------------- #
code(r"""
# Up-front audit: confirm every artifact the report displays is present.
_required = {
    "split.json": ROOT / "data" / "dataset" / "split.json",
    "classes.txt": ROOT / "data" / "output" / "classes.txt",
    "annotation sample (raw)": ROOT / "data" / "output" / "annotated" / "100.png",
    "annotation sample (clean)": ROOT / "data" / "output" / "annotated_clean" / "100.png",
    "YOLO results_summary.csv": YOLO_RUN / "results_summary.csv",
    "YOLO yolo26l/results.png": YOLO_RUN / "yolo26l" / "results.png",
    "RF-DETR results_summary.csv": RFDETR_RUN / "results_summary.csv",
    "RF-DETR large/metrics.csv": RFDETR_RUN / "rfdetr_large" / "metrics.csv",
    "comparison.csv": EVAL / "comparison.csv",
    "fig_map_vs_params.png": EVAL / "fig_map_vs_params.png",
    "failure (yolo26l)": EVAL / "failures" / "yolo26l" / "FP0_FN2__85.PNG",
    "failure (rfdetr_large)": EVAL / "failures" / "rfdetr_large" / "FP3_FN1__40.PNG",
}
_audit = pd.DataFrame(
    [{"artifact": k,
      "status": "✓" if Path(v).exists() else "⚠ MISSING",
      "path": str(Path(v).relative_to(ROOT))}
     for k, v in _required.items()]
)
display(_audit.style.hide(axis="index"))
_missing = (_audit["status"] != "✓").sum()
print("All artifacts present." if _missing == 0 else f"{_missing} artifact(s) missing — see table.")
""")

# --------------------------------------------------------------------------- #
# 5. §1 Dataset
# --------------------------------------------------------------------------- #
md(r"""
---
## 1. Dataset & split

The dataset is **288 aerial tiles** (~350 px) containing swimming pools, a single
detection class (`pool`). It is split **70 / 20 / 10** into train / val / test
with a fixed seed (`seed=0`), **stratified on whether a tile contains a pool**, so
each split holds a representative mix of positive (pool) and negative (no-pool)
tiles.

**Why this split.** A fixed, stratified seed makes the split reproducible and keeps
the positive/negative balance stable across train/val/test — important here because
~44% of tiles are negatives. The **same split is shared by both architectures**
(YOLO reads it directly; RF-DETR's COCO export is built from the identical
membership) so the head-to-head comparison is fair. Negatives are kept in all
splits as **hard-background** examples to suppress false positives.
""")

code(r"""
split = json.loads((ROOT / "data" / "dataset" / "split.json").read_text())
counts = split["counts"]
df_split = pd.DataFrame(counts).T[["total", "pos", "neg"]]
df_split.loc["TOTAL"] = df_split.sum()
df_split.index.name = "split"
print(f"class(es): {(ROOT / 'data' / 'output' / 'classes.txt').read_text().split()}"
      f"   |   seed={split['seed']}   ratios={split['ratios']}")
display(df_split)
""")

# --------------------------------------------------------------------------- #
# 6. §2 Step 1 Annotation
# --------------------------------------------------------------------------- #
md(r"""
---
## 2. Step 1 — Annotation pipeline (GroundingDINO + manual review)

Labels were bootstrapped with **GroundingDINO** (zero-shot, prompt `"swimming
pool"`, `box_threshold=0.25`, `text_threshold=0.20`), then **cleaned and manually
reviewed**. Automatic boxes are intentionally imperfect, so the pipeline:

1. **Auto-annotate** every tile with GroundingDINO → raw YOLO labels.
2. **Clean** (`01_annotations/clean_annotations.py`): drop boxes covering >70% of
   the image (`max_area_frac=0.70`) and de-duplicate overlaps via NMS
   (`nms_iou=0.50`), ranking smaller boxes higher (pools are small).
3. **Manual review** in Roboflow (assist-only — not used for training): fix wrong
   labels, remove false positives, add missed pools.

The panels below show **raw GroundingDINO output (top)** vs **cleaned labels
(bottom)** for the same tiles — the cleaning step removes spurious / oversized
boxes before human review.
""")

code(r"""
samples = ["100", "102", "108"]
raw = [ROOT / "data" / "output" / "annotated" / f"{s}.png" for s in samples]
clean = [ROOT / "data" / "output" / "annotated_clean" / f"{s}.png" for s in samples]
show_grid(raw + clean, ncols=3,
          titles=[f"raw {s}" for s in samples] + [f"clean {s}" for s in samples],
          suptitle="GroundingDINO raw (top) vs cleaned (bottom)")
""")

# --------------------------------------------------------------------------- #
# 7. §3 Step 2 YOLO26
# --------------------------------------------------------------------------- #
md(r"""
---
## 3. Step 2 — YOLO26 training (transfer learning, `n/s/m/l/x`)

All five YOLO26 scales were fine-tuned from pretrained COCO weights with
**identical hyperparameters** (only batch size varies by scale) for a fair
size comparison.

| Setting | Value |
|---|---|
| Optimizer | **AdamW** |
| Learning rate | `lr0=0.001`, `lrf=0.01` (final = lr0·lrf) |
| LR scheduler | **cosine** (`cos_lr=True`), 3 warm-up epochs |
| Weight decay | `5e-4` |
| Epochs | 100, early stop `patience=20` |
| Image size | **640** (upscales the ~350 px tiles → helps small-pool recall) |
| Batch size | 16 (`l`/`x`: 8) |
| Augmentations | mosaic=1.0 (closed last 10 epochs), HSV (0.015/0.7/0.4), fliplr=0.5, translate=0.1, scale=0.5; no vertical flip / rotation |
| Hardware | RTX 5060 Ti (16 GB) |

**Per-model metrics (val + held-out test), as reported by the Ultralytics
validator during training:**
""")

code(r"""
show_table(YOLO_RUN / "results_summary.csv", round_=3,
           highlight=["test_mAP50", "test_mAP50_95", "test_recall"])
""")

md(r"""
**Training curves (best scale, `yolo26l`).** Box/cls/dfl losses fall steadily and
val mAP50 / mAP50-95 plateau before early stopping triggers:
""")
code(r"""
show_image(YOLO_RUN / "yolo26l" / "results.png", caption="yolo26l training curves", width=1000)
""")

md(r"""
**Validation predictions vs ground truth (`yolo26l`)** — qualitative check that the
model localises pools well on held-out tiles:
""")
code(r"""
show_grid([YOLO_RUN / "yolo26l" / "val_batch0_pred.jpg",
           YOLO_RUN / "yolo26l" / "val_batch0_labels.jpg"],
          ncols=2, titles=["predictions", "ground truth"], figsize_per=(7, 7))
""")

md(r"""
**Confusion matrix and PR / F1 curves (`yolo26l`):**
""")
code(r"""
show_grid([YOLO_RUN / "yolo26l" / "confusion_matrix.png",
           YOLO_RUN / "yolo26l" / "BoxPR_curve.png",
           YOLO_RUN / "yolo26l" / "BoxF1_curve.png"],
          ncols=3, titles=["confusion matrix", "PR curve", "F1 curve"], figsize_per=(5, 5))
""")

# --------------------------------------------------------------------------- #
# 8. §4 Step 3 RF-DETR
# --------------------------------------------------------------------------- #
md(r"""
---
## 4. Step 3 — RF-DETR training (transformer detector, `nano…large`)

RF-DETR is a DETR-style transformer detector on a DINOv2 backbone, fine-tuned with
its built-in AdamW recipe, identical across scales:

| Setting | Value |
|---|---|
| Optimizer | AdamW (`lr=1e-4`, encoder `lr=1.5e-4`) |
| Effective batch | 16 (`batch_size=4` × `grad_accum=4`) |
| Epochs | 50, early stop `patience=10`, EMA weights |
| Backbone | DINOv2 (windowed), `num_queries=300` |
| Hardware | RTX 5060 Ti (16 GB) |

RF-DETR does not emit training-curve PNGs, so the curves below are **plotted from
each model's `metrics.csv`** (per-epoch validation mAP and loss):
""")
code(r"""
plot_rfdetr_curves(RFDETR_RUN, ["nano", "small", "medium", "base", "large"])
""")

md(r"""
**Per-model metrics (val + test), from the RF-DETR training summary:**
""")
code(r"""
show_table(RFDETR_RUN / "results_summary.csv", round_=3,
           highlight=["test_mAP50", "test_mAP50_95", "test_recall"])
""")

# --------------------------------------------------------------------------- #
# 9. §5 Unified evaluation
# --------------------------------------------------------------------------- #
md(r"""
---
## 5. Unified evaluation (fair head-to-head)

The two training scripts each used a *different* metric backend (Ultralytics vs.
supervision) at different thresholds, so their CSVs above are **not directly
comparable**. For the head-to-head, every one of the 10 checkpoints was
**re-scored through one shared pipeline** (`04_evaluation/evaluate_compare.py`):

- same test images (shared COCO test split, 30 tiles),
- same metric library (**supervision 0.28**),
- same conventions: mAP over all detections; precision/recall at **confidence ≥
  0.5**; box matches at **IoU ≥ 0.5**; all class IDs collapsed to a single `pool`
  class so YOLO's 1-class head and RF-DETR's 2-category COCO space line up.

Only the table below is valid for cross-architecture comparison (it differs from
the per-architecture summaries by design). Required metrics — **mAP50, mAP50-95,
precision, recall, parameter count** — are all included, plus a per-object-size
mAP breakdown and batch-1 latency.
""")
code(r"""
show_table(EVAL / "comparison.csv", round_=3,
           highlight=["mAP50", "mAP50_95", "recall", "precision"])
""")

code(r"""
show_grid([EVAL / "fig_map_vs_params.png", EVAL / "fig_speed_vs_accuracy.png",
           EVAL / "fig_metrics_bar.png", EVAL / "fig_size_recall.png"],
          ncols=2, titles=["accuracy vs params", "speed / accuracy",
                           "per-model metrics", "accuracy by object size"],
          figsize_per=(7, 5))
""")

# --------------------------------------------------------------------------- #
# 10. §6 Required experimental analysis
# --------------------------------------------------------------------------- #
md(r"""
---
## 6. Required experimental analysis

### Q1 — Which architecture performed best?
**RF-DETR Large**, overall: it tops the comparison on mAP50 (**0.777**) and
mAP50-95 (**0.591**) and ties for best recall (0.731). More broadly the families
sit at **different operating points**:

- **RF-DETR favours recall** — every variant recalls ~0.65–0.73 of pools vs
  ~0.50–0.58 for YOLO; it misses fewer pools.
- **YOLO favours precision** — `yolo26l`/`x` reach ~0.93 vs ~0.69–0.77 for
  RF-DETR; it emits fewer false positives.

So "best" depends on the cost of a miss vs. a false alarm: for pool *discovery*
(insurance/tax auditing, where a miss is the costly error) RF-DETR is preferable;
for low-false-alarm *targeting* lists, YOLO's precision is attractive.

### Q2 — Which model size offered the best speed/accuracy tradeoff?
**RF-DETR Nano / Small** are the sweet spot — the fastest models measured
(~12 ms/image at batch 1) *and* high detection rate (mAP50 ≈ 0.71), beating every
YOLO variant on both axes. `rfdetr_large` is the accuracy ceiling at a still-modest
~18 ms. `yolo26n` is the parameter champion (2.5 M) but only mid-pack on accuracy.
*Latency caveat: batch-1 timing includes per-call overhead and the two
architectures run at different input resolutions — treat it as relative, not a
throughput benchmark.*

### Q3 — Did RF-DETR outperform YOLO-based detectors?
**Partially — it depends on the metric.** RF-DETR wins consistently on **mAP50**
and **recall**; YOLO clearly wins on **precision**. On the strict **mAP50-95**
localisation metric the field is close (`rfdetr_large` 0.591 leads, but `yolo26l`
0.566 and `yolo26s` 0.542 beat the smaller RF-DETRs). Net: RF-DETR is the stronger
*detector* (finds more pools), YOLO the more *precise localiser per parameter*.

### Q4 — When are OBB annotations beneficial? *(out of scope for this submission)*
**OBB was not implemented here**, so this is a conceptual answer only. Oriented
boxes help most for **elongated, rotated pools** (lap / rectangular / kidney pools
photographed off-axis): a horizontal box around a diagonal rectangular pool
includes large slivers of non-pool area (lower IoU, looser localisation), which
suppresses exactly the mAP50-95 metric where all models here are weakest. OBB would
therefore be expected to improve localisation and mAP50-95 for rotated pools, while
offering little for small round/square pools that are already axis-aligned. This
remains a hypothesis — no OBB models were trained or measured.

### Q5 — Which failure cases occurred most often?
Aggregated over the test set for the best model of each architecture:

| Model | True positives | False positives | False negatives (misses) |
|---|---|---|---|
| yolo26l | 26 | 2 | 12 |
| rfdetr_large | 26 | 8 | 7 |

The dominant failure mode mirrors the precision/recall split: **YOLO's main error
is misses** (12 FN — small/shaded/occluded pools it never fires on); **RF-DETR's
main error is false positives** (8 FP — confident boxes on pool-coloured
distractors). Rendered examples in §7.

### Q6 — Which augmentations improved performance the most?
*No formal ablation was run*, so this is a qualitative read of the configured
pipeline. For small top-down aerial tiles the highest-leverage augmentations are
expected to be: **(1) Mosaic** (more pool instances + context variety per batch;
closed for the final 10 epochs to fine-tune on clean tiles), **(2) scale jitter
0.5** (robustness to pool size — the main axis of variation), **(3) HSV jitter**
(water-colour / lighting variation), **(4) horizontal flip**. Vertical flip and
rotation were left off. *To turn this into a measured ranking, run a small ablation
(mosaic on/off, scale 0.5 vs 0.9) — `train_yolo26.py` already exposes these.*
""")

# --------------------------------------------------------------------------- #
# 11. §7 Failure analysis
# --------------------------------------------------------------------------- #
md(r"""
---
## 7. Failure analysis

Overlays below use **green = true positive · orange = false positive (with
confidence) · red = missed ground truth**. Six representative cases (the worst
errors plus an "OK" contrast):
""")
code(r"""
fail = EVAL / "failures"
cases = [
    (fail / "yolo26l" / "FP0_FN2__85.PNG",       "yolo26l — 2 missed (shaded)"),
    (fail / "yolo26l" / "FP0_FN2__141.PNG",      "yolo26l — 2 missed"),
    (fail / "rfdetr_large" / "FP3_FN1__40.PNG",  "rfdetr_large — 3 false pos"),
    (fail / "rfdetr_large" / "FP1_FN0__164.PNG", "rfdetr_large — FP on no-pool tile"),
    (fail / "rfdetr_large" / "FP1_FN0__23.PNG",  "rfdetr_large — duplicate box"),
    (fail / "yolo26l" / "OK__104.PNG",           "yolo26l — OK (all correct)"),
]
show_grid([c[0] for c in cases], ncols=3, titles=[c[1] for c in cases], figsize_per=(5, 5))
""")

md(r"""
**Per-case cause → proposed fix**

| # | Case | What happens | Cause | Proposed fix |
|---|---|---|---|---|
| 1 | `yolo26l/…85` | 2 pools, 0 found | small + tree-shaded, low contrast | higher train/infer resolution (768–1024); stronger small-object aug (mosaic/copy-paste); lower conf for recall |
| 2 | `yolo26l/…141` | 2 pools, 0 found | same small-pool miss pattern (YOLO's 12 FN) | recall-oriented threshold; test-time augmentation |
| 3 | `rfdetr_large/…40` | 1 TP, 3 FP, 1 FN | fires on blue/turquoise non-pool (tarps, roofs) | hard-negative mining of blue surfaces; raise conf; colour-decorrelating aug |
| 4 | `*/…164` | FP on a pool-free tile | background resembles pool colour/texture | more hard-negative tiles in training |
| 5 | `*/…23` | 1 pool → 2 boxes | weak NMS / duplicate queries | tighter NMS IoU (YOLO); de-dup / higher threshold (RF-DETR) |

**Cross-cutting takeaways.** Small, shaded, partially-occluded pools are the
hardest positives for both architectures; blue/turquoise man-made surfaces are the
dominant false-positive trigger; `mAP_large` is noisy because the 30-image test
set has very few large pools (small-sample variance, not a real weakness).
""")

# --------------------------------------------------------------------------- #
# 12. §8 Conclusion
# --------------------------------------------------------------------------- #
md(r"""
---
## 8. Conclusion

- **Best overall:** RF-DETR Large (mAP50 0.777, mAP50-95 0.591, recall 0.731).
- **Architectural tradeoff:** RF-DETR maximises **recall** (finds more pools);
  YOLO maximises **precision** (fewer false alarms). Choose by error cost.
- **Best efficiency:** RF-DETR Nano/Small — fast *and* high mAP50; `yolo26n` is the
  smallest model if parameter count dominates.
- **Hardest cases:** small/shaded pools (misses) and blue-surface distractors
  (false positives).
- **Caveats:** the test set is only 30 images, so per-size breakdowns are
  high-variance; cross-architecture numbers are valid only from the §5 unified
  evaluation. **OBB (Step 4) was not attempted** and remains future work.
""")

# --------------------------------------------------------------------------- #
# Assemble & write
# --------------------------------------------------------------------------- #
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}

OUT.write_text(nbformat.writes(nb), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
