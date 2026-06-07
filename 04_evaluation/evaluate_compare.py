"""Unified, apples-to-apples evaluation of the YOLO26 and RF-DETR sweeps.

Why this script exists
----------------------
The two training scripts each evaluated with their *own* metric backend
(02 used Ultralytics' validator, 03 used supervision). Numbers produced that way
are NOT directly comparable — the mAP/precision/recall implementations and the
confidence thresholds differ. The assignment's Step 3 explicitly asks for a
*head-to-head* of "Transformer vs YOLO architectures", so this script re-runs
EVERY trained checkpoint through ONE shared pipeline:

  * same test images (the shared COCO test split, data/dataset_coco/test),
  * same metric library (supervision 0.28),
  * same conventions: mAP over all detections; precision/recall at conf >= 0.5;
    box matches at IoU >= 0.5; single "pool" class (all class ids collapsed to 0
    so YOLO's 1-class head and RF-DETR's 2-category COCO space line up).

Because the backend/threshold differ from the per-arch CSVs, the mAP values here
will not be identical to those in the training summaries — that is expected and
is the whole point: only these numbers are comparable across architectures.

Reads (auto-discovers the most recent run folder of each, or pass --yolo-run /
--rfdetr-run to pin one):
  02_yolo26_models/outputs/<run>/yolo26{n,s,m,l,x}/weights/best.pt
  03_rf_detr_models/outputs/<run>/rfdetr_{nano,small,medium,base,large}/checkpoint_best_ema.pth
  data/dataset_coco/test/{*.PNG, _annotations.coco.json}

Writes (04_evaluation/outputs/):
  comparison.csv            all models, unified metrics + params + latency
  fig_map_vs_params.png     accuracy (mAP50-95) vs model size
  fig_speed_vs_accuracy.png latency vs accuracy (speed/accuracy tradeoff)
  fig_metrics_bar.png       grouped bars: mAP50 / mAP50-95 / P / R per model
  fig_size_recall.png       mAP50-95 by object size (small/medium/large)
  failures/<model>/*.png    rendered failure cases (FN red, FP orange, TP green)
  failure_summary.csv       per-image FN/FP/TP for the best YOLO + best RF-DETR

Usage:
  python 04_evaluation/evaluate_compare.py
  python 04_evaluation/evaluate_compare.py --conf 0.5 --iou 0.5
  python 04_evaluation/evaluate_compare.py --models yolo26n rfdetr_large
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COCO_TEST = ROOT / "data" / "dataset_coco" / "test"
YOLO_OUTPUTS = ROOT / "02_yolo26_models" / "outputs"
RFDETR_OUTPUTS = ROOT / "03_rf_detr_models" / "outputs"
OUT_DIR = HERE / "outputs"

YOLO_TAGS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")
RFDETR_TAGS = ("rfdetr_nano", "rfdetr_small", "rfdetr_medium", "rfdetr_base", "rfdetr_large")

# Unified evaluation conventions (shared by both architectures).
CONF_PR = 0.5      # confidence threshold for precision/recall reporting
IOU_MATCH = 0.5    # IoU threshold for the TP/FP/FN matching in failure analysis
YOLO_IMGSZ = 640   # matches 02_yolo26_models/train_yolo26.py HYPERPARAMS["imgsz"]


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    tag: str
    arch: str               # "yolo26" | "rfdetr"
    weights: Path
    rfdetr_cls: str | None = None   # RFDETR class name, e.g. "RFDETRNano"


def _latest_run(outputs_dir: Path) -> Path | None:
    runs = sorted(p for p in outputs_dir.glob("run_*") if p.is_dir())
    return runs[-1] if runs else None


def _rfdetr_class_name(tag: str) -> str:
    # rfdetr_nano -> RFDETRNano, rfdetr_large -> RFDETRLarge
    return "RFDETR" + tag.split("_", 1)[1].capitalize()


def discover_models(yolo_run: str | None, rfdetr_run: str | None,
                    wanted: list[str] | None) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    yolo_root = (YOLO_OUTPUTS / yolo_run) if yolo_run else _latest_run(YOLO_OUTPUTS)
    if yolo_root:
        for tag in YOLO_TAGS:
            w = yolo_root / tag / "weights" / "best.pt"
            if w.exists():
                specs.append(ModelSpec(tag=tag, arch="yolo26", weights=w))

    rfdetr_root = (RFDETR_OUTPUTS / rfdetr_run) if rfdetr_run else _latest_run(RFDETR_OUTPUTS)
    if rfdetr_root:
        for tag in RFDETR_TAGS:
            w = rfdetr_root / tag / "checkpoint_best_ema.pth"
            if w.exists():
                specs.append(ModelSpec(tag=tag, arch="rfdetr", weights=w,
                                       rfdetr_cls=_rfdetr_class_name(tag)))

    if wanted:
        specs = [s for s in specs if s.tag in wanted]
    return specs


# --------------------------------------------------------------------------- #
# Shared test set (targets)
# --------------------------------------------------------------------------- #
def load_targets():
    """Return (paths, targets) from the COCO test split, class ids collapsed to 0."""
    import supervision as sv

    ds = sv.DetectionDataset.from_coco(
        images_directory_path=str(COCO_TEST),
        annotations_path=str(COCO_TEST / "_annotations.coco.json"),
    )
    paths, targets = [], []
    for path, _image, ann in ds:
        _collapse_class(ann)
        paths.append(Path(path))
        targets.append(ann)
    return paths, targets


def _collapse_class(det) -> None:
    """Force a single-class problem: every box becomes class 0 ('pool').

    YOLO's head emits class 0; the COCO export uses category 1 with a placeholder
    category 0. Collapsing both predictions and targets to 0 makes supervision's
    per-class matching architecture-agnostic.
    """
    if len(det) > 0:
        det.class_id = np.zeros(len(det), dtype=int)


# --------------------------------------------------------------------------- #
# Per-architecture prediction adapters -> sv.Detections
# --------------------------------------------------------------------------- #
def predict_yolo(weights: Path, paths: list[Path]) -> tuple[list, float, int]:
    """Run a YOLO26 checkpoint on every test image. Returns (preds, ms/img, params)."""
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO(str(weights))
    params = sum(p.numel() for p in model.model.parameters())

    # Warm-up (excluded from timing) so the first-call graph build isn't counted.
    model.predict(str(paths[0]), imgsz=YOLO_IMGSZ, conf=0.001, verbose=False)

    preds, t_total = [], 0.0
    for p in paths:
        t0 = time.perf_counter()
        res = model.predict(str(p), imgsz=YOLO_IMGSZ, conf=0.001, verbose=False)[0]
        t_total += time.perf_counter() - t0
        det = sv.Detections.from_ultralytics(res)
        _collapse_class(det)
        preds.append(det)
    return preds, 1000.0 * t_total / len(paths), params


def predict_rfdetr(cls_name: str, weights: Path, paths: list[Path]) -> tuple[list, float, int]:
    """Run an RF-DETR checkpoint on every test image. Returns (preds, ms/img, params)."""
    import rfdetr
    from PIL import Image

    model_cls = getattr(rfdetr, cls_name)
    model = model_cls(pretrain_weights=str(weights))
    params = _rfdetr_params(model)
    try:
        model.optimize_for_inference()
    except Exception:
        pass

    images = [Image.open(p).convert("RGB") for p in paths]  # RGBA -> RGB
    model.predict(images[0], threshold=0.0)  # warm-up

    preds, t_total = [], 0.0
    for img in images:
        t0 = time.perf_counter()
        det = model.predict(img, threshold=0.0)
        t_total += time.perf_counter() - t0
        _collapse_class(det)
        preds.append(det)
    return preds, 1000.0 * t_total / len(paths), params


def _rfdetr_params(model) -> int | None:
    for attr in ("model.model", "model"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return sum(p.numel() for p in obj.parameters())
        except (AttributeError, TypeError):
            continue
    return None


# --------------------------------------------------------------------------- #
# Metrics (the single shared backend)
# --------------------------------------------------------------------------- #
def compute_metrics(predictions: list, targets: list) -> dict:
    from supervision.metrics import MeanAveragePrecision, Precision, Recall

    out = {k: None for k in (
        "mAP50", "mAP50_95", "mAP_small", "mAP_medium", "mAP_large",
        "precision", "recall")}

    try:
        m = MeanAveragePrecision().update(predictions, targets).compute()
        out["mAP50"] = round(float(m.map50), 4)
        out["mAP50_95"] = round(float(m.map50_95), 4)
        # Object-size breakdown (COCO areas): small <32^2, medium, large.
        for key, sub in (("mAP_small", m.small_objects),
                         ("mAP_medium", m.medium_objects),
                         ("mAP_large", m.large_objects)):
            val = getattr(sub, "map50_95", None) if sub is not None else None
            out[key] = round(float(val), 4) if val is not None and not np.isnan(val) else None
    except Exception as exc:
        print(f"    [warn] mAP failed: {exc}")

    try:
        pr_pred = [d[d.confidence >= CONF_PR] for d in predictions]
        out["precision"] = round(float(Precision().update(pr_pred, targets).compute().precision_at_50), 4)
        out["recall"] = round(float(Recall().update(pr_pred, targets).compute().recall_at_50), 4)
    except Exception as exc:
        print(f"    [warn] precision/recall failed: {exc}")

    return out


# --------------------------------------------------------------------------- #
# Failure analysis (TP/FP/FN) + rendering
# --------------------------------------------------------------------------- #
def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thr: float):
    """Greedy IoU match -> (tp_pred_idx, fp_pred_idx, fn_gt_idx)."""
    ious = _iou_matrix(pred_xyxy, gt_xyxy)
    matched_gt, tp, fp = set(), [], []
    order = np.argsort(-ious.max(axis=1)) if len(gt_xyxy) else range(len(pred_xyxy))
    for pi in order:
        if len(gt_xyxy) == 0:
            fp.append(pi)
            continue
        gi = int(np.argmax(ious[pi]))
        if ious[pi, gi] >= iou_thr and gi not in matched_gt:
            matched_gt.add(gi)
            tp.append(pi)
        else:
            fp.append(pi)
    fn = [gi for gi in range(len(gt_xyxy)) if gi not in matched_gt]
    return tp, fp, fn


def render_failures(model_tag: str, paths, predictions, targets, summary_rows: list):
    """Render TP(green)/FP(orange)/FN(red) overlays; collect per-image counts."""
    import cv2

    out_dir = OUT_DIR / "failures" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    GREEN, ORANGE, RED = (0, 200, 0), (0, 165, 255), (0, 0, 255)  # BGR

    scored = []
    for path, pred, gt in zip(paths, predictions, targets):
        keep = pred[pred.confidence >= CONF_PR] if len(pred) else pred
        p_xyxy = keep.xyxy if len(keep) else np.empty((0, 4))
        g_xyxy = gt.xyxy if len(gt) else np.empty((0, 4))
        tp, fp, fn = match(p_xyxy, g_xyxy, IOU_MATCH)
        scored.append((path, keep, gt, tp, fp, fn))
        summary_rows.append({
            "model": model_tag, "image": path.name,
            "gt": len(g_xyxy), "tp": len(tp), "fp": len(fp), "fn": len(fn),
        })

    # Worst first: most errors (FP + FN) at the top so >=5 real failures surface.
    scored.sort(key=lambda r: len(r[4]) + len(r[5]), reverse=True)
    for path, keep, gt, tp, fp, fn in scored:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        p_xyxy = keep.xyxy if len(keep) else np.empty((0, 4))
        g_xyxy = gt.xyxy if len(gt) else np.empty((0, 4))
        for i in tp:
            _box(cv2, img, p_xyxy[i], GREEN)
        for i in fp:
            c = keep.confidence[i] if len(keep) else 0.0
            _box(cv2, img, p_xyxy[i], ORANGE, f"FP {c:.2f}")
        for i in fn:
            _box(cv2, img, g_xyxy[i], RED, "MISS")
        tag = "OK" if not fp and not fn else f"FP{len(fp)}_FN{len(fn)}"
        cv2.imwrite(str(out_dir / f"{tag}__{path.name}"), img)


def _box(cv2, img, xyxy, color, label: str | None = None):
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def color(arch):
        return "#1f77b4" if arch == "yolo26" else "#d62728"

    # 1) accuracy vs params
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(r["params_M"], r["mAP50_95"], color=color(r["arch"]), s=60)
        ax.annotate(r["model"].replace("yolo26", "y").replace("rfdetr_", "rf-"),
                    (r["params_M"], r["mAP50_95"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Parameters (M)"); ax.set_ylabel("test mAP50-95")
    ax.set_title("Accuracy vs model size")
    _arch_legend(ax, plt)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_map_vs_params.png", dpi=130); plt.close(fig)

    # 2) speed/accuracy tradeoff
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        if r["latency_ms"] is None:
            continue
        ax.scatter(r["latency_ms"], r["mAP50_95"], color=color(r["arch"]), s=60)
        ax.annotate(r["model"].replace("yolo26", "y").replace("rfdetr_", "rf-"),
                    (r["latency_ms"], r["mAP50_95"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Inference latency (ms/image)"); ax.set_ylabel("test mAP50-95")
    ax.set_title("Speed / accuracy tradeoff")
    _arch_legend(ax, plt)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_speed_vs_accuracy.png", dpi=130); plt.close(fig)

    # 3) grouped metric bars
    metrics = ["mAP50", "mAP50_95", "precision", "recall"]
    labels = [r["model"] for r in rows]
    x = np.arange(len(labels)); w = 0.2
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    for i, mkey in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, [r[mkey] or 0 for r in rows], w, label=mkey)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("score"); ax.set_title("Per-model metrics (unified backend)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_metrics_bar.png", dpi=130); plt.close(fig)

    # 4) mAP by object size
    sizes = ["mAP_small", "mAP_medium", "mAP_large"]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    for i, skey in enumerate(sizes):
        ax.bar(x + (i - 1) * w, [r[skey] or 0 for r in rows], w, label=skey.replace("mAP_", ""))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("mAP50-95"); ax.set_title("Accuracy by object size (small-pool focus)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_size_recall.png", dpi=130); plt.close(fig)


def _arch_legend(ax, plt):
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", label="YOLO26", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", label="RF-DETR", markersize=8),
    ])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-run", help="pin a YOLO run folder (default: latest)")
    ap.add_argument("--rfdetr-run", help="pin an RF-DETR run folder (default: latest)")
    ap.add_argument("--models", nargs="+", help="subset of model tags to evaluate")
    ap.add_argument("--no-failures", action="store_true", help="skip failure-case rendering")
    args = ap.parse_args()

    if not (COCO_TEST / "_annotations.coco.json").exists():
        raise SystemExit(f"{COCO_TEST} not ready — run 03_rf_detr_models/prepare_dataset.py first.")

    specs = discover_models(args.yolo_run, args.rfdetr_run, args.models)
    if not specs:
        raise SystemExit("No checkpoints found. Train the sweeps first (02 and 03).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths, targets = load_targets()
    print(f"Test images: {len(paths)} | models to evaluate: {len(specs)}")

    rows: list[dict] = []
    cached_preds: dict[str, list] = {}
    for spec in specs:
        print(f"\n>>> {spec.tag} ({spec.arch})")
        if spec.arch == "yolo26":
            preds, latency, params = predict_yolo(spec.weights, paths)
        else:
            preds, latency, params = predict_rfdetr(spec.rfdetr_cls, spec.weights, paths)
        cached_preds[spec.tag] = preds

        metrics = compute_metrics(preds, targets)
        row = {
            "model": spec.tag, "arch": spec.arch,
            "params_M": round(params / 1e6, 2) if params else None,
            "latency_ms": round(latency, 2),
            **metrics,
        }
        rows.append(row)
        print(f"    mAP50={metrics['mAP50']} mAP50-95={metrics['mAP50_95']} "
              f"P={metrics['precision']} R={metrics['recall']} "
              f"({row['params_M']}M, {row['latency_ms']}ms)")

    # Order rows: YOLO sweep first, then RF-DETR, each smallest -> largest.
    order = {t: i for i, t in enumerate(YOLO_TAGS + RFDETR_TAGS)}
    rows.sort(key=lambda r: order.get(r["model"], 99))

    # comparison.csv
    csv_path = OUT_DIR / "comparison.csv"
    fields = ["model", "arch", "params_M", "latency_ms", "mAP50", "mAP50_95",
              "mAP_small", "mAP_medium", "mAP_large", "precision", "recall"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    make_figures(rows)

    # Failure analysis on the best of each architecture (by mAP50-95).
    if not args.no_failures:
        summary_rows: list[dict] = []
        for arch in ("yolo26", "rfdetr"):
            cand = [r for r in rows if r["arch"] == arch and r["mAP50_95"] is not None]
            if not cand:
                continue
            best = max(cand, key=lambda r: r["mAP50_95"])
            print(f"\nFailure analysis: {best['model']} (best {arch})")
            render_failures(best["model"], paths, cached_preds[best["model"]], targets, summary_rows)
        if summary_rows:
            with (OUT_DIR / "failure_summary.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["model", "image", "gt", "tp", "fp", "fn"])
                w.writeheader()
                w.writerows(summary_rows)

    # Console table
    print(f"\n=== Unified comparison ({csv_path.relative_to(HERE)}) ===")
    print(" | ".join(fields))
    for r in rows:
        print(" | ".join(str(r.get(k)) for k in fields))
    print(f"\nArtifacts in {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
