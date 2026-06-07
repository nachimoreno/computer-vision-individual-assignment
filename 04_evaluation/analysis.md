# Evaluation & Experimental Analysis — Swimming-Pool Detection

**Task:** detect swimming pools in aerial imagery, comparing YOLO26 (`n/s/m/l/x`)
against RF-DETR (`nano/small/medium/base/large`).
**Hardware:** NVIDIA GeForce RTX 5060 Ti (16 GB, Blackwell), CUDA build of PyTorch.
**Test split:** 30 held-out images from the shared 70/20/10 split (`data/dataset_coco/test`).

> All numbers below come from a **single unified evaluation** ([evaluate_compare.py](evaluate_compare.py)):
> every checkpoint is re-scored on the *same* test images, with the *same* metric
> backend (supervision 0.28) and the *same* conventions — mAP over all detections,
> precision/recall at confidence ≥ 0.5, box matches at IoU ≥ 0.5, all class IDs
> collapsed to a single `pool` class. This is why the values here differ from the
> per-architecture training summaries (which each used a different evaluator at a
> different threshold): only these numbers are valid for a head-to-head comparison.

---

## 1. Results

| Model | Params (M) | Latency (ms/img) | mAP50 | mAP50-95 | mAP small | mAP medium | mAP large | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 2.50 | 20.5 | 0.682 | 0.537 | 0.443 | 0.633 | 0.152 | 0.833 | 0.577 |
| yolo26s | 9.95 | 17.5 | 0.640 | 0.542 | 0.455 | 0.661 | 0.006 | 0.824 | 0.539 |
| yolo26m | 21.77 | 16.4 | 0.670 | 0.489 | 0.309 | 0.612 | 0.033 | 0.875 | 0.539 |
| yolo26l | 26.18 | 16.2 | 0.648 | 0.566 | 0.495 | 0.680 | 0.069 | **0.933** | 0.539 |
| yolo26x | 58.81 | 24.9 | 0.679 | 0.525 | 0.502 | 0.620 | 0.016 | 0.929 | 0.500 |
| rfdetr_nano | 30.15 | 12.2 | 0.712 | 0.436 | 0.378 | 0.516 | 0.055 | 0.731 | 0.731 |
| rfdetr_small | 31.79 | **12.0** | 0.707 | 0.459 | **0.564** | 0.506 | 0.027 | 0.692 | 0.692 |
| rfdetr_medium | 33.37 | 13.3 | 0.680 | 0.499 | 0.313 | 0.579 | 0.212 | 0.773 | 0.654 |
| rfdetr_base | 31.86 | 13.4 | 0.730 | 0.531 | 0.373 | 0.658 | 0.075 | 0.760 | 0.731 |
| **rfdetr_large** | 33.62 | 18.3 | **0.777** | **0.591** | 0.537 | 0.679 | 0.108 | 0.731 | 0.731 |

![Accuracy vs model size](outputs/fig_map_vs_params.png)
![Speed / accuracy tradeoff](outputs/fig_speed_vs_accuracy.png)
![Per-model metrics](outputs/fig_metrics_bar.png)
![Accuracy by object size](outputs/fig_size_recall.png)

**Reported training configuration (YOLO26).** Optimizer AdamW, `lr0=0.001` with a
cosine scheduler (`lrf=0.01`, 3 warm-up epochs), weight decay `5e-4`, image size
640, up to 100 epochs with early stopping (patience 20), batch normalised to
nbs=64. Augmentations: mosaic (closed for the final 10 epochs), HSV jitter
(h=0.015, s=0.7, v=0.4), horizontal flip (p=0.5), translate 0.1, scale 0.5; no
vertical flip or rotation. RF-DETR used its built-in AdamW recipe (`lr=1e-4`,
encoder LR `1.5e-4`, effective batch 16 via gradient accumulation, early stopping
patience 10).

---

## 2. Required experimental analysis

### Q1 — Which architecture performed best?
**RF-DETR Large**, overall. It tops the table on the two headline metrics —
mAP50 **0.777** and mAP50-95 **0.591** — and ties for the best recall (0.731).
More broadly the two families have *different operating points*:

- **RF-DETR favours recall.** Every RF-DETR variant recalls ~0.65–0.73 of the
  pools, versus ~0.50–0.58 for YOLO. RF-DETR misses fewer pools.
- **YOLO favours precision.** yolo26l/x reach precision 0.93, versus ~0.69–0.77
  for RF-DETR. YOLO emits fewer false positives.

So "best" depends on the cost of a miss vs. a false alarm. For pool *discovery*
(insurance/tax auditing, where a missed pool is the expensive error) RF-DETR is
preferable; for low-false-alarm *targeting* lists, YOLO's precision is attractive.

### Q2 — Which model size offered the best speed/accuracy tradeoff?
See [fig_speed_vs_accuracy.png](outputs/fig_speed_vs_accuracy.png).

- **RF-DETR Nano / Small** are the sweet spot: the **fastest** models measured
  (~12 ms/image) *and* high detection rate (mAP50 ≈ 0.71). They beat every YOLO
  variant on both axes simultaneously.
- **rfdetr_large** is the accuracy ceiling at a still-modest 18 ms.
- **yolo26n** is the parameter-efficiency champion (2.5 M params) but sits only
  mid-pack on accuracy and, oddly, was not the fastest — at this tiny resolution
  and batch=1 the latencies are dominated by fixed per-call overhead, so the
  ms/image figures compress and don't track parameter count cleanly.

> ⚠️ Latency caveat: measured at batch 1 on 30 images, including Python/per-call
> overhead. Treat it as a relative indicator, not a throughput benchmark; YOLO
> and RF-DETR also run at different input resolutions.

### Q3 — Did RF-DETR outperform YOLO-based detectors?
**Partially — it depends on the metric.** On mAP50 and recall, RF-DETR wins
consistently (every RF-DETR variant ≥ 0.68 mAP50; the best YOLO is 0.682). On
**precision** YOLO clearly wins. On **mAP50-95**, the strict localisation metric,
the field is close: rfdetr_large (0.591) leads, but yolo26l (0.566) and yolo26s
(0.542) are competitive and beat the smaller RF-DETRs. Net: RF-DETR is the
stronger *detector* (finds more pools), YOLO the more *precise localiser per
parameter*.

### Q4 — When are OBB annotations beneficial?
**Step 4 (OBB) has not yet been trained** — there are no OBB checkpoints in the
repo, so this answer is analytical and will be backed with numbers once
`YOLO26-OBB`/`YOLO11-OBB` are run. Expected benefit: oriented boxes help most for
**elongated, rotated pools** (lap pools, kidney/rectangular pools photographed
off-axis). A horizontal box around a diagonal rectangular pool includes large
slivers of non-pool area (lower IoU, looser localisation), which directly
suppresses mAP50-95 — exactly the metric where all models here are weakest.
OBB should therefore lift mAP50-95 and localisation quality for rotated pools,
while offering little for small round/square pools that are already
axis-aligned. (To be confirmed empirically in Step 4.)

### Q5 — Which failure cases occurred most often?
Aggregated over the test set for the two best models:

| Model | True positives | False positives | False negatives (misses) |
|---|---|---|---|
| yolo26l | 26 | 2 | 12 |
| rfdetr_large | 26 | 8 | 7 |

The dominant failure mode differs by architecture, mirroring the precision/recall
split: **YOLO's main error is misses** (12 FN — small, shaded, or partially
occluded pools it never fires on), while **RF-DETR's main error is false
positives** (8 FP — confident boxes on pool-coloured distractors). See §3.

### Q6 — Which augmentations improved performance the most?
No formal augmentation ablation was run, so this is a qualitative read of the
configured pipeline rather than a measured ranking. Given the dataset (small
~350 px aerial tiles, top-down view, pools small in-frame), the augmentations
expected to matter most are:

1. **Mosaic** — multiplies effective pool instances per batch and forces context
   variety; the standard single biggest contributor for small-object YOLO. It is
   closed for the final 10 epochs so the model fine-tunes on clean tiles.
2. **Scale jitter (0.5)** — directly improves robustness to pool size, the main
   axis of variation here, and supports small-pool recall.
3. **HSV jitter** — handles water-colour and lighting variation (turquoise vs.
   dark/green water, sun glare).
4. **Horizontal flip** — cheap, safe doubling of orientation variety.

Vertical flip and rotation were left off in the horizontal-box setup; rotation in
particular is better deferred to the OBB experiment. *Recommendation:* run a
small ablation (mosaic on/off, scale 0.5 vs 0.9) to turn this into a measured
answer — the harness (`train_yolo26.py`) already exposes these as overrides.

---

## 3. Failure analysis

Rendered overlays live in [outputs/failures/](outputs/failures/) — **green = true
positive, orange = false positive (with confidence), red = missed ground truth**.
Per-image counts are in [outputs/failure_summary.csv](outputs/failure_summary.csv).
Five representative cases:

1. **Missed small/shaded pools — `yolo26l/FP0_FN2__85.PNG` (2 GT, 0 found).**
   Both pools are small and partly tree-shaded; YOLO produces no detection at all.
   *Cause:* low contrast + small object scale. *Fix:* train/infer at higher
   resolution (768–1024), increase small-object sampling via stronger mosaic/copy-
   paste, or lower the confidence threshold for recall-critical use.

2. **Total miss on a 2-pool tile — `yolo26l/FP0_FN2__141.PNG` (2 GT, 0 found).**
   Same pattern as above on a different tile, confirming misses are YOLO's
   systematic weakness here (12 FN total). *Fix:* recall-oriented threshold tuning
   and test-time augmentation.

3. **False positives on pool-coloured distractors — `rfdetr_large/FP3_FN1__40.PNG`
   (2 GT; 1 TP, 3 FP, 1 FN).** RF-DETR fires confident boxes on blue/turquoise
   non-pool features (tarps, shaded roofs, water tanks). *Cause:* colour-driven
   over-triggering, the price of RF-DETR's higher recall. *Fix:* hard-negative
   mining of blue rooftops/tarps, raise the confidence threshold, or add
   colour-decorrelating augmentation.

4. **False positive on a pool-free image — `*/FP1_FN0__164.PNG` (0 GT, 1 FP).**
   *Both* best models hallucinate a single pool on a negative tile. *Cause:*
   background object resembling pool texture/colour. *Fix:* include more
   hard-negative (pool-free) tiles in training — the dataset already carries
   negatives, so increasing their share should help.

5. **Duplicate / split detection — `*/FP1_FN0__23.PNG` (1 GT; 1 TP, 1 FP).**
   One pool yields two boxes (a correct match plus an overlapping extra). *Cause:*
   weak NMS / duplicate queries on a large or irregular pool. *Fix:* tighter NMS
   IoU for YOLO; for RF-DETR, post-hoc de-duplication or higher threshold.

**Cross-cutting takeaways**
- Small, shaded, and partially occluded pools are the hardest positives across
  both architectures (low `mAP_small` for several models).
- Blue/turquoise man-made surfaces are the dominant false-positive trigger.
- `mAP_large` is noisy and low for everyone because the 30-image test set
  contains very few large pools — small-sample variance, not a real weakness.

---

## 4. Caveats & reproduction

- **Small test set (30 images):** all metrics, especially the object-size
  breakdowns, are high-variance. Rankings are indicative, not definitive — a
  larger test set (or k-fold) would firm them up.
- **Unified vs. training-time metrics:** numbers here intentionally differ from
  the per-architecture training CSVs; only these are cross-comparable (see header).
- **Reproduce:**
  ```bash
  python 04_evaluation/evaluate_compare.py                 # full re-eval + figures + failures
  python 04_evaluation/evaluate_compare.py --no-failures   # metrics/figures only
  python 04_evaluation/evaluate_compare.py --models rfdetr_large yolo26l
  ```
