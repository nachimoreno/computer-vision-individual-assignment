# Swimming-Pool Detection in Aerial Imagery

Individual assignment for the IE Computer Vision course. The goal is to detect
**swimming pools** in aerial/satellite tiles and compare several modern object
detectors on the task. The pipeline goes end-to-end: auto-annotate raw tiles →
clean the labels → build a reproducible train/val/test split → train and
benchmark a family of models.

> **Single class:** every model detects one class — `pool`.

---

## What has been done so far

| Stage | Status | Artifacts |
|-------|--------|-----------|
| **1. Annotation** — zero-shot auto-labelling with Grounding DINO, then rule-based cleanup, then a manual review pass via Roboflow | ✅ Done | `data/output/labels/`, `data/output/labels_clean/`, `data/output/roboflow_annotated/` |
| **2. Dataset prep** — deterministic, stratified 70/20/10 split with negatives as hard background | ✅ Done | `data/dataset/` (`images/`, `labels/`, `data.yaml`, `split.json`) |
| **3. YOLO26 sweep** — train n/s/m scales with identical hyperparameters, resume-aware orchestration | ⚙️ Scripted, not yet run (`02_yolo26_models/outputs/` is empty) | `train_yolo26.py` |
| **4. RF-DETR benchmark** — fine-tune RF-DETR on the same dataset | ⚙️ Notebook prepared | `03_rf_detr_models/…ipynb` |
| **5. YOLO-OBB** — oriented bounding boxes | ⚙️ Notebook prepared | `02_yolo26_models/train-yolov8-obb.ipynb` |

### Dataset as it currently stands

- **288** source aerial tiles in `data/images/` (mixed `.PNG` / `.png`, ~350 px).
- **162 positives** (contain ≥1 pool) / **126 negatives** (pool-free background).
  90 of the negatives have no label file at all and are treated as empty
  hard-negative backgrounds to suppress false positives.
- Stratified split (seed 0), preserving the positive/negative ratio:

  | Split | Total | Positives | Negatives |
  |-------|------:|----------:|----------:|
  | train | 201   | 113       | 88        |
  | val   | 57    | 32        | 25        |
  | test  | 30    | 17        | 13        |

---

## Repository layout

```
.
├── data/
│   ├── images/                    # 288 source aerial tiles (committed input)
│   ├── output/
│   │   ├── classes.txt            # class list — just "pool"
│   │   ├── labels/                # raw Grounding DINO YOLO labels
│   │   ├── labels_clean/          # cleaned labels (area filter + NMS)
│   │   ├── annotated/             # raw labels drawn on images (QA)
│   │   ├── annotated_clean/       # cleaned labels drawn on images (QA)
│   │   ├── roboflow_upload/       # bundle pushed to Roboflow for review
│   │   └── roboflow_annotated/    # reviewed labels pulled back from Roboflow
│   └── dataset/                   # prepared YOLO split (git-ignored, regenerated)
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       ├── data.yaml              # absolute path for THIS machine
│       └── split.json             # record of which image went where
│
├── 01_annotations/                # Step 1 — annotation + cleanup
│   ├── build_annotations.ipynb        # Grounding DINO zero-shot labelling
│   ├── clean_annotations.py           # drop full-image boxes + NMS dedupe
│   ├── output_annotations.py          # draw boxes onto images for QA
│   ├── build_roboflow_upload.py       # bundle images+labels → zip for Roboflow
│   └── download_finished_annotations.py  # pull reviewed labels back
│
├── 02_yolo26_models/              # Steps 2 & 4 — YOLO training
│   ├── prepare_dataset.py             # build the stratified train/val/test split
│   ├── train_yolo26.py                # n/s/m scaling sweep, resume-aware
│   ├── train-yolo26-object-detection-on-custom-dataset.ipynb
│   ├── train-yolov8-obb.ipynb         # oriented bounding boxes (Step 4)
│   └── outputs/                       # per-run training folders (git-ignored)
│
├── 03_rf_detr_models/             # Step 3 — RF-DETR benchmark
│   └── how-to-finetune-rf-detr-on-detection-dataset.ipynb
│
├── AGENTS.md                      # guidance for agents/humans (env, conventions)
├── requirements.txt               # Python deps (PyTorch installed separately)
└── MBD_CS_IndividualAssignment2026.pdf  # the assignment brief
```

---

## Setup

All Python and shell commands **must run inside the `computer_vision` conda
environment** — the base system Python lacks the CV dependencies.

```bash
conda activate computer_vision
# or for one-off commands:  conda run -n computer_vision python <script>.py
```

### Install dependencies

Install **PyTorch first**, with the build that matches your machine, then the rest:

```bash
# RTX 5060 Ti (Blackwell, sm_120) — needs CUDA 12.8+ wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Intel Mac (CPU-only, dev / smoke-tests):
pip install "torch==2.2.2" "torchvision==0.17.2" "numpy<2"

# Everything else (index-agnostic):
pip install -r requirements.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **Hardware note:** the training desktop has an RTX 5060 Ti. Intel MacBooks are
> CPU-only (no CUDA, no MPS) and are for development / smoke-tests only — a full
> sweep there is impractically slow.

---

## How to operate it

The annotation stage (Step 1) is already complete — its outputs are committed, so
you normally start at **dataset prep**. The commands below assume the
`computer_vision` env is active.

### 1. (Optional) Re-run the annotation pipeline

Only needed if you change the source images or the labelling thresholds.

```bash
# a) Auto-label with Grounding DINO  → data/output/labels/
jupyter notebook 01_annotations/build_annotations.ipynb

# b) Clean: drop boxes > 70% of the image, NMS-dedupe at IoU 0.5
python 01_annotations/clean_annotations.py            # → labels_clean/

# c) Draw boxes for visual QA
python 01_annotations/output_annotations.py           # raw  → annotated/
python 01_annotations/output_annotations.py --clean   # clean → annotated_clean/

# d) Bundle for Roboflow review, then pull the reviewed labels back
python 01_annotations/build_roboflow_upload.py        # → roboflow_upload.zip
python 01_annotations/download_finished_annotations.py
```

### 2. Build the dataset split

Regenerates `data/dataset/` from `data/images/` + `data/output/labels_clean/`.
Deterministic (seed 0) and stratified, so it reproduces identically across
machines. **Re-run this on every new machine** — `data.yaml` is written with an
absolute path and the folder is git-ignored.

```bash
python 02_yolo26_models/prepare_dataset.py
```

### 3. Train the YOLO26 scaling sweep

Trains the n / s / m scales with **identical hyperparameters** (100 epochs,
imgsz 640, AdamW, cosine LR, mosaic) so the comparison is fair. Each fresh
invocation creates a timestamped, self-contained run folder under `outputs/`.

```bash
# Full n/s/m sweep → new outputs/run_<timestamp>/ folder
python 02_yolo26_models/train_yolo26.py

# Useful variations:
python 02_yolo26_models/train_yolo26.py --models n           # single scale
python 02_yolo26_models/train_yolo26.py --models n s         # subset
python 02_yolo26_models/train_yolo26.py --epochs 50 --batch 8
python 02_yolo26_models/train_yolo26.py --resume latest      # continue last run
python 02_yolo26_models/train_yolo26.py --resume run_20260602-144300
```

**Resume / checkpointing.** Two levels protect a long sweep:
- *Ultralytics* writes `last.pt` + `best.pt` every epoch — a killed run resumes
  mid-model with optimizer/epoch/LR state intact.
- *This script* tracks each model `pending → in_progress → done` in the run's
  `progress.json`. `--resume` skips finished models, resumes the in-progress one,
  and starts the rest fresh.

**Outputs per run** (`02_yolo26_models/outputs/run_<timestamp>/`):
- `yolo26<n|s|m>/` — Ultralytics dirs with weights, plots, eval results
- `progress.json` — sweep state for that run
- `results_summary.csv` — params + val/test metrics (mAP50, mAP50-95,
  precision, recall, inference ms) per model

> Generated artifacts (`outputs/`, `data/dataset/`) are git-ignored and
> regenerated by the scripts. Checkpoints (`*.pt`) aren't committed to plain git
> — sync them out-of-band or force-add a completed `best.pt` when needed.

### 4. RF-DETR benchmark and YOLO-OBB

```bash
jupyter notebook 03_rf_detr_models/how-to-finetune-rf-detr-on-detection-dataset.ipynb
jupyter notebook 02_yolo26_models/train-yolov8-obb.ipynb
```

Both train on the same prepared dataset so their metrics line up against the
YOLO26 sweep.

---

## Conventions

- Scripts resolve the repo root via `Path(__file__).resolve().parent.parent` and
  use **relative paths**, so they work on both the Mac dev box and the RTX
  desktop.
- Every script's module docstring lists its `Reads:` / `Writes:` paths.
- All hyperparameters the assignment requires us to report are set **explicitly**
  in `train_yolo26.py` (not left to Ultralytics "auto"), so the write-up can
  quote them directly.

See `AGENTS.md` for the full environment and cross-machine notes.
