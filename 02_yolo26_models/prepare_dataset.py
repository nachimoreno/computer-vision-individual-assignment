"""Assemble a YOLO detection dataset with a deterministic, stratified 70/20/10 split.

Reads:
  data/images/*.{PNG,png,jpg,...}   (288 source aerial tiles)
  data/output/labels_clean/*.txt    (cleaned YOLO labels, single class: pool)
  data/output/classes.txt           (class names)

Writes:
  data/dataset/images/{train,val,test}/*
  data/dataset/labels/{train,val,test}/*.txt
  data/dataset/data.yaml
  data/dataset/split.json   (record of which image went where)

Design notes
------------
- Images with NO clean label file are treated as NEGATIVES (pool-free
  background) and get an empty label file. This was an explicit project
  decision: ~90 of 288 images have no label and act as hard negatives to
  suppress false positives.
- The split is STRATIFIED on has-pool vs negative so train/val/test keep the
  same positive/negative ratio, and DETERMINISTIC (sorted stems + fixed seed)
  so re-running on a different machine reproduces the exact same split.
- `data.yaml` is written with an ABSOLUTE `path` for *this* machine. The
  dataset/ folder is git-ignored and regenerated per machine, so the absolute
  path is always correct locally while the script/code stay portable.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
LABELS_DIR = ROOT / "data" / "output" / "labels_clean"
CLASSES_FILE = ROOT / "data" / "output" / "classes.txt"
# Dataset lives under data/ to keep a clean split between source code and data.
DATASET_DIR = ROOT / "data" / "dataset"

IMG_EXTS = (".PNG", ".png", ".jpg", ".jpeg", ".JPG", ".JPEG")

SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 0


def find_image(stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def has_pool(label_path: Path | None) -> bool:
    """True if the image has at least one box (i.e. a positive, not a negative)."""
    if label_path is None or not label_path.exists():
        return False
    return any(line.strip() for line in label_path.read_text().splitlines())


def stratified_split(stems: list[str], positive: set[str]) -> dict[str, list[str]]:
    """Split each stratum (pos / neg) by the same ratios, then merge.

    Splitting per-stratum (rather than globally) guarantees the positive/negative
    ratio is preserved in every split even for small counts.
    """
    rng = random.Random(SEED)
    assignment: dict[str, list[str]] = {s: [] for s in SPLITS}

    for stratum in (sorted(positive), sorted(set(stems) - positive)):
        items = list(stratum)
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * SPLITS["train"])
        n_val = round(n * SPLITS["val"])
        assignment["train"] += items[:n_train]
        assignment["val"] += items[n_train : n_train + n_val]
        assignment["test"] += items[n_train + n_val :]

    return assignment


def main() -> None:
    # Build the universe of samples: every image, paired with its label (or None).
    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir() if p.suffix in IMG_EXTS and p.is_file()
    )
    if not image_paths:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    stems = [p.stem for p in image_paths]
    label_of = {s: (LABELS_DIR / f"{s}.txt") for s in stems}
    positive = {s for s in stems if has_pool(label_of[s])}
    negative = set(stems) - positive

    print(f"Images           : {len(stems)}")
    print(f"  positives (pool): {len(positive)}")
    print(f"  negatives (bg)  : {len(negative)}")
    missing_label = {s for s in stems if not label_of[s].exists()}
    print(f"  of which have no label file (→ empty negative): {len(missing_label)}")

    assignment = stratified_split(stems, positive)

    # Fresh dataset tree.
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in SPLITS:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_record: dict[str, dict] = {}
    for split, members in assignment.items():
        n_pos = n_neg = 0
        for stem in members:
            img = find_image(stem)
            if img is None:  # should not happen — stems come from images
                continue
            shutil.copy2(img, DATASET_DIR / "images" / split / img.name)

            # Copy the label if present, else write an empty file (negative).
            dst_label = DATASET_DIR / "labels" / split / f"{stem}.txt"
            src_label = label_of[stem]
            if src_label.exists():
                dst_label.write_text(src_label.read_text())
            else:
                dst_label.write_text("")

            if stem in positive:
                n_pos += 1
            else:
                n_neg += 1
        split_record[split] = {"total": len(members), "pos": n_pos, "neg": n_neg}
        print(f"{split:5s}: {len(members):3d} images  ({n_pos} pos / {n_neg} neg)")

    # classes.txt → names list.
    names = [c for c in CLASSES_FILE.read_text().splitlines() if c.strip()]

    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        "# Auto-generated by prepare_dataset.py — do not edit by hand.\n"
        "# `path` is absolute for THIS machine; the dataset is regenerated per machine.\n"
        f"path: {DATASET_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )

    (DATASET_DIR / "split.json").write_text(
        json.dumps(
            {"seed": SEED, "ratios": SPLITS, "counts": split_record, "assignment": assignment},
            indent=2,
        )
    )

    print(f"\nDataset ready at {DATASET_DIR.relative_to(ROOT)}")
    print(f"data.yaml      : {data_yaml.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
