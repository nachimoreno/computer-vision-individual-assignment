"""Build a COCO-format dataset for RF-DETR, reusing the YOLO sweep's exact split.

Reads:
  data/dataset/split.json          (canonical 70/20/10 split made by 02's prep)
  data/images/*.{PNG,png,...}       (source aerial tiles)
  data/output/labels_clean/*.txt    (cleaned YOLO labels, single class: pool)
  data/output/classes.txt           (class names)

Writes:
  data/dataset_coco/{train,valid,test}/<images>
  data/dataset_coco/{train,valid,test}/_annotations.coco.json

Why reuse the YOLO split?
-------------------------
The assignment compares RF-DETR head-to-head with YOLO26, so both must train and
evaluate on the *identical* train/val/test membership. Rather than re-deriving a
split, we read `data/dataset/split.json` produced by 02_yolo26_models/
prepare_dataset.py. Run that first. (RF-DETR uses the folder name `valid`, so the
YOLO `val` split is written there.)

COCO conventions
----------------
- Labels are converted YOLO (normalized cx,cy,w,h) → COCO (abs x,y,w,h).
- Negative images (no pools) are included with zero annotations — RF-DETR trains
  on them as hard-negative backgrounds, matching the YOLO setup.
- Categories mirror a Roboflow COCO export (a placeholder supercategory at id 0,
  the real `pool` class at id 1), which is the input shape RF-DETR expects.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLIT_JSON = ROOT / "data" / "dataset" / "split.json"
IMAGES_DIR = ROOT / "data" / "images"
LABELS_DIR = ROOT / "data" / "output" / "labels_clean"
CLASSES_FILE = ROOT / "data" / "output" / "classes.txt"
COCO_DIR = ROOT / "data" / "dataset_coco"

IMG_EXTS = (".PNG", ".png", ".jpg", ".jpeg", ".JPG", ".JPEG")

# YOLO split name -> RF-DETR/COCO folder name.
SPLIT_FOLDER = {"train": "train", "val": "valid", "test": "test"}


def find_image(stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_yolo(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    return [
        [float(x) for x in line.split()]
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def build_coco(stems: list[str], class_name: str) -> tuple[dict, list[Path]]:
    """Return (coco_dict, image_paths) for one split."""
    categories = [
        {"id": 0, "name": f"{class_name}s", "supercategory": "none"},
        {"id": 1, "name": class_name, "supercategory": f"{class_name}s"},
    ]
    images, annotations, sources = [], [], []
    ann_id = 1
    n_pos = n_neg = 0

    for img_id, stem in enumerate(sorted(stems)):
        img_path = find_image(stem)
        if img_path is None:
            continue
        w, h = Image.open(img_path).size
        images.append(
            {"id": img_id, "file_name": img_path.name, "width": w, "height": h}
        )
        sources.append(img_path)

        rows = parse_yolo(LABELS_DIR / f"{stem}.txt")
        if rows:
            n_pos += 1
        else:
            n_neg += 1
        for _cls, cx, cy, bw, bh in rows:
            # Clamp to image bounds — some annotations spill a few px over the
            # edge; COCO evaluators expect in-bounds boxes.
            x1 = max(0.0, (cx - bw / 2) * w)
            y1 = max(0.0, (cy - bh / 2) * h)
            x2 = min(float(w), (cx + bw / 2) * w)
            y2 = min(float(h), (cy + bh / 2) * h)
            bw_px, bh_px = x2 - x1, y2 - y1
            if bw_px <= 0 or bh_px <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw_px, 2), round(bh_px, 2)],
                    "area": round(bw_px * bh_px, 2),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    coco = {
        "info": {"description": "Swimming pools — COCO export for RF-DETR"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    return coco, sources, (n_pos, n_neg)


def main() -> None:
    if not SPLIT_JSON.exists():
        raise SystemExit(
            f"{SPLIT_JSON.relative_to(ROOT)} not found.\n"
            "Run 02_yolo26_models/prepare_dataset.py first so both architectures "
            "share the exact same train/val/test split."
        )

    assignment = json.loads(SPLIT_JSON.read_text())["assignment"]
    class_name = next(c for c in CLASSES_FILE.read_text().splitlines() if c.strip())

    if COCO_DIR.exists():
        shutil.rmtree(COCO_DIR)

    for yolo_split, members in assignment.items():
        folder = SPLIT_FOLDER[yolo_split]
        out_dir = COCO_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        coco, sources, (n_pos, n_neg) = build_coco(members, class_name)
        for src in sources:
            shutil.copy2(src, out_dir / src.name)
        (out_dir / "_annotations.coco.json").write_text(json.dumps(coco))

        print(
            f"{folder:5s}: {len(coco['images']):3d} images "
            f"({n_pos} pos / {n_neg} neg), {len(coco['annotations'])} boxes"
        )

    print(f"\nCOCO dataset ready at {COCO_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
