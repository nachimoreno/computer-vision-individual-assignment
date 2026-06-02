"""Draw YOLO-format bounding boxes onto source images and save them as files.

Reads:
  data/output/classes.txt
  data/output/labels/*.txt   (YOLO: class cx cy w h, normalized)
  images/*.PNG               (source images)

Writes:
  data/output/annotated/*.png
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import supervision as sv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--clean",
    action="store_true",
    help="Use labels_clean/ as input and write to annotated_clean/.",
)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
CLASSES_FILE = ROOT / "data" / "output" / "classes.txt"
LABELS_DIR = ROOT / "data" / "output" / ("labels_clean" if args.clean else "labels")
OUT_DIR = ROOT / "data" / "output" / ("annotated_clean" if args.clean else "annotated")

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Reading labels from {LABELS_DIR.relative_to(ROOT)}")
print(f"Writing images to  {OUT_DIR.relative_to(ROOT)}")

classes = CLASSES_FILE.read_text().strip().splitlines()

box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.INDEX)
label_annotator = sv.LabelAnnotator(
    text_scale=0.5, text_padding=4, color_lookup=sv.ColorLookup.INDEX
)


def find_image(stem: str) -> Path | None:
    for ext in (".PNG", ".png", ".jpg", ".jpeg", ".JPG"):
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def yolo_to_xyxy(rows: list[list[float]], w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    xyxy, class_ids = [], []
    for cls, cx, cy, bw, bh in rows:
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        xyxy.append([x1, y1, x2, y2])
        class_ids.append(int(cls))
    return np.array(xyxy, dtype=np.float32), np.array(class_ids, dtype=int)


written = skipped = 0
for label_path in sorted(LABELS_DIR.glob("*.txt")):
    img_path = find_image(label_path.stem)
    if img_path is None:
        print(f"skip {label_path.name}: no matching image")
        skipped += 1
        continue

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"skip {label_path.name}: cv2 could not read {img_path.name}")
        skipped += 1
        continue
    h, w = image.shape[:2]

    rows = [
        [float(x) for x in line.split()]
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        cv2.imwrite(str(OUT_DIR / f"{label_path.stem}.png"), image)
        written += 1
        continue

    xyxy, class_ids = yolo_to_xyxy(rows, w, h)
    detections = sv.Detections(xyxy=xyxy, class_id=class_ids)
    labels = [classes[c] if c < len(classes) else str(c) for c in class_ids]

    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )

    cv2.imwrite(str(OUT_DIR / f"{label_path.stem}.png"), annotated)
    written += 1

print(f"\nDone. Wrote {written} images to {OUT_DIR.relative_to(ROOT)} (skipped {skipped}).")
