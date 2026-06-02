"""Clean Grounding DINO auto-annotations: drop full-image boxes + NMS duplicates.

Reads:
  data/output/labels/*.txt   (YOLO: class cx cy w h, normalized)

Writes:
  data/output/labels_clean/*.txt

Prints a per-file report of what was removed and why.
"""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data" / "output" / "labels"
OUT_DIR = ROOT / "data" / "output" / "labels_clean"

MAX_AREA_FRAC = 0.70   # drop boxes covering more than 70% of the image
NMS_IOU = 0.50         # merge overlapping boxes above this IoU

OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_yolo(path: Path) -> np.ndarray:
    rows = [
        [float(x) for x in line.split()]
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def nms(xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Standard NMS. `scores` ranks which box to keep when two overlap."""
    if len(xyxy) == 0:
        return []
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thresh]
    return keep


total_in = total_out = dropped_area = dropped_nms = files_emptied = 0

for label_path in sorted(LABELS_DIR.glob("*.txt")):
    rows = parse_yolo(label_path)
    n_in = len(rows)
    total_in += n_in

    if n_in == 0:
        (OUT_DIR / label_path.name).write_text("")
        continue

    areas = rows[:, 3] * rows[:, 4]
    area_mask = areas <= MAX_AREA_FRAC
    dropped_area += int((~area_mask).sum())
    rows = rows[area_mask]

    if len(rows) == 0:
        files_emptied += 1
        (OUT_DIR / label_path.name).write_text("")
        continue

    # Rank smaller boxes higher — pools are small in aerial tiles, so when two
    # boxes overlap the smaller one is usually the better fit.
    xyxy = to_xyxy(rows)
    scores = -(rows[:, 3] * rows[:, 4])
    keep = nms(xyxy, scores, NMS_IOU)
    dropped_nms += len(rows) - len(keep)
    rows = rows[keep]

    total_out += len(rows)
    lines = [
        f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}"
        for r in rows
    ]
    (OUT_DIR / label_path.name).write_text("\n".join(lines) + "\n")

print(f"Files processed : {len(list(LABELS_DIR.glob('*.txt')))}")
print(f"Boxes in        : {total_in}")
print(f"Boxes out       : {total_out}")
print(f"  dropped (area > {MAX_AREA_FRAC:.0%}) : {dropped_area}")
print(f"  dropped (NMS @ IoU {NMS_IOU})       : {dropped_nms}")
print(f"Files now empty : {files_emptied}")
print(f"\nClean labels written to {OUT_DIR.relative_to(ROOT)}")
