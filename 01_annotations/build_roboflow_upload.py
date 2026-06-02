"""Bundle images + cleaned YOLO labels into a zip ready for Roboflow upload.

Builds:
  data/output/roboflow_upload/
    data.yaml
    images/*.PNG    (all 288 — including ones with no detections)
    labels/*.txt    (cleaned labels; missing ones = unannotated in Roboflow)

Then zips it as data/output/roboflow_upload.zip
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
LABELS_DIR = ROOT / "data" / "output" / "labels_clean"
BUILD_DIR = ROOT / "data" / "output" / "roboflow_upload"
ZIP_PATH = ROOT / "data" / "output" / "roboflow_upload"

if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
(BUILD_DIR / "images").mkdir(parents=True)
(BUILD_DIR / "labels").mkdir(parents=True)

n_images = 0
for img in IMAGES_DIR.iterdir():
    if img.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        shutil.copy2(img, BUILD_DIR / "images" / img.name)
        n_images += 1

n_labels = 0
for lbl in LABELS_DIR.glob("*.txt"):
    if lbl.read_text().strip():
        shutil.copy2(lbl, BUILD_DIR / "labels" / lbl.name)
        n_labels += 1

(BUILD_DIR / "data.yaml").write_text(
    "names:\n  - pool\nnc: 1\n"
)

shutil.make_archive(str(ZIP_PATH), "zip", BUILD_DIR)

print(f"Images bundled : {n_images}")
print(f"Labels bundled : {n_labels}  (empty/missing → unannotated in Roboflow)")
print(f"Zip written    : {ZIP_PATH.with_suffix('.zip').relative_to(ROOT)}")
