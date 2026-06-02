from roboflow import Roboflow
import shutil
import os

rf = Roboflow(api_key="q1sof4bgt5O8PHIRayfC")
project = rf.workspace("ignacios-workspace-ntvi0").project("computer-vision-individual-c2kt4")
version = project.version(1)
dataset = version.download("yolov8")

# Save dataset to specified location
output_path = "data/output/roboflow_annotated"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
if os.path.exists(output_path):
    shutil.rmtree(output_path)
shutil.move(dataset.location, output_path)

