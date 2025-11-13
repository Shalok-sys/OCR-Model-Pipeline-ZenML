from zenml import step
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

@step
def train_detector(data_yaml_path: str, epochs: int = 50):
    model = YOLO("yolov8n.pt")  # pretrained YOLOv8n
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        plots=False,   # do not generate plots
        save=True,     # save only final weights
        save_period=0  # avoid intermediate checkpoints
    )

    train_dir = Path("runs/detect/train")
    weights_dir = train_dir / "weights"
    final_model_path = weights_dir / "best.pt"
    results_csv_path = train_dir / "results.csv"

    return str(final_model_path)
