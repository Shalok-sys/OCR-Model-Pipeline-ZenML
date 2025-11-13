from zenml import step
from ultralytics import YOLO
from pathlib import Path
import cv2
import json

@step
def inference_detector(model_path: str, data_dir: str = "dataset"):
    test_dir = Path(data_dir) / "test" / "images"
    results_dir = Path("inference_results")
    results_dir.mkdir(exist_ok=True, parents=True)

    model = YOLO(model_path)
    print(f" Loaded model from {model_path}")

    detections_summary = {}

    for img_path in test_dir.glob("*.*"):
        results = model.predict(source=str(img_path), conf=0.4, save=False, verbose=False)
        boxes = results[0].boxes
        image = cv2.imread(str(img_path))

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class": model.names[cls],
                "confidence": round(conf, 3)
            })

        detections_summary[img_path.name] = detections
        print(f" {img_path.name}: {len(detections)} regions detected.")

    json_path = results_dir / "detections.json"
    with open(json_path, "w") as f:
        json.dump(detections_summary, f, indent=2)

    print(f"✅ Detection results saved to {json_path}")
    return str(json_path)
