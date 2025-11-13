from zenml import step
import easyocr
import cv2
import json
from pathlib import Path

@step
def run_ocr(detections_json_path: str, data_dir: str = "dataset"):
    reader = easyocr.Reader(['en'])
    test_dir = Path(data_dir) / "test" / "images"
    results_dir = Path("ocr_results")
    results_dir.mkdir(exist_ok=True, parents=True)

    with open(detections_json_path, "r") as f:
        detections = json.load(f)

    ocr_outputs = {}

    for img_name, boxes in detections.items():
        img_path = test_dir / img_name
        image = cv2.imread(str(img_path))
        img_results = []

        print(f" Running OCR on {img_name} ({len(boxes)} boxes)")

        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            roi = image[y1:y2, x1:x2]
            text_results = reader.readtext(roi)
            extracted_text = " ".join([t[1] for t in text_results])
            img_results.append({
                "region_label": box["class"],
                "confidence": box["confidence"],
                "recognized_text": extracted_text
            })
            print(f"  ↳ [{box['class']}] → {extracted_text}")

        ocr_outputs[img_name] = img_results

    output_path = results_dir / "ocr_output.json"
    with open(output_path, "w") as f:
        json.dump(ocr_outputs, f, indent=2)

    print(f"✅ OCR results saved to {output_path}")
    return str(output_path)
