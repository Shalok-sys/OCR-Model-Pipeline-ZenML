from zenml import step
import json
from pathlib import Path

@step
def evaluate_ocr_results(ocr_json_path: str):
    with open(ocr_json_path, "r") as f:
        ocr_results = json.load(f)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / "final_results.txt"

    with open(log_file, "w") as log:
        log.write(" OCR Pipeline Results\n")
        log.write("=" * 50 + "\n\n")

        for img_name, fields in ocr_results.items():
            log.write(f" Image: {img_name}\n")
            for field in fields:
                label = field["region_label"]
                conf = field["confidence"]
                text = field["recognized_text"]
                log.write(f"  [{label}] ({conf}): {text}\n")
            log.write("-" * 50 + "\n")

    print(f"✅ Final evaluation results saved to {log_file}")
    return str(log_file)
