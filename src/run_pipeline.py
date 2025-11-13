from zenml import pipeline

# Import all steps
from steps.ingest_data import ingest_data
from steps.train_detector import train_detector
from steps.inference_detector import inference_detector
from steps.run_ocr import run_ocr
from steps.evaluate_ocr_results import evaluate_ocr_results

@pipeline(enable_cache=False)
def document_ocr_pipeline(data_dir: str = "dataset"):
    dataset_path = ingest_data(data_dir)
    model_path = train_detector(dataset_path)
    detections_path = inference_detector(model_path, data_dir)
    ocr_results_path = run_ocr(detections_path, data_dir)
    evaluation_log = evaluate_ocr_results(ocr_results_path)
    return evaluation_log

if __name__ == "__main__":
    run = document_ocr_pipeline()
