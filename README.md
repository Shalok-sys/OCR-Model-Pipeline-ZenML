## 📄 Document OCR Pipeline with ZenML, YOLOv8, and EasyOCR

This repository contains a ZenML pipeline for a complete Document Optical Character Recognition (OCR) workflow. The pipeline trains a **YOLOv8** model for document field detection and then uses **EasyOCR** to extract text from the detected regions.

-----

### ⚙️ Pipeline Overview

The complete workflow is orchestrated using ZenML, enabling modularity and reproducibility. The pipeline consists of the following steps:

1.  **`ingest_data`**: Loads the dataset configuration file (`data.yaml`).
2.  **`train_detector`**: Trains an object detection model (YOLOv8n) on the dataset.
3.  **`inference_detector`**: Uses the trained model to detect fields on the test images, saving the bounding boxes and class labels to a JSON file.
4.  **`run_ocr`**: Reads the detected bounding boxes and applies EasyOCR to extract text from those specific regions of interest (ROIs) in the test images.
5.  **`evaluate_ocr_results`**: Writes the final detection and OCR results to a log file for review.

-----

### 🗃️ Files Included

| File Name | Description |
| :--- | :--- |
| `run_pipeline.py` | Defines and runs the main ZenML `document_ocr_pipeline`. This file orchestrates all steps. |
| `ingest_data.py` | ZenML step for loading the dataset configuration. |
| `train_detector.py` | ZenML step for training the YOLOv8 object detection model. |
| `inference_detector.py` | ZenML step for running inference with the trained YOLOv8 model on test images. |
| `run_ocr.py` | ZenML step that performs OCR using EasyOCR on the regions detected by the YOLOv8 model. |
| `evaluate_ocr_results.py` | ZenML step that logs the final combined detection and OCR results to a text file. |

-----

### 🚀 How to Run the Pipeline

1.  **Prerequisites:** Ensure you have ZenML, `ultralytics` (for YOLOv8), `easyocr`, `opencv-python` (`cv2`), and `PyYAML` installed.

    ```bash
    pip install zenml ultralytics easyocr opencv-python pyyaml
    ```

    *Note: EasyOCR may require additional dependencies for specific backends.*

2.  **Dataset Structure:** The pipeline expects a dataset structured for YOLO training, including a `data.yaml` file, and test images located in `dataset/test/images`.

3.  **Execution:** Run the main pipeline file from your terminal:

    ```bash
    python run_pipeline.py
    ```

-----

### 📝 Key Step Details

#### `run_ocr.py`

This step is crucial for the text extraction phase.

  * It takes the JSON output of the detector (bounding boxes, class, and confidence) and the `data_dir`.
  * It initializes an `easyocr.Reader` for English (`['en']`).
  * It iterates through each detected box in the images.
  * For each box, it crops the **Region of Interest (ROI)** from the original image using `cv2.imread` and slicing (`image[y1:y2, x1:x2]`).
  * `reader.readtext(roi)` is called to perform the OCR.
  * The final text results are combined and saved to `ocr_results/ocr_output.json`.

#### `inference_detector.py`

This step uses the trained YOLOv8 model to locate document fields.

  * It loads the `best.pt` model using `YOLO(model_path)`.
  * It runs `model.predict` on all test images.
  * It extracts the bounding box coordinates (`xyxy`), class label (`cls`), and confidence score (`conf`) for each detection.
  * The detections are formatted and saved to `inference_results/detections.json`.

#### `run_pipeline.py`

This file ties everything together, ensuring the output of one step becomes the input for the next, defining the overall **ZenML DAG (Directed Acyclic Graph)**. The pipeline is configured with `enable_cache=False` to ensure a fresh run every time, bypassing intermediate results caching.
