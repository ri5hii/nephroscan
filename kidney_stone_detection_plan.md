# Kidney Stone Detection Software Plan

**User Request:**\
Create a kidney stone detection software that: - Takes CT scans and
X-ray images (any type) - Preprocesses them - Runs an AI model to detect
stones - Classifies into *normal* and *stone* category - Draws tight
bounding boxes around stones - Outputs metadata JSON with serial number,
bounding box color, coordinates, size, class, etc. - Provides an .exe
with a guide where you can train/test a dataset and then use random CT
scans/X-rays to generate medical-grade reports.

------------------------------------------------------------------------

## Recommended Tech Stack & Approach

### Frontend / Desktop UI

-   **Framework**: PyQt6 or Electron + Python backend (Flask/FastAPI).\
    *PyQt6 is simpler for a single .exe and integrates well with Python
    ML code.*
-   Features:
    -   File browser for DICOM/PNG upload
    -   "Train Model" tab: select dataset path, set hyper-parameters
    -   "Inference" tab: preview bounding boxes, export JSON/PDF

### Backend / ML

-   **Language**: Python 3.11+
-   **Deep Learning Framework**: PyTorch
-   **Computer Vision**: `torchvision`, `OpenCV`, `pydicom`,
    `albumentations` (for augmentations), USE YOLOV8

### Packaging

-   PyInstaller or cx_Freeze to create a Windows .exe that bundles model
    weights and dependencies.

------------------------------------------------------------------------

## Model Architecture

Kidney stones are small, high-contrast objects → object detection +
classification.

  -----------------------------------------------------------------------
  Option                        Why                Notes
  ----------------------------- ------------------ ----------------------
  YOLOv8/YOLOv9 (Ultralytics)   State-of-the-art   
                                detection, easy    
                                export, good for   
                                bounding boxes &   
                                class.             

  Faster R-CNN                  High accuracy but  
                                slower and         
                                heavier.           

  MONAI + nnU-Net               Good for full 3D   
                                CT segmentation.   
  -----------------------------------------------------------------------

**Recommended**: YOLOv8-small for 2D X-ray/CT slice detection.

------------------------------------------------------------------------

## Data Handling & Pre-processing

1.  Accept DICOM and image formats.
2.  Convert to 8-bit grayscale (CT Hounsfield normalization: window
    \[--200, 500 HU\]).
3.  Augmentations: random rotation/flip, CLAHE contrast, Gaussian noise.

------------------------------------------------------------------------

## Training Pipeline

1.  Annotate images using Label Studio or Roboflow (export in YOLO
    format).

2.  Train YOLOv8:

    ``` bash
    yolo detect train data=dataset.yaml model=yolov8s.pt epochs=100 imgsz=640
    ```

3.  Save best weights (`best.pt`).

------------------------------------------------------------------------

## Inference & JSON Metadata

-   Generate JSON:

``` json
{
  "image_id": "scan_001",
  "detections": [
    {
      "serial_no": 1,
      "class": "stone",
      "bbox": [x_min, y_min, x_max, y_max],
      "color": "#FF0000",
      "size_px": area
    }
  ]
}
```

-   Draw bounding boxes with OpenCV.

------------------------------------------------------------------------

## Report Generation

-   Use ReportLab or WeasyPrint to create PDFs with:
    -   Patient details
    -   Embedded image with boxes
    -   Summary table of detections.

------------------------------------------------------------------------

## Deployment

-   Build exe: `bash  pyinstaller --onefile --noconsole app.py`
-   Include model weights and config in a `models/` folder.

------------------------------------------------------------------------

## Hardware & Scaling

-   Training: NVIDIA GPU (RTX 3060+)
-   Inference: CPU is fine; YOLOv8-small runs fast.

------------------------------------------------------------------------

## Summary Table

  Layer             Choice
  ----------------- -------------------------
  Language          Python 3.11
  Model             YOLOv8-small
  Medical I/O       pydicom, OpenCV
  UI                PyQt6
  Packaging         PyInstaller
  Report            ReportLab / WeasyPrint
  Annotation Tool   Label Studio / Roboflow

------------------------------------------------------------------------

### Tips for Medical-Grade Quality

-   Use curated datasets like Kidney Stone CT Dataset (KiTS) for 3D CT.
-   Cross-validate, measure sensitivity/specificity, FROC.
-   Keep an audit trail (versioned models, data splits).

**In short**: Python + PyTorch + YOLOv8 for detection, PyQt6 for UI,
packaged with PyInstaller, handling DICOM/CT, outputting bounding boxes
& JSON metadata, and generating medical-grade PDF reports.
