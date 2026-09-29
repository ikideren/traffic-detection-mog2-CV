# Traffic Vehicle Detection

**DETECT. TRACK. OPTIMIZE.**

This project is a computer vision application for detecting and tracking vehicles in traffic scenes. The primary method is **MOG2 background subtraction**, which processes traffic video, filters foreground objects, tracks vehicle centroids, and estimates a recommended green-light duration based on detected traffic volume.

Two additional approaches are included for comparison:

- **SVM + HOG**, using handcrafted visual features and a linear classifier
- **YOLOv8**, using a pretrained deep learning object-detection model

> **Note:** This project is intended for computer vision experimentation and traffic-analysis demonstrations. Detection results depend on camera position, lighting, video quality, and the selected parameters.

## Table of Contents

- [Background](#background)
- [Features](#features)
- [How It Works](#how-it-works)
- [Models and Methods](#models-and-methods)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Limitations and Future Work](#limitations-and-future-work)
- [Team](#team)

## Background

Traffic monitoring systems need to identify vehicles reliably and quickly to support analysis of road conditions and traffic flow. Traditional computer vision methods can be useful in controlled camera environments because they are relatively lightweight and easy to inspect.

This project focuses on MOG2 as the main detection pipeline. SVM + HOG and YOLOv8 are provided as comparison methods to examine the differences between background-subtraction, feature-based classification, and modern deep learning detection.

## Features

- Detect moving vehicles from traffic video using MOG2
- Restrict processing to a dynamically calculated region of interest
- Remove noise using morphological opening and closing
- Track detected objects using centroid and distance-based matching
- Assign IDs to tracked vehicles
- Estimate green-light duration from the detected vehicle count
- Export intermediate MOG2 processing layers as MP4 files
- Compare the primary method with SVM + HOG and YOLOv8
- Train the SVM classifier from HOG features extracted from the included dataset

### Vehicle Categories

The training dataset contains the following vehicle categories:

| # | Class |
| --- | --- |
| 1 | Articulated truck |
| 2 | Bicycle |
| 3 | Bus |
| 4 | Car |
| 5 | Motorcycle |
| 6 | Non-motorized vehicle |
| 7 | Pedestrian |
| 8 | Pickup truck |
| 9 | Single-unit truck |
| 10 | Work van |
| 11 | Background |

## How It Works

### Main MOG2 Pipeline

1. Read frames from the configured traffic video.
2. Create a region of interest covering the road area.
3. Apply OpenCV MOG2 background subtraction to the ROI.
4. Threshold the foreground mask to isolate moving objects.
5. Apply morphological opening and closing to reduce noise and fill gaps.
6. Find contours and filter them by area and aspect ratio.
7. Track object centroids across frames using nearest-distance matching.
8. Draw bounding boxes and tracking IDs on the output frame.
9. Estimate green-light duration from the vehicle count.
10. Export four processing layers and display them in OpenCV windows.

The MOG2 script currently calculates the green-light duration using:

- Base time: 15 seconds
- Additional clearing time: 2.5 seconds per detected vehicle
- Maximum duration: 60 seconds

### SVM + HOG Comparison

The SVM pipeline converts grayscale vehicle images to **64x64** pixels, extracts HOG features, and trains a `LinearSVC` classifier to distinguish background from vehicles. During detection, Selective Search generates candidate regions, the SVM classifies them, and non-maximum suppression removes overlapping boxes.

### YOLOv8 Comparison

The YOLO pipeline uses the pretrained `yolov8n.pt` model and filters detections to cars, motorcycles, buses, and trucks. YOLO performs object detection and non-maximum suppression automatically.

## Models and Methods

| Method | Role | Input | Output |
| --- | --- | --- | --- |
| MOG2 | Primary method | Traffic video | Moving-vehicle masks, tracking IDs, MP4 layers |
| SVM + HOG | Comparison method | Still image | Vehicle bounding boxes |
| YOLOv8n | Comparison method | Still image | Vehicle bounding boxes and classes |

## Dataset

The dataset is stored under `dataset/train1/` and is organized into class-specific folders. The SVM preprocessing script uses the following labels:

- Positive examples: cars, buses, motorcycles, pickup trucks, articulated trucks, single-unit trucks, and work vans
- Negative examples: background

The bicycle, non-motorized vehicle, and pedestrian folders are present in the dataset but are not included as positive classes by the current SVM preprocessing script.

## Tech Stack

- **Language:** Python
- **Computer vision:** OpenCV and OpenCV contrib
- **Feature extraction:** scikit-image HOG
- **Classical machine learning:** scikit-learn LinearSVC
- **Model serialization:** joblib
- **Deep learning detection:** Ultralytics YOLOv8
- **Numerical processing:** NumPy
- **Progress reporting:** tqdm

## Project Structure

```
Final Project/
+-- dataset/
|   `-- train1/
|       |-- articulated_truck/
|       |-- background/
|       |-- bicycle/
|       |-- bus/
|       |-- car/
|       |-- motorcycle/
|       |-- non-motorized_vehicle/
|       |-- pedestrian/
|       |-- pickup_truck/
|       |-- single_unit_truck/
|       `-- work_van/
+-- MOG/
|   `-- detect_mog2.py
+-- SVM+HOG/
|   |-- detect_traffic.py
|   |-- extract_negatives.py
|   |-- preprocess.py
|   `-- train_svm.py
+-- Yolo/
|   |-- detect_yolo.py
|   `-- yolov8n.pt
+-- requirements.txt
`-- README.md
```

The following runtime files are referenced by the scripts but are not included in the source tree listing above:

- `MOG/traffic_video6.mp4` for the MOG2 pipeline
- `test_intersection.jpg` for the SVM + HOG and YOLO comparison scripts
- `hog_features_labels.pkl` and `svm_model.pkl`, generated by the SVM training workflow

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip
- A video file for the MOG2 pipeline
- A test image for the comparison pipelines

### Installation

```bash
# 1. Clone the repository
# git clone <repository-url>
# cd "Final Project"

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Run MOG2 Detection

Place the traffic video at `MOG/traffic_video6.mp4`, then run this command from the project root:

```bash
python MOG/detect_mog2.py
```

The script displays four OpenCV windows and exports these files to `MOG/`:

- `layer1_roi.mp4`
- `layer2_raw_mog.mp4`
- `layer3_morphological.mp4`
- `layer4_final_tracking.mp4`

Press `q` to stop processing.

### Train the SVM + HOG Model

Run the preprocessing and training scripts from the project root:

```bash
python SVM+HOG/preprocess.py
python SVM+HOG/train_svm.py
```

This generates:

- `hog_features_labels.pkl`
- `svm_model.pkl`

To run the SVM detector, place `test_intersection.jpg` in the project root and run:

```bash
python SVM+HOG/detect_traffic.py
```

### Run YOLOv8 Detection

Place `test_intersection.jpg` in the `Yolo/` directory, then run:

```bash
cd Yolo
python detect_yolo.py
```

The YOLO model may download automatically if `yolov8n.pt` is not available locally. The detector filters results to cars, motorcycles, buses, and trucks.

## Limitations and Future Work

**Limitations**

- MOG2 is sensitive to camera movement, shadows, sudden lighting changes, and background motion.
- The centroid tracker can lose objects when vehicles overlap or move quickly.
- The vehicle counter is based on cumulative tracking IDs rather than a calibrated line-crossing count.
- The SVM and YOLO comparison scripts process still images rather than the main traffic video.
- Green-light recommendations are heuristic values and are not connected to a real traffic signal controller.
- The current scripts rely on hard-coded file paths, thresholds, and regions of interest.

**Future work**

- Add configurable command-line arguments for input paths and detection parameters.
- Improve tracking with a Kalman filter or a dedicated multi-object tracker.
- Add line-crossing logic to count vehicles entering or leaving a road segment.
- Evaluate all methods using precision, recall, F1-score, and processing speed.
- Support real-time camera streams.
- Calibrate traffic-light timing using historical traffic data and road capacity.
- Add visual result summaries and automated comparison reports.

## Team

| Name | Role |
| --- | --- |
| Darren Christian Pramana | Proposal, computer vision research, and model development |
| Edward Nicholas Adidjaja | Proposal, implementation, and project development |
| Christian Leonardo Halim | Proposal, implementation, and project development |
| Jozio Damaier Gidalti | Proposal, implementation, and project development |

Computer Science, Bina Nusantara University

---

> Built as a team project for traffic-vehicle detection and computer vision method comparison.
