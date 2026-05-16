# Drone Object Detection & Counting
A computer vision pipeline using YOLOv8 to detect humans and cars, calculate crowd counts, and track objects across aerial drone footage. Built for the Antlings AI/ML Internship Technical Assessment.

## 📌 Project Overview
This project implements an end-to-end machine learning pipeline capable of analyzing drone/aerial images. The system detects specific targets (humans and cars), counts the total number of humans in a frame, and tracks objects across sequential frames using the Ultralytics YOLOv8 framework and ByteTrack.

---

## 🛠️ Task-01: Dataset Understanding & Preprocessing
The model was trained using the **VisDrone Dataset**, a large-scale visual object detection benchmark captured by drones.

* **Format:** The dataset is pre-formatted in the YOLO structure (parallel `images` and `labels` directories) and split into `train`, `val`, and `test-dev` sets. 
* **Target Classes:** The dataset contains 10 target classes. We specifically filtered for `pedestrian` (0), `people` (1), and `car` (3).
* **Preprocessing:** The YOLOv8 pipeline automatically handles image resizing (to `640x640` pixels) while maintaining aspect ratios using padding. Pixel values are normalized for optimized GPU processing.
* **Challenges Noticed:** Drone footage inherently presents challenges such as extremely small object sizes due to high altitudes, severe angle variations, and dense overlapping (occlusion) in crowded urban areas.

---

## 🚀 Task-02: Model Training Pipeline
For the core detection model, **YOLOv8 Nano (`yolov8n.pt`)** was selected. 

* **Approach:** YOLOv8 was chosen over Faster R-CNN or SSD due to its superior inference speed, automated preprocessing capabilities, and seamless integration with tracking algorithms. 
* **Training:** The model was fine-tuned on the VisDrone training set for 10 epochs using Google Colab's T4 GPU. 

---

## 🧮 Task-03: Detection & Human Counting
Inference was executed on the unseen `test-dev` set.
* **Filtering:** The model's predictions were explicitly filtered to only return bounding boxes for human-related classes (0 and 1) and cars (3).
* **Counting Logic:** A custom Python counting loop iterates through the output tensors per frame, tallying the number of boxes that match the human class IDs, and prints the total human count alongside the visualized bounding boxes.

---

## 🎯 Task-04: Object Tracking (Bonus)
Object tracking was implemented to assign and maintain unique IDs for detected vehicles and humans across sequential frames.
* **Implementation:** This was achieved using YOLO's natively integrated **ByteTrack** architecture. ByteTrack was chosen for its high accuracy in handling occlusions and identity maintenance, which is vital for dense aerial drone footage.

---

## 📊 Task-05: Evaluation & Analysis

### Visual Outputs
* **Prediction Outputs:** The model successfully draws bounding boxes around targeted classes (0, 1, and 3).
* **Counting Visualization:** Processed test images display a dynamic title indicating the exact Total Human Count tallied from the filtered bounding boxes.
* **Processed Images:** Tracking results visually display unique ID numbers assigned to the bounding boxes. Training metrics (`results.png`) are included in the repository.

### Discussion
* **Strengths:** The model successfully learned to identify distinct features of cars and humans from a top-down perspective, adapting well to the drone's specific viewing angle.
* **Limitations:** Due to the limited 10-epoch training run, the model occasionally yields lower confidence scores (or false negatives) for extremely small or distant pedestrians. 
* **Challenges Faced:** The primary challenges were heavy class imbalance (far more cars than humans in the dataset), dense occlusion in urban scenes, and constantly varying drone altitudes drastically changing target scales.

---

## 📁 Repository Contents
* `AntsTask.ipynb`: The complete Google Colab notebook containing the end-to-end pipeline.
* `images/`: Directory containing sample outputs, including basic detection with counting, and ByteTrack ID visualizations.
* `results.png`: Training metrics graph (Precision, Recall, mAP).

## 🎥 Demonstration Video:
https://drive.google.com/file/d/18JToslMaPnfDYYchapgKnesSADyYQFwZ/view?usp=sharing
