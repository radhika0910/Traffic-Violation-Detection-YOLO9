# Traffic Violation Detection System (YOLOv9 + EasyOCR)

This project implements the automated traffic violation detection system described in the abstract. It leverages the YOLOv9 object detection architecture for real-time inference and EasyOCR for accurate Indian license plate character recognition.

## Features
- Motorcycle and non-helmet rider detection via **YOLOv9**
- Real-time video processing supporting `.mp4`, IP Cams, and RTSP streams
- Pre-processing optimizations and Optical Character Recognition (**EasyOCR**) for License Plates
- Database integration for centralized evidence logging (MySQL)

## Prerequisites
- Minimum Python 3.10
- GPU Support (NVIDIA RTX 3060/4060 or better) strongly recommended for EasyOCR and YOLO inference

## Setup Instructions

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Settings**
   Open `config.yaml` and adjust the configuration for database credentials, models, and video paths.

3. **Database**
   If you intend to use the MySQL logging system, start your local MySQL instance and ensure a database named `traffic_violations` exists matching your `config.yaml`. The project will automatically create the required schema tables.

4. **YOLO Checkpoints**
   You'll need an exported YOLOv9 weight file (e.g., `yolov9-c.pt`) mapped to the classifications defined in `src/detection.py`. Put it inside a `weights/` folder.

5. **Execution**
   ```bash
   python main.py
   ```
