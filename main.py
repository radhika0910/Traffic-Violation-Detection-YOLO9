import cv2
import yaml
import os
import time

from src.detection import Detector
from src.ocr import PlateRecognizer
from src.violation_logic import check_violations
# from src.db import DatabaseLogger

def main():
    print("[INFO] Loading Configuration...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['output']['save_dir'], exist_ok=True)

    print("[INFO] Initializing Detector (YOLOv9)...")
    try:
        detector = Detector(config)
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector: {e}")
        return

    print("[INFO] Initializing Plate Recognizer (EasyOCR)...")
    ocr = PlateRecognizer(config)
    
    # Initialize DB (uncomment after setting up MySQL)
    # print("[INFO] Connecting to database...")
    # db_logger = DatabaseLogger(config) 

    source = config['camera']['source']
    
    # Check if stream source is numeric (for local camera) or string (file/rtsp)
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {source}")
        return

    print("[INFO] Starting Video Stream processing...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        detections = detector.detect(frame)
        violations = check_violations(detections, config['model']['violation_threshold'])
        
        for v in violations:
            plate_box = v['plate']['box']
            px1, py1, px2, py2 = plate_box
            
            # Crop the license plate from the frame
            plate_img = frame[py1:py2, px1:px2]
            
            # Extra safety check to prevent passing empty images to EasyOCR
            if plate_img.size == 0:
                continue
            
            plate_text, conf = ocr.recognize(plate_img)
            
            if plate_text:
                print(f"[VIOLATION] Plate: {plate_text} (Confidence: {conf:.2f})")
                
                # Save snapshot evidence
                timestamp_str = str(int(time.time()))
                snapshot_name = f"{timestamp_str}_snapshot.jpg"
                plate_name = f"{timestamp_str}_plate.jpg"
                snapshot_path = os.path.join(config['output']['save_dir'], snapshot_name)
                plate_path = os.path.join(config['output']['save_dir'], plate_name)
                
                # Draw boxes and save
                snap_img = frame.copy()
                cv2.rectangle(snap_img, (plate_box[0], plate_box[1]), (plate_box[2], plate_box[3]), (0, 0, 255), 2)
                cv2.imwrite(snapshot_path, snap_img)
                cv2.imwrite(plate_path, plate_img)
                
                # Log to DB
                # db_logger.log_violation("CAM_01", plate_text, conf, "No Helmet", snapshot_path, plate_path)

        # cv2.imshow is disabled - use 'streamlit run app.py' for the live web UI
        # Headless progress print so you know its running
        print(f"[PROCESSING] Detections: {len(detections)} | Violations: {len(violations)}", end='\r')

    cap.release()
    print("\n[INFO] Stream ended.")

if __name__ == "__main__":
    main()
