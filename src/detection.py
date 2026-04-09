from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTANT NOTE ON MODELS
# ──────────────────────────────────────────────────────────────────────────────
# yolov9c.pt is a COCO-pretrained general model.
# COCO has 80 classes – the ones relevant for this system are:
#   class 0  → person
#   class 3  → motorcycle
#
# It does NOT have "Helmet", "No-Helmet", or "License Plate" classes.
# To detect those you need a custom-trained model (e.g. from Roboflow Universe).
#
# Until a custom model is available the system will:
#   • Correctly detect motorcycles (COCO class 3)
#   • Correctly detect persons (COCO class 0) as riders
#   • Flag a "potential violation" when a person is on a motorcycle
#     (since COCO can't distinguish helmeted vs. un-helmeted riders)
# ──────────────────────────────────────────────────────────────────────────────

# Full 80-class COCO label map
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

# Classes we actually care about (keep only these in output)
RELEVANT_CLASSES = {'person', 'motorcycle', 'bicycle'}

class Detector:
    def __init__(self, config):
        weights = config['model']['yolov9_weights']
        self.model = YOLO(weights)
        self.conf_thresh = config['model']['confidence_threshold']

        # Detect whether we are using a custom model
        self.is_custom_model = self._check_custom_model()
        
        # Load secondary helmet model if available
        import os
        self.helmet_model = None
        if os.path.exists('helmet_model.pt'):
            self.helmet_model = YOLO('helmet_model.pt')
            print("[INFO] Secondary Helmet Model Loaded successfully.")

    def _check_custom_model(self):
        """Returns True if the loaded primary model has custom traffic violation classes."""
        names = list(self.model.names.values())
        custom_markers = {'Helmet', 'No-Helmet', 'License Plate',
                          'helmet', 'no-helmet', 'license plate', 'licenseplate'}
        return bool(custom_markers.intersection(set(names)))

    def detect(self, frame):
        # 1. Primary Model (YOLOv9c - COCO)
        results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf        = float(box.conf[0])
            class_id    = int(box.cls[0])

            if self.is_custom_model:
                class_name = self.model.names[class_id]
            else:
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else 'unknown'
                if class_name not in RELEVANT_CLASSES:
                    continue
                if class_name == 'person':
                    class_name = 'Rider'
                elif class_name == 'motorcycle':
                    class_name = 'Motorcycle'

            detections.append({
                'box':        (x1, y1, x2, y2),
                'conf':       conf,
                'class_id':   class_id,
                'class_name': class_name,
            })
            
        # 2. Secondary Model (Helmet Specific)
        if self.helmet_model is not None:
             h_results = self.helmet_model(frame, conf=self.conf_thresh, verbose=False)[0]
             for box in h_results.boxes:
                 x1, y1, x2, y2 = map(int, box.xyxy[0])
                 conf = float(box.conf[0])
                 class_id = int(box.cls[0])
                 raw_name = self.helmet_model.names[class_id]
                 
                 # Normalize labels to what violation_logic expects
                 if raw_name == 'With Helmet':
                     class_name = 'Helmet'
                     # HELMET FALSE POSITIVE FILTER: 
                     # Only accept 'Helmet' if the model is very confident (e.g. > 75%)
                     # This helps filter out scarves and turbans which look like helmets at low conf
                     if conf < max(self.conf_thresh, 0.75):
                         continue
                 elif raw_name == 'Without Helmet':
                     class_name = 'No-Helmet'
                 else:
                     class_name = raw_name
                     
                 detections.append({
                    'box':        (x1, y1, x2, y2),
                    'conf':       conf,
                    'class_id':   class_id + 100, # offset ID
                    'class_name': class_name,
                 })

        return detections
