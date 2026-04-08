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
        # A custom model will have class names like 'Motorcycle', 'Helmet', etc.
        self.is_custom_model = self._check_custom_model()

    def _check_custom_model(self):
        """Returns True if the loaded model has custom traffic violation classes."""
        names = list(self.model.names.values())
        custom_markers = {'Helmet', 'No-Helmet', 'License Plate',
                          'helmet', 'no-helmet', 'license plate', 'licenseplate'}
        return bool(custom_markers.intersection(set(names)))

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf        = float(box.conf[0])
            class_id    = int(box.cls[0])

            if self.is_custom_model:
                # Custom model: use its own class names directly
                class_name = self.model.names[class_id]
            else:
                # COCO model: map using the COCO label list
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else 'unknown'

                # Only keep traffic-relevant classes to reduce noise
                if class_name not in RELEVANT_CLASSES:
                    continue

                # Normalise to internal names the violation_logic expects
                # For COCO we approximate: a person near a motorcycle = potential rider
                if class_name == 'person':
                    class_name = 'Rider'          # rider (may or may not have helmet)
                elif class_name == 'motorcycle':
                    class_name = 'Motorcycle'

            detections.append({
                'box':        (x1, y1, x2, y2),
                'conf':       conf,
                'class_id':   class_id,
                'class_name': class_name,
            })

        return detections
