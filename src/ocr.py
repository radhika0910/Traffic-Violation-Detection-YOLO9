import easyocr
import cv2
import re

class PlateRecognizer:
    def __init__(self, config):
        languages = config['ocr']['languages']
        # Initialize easyocr reader
        self.reader = easyocr.Reader(languages, gpu=True)
        self.conf_thresh = config['ocr']['confidence_threshold']

    def preprocess_plate(self, plate_img):
        # 1. Perspective Correction
        # Assuming plate_img is a rectangular crop, but if tilted, 
        # a more advanced system would detect corners. 
        # For a standard crop, we ensure it's resized to a standard aspect ratio
        h, w = plate_img.shape[:2]
        # Basic perspective correction if corners are known, but for a bounding box crop,
        # we can enhance readability by resizing to a standard license plate ratio (e.g., 500x150)
        plate_img = cv2.resize(plate_img, (500, 150))

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # 3. Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 4. Noise Reduction
        blur = cv2.bilateralFilter(enhanced, 11, 17, 17)
        
        # 5. Binarization
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def recognize(self, plate_img):
        best_text = ""
        best_conf = 0.0
        
        # Try two passes: 1. Processed, 2. Raw
        passes = [self.preprocess_plate(plate_img), plate_img]
        
        for idx, img in enumerate(passes):
            # Extract text (allowing alphanumeric chars)
            results = self.reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            for (bbox, text, prob) in results:
                # Clean text: remove spaces and non-alphanumeric chars
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                # Length check: most plates are 4-10 chars. User's example might be shorter.
                if len(clean_text) >= 4 and len(clean_text) <= 12:
                    if prob > best_conf:
                        best_text = clean_text
                        best_conf = prob
                        
            # If we got a high confidence hit in pass 1, skip pass 2
            if best_conf > 0.8:
                break

        return best_text, best_conf
