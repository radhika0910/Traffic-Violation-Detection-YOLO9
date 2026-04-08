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
        # Apply preprocessing
        processed = self.preprocess_plate(plate_img)
        
        # Extract text (allowing alphanumeric chars suitable for Indian plates)
        results = self.reader.readtext(processed, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        best_text = ""
        best_conf = 0.0
        
        indian_plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')
        
        for (bbox, text, prob) in results:
            # Clean text: remove spaces and non-alphanumeric chars
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            if prob > self.conf_thresh and prob > best_conf:
                # Format validation: check if it roughly matches Indian license plate format
                # We do a basic check, or if it strictly matches the regex
                if indian_plate_pattern.match(clean_text):
                    best_text = clean_text
                    best_conf = prob
                # Fallback: if it doesn't strictly match but has high confidence and reasonable length
                elif len(clean_text) >= 8 and len(clean_text) <= 10 and prob > best_conf + 0.1:
                     best_text = clean_text
                     best_conf = prob
                elif not best_text: # keep the highest if nothing matched format
                    best_text = clean_text
                    best_conf = prob

        return best_text, best_conf
