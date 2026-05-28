from paddleocr import PaddleOCR
import cv2

class OCREngine:
    def __init__(self, lang='en'):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        
    def extract_text(self, img_path):
        """
        Extracts text from an image and returns a list of results.
        Supports both traditional PaddleOCR and new PaddleX-integrated formats.
        """
        result = self.ocr.ocr(img_path)
        if not result:
            return []
            
        # New PaddleX-integrated format: [{'rec_texts': [...], ...}]
        if isinstance(result[0], dict) and 'rec_texts' in result[0]:
            return result[0]['rec_texts']
            
        # Traditional format: [[[box], (text, score)], ...]
        if isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], list):
            return [line[1][0] for line in result[0]]
            
        return []
    
    def get_full_text(self, img_path):
        """
        Returns all extracted text as a single string.
        """
        texts = self.extract_text(img_path)
        return "\n".join(texts)
