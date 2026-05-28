import re
from abc import ABC, abstractmethod
from ocr.engine import OCREngine
from llm import LLMManager

class BasePipeline(ABC):
    def __init__(self, ocr_engine: OCREngine, llm_manager: LLMManager):
        self.ocr_engine = ocr_engine
        self.llm_manager = llm_manager
        
    @abstractmethod
    def run(self, img_path: str) -> dict:
        pass

class RawOCRPipeline(BasePipeline):
    """
    Pipeline 1: Image -> PaddleOCR -> Heuristic Mapping
    """
    def run(self, img_path: str) -> dict:
        full_text = self.ocr_engine.get_full_text(img_path)
        # Simple heuristics for baseline
        entities = {
            "company": "",
            "date": "",
            "address": "",
            "total": ""
        }
        
        lines = full_text.split('\n')
        if lines:
            entities["company"] = lines[0] # Usually first line
            
        # Try to find total (last number usually)
        amounts = re.findall(r'\d+\.\d{2}', full_text)
        if amounts:
            entities["total"] = amounts[-1]
            
        # Try to find date
        date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', full_text)
        if date_match:
            entities["date"] = date_match.group(0)
            
        # Address is usually lines between company and items - very hard to heuristic
        # We'll just take the 2nd line as a placeholder
        if len(lines) > 1:
            entities["address"] = lines[1]
            
        return {
            "entities": entities,
            "raw_text": full_text
        }

class MultimodalLLMPipeline(BasePipeline):
    """
    Pipeline 2: Image -> Multimodal LLM (Llama 4 Scout / 3.2 Vision)
    """
    def run(self, img_path: str) -> dict:
        entities = self.llm_manager.extract_from_image(img_path)
        if not entities:
            # Fallback to Raw OCR as per plan
            return RawOCRPipeline(self.ocr_engine, self.llm_manager).run(img_path)
            
        return {
            "entities": entities,
            "raw_text": "Vision-only extraction (no OCR text)"
        }

class OCRTextLLMPipeline(BasePipeline):
    """
    Pipeline 3: Image -> PaddleOCR -> Llama 3.3 (Extraction)
    """
    def run(self, img_path: str) -> dict:
        full_text = self.ocr_engine.get_full_text(img_path)
        entities = self.llm_manager.extract_entities(full_text)
        return {
            "entities": entities,
            "raw_text": full_text
        }

class FullHybridPipeline(BasePipeline):
    """
    Pipeline 4: Image -> OCR -> Llama 3.3 (Rectification) -> Llama 3.3 (Extraction) -> Arithmetic Validation
    """
    def validate_arithmetic(self, entities: dict) -> dict:
        def to_float(val):
            if not val: return 0.0
            # Remove commas and other non-numeric chars except decimal point
            clean_val = re.sub(r'[^\d.]', '', str(val))
            try:
                return float(clean_val)
            except ValueError:
                return 0.0

        try:
            subtotal = to_float(entities.get("subtotal", 0))
            tax = to_float(entities.get("tax", 0))
            total = to_float(entities.get("total", 0))
            
            if (subtotal > 0 or tax > 0) and total > 0:
                # Allow for small rounding differences (e.g. 0.05)
                is_valid = abs((subtotal + tax) - total) < 0.05
                entities["validation"] = {
                    "arithmetic_valid": is_valid,
                    "check": f"{subtotal:.2f} + {tax:.2f} = {total:.2f}"
                }
            else:
                entities["validation"] = {"arithmetic_valid": "insufficient_data"}
        except Exception as e:
            entities["validation"] = {"arithmetic_valid": "error", "message": str(e)}
        return entities

    def run(self, img_path: str) -> dict:
        raw_text = self.ocr_engine.get_full_text(img_path)
        # Step 1: Rectification
        rectified_text = self.llm_manager.rectify_text(raw_text)
        # Step 2: Extraction
        entities = self.llm_manager.extract_entities(rectified_text)
        # Step 3: Arithmetic Validation
        entities = self.validate_arithmetic(entities)
        
        return {
            "entities": entities,
            "raw_text": raw_text,
            "rectified_text": rectified_text
        }
