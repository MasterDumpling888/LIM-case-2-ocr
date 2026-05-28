import os
import base64
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from .prompts import RECTIFICATION_PROMPT, EXTRACTION_PROMPT, VISION_PROMPT, QA_PROMPT

load_dotenv()

class LLMManager:
    def __init__(self):
        self.primary_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Multimodal model - using Llama 4 Scout
        self.vision_model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.vision_llm = ChatGroq(
            model=self.vision_model_name,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        self._setup_chains()
        
    def _setup_chains(self):
        self.rectification_chain = RECTIFICATION_PROMPT | self.primary_llm | StrOutputParser()
        self.extraction_chain = EXTRACTION_PROMPT | self.primary_llm | JsonOutputParser()
        self.vision_chain = VISION_PROMPT | self.vision_llm | JsonOutputParser()
        self.qa_chain = QA_PROMPT | self.primary_llm | StrOutputParser()
        
    def rectify_text(self, ocr_text: str) -> str:
        return self.rectification_chain.invoke({"ocr_text": ocr_text})
        
    def extract_entities(self, text: str) -> dict:
        return self.extraction_chain.invoke({"text": text})
        
    def extract_from_image(self, image_path: str) -> dict:
        """
        Extract entities directly from image using multimodal LLM.
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Groq expects data:image/jpeg;base64,...
            mime_type = "image/jpeg"
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
                
            image_url = f"data:{mime_type};base64,{encoded_string}"
            return self.vision_chain.invoke({"image_url": image_url})
        except Exception as e:
            print(f"Vision extraction failed: {e}")
            return {}
            
    def answer_question(self, extracted_content: str, question: str) -> str:
        return self.qa_chain.invoke({
            "extracted_content": extracted_content,
            "question": question
        })
