import gradio as gr
import os
import json
import cv2
from ocr.engine import OCREngine
from llm import LLMManager
from pipelines import RawOCRPipeline, MultimodalLLMPipeline, OCRTextLLMPipeline, FullHybridPipeline
from utils.data_loader import SROIEDataset

# Initialize components
ocr_engine = OCREngine()
llm_manager = LLMManager()

pipelines = {
    "Raw OCR": RawOCRPipeline(ocr_engine, llm_manager),
    "Multimodal LLM": MultimodalLLMPipeline(ocr_engine, llm_manager),
    "OCR + Entity LLM": OCRTextLLMPipeline(ocr_engine, llm_manager),
    "Full Hybrid": FullHybridPipeline(ocr_engine, llm_manager)
}

# Pre-load dataset mapping for ground truth lookup (optional)
dataset_mapping = {}
for split in ["train", "test"]:
    ds_path = f"data/SROIE2019/{split}"
    if os.path.exists(ds_path):
        ds = SROIEDataset(ds_path)
        for i in range(len(ds)):
            example = ds.get_example(i)
            dataset_mapping[os.path.basename(example["img_path"])] = example["entities"]
    else:
        print(f"Warning: Dataset path {ds_path} not found. Skipping ground truth pre-loading.")

def process_receipt(image, pipeline_name):
    if image is None:
        return "Please upload an image.", {}, "", "", {}
        
    # image is a numpy array from Gradio
    # Resize if too large to prevent crashes and speed up processing
    max_dim = 1300
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from {w}x{h} to {new_w}x{new_h}")
        
    temp_path = "temp_receipt.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    pipeline = pipelines[pipeline_name]
    output = pipeline.run(temp_path)
    
    entities = output.get("entities", {})
    raw_text = output.get("raw_text", "")
    rectified_text = output.get("rectified_text", "N/A (Only for Hybrid Pipeline)")
    
    # Try to find ground truth if it's a known file
    filename = "temp_receipt.jpg" # This is local, but in real demo it would be the original filename
    # For demo purposes, we'll try to match by content or just show N/A
    ground_truth = "N/A (Not in dataset)"
    
    return raw_text, entities, rectified_text, ground_truth

def chatbot_response(message, history, extracted_entities):
    if not extracted_entities:
        return "Please process a receipt first."
    
    content_str = json.dumps(extracted_entities, indent=2)
    response = llm_manager.answer_question(content_str, message)
    return response

with gr.Blocks(title="Receipt OCR Extraction System") as demo:
    gr.Markdown("#Receipt OCR Extraction System")
    gr.Markdown("Extract structured data from receipts using Hybrid OCR + LLM pipelines.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload Receipt")
            pipeline_select = gr.Dropdown(
                choices=list(pipelines.keys()), 
                value="Full Hybrid", 
                label="Select Pipeline"
            )
            run_btn = gr.Button("Process Receipt", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Extracted Entities"):
                    output_json = gr.JSON(label="Structured Output")
                with gr.TabItem("Intermediate Steps"):
                    with gr.Row():
                        raw_ocr_out = gr.Textbox(label="Raw OCR Text", lines=10)
                        rectified_text_out = gr.Textbox(label="LLM Rectified Text", lines=10)
                with gr.TabItem("Q&A Chat"):
                    chatbot = gr.Chatbot(label="Ask about the receipt")
                    msg = gr.Textbox(placeholder="What was the total amount?", label="Question")
                    clear = gr.Button("Clear Chat")

    # State to store extracted entities for chatbot
    extracted_state = gr.State({})

    def on_process(image, pipeline_name):
        raw_text, entities, rectified_text, gt = process_receipt(image, pipeline_name)
        # Use rectified text if available, otherwise raw text
        full_context_text = rectified_text if rectified_text != "N/A (Only for Hybrid Pipeline)" else raw_text
        chat_context = {
            "structured_entities": entities,
            "full_receipt_text": full_context_text
        }
        return raw_text, entities, rectified_text, chat_context

    run_btn.click(
        on_process, 
        inputs=[input_image, pipeline_select], 
        outputs=[raw_ocr_out, output_json, rectified_text_out, extracted_state]
    )
    
    def respond(message, chat_history, state):
        bot_message = chatbot_response(message, chat_history, state)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    msg.submit(respond, [msg, chatbot, extracted_state], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=False) # No publicly shareable link
