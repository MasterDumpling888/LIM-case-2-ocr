# LIM-stratpoint-case-2 Receipt OCR Extraction

## Overview

This project is a case study for Stratpoint. It focuses on developing an AI-powered system designed to automate structured data extraction from financial receipts. The system was developed using the SROIE V2 dataset and combines local OCR engines with advanced Large Language Models (LLMs) for semantic correction and validation.

### Key Features Include:

- **Hybrid OCR Pipelines:** Evaluates four distinct strategies: Raw OCR, Pure Multimodal, OCR + Entity LLM, and a Full Hybrid Pipeline.
- **Semantic Rectification:** Automatically corrects OCR misspellings (e.g., "sollor" -> "seller") and completes compound entities using Llama 3.3 70B.
- **Arithmetic Validation:** Implements a Python-based validation layer to verify `Subtotal + Tax = Total` logic, preventing LLM hallucinations.
- **Multimodal Intelligence:** Leverages Llama 4 Scout (via Groq) for direct image-to-JSON extraction without intermediate OCR steps.
- **Interactive Q&A:** A Gradio web interface featuring a chat assistant that can answer detailed questions about specific line items and prices.

## Architecture

1.  **Data Layer:**
    - SROIE V2 Dataset (Receipt images and ground truth JSON).
2.  **OCR Layer:**
    - `src/ocr/engine.py`: PaddleOCR wrapper with automated image resizing for high-resolution stability.
3.  **LLM Layer:**
    - `src/llm/prompts.py`: Optimized templates for rectification, extraction, and vision tasks.
    - `src/llm/chains.py`: LangChain LCEL orchestration using Groq's high-speed inference.
4.  **Pipeline Layer:**
    - `src/pipelines.py`: Implementation of the 4 evaluative standpoints required by the challenge.
5.  **Interface Layer:**
    - `src/app.py`: Enhanced Gradio UI with intermediate step visualization and Q&A chat.

## Installation & Setup

### Prerequisites

- Python 3.11+
- (Conda) Virtual environment (optional but recommended)

### Setup

After cloning the repo and activating the python environment, please install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the root directory and add the GROQ_API_KEY:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Can be obtained from https://console.groq.com/api-keys

## Usage

Launch the interface by running:

```bash
export PYTHONPATH=$PYTHONPATH:src
python src/app.py
```

This will provide a local URL (e.g., `http://127.0.0.1:7860`).

TADA!

## Replicating the Pipeline

If you want to run the full benchmark suite against the SROIE dataset ground truth, follow these steps:

### 1. Data

- **SROIE V2 Dataset:**
  1. Ensure the dataset is present in the `data/SROIE2019/` directory (might need to make a `data` directory).
  2. The partitions should be organized into `train/` and `test/` folders, each containing `img/`, `box/`, and `entities/` subdirectories.

### 2. Running the Benchmark

Execute the evaluation script:

```bash
export PYTHONPATH=$PYTHONPATH:src
python src/evaluation/benchmark.py
```

This will generate a `benchmark_results.csv` and display a summary table of Precision, Recall, and F1 scores across all 4 pipelines.

That's it pancit!
