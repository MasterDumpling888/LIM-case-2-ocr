import os
import argparse
import pandas as pd
from tqdm import tqdm
from src.utils.data_loader import SROIEDataset
from src.ocr.engine import OCREngine
from src.llm import LLMManager
from src.pipelines import RawOCRPipeline, MultimodalLLMPipeline, OCRTextLLMPipeline, FullHybridPipeline
from src.evaluation.metrics import calculate_entity_metrics, aggregate_metrics

def run_benchmark(limit=None):
    # Initialize components
    ocr_engine = OCREngine()
    llm_manager = LLMManager()
    
    pipelines = {
        "Raw OCR": RawOCRPipeline(ocr_engine, llm_manager),
        "Multimodal LLM": MultimodalLLMPipeline(ocr_engine, llm_manager),
        "OCR + Entity LLM": OCRTextLLMPipeline(ocr_engine, llm_manager),
        "Full Hybrid": FullHybridPipeline(ocr_engine, llm_manager)
    }
    
    # Load dataset (combined train and test)
    train_dataset = SROIEDataset("data/SROIE2019/train")
    test_dataset = SROIEDataset("data/SROIE2019/test")
    
    all_examples = []
    # Combined dataset list
    for i in range(len(train_dataset)):
        all_examples.append(("train", train_dataset, i))
    for i in range(len(test_dataset)):
        all_examples.append(("test", test_dataset, i))
        
    if limit:
        import random
        random.seed(42)
        all_examples = random.sample(all_examples, min(limit, len(all_examples)))
        
    print(f"Running benchmark on {len(all_examples)} examples...")
    
    results = {name: [] for name in pipelines.keys()}
    
    for split, ds, idx in tqdm(all_examples):
        example = ds.get_example(idx)
        img_path = example["img_path"]
        gt_entities = example["entities"]
        
        if not gt_entities:
            continue
            
        for name, pipeline in pipelines.items():
            try:
                output = pipeline.run(img_path)
                metrics = calculate_entity_metrics(gt_entities, output["entities"])
                results[name].append(metrics)
            except Exception as e:
                print(f"Error running {name} on {img_path}: {e}")
                
    # Aggregate results
    summary = []
    for name, pipeline_results in results.items():
        if not pipeline_results:
            continue
        agg = aggregate_metrics(pipeline_results)
        summary.append({
            "Pipeline": name,
            "Precision": f"{agg['precision']:.4f}",
            "Recall": f"{agg['recall']:.4f}",
            "F1 Score": f"{agg['f1']:.4f}",
            "Avg Entities Extracted": f"{agg['avg_entities_extracted']:.2f}"
        })
        
    df = pd.DataFrame(summary)
    print("\nBenchmark Results:")
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to benchmark")
    args = parser.parse_args()
    
    run_benchmark(args.limit)
