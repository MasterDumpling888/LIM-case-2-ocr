import re

def clean_string(s):
    if not s:
        return ""
    # Lowercase, remove special chars, normalize whitespace
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = " ".join(s.split())
    return s

def calculate_entity_metrics(gt_entities, pred_entities):
    """
    Calculates precision, recall, and F1 for a single example.
    gt_entities: dict with keys [company, date, address, total, subtotal, tax]
    pred_entities: dict with same keys
    """
    keys = ["company", "date", "address", "total", "subtotal", "tax"]
    tp = 0
    fp = 0
    fn = 0
    
    for key in keys:
        gt_val = clean_string(str(gt_entities.get(key, "")))
        pred_val = clean_string(str(pred_entities.get(key, "")))
        
        if gt_val == "" and pred_val == "":
            continue # True Negative - not counted in P/R/F1 usually
        elif gt_val != "" and pred_val == "":
            fn += 1
        elif gt_val == "" and pred_val != "":
            fp += 1
        else:
            # Both have values
            if gt_val in pred_val or pred_val in gt_val:
                tp += 1
            else:
                fp += 1
                fn += 1 # It's a mismatch, so both FP and FN
                
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extracted_count": sum(1 for v in pred_entities.values() if v)
    }

def aggregate_metrics(all_results):
    total_tp = sum(r["tp"] for r in all_results)
    total_fp = sum(r["fp"] for r in all_results)
    total_fn = sum(r["fn"] for r in all_results)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_extracted = sum(r["extracted_count"] for r in all_results) / len(all_results) if all_results else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_entities_extracted": avg_extracted
    }
