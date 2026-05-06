import argparse
import json
import os
import torch
import hashlib
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from sklearn.metrics import roc_auc_score
from summa import keywords

import warnings
from transformers import logging


def hash_ids(ids, seed=0, mod=None):
    s = ",".join(map(str, ids)) + f":{seed}"
    h = int(hashlib.sha256(s.encode()).hexdigest(), 16)
    return h if mod is None else h % mod

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--input_file", type=str, default="attacked_data.json")
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    args = parser.parse_args()

    delta = args.delta

    # 1. Load the Attacked Data
    print(f"Loading attacked data from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # Locate the correct entry for this gamma/delta
    target_entry = None
    for entry in data:
        if entry.get('delta') == delta:
            target_entry = entry
            break
    
    # Fallback: if structure is flat or only one entry exists
    if not target_entry and len(data) > 0:
        target_entry = data[0]

    samples = target_entry.get('samples', [])
    if not samples:
        print("No samples found for the specified gamma/delta.")
        return

    # 2. Setup Tools (Model & Tokenizer)
    print("Initializing Model and Tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google-t5/t5-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformers_config = TransformersConfig(
        model=AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device),
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=200,
        min_length=64,
        do_sample=True,
        no_repeat_ngram_size=4,
    )

    # 3. Analyze Attack
    print(f"Analyzing {len(samples)} samples...")
    
    attacked_scores = []
    original_nwm_scores = []  # needed for ROC baseline
    results_list = []
    num_failed = 0
    num_total = 0
    # Threshold for detection (usually derived or set to 0/4.0 depending on strictness)
    # KGW typically uses z-score > 4.0 for high confidence, or > 0 for binary classification in AUC
    
    for sample in tqdm(samples, desc="Detecting"):
        prompt = sample['prompt']
        attacked_text = sample.get('attacked_text', "")
        
        if not attacked_text:
            continue

        # --- REGENERATE HASH KEY ---
        # The detector MUST know the specific hash key for this prompt to 
        # recreate the correct Green List.
        # token = keywords.keywords(prompt, ratio=0.3)
        # ids = tokenizer.encode(token, add_special_tokens=False)
        # hash_key = hash_ids(ids, len(prompt), 998244353)

        # --- LOAD DETECTOR ---
        myWatermark = AutoWatermark.load(
            'SIR',
            algorithm_config='config/SIR.json',
            transformers_config=transformers_config,
            delta=delta,
            # hash_key=hash_key
        )

        # Detect on Attacked Text
        detect_result = myWatermark.detect_watermark(attacked_text)
        new_score = detect_result.get('score', None)

        num_total += 1
        if new_score is None or not np.isfinite(new_score):
            num_failed += 1
            continue
        # Store for aggregation
        attacked_scores.append(new_score)
        original_nwm_scores.append(sample['nwm_score'])

        results_list.append({
            "prompt": prompt,
            "attacked_text": attacked_text,
            "original_wm_score": sample['wm_score'],
            "attacked_wm_score": new_score,
            "is_detected": detect_result['is_watermarked']
        })

    # 4. Calculate New Metrics
    # ROC AUC: compares Attacked Watermarked vs Original Non-Watermarked
    # We want to see if we can still distinguish attacked text from natural text.
    y_true = [1] * len(attacked_scores) + [0] * len(original_nwm_scores)
    y_scores = attacked_scores + original_nwm_scores
    
    new_roc_auc = roc_auc_score(y_true, y_scores)
    
    avg_original_score = np.mean([r['original_wm_score'] for r in results_list])
    avg_attack_score = np.mean(attacked_scores)
    detection_rate = np.mean([r['is_detected'] for r in results_list])

    print("\n" + "="*40)
    print(f"Attack Analysis Results (delta={delta})")
    print("="*40)
    print(f"Original Avg Z-Score:  {avg_original_score:.4f}")
    print(f"Attacked Avg Z-Score:  {avg_attack_score:.4f}")
    print(f"Attacked ROC AUC:      {new_roc_auc:.4f}")
    print(f"Attacked Detection Rate: {detection_rate:.2%}")
    print("="*40)

    print("\n" + "Total:",num_total)
    print("Failed:", num_failed)
    # 5. Save Output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"analysis_SIR_{delta}.json")
    
    final_data = {
        "delta": delta,
        "metrics": {
            "roc_auc": new_roc_auc,
            "avg_z_score_drop": avg_original_score - avg_attack_score,
            "detection_rate": detection_rate
        },
        "samples": results_list
    }

    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Saved analysis to {output_path}")

if __name__ == "__main__":
    
    logging.set_verbosity_error()
    warnings.filterwarnings(
        "ignore",
        message="Some weights of BertModel were not initialized"
    )

    main()
