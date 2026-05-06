import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
from summa import keywords
import hashlib
from tqdm import tqdm

from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from visualize.font_settings import FontSettings
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.visualizer import DiscreteVisualizer
from watermark.auto_config import AutoConfig
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

import gc
import sys
import json
import torch
from watermark.auto_watermark import AutoWatermarkForVLLM
from utils.transformers_config import TransformersConfig
from sklearn.metrics import roc_auc_score
from evaluation.tools.text_quality_analyzer import BLEUCalculator

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, required=True)
parser.add_argument("--delta", type=float, required=True)
parser.add_argument("--output_dir", type=str, default="results")
args = parser.parse_args()

gamma = args.gamma
delta = args.delta


def hash_ids(ids, seed=0, mod=None):
    s = ",".join(map(str, ids)) + f":{seed}"
    h = int(hashlib.sha256(s.encode()).hexdigest(), 16)
    return h if mod is None else h % mod

# Parameters
# KGW
# gamma=0.5
# delta=0.1
# gamma_list = [0.1, 0.25, 0.5, 0.9]
# delta_list = [0.1, 0.5, 1.0, 2.0, 5.0]
all_results = []

# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()

# Load data
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
    lines = [json.loads(line) for line in lines[:5000]]

# Transformers config
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

# prompts = [line['prompt'] for line in lines]
# references = [line['natural_text'] for line in lines]

# print(f"\n===== Running gamma={gamma}, delta={delta} =====")

nowatermark_text=[]
watermark_text=[]
detect_results_watermark = []
detect_results_nowatermark = []
BLEU_watermark = []
BLEU_nowatermark = []
# BLEU_delta = []
BLEU_wm_nwm = []

with torch.no_grad():
    for i, line in enumerate(tqdm(lines, desc=f"γ={gamma}, δ={delta}", unit="line")):

        prompt = line['prompt']
        reference = line['natural_text']


        token = keywords.keywords(prompt, ratio=0.3)
        ids = tokenizer.encode(token, add_special_tokens=False)

        hash_key = hash_ids(ids, len(prompt), 998244353)
        myWatermark = AutoWatermark.load(
            'KGW',
            algorithm_config='config/KGW.json',
            transformers_config=transformers_config,
            gamma=gamma,
            delta=delta,
            hash_key=hash_key
        )

        # ---- Watermarked ----
        wm_text = myWatermark.generate_watermarked_text(prompt)
        watermark_text.append(wm_text)
        detect_results_watermark.append(myWatermark.detect_watermark(wm_text))

        # ---- Non-watermarked ----
        nwm_text = myWatermark.generate_unwatermarked_text(prompt)
        nowatermark_text.append(nwm_text)
        detect_results_nowatermark.append(myWatermark.detect_watermark(nwm_text))

        # BLEU
        bleu_calculator = BLEUCalculator()

        BLEU_watermark.append(bleu_calculator.analyze(wm_text,reference))
        BLEU_nowatermark.append(bleu_calculator.analyze(nwm_text,reference))
        BLEU_wm_nwm.append(bleu_calculator.analyze(wm_text,nwm_text))

        # if i % 5 == 0:
        #     tqdm.write(
        #         f"[{i}] len={len(prompt)} "
        #         # f"wm_score={score_watermark:.4f} "
        #         # f"nwm_score={score_nowatermark:.4f}"
        #         # f"\nwatermark_text={wm_text}"
        #         # f"\nnowatermark_text={nwm_text}"
        #     )

# nowatermark_ppl = np.mean([-output.outputs[0].cumulative_logprob/len(output.outputs[0].token_ids) for output in outputs])
# nowatermark_detect_results = np.mean([r['is_watermarked'] for r in detect_results_nowatermark])
# print(f"nowatermark_ppl: {nowatermark_ppl:.3f}")
# print(f"nowatermark_detect_results: {nowatermark_detect_results:.3f}")


# watermark_ppl = np.mean([-output.outputs[0].cumulative_logprob/len(output.outputs[0].token_ids) for output in outputs])
# watermark_detect_results = np.mean([r['is_watermarked'] for r in detect_results_watermark])
# print(f"watermark_ppl: {watermark_ppl:.3f}")
# print(f"watermark_detect_results: {watermark_detect_results:.3f}")

wm_detect_rate = np.mean([r['is_watermarked'] for r in detect_results_watermark])
nwm_detect_rate = np.mean([r['is_watermarked'] for r in detect_results_nowatermark])

# ROC AUC
y_true = (
    [1] * len(detect_results_watermark) +
    [0] * len(detect_results_nowatermark)
)

y_score = (
    [r["score"] for r in detect_results_watermark] +
    [r["score"] for r in detect_results_nowatermark]
)

roc_auc = roc_auc_score(y_true, y_score)
# print(f"ROC AUC: {roc_auc:.4f}")

# BLEU
bleu_wm_vs_nwm = float(np.mean(BLEU_wm_nwm))
bleu_wm_ref = float(np.mean(BLEU_watermark))
bleu_nwm_ref = float(np.mean(BLEU_nowatermark))

# print(f"BLEU(watermark vs no-watermark): {bleu_wm_vs_nwm:.2f}")
# print(f"BLEU(watermark vs reference): {bleu_wm_ref:.2f}")
# print(f"BLEU(no-watermark vs reference): {bleu_nwm_ref:.2f}")

experiment_result = {
    "gamma": gamma,
    "delta": delta,
    "metrics": {
        "roc_auc": float(roc_auc),
        "wm_detect_rate": float(wm_detect_rate),
        "nwm_detect_rate": float(nwm_detect_rate),
        "bleu_wm_ref": bleu_wm_ref,
        "bleu_nwm_ref": bleu_nwm_ref,
        "bleu_wm_vs_nwm": bleu_wm_vs_nwm,
    },
    "samples": [
        {
            "prompt": lines[i]["prompt"],
            "reference": lines[i]["natural_text"],
            "watermarked": watermark_text[i],
            "non_watermarked": nowatermark_text[i],
            "wm_score": detect_results_watermark[i]["score"],
            "nwm_score": detect_results_nowatermark[i]["score"],
        }
        for i in range(len(watermark_text))
    ],
}

all_results.append(experiment_result)

os.makedirs("results", exist_ok=True)
results_name = f"T5_C4_5k_{gamma}_{delta}_202601101244"

with open(f"results/{results_name}.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("All results saved to results")
