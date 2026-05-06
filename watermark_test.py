import torch
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transformers config
model_name = "google-t5/t5-base"
transformers_config = TransformersConfig(model=AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device),
                                         tokenizer=AutoTokenizer.from_pretrained(model_name),
                                       #   vocab_size=50272,
                                         device=device,
                                         max_new_tokens=200,
                                         min_length=128,
                                         do_sample=True,
                                         
                                         no_repeat_ngram_size=4,
                                       )
  
# Load watermark algorithm
myWatermark = AutoWatermark.load('KGW', 
                                 algorithm_config='config/KGW.json',
                                 transformers_config=transformers_config,
                                 gamma=0.5,
                                 hash_key=675787149
                                )

# Prompt
prompt = 'Summary: MarkLLM is an open-source toolkit developed to facilitate the research and application of watermarking technologies within large language models (LLMs). As the use of large language models (LLMs) expands, ensuring the authenticity and origin of machine-generated text becomes critical. MarkLLM simplifies the access, understanding, and assessment of watermarking technologies, making it accessible to both researchers and the broader community.'

# Generate and detect
watermarked_text = myWatermark.generate_watermarked_text(prompt)
print("Watermarked text:")
print(watermarked_text)

detect_result = myWatermark.detect_watermark(watermarked_text)
print(detect_result)

unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
print("Unwatermarked text:")
print(unwatermarked_text)

detect_result = myWatermark.detect_watermark(unwatermarked_text)
print(detect_result)


