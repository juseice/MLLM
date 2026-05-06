from summa import keywords
from transformers import AutoTokenizer
import hashlib

def hash_ids(ids, seed=0, mod=None):
    s = ",".join(map(str, ids)) + f":{seed}"
    h = int(hashlib.sha256(s.encode()).hexdigest(), 16)
    return h if mod is None else h % mod

model_name = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# text = 'Summary: MarkLLM is an open-source toolkit developed to facilitate the research and application of watermarking technologies within large language models (LLMs). As the use of large language models (LLMs) expands, ensuring the authenticity and origin of machine-generated text becomes critical. MarkLLM simplifies the access, understanding, and assessment of watermarking technologies, making it accessible to both researchers and the broader community.'
text = 'There are several good decoder-only (causal) language models well under 1B parameters, and many of them are much better suited than BERT if your goal is instruction / generation / chat distillation.'

token = keywords.keywords(text, ratio=0.3)
print(token)

ids = tokenizer.encode(token, add_special_tokens=False)
print(ids)

print(hash_ids(ids,len(text),998244353))
