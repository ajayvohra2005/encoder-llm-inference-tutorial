import math
import os
import time
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch_xla.core.xla_model as xm

os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --enable-fast-loading-neuron-binaries"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NEURON_RT_VISIBLE_CORES'] = "0"
os.environ["NEURON_COMPILE_CACHE_URL"] = "/tmp/xla_cache/"

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
model.eval()
model.to(xm.xla_device())

# prepare input
text = ["The giant panda is a bear species endemic to China."]
encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", truncation=True, max_length=128)
encoded_input.to(xm.xla_device())
# forward pass

for _ in range(128):
    start = time.time()
    output = model(**encoded_input)
    xm.mark_step()
    end = time.time()
    print(f"time: {end - start} {output.logits}")