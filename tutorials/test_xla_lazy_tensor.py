
import os
import time
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend as xb
    import torch_xla
except ImportError:
    xm = None
    xr = None
    xb = None

def get_device() -> torch.device:
    if xm:
        return xm.xla_device()
    elif torch.cuda.is_available():
        __current_device = torch.device('cuda:0')
        torch.cuda.set_device(__current_device)
        return __current_device
    else:
        return torch.device("cpu")
    
os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --enable-fast-loading-neuron-binaries"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NEURON_RT_VISIBLE_CORES'] = "0"
os.environ["NEURON_COMPILE_CACHE_URL"] = "/tmp/xla_cache/"

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
model.eval()
model.to(get_device())

# prepare input
text = ["The giant panda is a bear species endemic to China."]
encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", truncation=True, max_length=128)
encoded_input.to(get_device())
# forward pass

avg_time = 0.0
for i in range(128):
    start = time.time()
    output = model(**encoded_input)
    # print(torch_xla._XLAC._get_xla_tensors_text([output['logits']]))
    if xm:
        xm.mark_step()
    # print(torch_xla._XLAC._get_xla_tensors_text([output['logits']]))
    end = time.time()
    print(f"logits: {output['logits']}, time: {end-start}")
    avg_time = ((avg_time*i) + (end-start))/(i+1)

print(f"Avg time: {avg_time}")