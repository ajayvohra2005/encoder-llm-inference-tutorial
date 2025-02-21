import os
import time
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.xla_backend as xb
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
model.eval()
model.to(device=get_device())

# Traces the forward method and constructs a `ScriptModule`
texts = ["The giant panda is a bear species endemic to China."]
# prepare input
encoded_input = tokenizer(texts, return_tensors='pt', padding="max_length", truncation=True, max_length=128)
encoded_input = encoded_input.to(device=get_device())

avg_time1 = 0.0

for i in range(128):
    start = time.time()
    output = model(**encoded_input)
    if xm:
        xm.mark_step()
    end = time.time()
    print(f"logits: {output['logits']}, time: {end-start}")
    avg_time1 = ((avg_time1*i) + (end-start))/(i+1)

if not os.path.exists("model-cuda.pt"):
    trace = torch.jit.trace(model, strict=False, example_kwarg_inputs=dict(encoded_input))
    print(f"Torch Script graph: {trace.graph}")
    torch.jit.save(trace, 'model-cuda.pt')

model = torch.jit.load('model-cuda.pt')

avg_time = 0.0
for _ in range(128):
    start = time.time()
    output = model(**encoded_input)
    end = time.time()
    print(f"logits: {output['logits']}, time: {end-start}")
    avg_time = ((avg_time*i) + (end-start))/(i+1)

print(f"Avg time: Non-traced Model : {avg_time1} TorchScript Traced Model: {avg_time}")