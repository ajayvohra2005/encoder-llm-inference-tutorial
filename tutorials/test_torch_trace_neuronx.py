import os
import time
import torch
import torch_neuronx
from transformers import AutoModelForMaskedLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NEURON_RT_VISIBLE_CORES'] = "0"

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
model.eval()
# Traces the forward method and constructs a `ScriptModule`
texts = ["The giant panda is a bear species endemic to China."]
# prepare input
encoded_input = tokenizer(texts, return_tensors='pt', padding="max_length", truncation=True, max_length=128)
inputs = tuple(encoded_input.values())
if not os.path.exists("model-neuron.pt"):
    trace = torch_neuronx.trace(model, example_inputs=inputs, compiler_workdir="/tmp/trace_workdir", 
                                compiler_args="--target trn1 --model-type=transformer --enable-fast-loading-neuron-binaries")
    torch.jit.save(trace, 'model-neuron.pt')
# Executes on a NeuronCore
model = torch.jit.load('model-neuron.pt')

avg_time = 0.0
for i in range(128):
    start = time.time()
    output = model(*inputs)
    end = time.time()
    print(f"time: {end - start} {output['logits']}")
    avg_time = ((avg_time*i) + (end-start))/(i+1)

print(f"Avg time: {avg_time}")