import argparse
import os
import random
import time
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import logging
import torch_xla.core.xla_model as xm
import math
import itertools
import torch.nn.functional as F

os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --enable-fast-loading-neuron-binaries"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NEURON_RT_VISIBLE_CORES'] = "0"
os.environ["NEURON_COMPILE_CACHE_URL"] = "/tmp/neuronx_cache"

def powers_of_2(n):
    return [2**i for i in range(int(math.log2(n))+1)]

def min_power_of_2(n):
    return 2**math.ceil(math.log2(n))

def get_bucket_batch_size(n):
    assert n > 0, f"batch_size {n} is not > 0"
    n = min_power_of_2(n)
    for bs in bucket_batch_size:
        if bs >= n:
            return bs
        
    return max_batch_size

def get_bucket_seq_len(n):
    n = min_power_of_2(n)
    for seq_len in bucket_seq_len:
        if seq_len >= n:
            return seq_len
        
    return max_seq_len

def unpad_tensor(tensor, pad_value):
    return tensor[tensor != pad_value]

def compile_model():
    permutations = list(itertools.product(bucket_batch_size, bucket_seq_len))
    for batch_size,seq_len in permutations:
        print(f"Compiling model for batch size: {batch_size}, seq length {seq_len}")
        texts = [ example_text ] * batch_size
        inputs = tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt', max_length=seq_len)   
        _bucket_batch_inference(inputs)

def load_model(properties):
    global model, tokenizer
    global bucket_batch_size
    global bucket_seq_len 
    global max_batch_size
    global max_seq_len
    global example_text
    
    logging.info("Enter: load_model")
    logging.info(f"properties: {properties}")
    
    example_text = 'The giant panda is a bear species endemic to China.'
    
    # model location on the serving host
    model_location = properties.get("model_id")
    bucket_batch_size = list(properties.get("bucket_batch_size"))
    bucket_batch_size.sort()

    for bss in bucket_batch_size:
        assert (bss & (bss-1) == 0), f"bucket batch size {bs} is not power of 2"
    max_batch_size = max(bucket_batch_size)
  
    bucket_seq_len = list(properties.get("bucket_seq_len"))
    bucket_seq_len.sort()

    for bsl in bucket_seq_len:
        assert (bsl & (bsl-1) == 0), f"bucket seq len {bsl} is not power of 2"
    max_seq_len = max(bucket_seq_len)
  
    
    logging.info(f"Creating model and tokenizer using: {model_location}")
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    model = AutoModelForMaskedLM.from_pretrained(model_location)
    model.eval()

    logging.info(f"Move model to device")
    path = os.getcwd()
    os.chdir("/tmp")
    
    model.to(xm.xla_device())
    compile_model()
    
    os.chdir(path)
    logging.info("Exit: load_model")

def _bucket_batch_inference(inputs: dict) -> list:  
    with torch.no_grad():
        inputs.to(xm.xla_device())
        logits = model(**inputs, return_dict=True).logits.detach().cpu().numpy()
        return logits
        
def run_inference(texts: list):
    # Assumption tokenizer.pad_token value is 1

    input_batch_size = len(texts)
    assert input_batch_size <= max_batch_size, f"batch_size: {input_batch_size}  is > max_batch_size: {max_batch_size}"
    pad_batch_size = get_bucket_batch_size(input_batch_size)

    texts.extend([ example_text for _ in range(pad_batch_size - input_batch_size) ] )
    inputs = tokenizer(texts, padding="longest", truncation=True, return_tensors='pt', max_length=max_seq_len)

    input_ids = torch.split(inputs['input_ids'], 1, dim=0)

    ragged_input_ids = [ unpad_tensor(tensor, 1) for tensor in input_ids ]

    input_seq_len = inputs['input_ids'].shape[-1]
    pad_seq_len = get_bucket_seq_len(input_seq_len)
    padding = pad_seq_len - input_seq_len
    inputs['input_ids'] = F.pad(inputs['input_ids'], (0, padding), 'constant', 1)
    inputs['attention_mask'] = F.pad(inputs['attention_mask'], (0, padding), 'constant', 0)
    
    logits  = _bucket_batch_inference(inputs)
    logits = logits[:input_batch_size].tolist()
    logits = [ tensor[:ragged_input_ids[i].shape[0]] for i,tensor in enumerate(logits) ]

    return logits

def handle(properties: dict, data: dict={}):
    """
    inputs: Contains the configurations from serving.properties
    """

    if os.getenv("MODEL_LOADED", None) != "true":
        load_model(properties)
        os.environ["MODEL_LOADED"] = "true"

    if not data:
        return {}
    
    texts = data["inputs"]

    logits = run_inference(texts)
    return logits

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='test_encoder_llm_inference',
                    description='Test encoder LLM inference',
                    epilog='Text at the bottom of help')
    
    parser.add_argument("--model-id",  type=str, help="Model id", default='FacebookAI/xlm-roberta-base')
    args = parser.parse_args()

    properties = dict()
    properties['model_id'] = args.model_id
    properties['bucket_batch_size'] = [1,2,4,8]
    properties['bucket_seq_len'] = [16,32,64,128]

    handle(properties=properties)
    
    data = {}
    example_texts = [
                    "Batched inputs are often different lengths, so they can’t be converted to fixed-size tensors. Padding and truncation are strategies for dealing with this problem, to create rectangular tensors from batches of varying lengths. Padding adds a special padding token to ensure shorter sequences will have the same length as either the longest sequence in a batch or the maximum length accepted by the model. Truncation works in the other direction by truncating long sequences.",
                    "In most cases, padding your batch to the length of the longest sequence and truncating to the maximum length a model can accept works pretty well. However, the API supports more strategies if you need them. The three arguments you need to are: padding, truncation and max_length.",
                    "See torch.nn.CircularPad2d, torch.nn.ConstantPad2d, torch.nn.ReflectionPad2d, and torch.nn.ReplicationPad2d for concrete examples on how each of the padding modes works. Constant padding is implemented for arbitrary dimensions. Circular, replicate and reflection padding are implemented for padding the last 3 dimensions of a 4D or 5D input tensor, the last 2 dimensions of a 3D or 4D input tensor, or the last dimension of a 2D or 3D input tensor.",
                    "Apple Inc. is a technology company headquartered in Cupertino, California. "
                    "Named for a local creek by Spanish explorer Juan Bautista de Anza's cartographer bearing the name of Saint Joseph of Cupertino, Cupertino was officially incorporated in 1955, though it saw economic activity in the early 19th century. " ,
                    "The company was founded to produce and market personal computer. Its second computer, the Apple II, became a best seller as one of the first mass-produced microcomputers.",
                    "It takes two arguments, a and b, representing the lower and upper bounds of the range, respectively.",
                    "random.randint() is a function in Python's random module used to generate a random integer within a specified range (inclusive of both the start and end points). It takes two arguments, a and b, representing the lower and upper bounds of the range, respectively. It returns a randomly selected integer N such that a <= N <= b."
                    "The randint Python function is a built-in method that lets you generate random integers using the random module."
    ]
    n_texts = len(example_texts)
    avg = 0.0
    for i in range(128):
        bs = random.randint(1, max_batch_size)
        data['inputs'] = [ example_texts[random.randint(0, n_texts-1)] for _ in range(bs) ]
        start = time.time()
        logits = handle(properties=properties, data=data)
        inf_time = time.time() - start
        print(f"inference: batch size {bs}, time: {inf_time}")
        avg = (avg*i+inf_time)/(i+1)

    print(f"Average latency time: {avg} seconds")
    