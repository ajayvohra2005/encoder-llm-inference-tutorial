import json
import os
import time

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch_xla.core.xla_model as xm
import math
import numpy as np
import itertools
import torch.nn.functional as F
import asyncio, threading

import triton_python_backend_utils as pb_utils

_MODEL_ARGS_FILENAME = "model.json"

class TritonPythonModel:

    def initialize(self, args):

        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        logits_config = pb_utils.get_output_config_by_name(self.model_config, "logits")
        self.logits_dtype = pb_utils.triton_string_to_numpy(logits_config["data_type"])
        self.example_text = 'The giant panda, sometimes called a panda bear, or simply panda, is a bear species endemic to China.'
        self._init_service()

        self.logger.log_info("TritonPythonModel initialized")

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        
        inputs = [{"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]}]
        outputs = [{"name": "logits", "data_type": "TYPE_FP32", "dims": [-1]}]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_model_transaction_policy(dict(decoupled=False))
        auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config

    @staticmethod
    def powers_of_2(n:int) -> int:
        return [2**i for i in range(int(math.log2(n))+1)]

    @staticmethod
    def min_power_of_2(n:int) -> int:
        return 2**math.ceil(math.log2(n))

    def _get_bucket_batch_size(self, n:int) -> int:
        assert n > 0, f"batch_size {n} is not > 0"
        n = self.min_power_of_2(n)
        for bs in self.bucket_batch_size:
            if bs >= n:
                return bs
            
        return self.max_batch_size

    def _get_bucket_seq_len(self, n:int) -> int:
        n = self.min_power_of_2(n)
        for seq_len in self.bucket_seq_len:
            if seq_len >= n:
                return seq_len
            
        return self.max_seq_len

    @staticmethod
    def unpad_tensor(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
        return tensor[tensor != pad_value]

    def _bucket_batch_inference(self, inputs: dict) -> list:  
        with torch.no_grad():
            inputs.to(xm.xla_device())
            logits = self.model(**inputs, return_dict=True).logits.detach().cpu().numpy()
            return logits
            
    def _run_inference(self, texts: list) -> list:
        # Assumption tokenizer.pad_token value is 1
        start = time.time()
        pad_value = 1
        input_batch_size = len(texts)
        assert input_batch_size <= self.max_batch_size, f"input_batch_size: {input_batch_size}  is > max_batch_size: {self.max_batch_size}"
        pad_batch_size = self._get_bucket_batch_size(input_batch_size)

        texts.extend([ self.example_text for _ in range(pad_batch_size - input_batch_size) ] )
        inputs = self.tokenizer(texts, padding="longest", truncation=True, return_tensors='pt', max_length=self.max_seq_len)
        input_ids = torch.split(inputs['input_ids'], 1, dim=0)

        ragged_input_ids = [ self.unpad_tensor(tensor, pad_value) for tensor in input_ids ]

        input_seq_len = inputs['input_ids'].shape[-1]
        pad_seq_len = self._get_bucket_seq_len(input_seq_len)
        padding = pad_seq_len - input_seq_len
        inputs['input_ids'] = F.pad(inputs['input_ids'], (0, padding), 'constant', pad_value)
        inputs['attention_mask'] = F.pad(inputs['attention_mask'], (0, padding), 'constant', 0)
        
        logits  = self._bucket_batch_inference(inputs)
        logits = logits[:input_batch_size].tolist()
        logits = [ tensor[:ragged_input_ids[i].shape[0]] for i,tensor in enumerate(logits) ]
        int_time = time.time() - start
        self.logger.log_info(f"Model input_batch_size: {input_batch_size} input_seq_len: {input_seq_len}, inference time: {int_time}")
        assert len(logits) == input_batch_size, f"num logits {len(logits)} != batch_size: {input_batch_size}"
        return logits

    def _compile_model(self):
        permutations = list(itertools.product(self.bucket_batch_size, self.bucket_seq_len))
        for batch_size,seq_len in permutations:
            self.logger.log_info(f"Compiling model for batch size: {batch_size}, seq length {seq_len}")
            texts = [ self.example_text ] * batch_size
            inputs = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt', max_length=seq_len)   
            self._bucket_batch_inference(inputs)

    def _init_service(self):

        max_batch_size = int(self.model_config.get('max_batch_size', 8))
        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config) 
        assert (
            not using_decoupled 
        ), "Triton Server Python backend must not use decoupled model transaction policy"

        model_args_filepath = os.path.join( 
            pb_utils.get_model_dir(), _MODEL_ARGS_FILENAME
        )
        assert os.path.isfile(
            model_args_filepath
        ), f"'{_MODEL_ARGS_FILENAME}' containing model args must be provided in '{pb_utils.get_model_dir()}'"
        with open(model_args_filepath) as file:
            properties = json.load(file)

        self.bucket_batch_size = list(properties.get("bucket_batch_size", [1,2,4,8]))
        self.bucket_batch_size.sort()

        for bss in self.bucket_batch_size:
            assert (bss & (bss-1) == 0), f"bucket batch size {bs} is not power of 2"
        self.max_batch_size = max(self.bucket_batch_size)

        self.bucket_seq_len = list(properties.get("bucket_seq_len", [32,64,128]))
        self.bucket_seq_len.sort()

        for bsl in self.bucket_seq_len:
            assert (bsl & (bsl-1) == 0), f"bucket seq len {bsl} is not power of 2"
        self.max_seq_len = max(self.bucket_seq_len)

        assert ( self.max_batch_size == max_batch_size), \
        f"Triton Server max_batch_size {max_batch_size}  is not equal to model max_batch_size: {self.max_batch_size}"

        assert ( self.max_batch_size >= 1 and self.max_batch_size <= 8), \
        "max_batch_size {self.max_batch_size}  is not between 1 and 8"

        model_location = properties.get("model_id_or_path")
        self.tokenizer = AutoTokenizer.from_pretrained(model_location)
        self.model = AutoModelForMaskedLM.from_pretrained(model_location)

        self.model.eval()
        
        self.logger.log_info(f"Move model to device")
        path = os.getcwd()
        os.chdir("/tmp")
        
        self.model.to(xm.xla_device())
        self._compile_model()
        
        os.chdir(path)
        self.logger.log_info("Exit: load_model")


    def execute(self, requests):
        responses = []
           
        texts = []
        for request in requests:
            inputs = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy().tolist()
            text = [ input[0].decode("utf-8") if isinstance(input[0], bytes) else input[0] for input in inputs]
            assert len(text) == 1
            texts.append(text[0])
            if texts:
                logits = self._run_inference(texts)
                output_tensors = [ pb_utils.Tensor("logits", np.array(result).astype(self.logits_dtype)) for result in logits ]
                inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
                responses.append(inference_response)

        assert len(responses) == len(requests)
        return responses
    
    def finalize(self):
        self.logger.log_info("Cleaning up...")