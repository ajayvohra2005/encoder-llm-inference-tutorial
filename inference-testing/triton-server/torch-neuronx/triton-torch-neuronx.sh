#!/bin/bash

[ ! -d /cache ] && echo "/cache dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[  -z "$MODEL_ID_OR_PATH"  ] && echo "MODEL_ID_OR_PATH environment variable must exist" && exit 1


CACHE_DIR=/cache

cat > /tmp/model.py <<EOF
import json
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import logging
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
        self.__tasks = set()
        self._init_service()
        self.__tasks_inited = False

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
            
    def run_inference(self, texts: list) -> list:
        # Assumption tokenizer.pad_token value is 1
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

        return logits

    def compile_model(self):
        permutations = list(itertools.product(self.bucket_batch_size, self.bucket_seq_len))
        for batch_size,seq_len in permutations:
            self.logger.log_info(f"Compiling model for batch size: {batch_size}, seq length {seq_len}")
            texts = [ self.example_text ] * batch_size
            inputs = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt', max_length=seq_len)   
            self._bucket_batch_inference(inputs)

    def _init_service(self):

        max_batch_size = int(self.model_config.get('max_batch_size', 0))
        assert (
            max_batch_size == 0
        ), "Triton Server model config max_batch_size must be set to 0"

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config) 
        assert (
            using_decoupled 
        ), "Triton Server Python backend must use decoupled model transaction policy"

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
        self.compile_model()
        
        os.chdir(path)
        self.logger.log_info("Exit: load_model")

        self.logger.log_info("Create request asyncio queue: maxsize {self.max_batch_size}")
        self.__request_queue = asyncio.Queue(maxsize=self.max_batch_size)

        self.logger.log_info("Create response asyncio queue: maxsize {self.max_batch_size}")
        self.__response_queue = asyncio.Queue(maxsize=self.max_batch_size)


    async def __init_tasks(self):
        self.logger.log_info("Start respond loop")
        task = asyncio.create_task(self.__respond_loop())
        self.__tasks.add(task)
        task.add_done_callback(self.__tasks.discard)
        
        self.logger.log_info("Start inference loop")
        task = asyncio.create_task(self.__inference_loop())
        self.__tasks.add(task)
        task.add_done_callback(self.__tasks.discard)

        self.__tasks_inited = True

    async def execute(self, requests):
        if not self.__tasks_inited:
            try:
                await self.__init_tasks()
            except KeyError:
                self.logger.log_error("Future not found or has already completed.")

        for request in requests:
            try:
                await self.__request_queue.put(request)
            except KeyError:
                self.logger.log_error("Future not found or has already completed.")

    async def __check_requests(self):
        if len(self.__requests) == 0:
            new_request = await self.__request_queue.get()
            self.__request_queue.task_done()
            self.__requests.append(new_request)

        while len(self.__requests) < self.max_batch_size:
            try:
                await asyncio.sleep(.001) # Adjust based on model latency
                new_request = self.__request_queue.get_nowait()
                self.__request_queue.task_done()
                self.__requests.append(new_request)
            except asyncio.QueueEmpty:
                break

    def __inference(self):

        texts = []
        for request in self.__requests:
            print(f"request: {request}", flush=True)
            inputs = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy().tolist()
            text = [ input.decode("utf-8") if isinstance(input, bytes) else input for input in inputs]

            print(f"text: {text}", flush=True)
            assert len(text) == 1
            texts.append(text[0])
        
        if texts:
            logits = self.run_inference(texts)
            for result in logits:
                self.__results.append(result)

    async def __inference_loop(self):
        while True:
            try:
                await self.__check_requests()
                self.__inference()

                assert len(self.__requests) == len(self.__results), \
                f"requests: {len(self.__requests)} != results: {len(self.__results)}"

                finished = []
                for i in range(len(self.__requests)):
                    res = self.__results[i]
                    req = self.__requests[i]
                    try:
                        self.__response_queue.put_nowait((req, res))
                    except asyncio.QueueFull:
                        self.logger.log_info("response queue is full; await put")
                        await self.__response_queue.put((req, res))
                finished.append((req, res))
                
                for item in finished:
                    req, res = item
                    self.__requests.remove(req)
                    self.__results.remove(res)

                assert len(self.__requests) == len(self.__results), \
                f"requests: {len(self.__requests)} != results: {len(self.__results)}"

            except Exception as e:
                self.logger.log_error(f"Unpexpected error: {e}. Inflight requests discarded. Reset engine.")
                self.reset()

    def reset(self):
        self.__requests = []
        self.__results = []

    async def __respond_loop(self):
        self.reset()

        while True:
            try:
                req, res = await self.__response_queue.get()
                self.__response_queue.task_done()
                
                t = threading.Thread(target=self.__send_response, 
                kwargs={"request": req, "response": res})
                t.start()
            except Exception as e:
                self.logger.log_info(f"Error: respond loop exception {e}")

    def __send_response(self, request, response: list):
        try:
            response_sender = request.get_response_sender()
            
            out_tensor = pb_utils.Tensor("logits", np.array(response).astype(self.logits_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            response_sender.send(inference_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        except Exception as e:
            self.logger.log_info(f"send error: {request}")
            self.logger.log_info(e)

    def finalize(self):
        self.logger.log_info("Cleaning up...")

EOF

cat > /tmp/config.pbtxt <<EOF
  backend: "python"
  max_batch_size: 0
  model_transaction_policy {
    decoupled: true
  }

  input [ 
    {
      name: "text_input"
      data_type: TYPE_STRING
      dims: [1]
    }
  ] 
  output [
    {
      name: "logits"
      data_type: TYPE_FP32
      dims: [-1]
    }
  ]

  instance_group [
    {
      count: 1
      kind: KIND_MODEL
    }
  ]
  
EOF

cat > /tmp/model.json <<EOF
  {
    "model_id_or_path": "$MODEL_ID_OR_PATH",
    "bucket_batch_size": [1,2,4,8],
    "bucket_seq_len": [16,32,64,128]
  }

EOF

export MODEL_REPO=/opt/ml/model/model_repo
mkdir -p $MODEL_REPO
VERSION=1
MODEL_NAME=model
mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION
cp /tmp/model.py $MODEL_REPO/$MODEL_NAME/$VERSION/model.py
cp /tmp/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /tmp/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt
export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export OMP_NUM_THREADS=16
export MODEL_SERVER_CORES=1
export FI_EFA_FORK_SAFE=1
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"