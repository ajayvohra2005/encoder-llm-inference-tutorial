
## Overview

This tutorial is for testing inference with Encoder LLM models using Triton Inference Server.

## Tutorial Steps

### Step 1. Launch Deep Learning Ubuntu Desktop

This tutorial assumes a `trn1.32xlarge` machine for Neuron examples, and a `g5.48xlarge` for CUDA examples. You may want to launch the [AWS Deep Learning Desktop](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop) with  `trn1.32xlarge` instance type for neuron, or `g5.48xlarge` for gpu.

### Step 2. Get Hugging Face Token For Gated Models

To access [Hugging Face gated models](https://huggingface.co/docs/hub/en/models-gated), get a Hugging Face token. You will need to specify it in the `HF_TOKEN` below. 

### Step 3. Build container

To build the container for triton inference server with neuronx, execute this on `trn1.32xlarge` machine:

    ./scripts/build-tritonserver-neuronx.sh
    
### Step 4. Run Testing

### Triton Inference Server


To launch:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/torch-neuronx/compose-triton-torch-neuronx.sh up

To stop the server:

    HF_TOKEN=your-token MODEL_ID=hf-model-id \
        ./triton-server/torch-neuronx/compose-triton-torch-neuronx.sh down
        
### Optonal Environment Variables

You may set following optional environment variables when launching the server above. 

The default values may not provide the best-performance for your given machine. Please experiment with different values of MODEL_SERVER_CORES ( must be set <= number of cores on the machine).

 Unless your model is using Tensor Parallelism, NEURON_RT_NUM_CORES must always be set to 1.

| Name      | Default Value | Semantics |
| ----------- | ----------- | ----------- |
| MODEL_SERVER_CORES      | Number of cores on machine       | Number of Triton Server instances = Number-of-cores-on-machine // MODEL_SERVER_CORES       |
| NEURON_RT_NUM_CORES   | 1        | Number of model instances within a Triton Inference Server = MODEL_SERVER_CORES // NEURON_RT_NUM_CORES |