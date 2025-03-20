#!/bin/bash

[ ! -d /cache ] && echo "/cache dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[  -z "$MODEL_ID_OR_PATH"  ] && echo "MODEL_ID_OR_PATH environment variable must exist" && exit 1

NUM_CORES=$(($(neuron-ls -j | grep neuron_device | wc -l)*2))
: ${MODEL_SERVER_CORES:=$NUM_CORES}
: ${NEURON_RT_NUM_CORES:=1}
: ${OMP_NUM_THREADS:=16}

CACHE_DIR=/cache
MODEL_COUNT=$(( $MODEL_SERVER_CORES / $NEURON_RT_NUM_CORES ))
cat > /tmp/config.pbtxt <<EOF
  backend: "python"
  max_batch_size: 8
  model_transaction_policy {
    decoupled: false
  }
  dynamic_batching {
    max_queue_delay_microseconds: 1000
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
      count: $MODEL_COUNT
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
cp /scripts/triton_python_model.py $MODEL_REPO/$MODEL_NAME/$VERSION/model.py
cp /tmp/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /tmp/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt
export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export FI_EFA_FORK_SAFE=1
export NEURON_RT_NUM_CORES
export OMP_NUM_THREADS
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"