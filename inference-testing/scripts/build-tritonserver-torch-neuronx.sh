#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/tritonserver-torch-neuronx
docker buildx build -t tritonserver-torch-neuronx:latest .