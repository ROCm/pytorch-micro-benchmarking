#!/bin/bash

# run model ops in fp32, fp16 and bf16
echo -e "########## Running model ops with fp32 type ###########"
./torch_tensor_ops_bench.py --run-model-ops --dtype fp32 |& tee model_ops_fp32.log
echo -e "\n########## Running model ops with fp16 type ###########"
./torch_tensor_ops_bench.py --run-model-ops --dtype fp16 |& tee model_ops_fp16.log
echo -e "\n########## Running model ops with bf16 type ###########"
./torch_tensor_ops_bench.py --run-model-ops --dtype bf16 |& tee model_ops_bf16.log

# run predefined ops with generic tensor size of 64-1024-1024
echo -e "\n########## Running pre-defined ops with fp32 type ###########"
./torch_tensor_ops_bench.py --run-predefined --dtype fp32 |& tee predefined_ops_fp32.log
echo -e "\n########## Running pre-defined ops with fp16 type ###########"
./torch_tensor_ops_bench.py --run-predefined --dtype fp16 |& tee predefined_ops_fp16.log
