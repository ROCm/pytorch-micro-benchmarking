#!/bin/bash

model=$1
tag=$2
iter=1
if [[ $3 != "" ]]; then
    iter=$3
fi
DATE=$(date +%y%m%d-%H%M%S)
outdir=perf_run_${model}_${tag}_${DATE}
mkdir ${outdir}

function run_pyt_cnn() {
    model=$1
    gpus_per_node=$2
    batch_per_gpu=$3
    use_amp=$4
    use_horovod=$5
    tag=$6
    LOG=${outdir}/run_${model}_gbs$((gpus_per_node*batch_per_gpu))_${gpus_per_node}GPUs_${tag}
    DATE=$(date +%y%m%d-%H%M%S)

    cmd="micro_benchmarking_pytorch.py --model ${model} --distributed_dataparallel --batch-size ${batch_per_gpu} --iterations 100 --dist-backend nccl"
    if [[ ${use_horovod} == "true" ]]; then
        cmd="horovodrun -np ${gpus_per_node} python3 ${cmd} --horovod"
        LOG+="_hvd"
    else
        cmd="python3 -u -m multiproc --nproc_per_node ${gpus_per_node} --nnodes 1 ${cmd}"
    fi
    if [[ ${use_amp} == "true" ]]; then
        cmd+=" --amp-opt-level 1"
        LOG+="_amp"
    fi
    LOG+="_${DATE}.log"
    echo ${cmd} | tee ${LOG}
    ${cmd} 2>&1 | tee -a ${LOG}
}

for bs_per_gpu in `echo 128 256 512`; do
    for gpus_per_node in `echo 8 2`; do
        for i in `seq 1 ${iter}`; do
            run_pyt_cnn ${model} ${gpus_per_node} ${bs_per_gpu} true false ${tag}
        done
    done
done

for bs_per_gpu in `echo 128 256`; do
    for gpus_per_node in `echo 8 2`; do
        for i in `seq 1 ${iter}`; do
            run_pyt_cnn ${model} ${gpus_per_node} ${bs_per_gpu} false false ${tag}
        done
    done
done