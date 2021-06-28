# pytorch-micro-benchmarking
We supply a small microbenchmarking script for PyTorch training on ROCm.

To execute:
`python micro_benchmarking_pytorch.py --network <network name> [--batch-size <batch size> ] [--iterations <number of iterations>] [--fp16 <0 or 1> ] [--dataparallel|--distributed_dataparallel] [--device_ids <comma separated list (no spaces) of GPU indices (0-indexed) to run dataparallel/distributed_dataparallel api on>]`

Possible network names are: `alexnet`, `densenet121`, `inception_v3`, `resnet50`, `resnet101`, `SqueezeNet`, `vgg16` etc.

Default are 10 training iterations, `fp16` off (i.e., 0), and a batch size of 64.

`--distributed_dataparallel` will spawn multiple sub-processes and adjust world_size and rank accordingly. Py3.6 ONLY.

To run FlopsProfiler (with deepspeed.profiling.flops_profiler imported):
`python micro_benchmarking_pytorch.py --network resnet50 --amp-opt-level=2 --batch-size=256 --iterations=20 --flops-prof-step 10`

## Performance tuning
If performance on a specific card and/or model is found to be lacking, typically some gains can be made by tuning MIOpen. For this, `export MIOPEN_FIND_ENFORCE=3` prior to running the model. This will take some time if untuned configurations are encountered and write to a local performance database. More information on this can be found in the [MIOpen documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/perfdatabase.html).

