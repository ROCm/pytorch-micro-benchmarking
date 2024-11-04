# pytorch-micro-benchmarking
We supply a small microbenchmarking script for PyTorch training on ROCm.

To execute:
`python micro_benchmarking_pytorch.py --network <network name> [--batch-size <batch size> ] [--iterations <number of iterations>] [--fp16 <0 or 1> ] `

Possible network names are: `alexnet`, `densenet121`, `inception_v3`, `resnet50`, `resnet101`, `SqueezeNet`, `vgg16` etc.

Default are 10 training iterations, `fp16` off (i.e., 0), and a batch size of 64.

For mGPU runs, use `torchrun` for best performance. It will spawn multiple sub-processes for each of the GPUs and adjust world_size and rank accordingly. `torchrun` also defaults to using distributed dataprallel.
_NOTE_: `--dataprallel` is deprecated. `--distributed_dataprallel` option will also be deprecated as this path can be exercised now with `torchrun`.

Eg. 
for a 1-GPU resnet50 run:
```
python3 micro_benchmarking_pytorch.py --network resnet50
```
for a 2-GPU run on a single node, use `torchrun` to spawn multiple sub-proceses (one for each GPU). 
```
torchrun --nproc-per-node 8 micro_benchmarking_pytorch.py --network resnet50

```

To run FlopsProfiler (with deepspeed.profiling.flops_profiler imported):
`python micro_benchmarking_pytorch.py --network resnet50 --amp-opt-level=2 --batch-size=256 --iterations=20 --flops-prof-step 10`

## Performance tuning
If performance on a specific card and/or model is found to be lacking, typically some gains can be made by tuning MIOpen. For this, `export MIOPEN_FIND_ENFORCE=3` prior to running the model. This will take some time if untuned configurations are encountered and write to a local performance database. More information on this can be found in the [MIOpen documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/perfdatabase.html).

## PyTorch 2.0
Added the `--compile` option opens up PyTorch 2.0 capabilities, which comes with several options. Here are some notes from upstream: 
```
    Optimizes given model/function using TorchDynamo and specified backend.

    Args:
       model (Callable): Module/function to optimize
       fullgraph (bool): Whether it is ok to break model into several subgraphs
       dynamic (bool): Use dynamic shape tracing
       backend (str or Callable): backend to be used
       mode (str): Can be either "default", "reduce-overhead" or "max-autotune"
       options (dict): A dictionary of options to pass to the backend.
       disable (bool): Turn torch.compile() into a no-op for testing

    Example::

        @torch.compile(options={"matmul-padding": True}, fullgraph=True)
        def foo(x):
            return torch.sin(x) + torch.cos(x)
```

With the required `--compile` option, these additional options are now available from the command line with the `--compileContext` flag. Here are a few examples:

```bash
python micro_benchmarking_pytorch.py --network resnet50 --compile # default run
```

```bash
python micro_benchmarking_pytorch.py --network resnet50 --compile --compileContext "{'mode': 'max-autotune', 'fullgraph': 'True'}"
```

```bash
python micro_benchmarking_pytorch.py --network resnet50 --compile --compileContext "{'options': {'static-memory': 'True', 'matmul-padding': 'True'}}"
```
Note: you cannot pass the `mode` and `options` options together.
