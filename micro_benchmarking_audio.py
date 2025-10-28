import torch
import time
import argparse
import os
import sys
import ast
import copy
import math
import torch.nn as nn
import torch.multiprocessing as mp
from fp16util import network_to_half, get_param_copy
import torch.nn.functional as F
from audio.audio_model import get_network_names, get_network, get_input_type, get_input, get_output_selection
from audio.audio_loss import get_criterion, calculate_loss

try:
    import torch._dynamo
    torch._dynamo.config.verbose=True
    HAVE_DYNAMO = True
except:
    HAVE_DYNAMO = False

IS_PT2 = hasattr(torch, "compile")

is_torchrun = False
if "LOCAL_RANK" in os.environ:
    # this indicates we're using torchrun
    is_torchrun = True

try:
    import apex
    HAVE_APEX = True
except:
    HAVE_APEX = False


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def forwardbackward(inp, optimizer, network, amp_opt_level, network_name, batch_size, criterion, flops_prof_step=0):
    optimizer.zero_grad()
    if flops_prof_step:
        prof = FlopsProfiler(network)
        prof.start_profile()

    out = network(**inp)
    output_index = get_output_selection(network_name) 
    if output_index is not None:
        out = out[output_index]
    
    loss = calculate_loss(network_name, criterion, out)
    
    # End profiler here if only to profile forward pass

    if amp_opt_level:
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    if flops_prof_step:
        prof.print_model_profile(profile_step=flops_prof_step)
        prof.end_profile()

    optimizer.step()

def rendezvous(distributed_parameters):
    print("Initializing process group...")
    torch.distributed.init_process_group(backend=distributed_parameters['dist_backend'], init_method=distributed_parameters['dist_url'], rank=distributed_parameters['rank'], world_size=distributed_parameters['world_size'])
    print("Rendezvous complete. Created process group...")

def run_benchmarking_wrapper(params):
    params.flops_prof_step = max(0, min(params.flops_prof_step, params.iterations - 1))
    if (params.device_ids):
        params.device_ids = [int(x) for x in params.device_ids.split(",")]
    else:
        params.device_ids = None
    params.distributed_parameters = {}
    if is_torchrun:
        params.distributed_parameters['rank'] = int(os.environ["LOCAL_RANK"])
        params.distributed_parameters['world_size'] = int(os.environ["WORLD_SIZE"])
        params.distributed_parameters['dist_backend'] = "nccl"
        params.distributed_parameters['dist_url'] = 'tcp://' + os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"]
    else:
        params.distributed_parameters['rank'] = params.rank
        params.distributed_parameters['world_size'] = params.world_size
        params.distributed_parameters['dist_backend'] = params.dist_backend
        params.distributed_parameters['dist_url'] = params.dist_url

    # Some arguments are required for distributed_dataparallel
    if params.distributed_dataparallel:
        assert params.distributed_parameters['rank'] is not None and \
               params.distributed_parameters['world_size'] is not None and \
               params.distributed_parameters['dist_backend'] is not None and \
               params.distributed_parameters['dist_url'] is not None, "rank, world-size, dist-backend and dist-url are required arguments for distributed_dataparallel"

    if is_torchrun:
        params.ngpus = params.distributed_parameters['world_size']
    elif params.distributed_dataparallel:
        params.ngpus = len(params.device_ids) if params.device_ids else torch.cuda.device_count()
    else:
        params.ngpus = 1

    if is_torchrun:
        run_benchmarking(params.distributed_parameters['rank'], params)
    elif params.distributed_dataparallel:
        # Assumption below that each process launched with --distributed_dataparallel has the same number of devices visible/specified
        params.distributed_parameters['world_size'] = params.ngpus * params.distributed_parameters['world_size']
        params.distributed_parameters['rank'] = params.ngpus * params.distributed_parameters['rank']
        mp.spawn(run_benchmarking, nprocs=params.ngpus, args=(params,))
    else:
        run_benchmarking(0, params)

def run_benchmarking(local_rank, params):
    device_ids = params.device_ids
    ngpus = params.ngpus
    net = params.network
    run_fp16 = params.fp16
    amp_opt_level = params.amp_opt_level
    distributed_dataparallel = params.distributed_dataparallel
    distributed_parameters = params.distributed_parameters
    batch_size = params.batch_size
    kineto = params.kineto
    iterations = params.iterations
    autograd_profiler = params.autograd_profiler
    flops_prof_step = params.flops_prof_step

    if is_torchrun:
        torch.cuda.set_device("cuda:%d" % local_rank)
    elif device_ids:
        assert ngpus == len(device_ids)
        torch.cuda.set_device("cuda:%d" % device_ids[local_rank])
    else:
        torch.cuda.set_device("cuda:0")

    network = get_network(net)
    criterion = get_criterion(net)
    if "shufflenet" == net:
        network.apply(weight_init)

    if (run_fp16):
        network = network_to_half(network)

    if params.compile:
        compile_ctx = {"mode": None,
                       "dynamic": False,
                       "fullgraph": False,
                       "backend": "inductor",
                       "options": None,
                       "disable": False}
        options = None  # needed for internal pytorch checks
        if params.compileContext:
            compile_ctx.update(ast.literal_eval(params.compileContext))
            if compile_ctx["mode"] is not None and compile_ctx["options"] is not None:
                raise RuntimeError("Cannot specify mode and options simultaneously")
            if compile_ctx["options"] is not None:
                options = {}  # needed to save multiple options
                for compiler_pass in compile_ctx["options"].keys():
                    options.update({compiler_pass: bool(compile_ctx["options"][compiler_pass])})
        if IS_PT2:
            network = torch.compile(network,
                                    mode=compile_ctx["mode"],
                                    dynamic=bool(compile_ctx["dynamic"]),
                                    fullgraph=bool(compile_ctx["fullgraph"]),
                                    backend=compile_ctx["backend"],
                                    options=options,
                                    disable=compile_ctx["disable"])
        else:
            print ("ERROR: requested torch.compile but this isn't pytorch 2.x")
            sys.exit(1)

    param_copy = network.parameters()
    if (run_fp16):
        param_copy = get_param_copy(network)
    optimizer = torch.optim.SGD(param_copy, lr = 0.01, momentum = 0.9)

    if (amp_opt_level):
        network, optimizer = apex.amp.initialize(network, optimizer, opt_level="O%d"%amp_opt_level)

    if is_torchrun:
        rendezvous(distributed_parameters)
        devices_to_run_on = [local_rank]
        print ("INFO: Rank {} running distributed_dataparallel on devices: {}".format(distributed_parameters['rank'], str(devices_to_run_on)))
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=devices_to_run_on)
        batch_size = int(batch_size / ngpus)
    elif (distributed_dataparallel):
        distributed_parameters['rank'] += local_rank
        rendezvous(distributed_parameters)
        devices_to_run_on = [(device_ids[local_rank] if device_ids else local_rank)]
        print ("INFO: Rank {} running distributed_dataparallel on devices: {}".format(distributed_parameters['rank'], str(devices_to_run_on)))
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=devices_to_run_on)
        batch_size = int(batch_size / ngpus)

    inp = get_input(net, network, batch_size)
        
    if (run_fp16):
        inp = inp.half()
    
    ## warmup.
    print ("INFO: running forward and backward for warmup.")
    forwardbackward(inp, optimizer, network, amp_opt_level, net, batch_size, criterion)
    forwardbackward(inp, optimizer, network, amp_opt_level, net, batch_size, criterion)

    time.sleep(1)
    torch.cuda.synchronize()

    ## benchmark.
    print ("INFO: running the benchmark..")
    if kineto:
        from torch.profiler import schedule, profile, ProfilerActivity, record_function
        profiler_schedule = schedule(
            skip_first = 0,
            wait = 1,
            warmup = 2,
            active = 2,
            repeat = 1,
        )

        def trace_ready_callback(prof):
            print("----------- Trace Ready -----------")
            prof.export_chrome_trace(f"trace{prof.step_num}.json")

        tm = time.time()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiler_schedule,
            on_trace_ready=trace_ready_callback) as prof:
            for i in range(iterations):
                with record_function(f"iteration {i}"):
                    forwardbackward(inp, optimizer, network, amp_opt_level, net, batch_size, criterion)
                prof.step()
            torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total"))
    else:
        tm = time.time()
        with torch.autograd.profiler.emit_nvtx(enabled=autograd_profiler):
            for i in range(iterations):
                if i == flops_prof_step:
                    forwardbackward(inp, optimizer, network, amp_opt_level, net, batch_size, criterion, i)
                else:
                    forwardbackward(inp, optimizer, network, amp_opt_level, net, batch_size, criterion)
        torch.cuda.synchronize()

    tm2 = time.time()
    time_per_batch = (tm2 - tm) / iterations

    if run_fp16:
        dtype = 'FP16'
    elif amp_opt_level == 1:
        dtype = 'AMP-O1: Insert automatic FP16 casts around safe Pytorch functions and Tensor methods.'
    elif amp_opt_level == 2:
        dtype = 'AMP-O2: FP16 training with FP32 batchnorm and FP32 master weights.'
    elif amp_opt_level == 3:
        dtype = 'AMP-O3: Pure FP16 training.'
    elif amp_opt_level == 4:
        dtype = 'AMP-O4: Insert automatic BFLOAT16 casts around safe Pytorch functions and Tensor methods.'
    elif amp_opt_level == 5:
        dtype = 'AMP-O5: BFLOAT16 training with FP32 batchnorm and FP32 master weights.'
    else:
        dtype = 'FP32'

    print ("OK: finished running benchmark..")
    print ("--------------------SUMMARY--------------------------")
    print ("Microbenchmark for network : {}".format(net))
    if distributed_dataparallel or is_torchrun:
      print ("--------This process: rank " + str(distributed_parameters['rank']) + "--------");
      print ("Num devices: 1")
    else:
      print ("Num devices: {}".format(ngpus))
    print ("Dtype: {}".format(dtype))
    print ("Mini batch size [", get_input_type(net), "] : {}".format(batch_size))
    print ("Time per mini-batch : {}".format(time_per_batch))
    print ("Throughput [", get_input_type(net), "/sec] : {}".format(batch_size/time_per_batch))
    if (distributed_dataparallel or is_torchrun) and distributed_parameters['rank'] == 0:
      print ("")
      print ("--------Overall (all ranks) (assuming same num/type devices for each rank)--------")
      world_size = distributed_parameters['world_size']
      print ("Num devices: {}".format(world_size))
      print ("Dtype: {}".format(dtype))
      print ("Mini batch size [", get_input_type(net), "] : {}".format(batch_size*world_size))
      print ("Time per mini-batch : {}".format(time_per_batch))
      print ("Throughput [", get_input_type(net), "/sec] : {}".format(batch_size*world_size/time_per_batch))

def main():
    run_benchmarking_wrapper(copy.deepcopy(args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, choices=get_network_names(), required=True, help="Network to run.")
    parser.add_argument("--batch-size" , type=int, required=False, default=64, help="Batch size (will be split among devices used by this invocation)")
    parser.add_argument("--iterations", type=int, required=False, default=20, help="Iterations")
    parser.add_argument("--flops-prof-step", type=int, required=False, default=0, help="The flops profiling step")
    parser.add_argument("--kineto", action='store_true', required=False, help="Turn kineto profiling on")
    parser.add_argument("--autograd_profiler", action='store_true', required=False, help="Use PyTorch autograd (old) profiler")
    parser.add_argument("--fp16", type=int, required=False, default=0,help="FP16 mixed precision benchmarking")
    parser.add_argument("--amp-opt-level", type=int, required=False, default=0,help="apex.amp mixed precision benchmarking opt level")
    parser.add_argument("--distributed_dataparallel", action='store_true', required=False, help="Use torch.nn.parallel.DistributedDataParallel api to run on multiple processes/nodes. The multiple processes need to be launched manually, this script will only launch ONE process per invocation. Either use --distributed_dataparallel and manually launch multiple processes or launch this script with `torchrun`")
    parser.add_argument("--device_ids", type=str, required=False, default=None, help="Comma-separated list (no spaces) to specify which HIP devices (0-indexed) to run distributedDataParallel api on. Might need to use HIP_VISIBLE_DEVICES to limit visiblity of devices to different processes.")
    parser.add_argument("--rank", type=int, required=False, default=None, help="Rank of this process. Required for --distributed_dataparallel")
    parser.add_argument("--world-size", type=int, required=False, default=None, help="Total number of ranks/processes. Required for --distributed_dataparallel")
    parser.add_argument("--dist-backend", type=str, required=False, default=None, help="Backend used for distributed training. Can be one of 'nccl' or 'gloo'. Required for --distributed_dataparallel")
    parser.add_argument("--dist-url", type=str, required=False, default=None, help="url used for rendezvous of processes in distributed training. Needs to contain IP and open port of master rank0 eg. 'tcp://172.23.2.1:54321'. Required for --distributed_dataparallel")
    parser.add_argument("--compile", action='store_true', required=False, help="use pytorch 2.0")
    parser.add_argument("--compileContext", default={}, required=False, help="additional compile options")

    args = parser.parse_args()

    if args.flops_prof_step:
        try:
            from deepspeed.profiling.flops_profiler import FlopsProfiler
        except:
            print("ERROR: You must install (or copy) deepspeed.profiling to use --flops-prof-step")
            sys.exit(1)

    if args.fp16 and args.amp_opt_level:
        print ("ERROR: Cannot use both --fp16 and --amp-opt-level")
        sys.exit(1)
    if args.amp_opt_level and not HAVE_APEX:
        print ("ERROR: You must install apex to use --amp-opt-level")
        sys.exit(1)

    main()