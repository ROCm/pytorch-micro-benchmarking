import torch
import torchvision
import random
import time
import argparse
import os
import sys
import math
import torch.nn as nn
import torch.multiprocessing as mp
from fp16util import network_to_half, get_param_copy
from shufflenet import shufflenet
from shufflenet_v2 import shufflenet as shufflenet_v2
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

# num_classes=1000
models = {
        "alexnet" :            torchvision.models.alexnet,
        "densenet121" :        torchvision.models.densenet121,
        "densenet161" :        torchvision.models.densenet161,
        "densenet169" :        torchvision.models.densenet169,
        "densenet201" :        torchvision.models.densenet201,
        "googlenet" :          torchvision.models.googlenet,
        "inception_v3" :       torchvision.models.inception_v3,
        "mnasnet0_5" :         torchvision.models.mnasnet0_5,
        "mnasnet0_75" :        torchvision.models.mnasnet0_75,
        "mnasnet1_0" :         torchvision.models.mnasnet1_0,
        "mnasnet1_3" :         torchvision.models.mnasnet1_3,
        "mobilenet_v2" :       torchvision.models.mobilenet_v2,
        "resnet18" :           torchvision.models.resnet18,
        "resnet34" :           torchvision.models.resnet34,
        "resnet50" :           torchvision.models.resnet50,
        "resnet101" :          torchvision.models.resnet101,
        "resnet152" :          torchvision.models.resnet152,
        "resnext50" :          torchvision.models.resnext50_32x4d,
        "resnext50_32x4d" :    torchvision.models.resnext50_32x4d,
        "resnext101" :         torchvision.models.resnext101_32x8d,
        "resnext101_32x8d" :   torchvision.models.resnext101_32x8d,
        "shufflenet" :         shufflenet,
        "shufflenet_v2" :      shufflenet_v2,
        "shufflenet_v2_x05" :  torchvision.models.shufflenet_v2_x0_5,
        "shufflenet_v2_x10" :  torchvision.models.shufflenet_v2_x1_0,
        "shufflenet_v2_x15" :  torchvision.models.shufflenet_v2_x1_5,
        "shufflenet_v2_x20" :  torchvision.models.shufflenet_v2_x2_0,
        "shufflenet_v2_x0_5" : torchvision.models.shufflenet_v2_x0_5,
        "shufflenet_v2_x1_0" : torchvision.models.shufflenet_v2_x1_0,
        "shufflenet_v2_x1_5" : torchvision.models.shufflenet_v2_x1_5,
        "shufflenet_v2_x2_0" : torchvision.models.shufflenet_v2_x2_0,
        "SqueezeNet" :         torchvision.models.squeezenet1_0,
        "squeezenet1_0" :      torchvision.models.squeezenet1_0,
        "SqueezeNet1.1" :      torchvision.models.squeezenet1_1,
        "squeezenet1_1" :      torchvision.models.squeezenet1_1,
        "vgg11" :              torchvision.models.vgg11,
        "vgg13" :              torchvision.models.vgg13,
        "vgg16" :              torchvision.models.vgg16,
        "vgg19" :              torchvision.models.vgg19,
        "vgg11_bn" :           torchvision.models.vgg11_bn,
        "vgg13_bn" :           torchvision.models.vgg13_bn,
        "vgg16_bn" :           torchvision.models.vgg16_bn,
        "vgg19_bn" :           torchvision.models.vgg19_bn,
        "wide_resnet50_2" :    torchvision.models.wide_resnet50_2,
        "wide_resnet101_2" :   torchvision.models.wide_resnet101_2,
}

# newer torchvision models, for backwards compat
try:
    models["mobilenet_v3_large"] = torchvision.models.mobilenet_v3_large
    models["mobilenet_v3_small"] = torchvision.models.mobilenet_v3_small
except AttributeError:
    pass

# segmentation models, num_classes=21
segmentation_models = {
        "fcn_resnet50" :        torchvision.models.segmentation.fcn_resnet50,
        "fcn_resnet101" :       torchvision.models.segmentation.fcn_resnet101,
        "deeplabv3_resnet50" :  torchvision.models.segmentation.deeplabv3_resnet50,
        "deeplabv3_resnet101" : torchvision.models.segmentation.deeplabv3_resnet101,
}

# newer torchvision segmentation models, for backwards compat
try:
    segmentation_models["deeplabv3_mobilenet_v3_large"] = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large
    segmentation_models["lraspp_mobilenet_v3_large"] = torchvision.models.segmentation.lraspp_mobilenet_v3_large,
except AttributeError:
    pass

def get_network_names():
    return sorted(list(models.keys()) + list(segmentation_models.keys()))

def get_network(net):
    # aux_logits=False only used by inception_v3
    if "inception_v3" == net:
        return models[net](aux_logits=False).to(device="cuda")
    elif net in models:
        return models[net]().to(device="cuda")
    elif net in segmentation_models:
        return segmentation_models[net]().to(device="cuda")
    else:
        print ("ERROR: not a supported model '%s'" % net)
        sys.exit(1)

def forwardbackward(inp, optimizer, network, target, amp_opt_level):
    optimizer.zero_grad()
    out = network(inp)
    # WIP: googlenet, deeplabv3_*, fcn_* missing log_softmax for this to work
    loss = torch.nn.functional.cross_entropy(out, target)
    if amp_opt_level:
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

def rendezvous(distributed_parameters):
    print("Initializing process group...")
    torch.distributed.init_process_group(backend=distributed_parameters['dist_backend'], init_method=distributed_parameters['dist_url'], rank=distributed_parameters['rank'], world_size=distributed_parameters['world_size'])
    print("Rendezvous complete. Created process group...")

def run_benchmarking_wrapper(net, batch_size, iterations, amp_opt_level, run_fp16, dataparallel, distributed_dataparallel, device_ids=None, distributed_parameters=None):
    if (dataparallel or distributed_dataparallel):
        ngpus = len(device_ids) if device_ids else torch.cuda.device_count()
    else:
        ngpus = 1

    if (distributed_dataparallel):
        # Assumption below that each process launched with --distributed_dataparallel has the same number of devices visible/specified
        distributed_parameters['world_size'] = ngpus * distributed_parameters['world_size']
        distributed_parameters['rank'] = ngpus * distributed_parameters['rank']
        mp.spawn(run_benchmarking, nprocs=ngpus, args=(ngpus, net, batch_size, iterations, amp_opt_level, run_fp16, dataparallel, distributed_dataparallel, device_ids, distributed_parameters))
    else:
        run_benchmarking(0, ngpus, net, batch_size, iterations, amp_opt_level, run_fp16, dataparallel, distributed_dataparallel, device_ids=None, distributed_parameters=None)

def run_benchmarking(local_rank, ngpus, net, batch_size, iterations, amp_opt_level, run_fp16, dataparallel, distributed_dataparallel, device_ids=None, distributed_parameters=None):
    if device_ids:
        assert ngpus == len(device_ids)
        torch.cuda.set_device("cuda:%d" % device_ids[local_rank])
    else:
        torch.cuda.set_device("cuda:0")

    network = get_network(net)
    if "shufflenet" == net:
        model.apply(weight_init)

    if (run_fp16):
        network = network_to_half(network)

    if (dataparallel):
        devices_to_run_on = device_ids if device_ids else list(range(ngpus))
        print ("INFO: Running dataparallel on devices: {}".format(str(devices_to_run_on)))
        network = torch.nn.DataParallel(network, device_ids=devices_to_run_on)
    elif (distributed_dataparallel):
        distributed_parameters['rank'] += local_rank
        rendezvous(distributed_parameters)
        devices_to_run_on = [(device_ids[local_rank] if device_ids else local_rank)]
        print ("INFO: Rank {} running distributed_dataparallel on devices: {}".format(distributed_parameters['rank'], str(devices_to_run_on)))
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=devices_to_run_on)
        batch_size = int(batch_size / ngpus)

    if (net == "inception_v3"):
        inp = torch.randn(batch_size, 3, 299, 299, device="cuda")
    else:
        inp = torch.randn(batch_size, 3, 224, 224, device="cuda")
    if (run_fp16):
        inp = inp.half()
    if net in models:
        # number of classes is 1000 for imagenet
        target = torch.randint(0, 1000, (batch_size,), device="cuda")
    elif net in segmentation_models:
        # number of classes is 21 for segmentation
        target = torch.randint(0, 21, (batch_size,), device="cuda")
    param_copy = network.parameters()
    if (run_fp16):
        param_copy = get_param_copy(network)
    optimizer = torch.optim.SGD(param_copy, lr = 0.01, momentum = 0.9)

    if (amp_opt_level):
        network, optimizer = apex.amp.initialize(network, optimizer, opt_level="O%d"%amp_opt_level)

    ## warmup.
    print ("INFO: running forward and backward for warmup.")
    forwardbackward(inp, optimizer, network, target, amp_opt_level)
    forwardbackward(inp, optimizer, network, target, amp_opt_level)

    time.sleep(1)
    torch.cuda.synchronize()

    ## benchmark.
    print ("INFO: running the benchmark..")
    tm = time.time()
    for i in range(iterations):
        forwardbackward(inp, optimizer, network, target, amp_opt_level)
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
    if (distributed_dataparallel):
      print ("--------This process: rank " + str(distributed_parameters['rank']) + "--------");
      print ("Num devices: 1")
    else:
      print ("Num devices: {}".format(ngpus))
    print ("Dtype: {}".format(dtype))
    print ("Mini batch size [img] : {}".format(batch_size))
    print ("Time per mini-batch : {}".format(time_per_batch))
    print ("Throughput [img/sec] : {}".format(batch_size/time_per_batch))
    if (distributed_dataparallel):
      print ("")
      print ("--------Overall (all ranks) (assuming same num/type devices for each rank)--------")
      world_size = distributed_parameters['world_size']
      print ("Num devices: {}".format(world_size))
      print ("Dtype: {}".format(dtype))
      print ("Mini batch size [img] : {}".format(batch_size*world_size))
      print ("Time per mini-batch : {}".format(time_per_batch))
      print ("Throughput [img/sec] : {}".format(batch_size*world_size/time_per_batch))

def main():
    net = args.network
    batch_size = args.batch_size
    iterations = args.iterations
    run_fp16 = args.fp16
    amp_opt_level = args.amp_opt_level
    dataparallel = args.dataparallel
    distributed_dataparallel = args.distributed_dataparallel
    device_ids_str = args.device_ids
    if (args.device_ids):
        device_ids = [int(x) for x in device_ids_str.split(",")]
    else:
        device_ids = None
    distributed_parameters = {}
    distributed_parameters['rank'] = args.rank
    distributed_parameters['world_size'] = args.world_size
    distributed_parameters['dist_backend'] = args.dist_backend
    distributed_parameters['dist_url'] = args.dist_url
    # Some arguments are required for distributed_dataparallel
    if distributed_dataparallel:
        assert args.rank is not None and \
               args.world_size is not None and \
               args.dist_backend is not None and \
               args.dist_url is not None, "rank, world-size, dist-backend and dist-url are required arguments for distributed_dataparallel"

    run_benchmarking_wrapper(net, batch_size, iterations, amp_opt_level, run_fp16, dataparallel, distributed_dataparallel, device_ids, distributed_parameters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, choices=get_network_names(), required=True, help="Network to run.")
    parser.add_argument("--batch-size" , type=int, required=False, default=64, help="Batch size (will be split among devices used by this invocation)")
    parser.add_argument("--iterations", type=int, required=False, default=20, help="Iterations")
    parser.add_argument("--fp16", type=int, required=False, default=0,help="FP16 mixed precision benchmarking")
    parser.add_argument("--amp-opt-level", type=int, required=False, default=0,help="apex.amp mixed precision benchmarking opt level")
    parser.add_argument("--dataparallel", action='store_true', required=False, help="Use torch.nn.DataParallel api to run single process on multiple devices. Use only one of --dataparallel or --distributed_dataparallel")
    parser.add_argument("--distributed_dataparallel", action='store_true', required=False, help="Use torch.nn.parallel.DistributedDataParallel api to run on multiple processes/nodes. The multiple processes need to be launched manually, this script will only launch ONE process per invocation. Use only one of --dataparallel or --distributed_dataparallel")
    parser.add_argument("--device_ids", type=str, required=False, default=None, help="Comma-separated list (no spaces) to specify which HIP devices (0-indexed) to run dataparallel or distributedDataParallel api on. Might need to use HIP_VISIBLE_DEVICES to limit visiblity of devices to different processes.")
    parser.add_argument("--rank", type=int, required=False, default=None, help="Rank of this process. Required for --distributed_dataparallel")
    parser.add_argument("--world-size", type=int, required=False, default=None, help="Total number of ranks/processes. Required for --distributed_dataparallel")
    parser.add_argument("--dist-backend", type=str, required=False, default=None, help="Backend used for distributed training. Can be one of 'nccl' or 'gloo'. Required for --distributed_dataparallel")
    parser.add_argument("--dist-url", type=str, required=False, default=None, help="url used for rendezvous of processes in distributed training. Needs to contain IP and open port of master rank0 eg. 'tcp://172.23.2.1:54321'. Required for --distributed_dataparallel")

    args = parser.parse_args()

    if args.fp16 and args.amp_opt_level:
        print ("ERROR: Cannot use both --fp16 and --amp-opt-level")
        sys.exit(1)
    if args.amp_opt_level and not HAVE_APEX:
        print ("ERROR: You must install apex to use --amp-opt-level")
        sys.exit(1)

    main()
