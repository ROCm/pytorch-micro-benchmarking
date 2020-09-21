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

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
#'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2'
def get_network(net):
    if (net == "alexnet"):
        return torchvision.models.alexnet().to(device="cuda")
    elif (net == "densenet121"):
        return torchvision.models.densenet121().to(device="cuda")
    elif (net == "densenet161"):
        return torchvision.models.densenet161().to(device="cuda")
    elif (net == "densenet169"):
        return torchvision.models.densenet169().to(device="cuda")
    elif (net == "densenet201"):
        return torchvision.models.densenet201().to(device="cuda")
    elif (net == "googlenet"):
        return torchvision.models.googlenet().to(device="cuda")
    elif (net == "inception"):
        return torchvision.models.inception(aux_logits=False).to(device="cuda")
    elif (net == "inception_v3"):
        return torchvision.models.inception_v3(aux_logits=False).to(device="cuda")
    elif (net == "mobilenet_v2"):
        return torchvision.models.mobilenet_v2().to(device="cuda")
    elif (net == "resnet18"):
        return torchvision.models.resnet18().to(device="cuda")
    elif (net == "resnet34"):
        return torchvision.models.resnet34().to(device="cuda")
    elif (net == "resnet50"):
        return torchvision.models.resnet50().to(device="cuda")
    elif (net == "resnet101"):
        return torchvision.models.resnet101().to(device="cuda")
    elif (net == "resnet152"):
        return torchvision.models.resnet152().to(device="cuda")
    elif (net == "resnext50"):
        return torchvision.models.resnext50_32x4d().to(device="cuda")
    elif (net == "resnext101"):
        return torchvision.models.resnext101_32x8d().to(device="cuda")
    elif (net == "shufflenet"):
        model = shufflenet().to(device="cuda")
        model.apply(weight_init)
        return model
    elif (net == "shufflenet_v2_x05"):
        return torchvision.models.shufflenet_v2_x0_5().to(device="cuda")
    elif (net == "shufflenet_v2_x10"):
        return torchvision.models.shufflenet_v2_x1_0().to(device="cuda")
    elif (net == "shufflenet_v2_x15"):
        return torchvision.models.shufflenet_v2_x1_5().to(device="cuda")	
    elif (net == "SqueezeNet"):
        return torchvision.models.squeezenet1_0().to(device="cuda")
    elif (net == "SqueezeNet1.1"):
        return torchvision.models.squeezenet1_1().to(device="cuda")
    elif (net == "vgg11"):
        return torchvision.models.vgg11().to(device="cuda")
    elif (net == "vgg13"):
        return torchvision.models.vgg13().to(device="cuda")
    elif (net == "vgg16"):
        return torchvision.models.vgg16().to(device="cuda")
    elif (net == "vgg19"):
        return torchvision.models.vgg19().to(device="cuda")
    elif (net == "vgg11_bn"):
        return torchvision.models.vgg11_bn().to(device="cuda")
    elif (net == "vgg13_bn"):
        return torchvision.models.vgg13_bn().to(device="cuda")
    elif (net == "vgg16_bn"):
        return torchvision.models.vgg16_bn().to(device="cuda")
    elif (net == "vgg19_bn"):
        return torchvision.models.vgg19_bn().to(device="cuda")
    # segmentation models
    elif (net == "deeplabv3_resnet50"):
        return torchvision.models.segmentation.deeplabv3_resnet50().to(device="cuda")
    elif (net == "deeplabv3_resnet101"):
        return torchvision.models.segmentation.deeplabv3_resnet101().to(device="cuda")
    elif (net == "fcn_resnet50"):
        return torchvision.models.segmentation.deeplabv3_resnet50().to(device="cuda")
    elif (net == "fcn_resnet101"):
        return torchvision.models.segmentation.deeplabv3_resnet101().to(device="cuda")
    else:
        print ("ERROR: not a supported model.")
        sys.exit(1)

def forwardbackward(inp, optimizer, network, target):
    optimizer.zero_grad()
    out = network(inp)
    # WIP: googlenet, deeplabv3_*, fcn_* missing log_softmax for this to work
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    optimizer.step()

def rendezvous(distributed_parameters):
    print("Initializing process group...")
    torch.distributed.init_process_group(backend=distributed_parameters['dist_backend'], init_method=distributed_parameters['dist_url'], rank=distributed_parameters['rank'], world_size=distributed_parameters['world_size'])
    print("Rendezvous complete. Created process group...")

def run_benchmarking_wrapper(net, batch_size, iterations, run_fp16, dataparallel, distributed_dataparallel, device_ids=None, distributed_parameters=None):
    if (dataparallel or distributed_dataparallel):
        ngpus = len(device_ids) if device_ids else torch.cuda.device_count()
    else:
        ngpus = 1

    if (distributed_dataparallel):
        # Assumption below that each process launched with --distributed_dataparallel has the same number of devices visible/specified
        distributed_parameters['world_size'] = ngpus * distributed_parameters['world_size']
        distributed_parameters['rank'] = ngpus * distributed_parameters['rank']
        mp.spawn(run_benchmarking, nprocs=ngpus, args=(ngpus, net, batch_size, iterations, run_fp16, dataparallel, distributed_dataparallel, device_ids, distributed_parameters))
    else:
        run_benchmarking(0, ngpus, net, batch_size, iterations, run_fp16, dataparallel, distributed_dataparallel, device_ids=None, distributed_parameters=None)

def run_benchmarking(local_rank, ngpus, net, batch_size, iterations, run_fp16, dataparallel, distributed_dataparallel, device_ids=None, distributed_parameters=None):
    if device_ids:
        assert ngpus == len(device_ids)
        torch.cuda.set_device("cuda:%d" % device_ids[local_rank])
    else:
        torch.cuda.set_device("cuda:0")

    network = get_network(net)
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
    target = torch.arange(batch_size, device="cuda")
    param_copy = network.parameters()
    if (run_fp16):
        param_copy = get_param_copy(network)
    optimizer = torch.optim.SGD(param_copy, lr = 0.01, momentum = 0.9)

    ## warmup.
    print ("INFO: running forward and backward for warmup.")
    forwardbackward(inp, optimizer, network, target)
    forwardbackward(inp, optimizer, network, target)

    time.sleep(1)
    torch.cuda.synchronize()

    ## benchmark.
    print ("INFO: running the benchmark..")
    tm = time.time()
    for i in range(iterations):
        forwardbackward(inp, optimizer, network, target)
    torch.cuda.synchronize()
    
    tm2 = time.time()
    time_per_batch = (tm2 - tm) / iterations

    print ("OK: finished running benchmark..")
    print ("--------------------SUMMARY--------------------------")
    print ("Microbenchmark for network : {}".format(net))
    if (distributed_dataparallel):
      print ("--------This process: rank " + str(distributed_parameters['rank']) + "--------");
      print ("Num devices: 1")
    else:
      print ("Num devices: {}".format(ngpus))
    print ("Mini batch size [img] : {}".format(batch_size))
    print ("Time per mini-batch : {}".format(time_per_batch))
    print ("Throughput [img/sec] : {}".format(batch_size/time_per_batch))
    if (distributed_dataparallel):
      print ("")
      print ("--------Overall (all ranks) (assuming same num/type devices for each rank)--------")
      world_size = distributed_parameters['world_size']
      print ("Num devices: {}".format(world_size))
      print ("Mini batch size [img] : {}".format(batch_size*world_size))
      print ("Time per mini-batch : {}".format(time_per_batch))
      print ("Throughput [img/sec] : {}".format(batch_size*world_size/time_per_batch))

def main():
    net = args.network
    batch_size = args.batch_size
    iterations = args.iterations
    run_fp16 = args.fp16
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

    run_benchmarking_wrapper(net, batch_size, iterations, run_fp16, dataparallel, distributed_dataparallel, device_ids, distributed_parameters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, 
        choices=['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'shufflenet', 'shufflenet_v2_x05', 'shufflenet_v2_x10', 'shufflenet_v2_x15', 'SqueezeNet', 'SqueezeNet1.1', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inception', 'inception_v3', 'resnext50', 'resnext101', 'mobilenet_v2', 'googlenet' , 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'fcn_resnet50', 'fcn_resnet101' ],
        required=True, help="Network to run.")
    parser.add_argument("--batch-size" , type=int, required=False, default=64, help="Batch size (will be split among devices used by this invocation)")
    parser.add_argument("--iterations", type=int, required=False, default=20, help="Iterations")
    parser.add_argument("--fp16", type=int, required=False, default=0,help="FP16 mixed precision benchmarking")
    parser.add_argument("--dataparallel", action='store_true', required=False, help="Use torch.nn.DataParallel api to run single process on multiple devices. Use only one of --dataparallel or --distributed_dataparallel")
    parser.add_argument("--distributed_dataparallel", action='store_true', required=False, help="Use torch.nn.parallel.DistributedDataParallel api to run on multiple processes/nodes. The multiple processes need to be launched manually, this script will only launch ONE process per invocation. Use only one of --dataparallel or --distributed_dataparallel")
    parser.add_argument("--device_ids", type=str, required=False, default=None, help="Comma-separated list (no spaces) to specify which HIP devices (0-indexed) to run dataparallel or distributedDataParallel api on. Might need to use HIP_VISIBLE_DEVICES to limit visiblity of devices to different processes.")
    parser.add_argument("--rank", type=int, required=False, default=None, help="Rank of this process. Required for --distributed_dataparallel")
    parser.add_argument("--world-size", type=int, required=False, default=None, help="Total number of ranks/processes. Required for --distributed_dataparallel")
    parser.add_argument("--dist-backend", type=str, required=False, default=None, help="Backend used for distributed training. Can be one of 'nccl' or 'gloo'. Required for --distributed_dataparallel")
    parser.add_argument("--dist-url", type=str, required=False, default=None, help="url used for rendezvous of processes in distributed training. Needs to contain IP and open port of master rank0 eg. 'tcp://172.23.2.1:54321'. Required for --distributed_dataparallel")

    args = parser.parse_args()

    main()
