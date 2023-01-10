import math
import torch
import torch.nn as nn
import numpy as np
from apex.contrib.groupbn import BatchNorm2d_NHWC as gbn_persistent

__all__ = ['ResNet', 'build_resnet', 'resnet_versions', 'resnet_configs']

# ResNetBuilder {{{

class ResNetBuilder(object):
    def __init__(self, version, config, local_rank=0, get_logs=False):
        self.config = config

        self.L = sum(version['layers'])
        self.M = version['block'].M
        self.local_rank=local_rank
        self.get_logs=get_logs

    def conv(self, kernel_size, in_planes, out_planes, name, stride=1):
        if kernel_size == 3:
            conv = self.config['conv'](
                    in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)
        elif kernel_size == 1:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                             bias=False)
        elif kernel_size == 5:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                             padding=2, bias=False)
        elif kernel_size == 7:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                             padding=3, bias=False)
        else:
            return None

        if self.config['nonlinearity'] == 'relu':
            nn.init.kaiming_normal_(conv.weight,
                    mode=self.config['conv_init'],
                    nonlinearity=self.config['nonlinearity'])

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1, name=None):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, name=name, stride=stride)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, name=None):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, name=name, stride=stride)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, name=None):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, name=name, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes, last_bn=False):
        bn = nn.BatchNorm2d(planes)
        gamma_init_val = 0 if last_bn and self.config['last_bn_0_init'] else 1
        nn.init.constant_(bn.weight, gamma_init_val)
        nn.init.constant_(bn.bias, 0)

        return bn

    def activation(self):
        return self.config['activation']()

    def fc(self, planes, num_classes=1000):
        fc = nn.Linear(planes, num_classes)
        return fc
    
    def groupbn(self, planes, fuse_relu=True, bn_group=1, torch_channels_last=True, name_beta=None, name_gamma=None):
        bn = gbn_persistent(planes, fuse_relu=fuse_relu, bn_group=bn_group, torch_channels_last=torch_channels_last, cta_launch_margin=0)
        return bn

# ResNetBuilder }}}

# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
# BasicBlock }}}

# Bottleneck {{{
class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, stage=1, unit=1):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(inplanes, planes, name="stage" + str(stage) + "_unit" + str(unit) + "_conv1_weight")
        self.group_bn_1 = builder.groupbn(planes, fuse_relu=True, torch_channels_last=True, name_beta= "stage" + str(stage) + "_unit" + str(unit) + "_bn1_beta", name_gamma="stage" + str(stage) + "_unit" + str(unit) + "_bn1_gamma")
        self.conv2 = builder.conv3x3(planes, planes, stride=stride, name="stage" + str(stage) + "_unit" + str(unit) + "_conv2_weight")
        self.group_bn_2 = builder.groupbn(planes, fuse_relu=True, torch_channels_last=True, name_beta= "stage" + str(stage) + "_unit" + str(unit) + "_bn2_beta", name_gamma="stage" + str(stage) + "_unit" + str(unit) + "_bn2_gamma")
        self.conv3 = builder.conv1x1(planes, planes * self.expansion, name="stage" + str(stage) + "_unit" + str(unit) + "_conv3_weight")
        self.relu = builder.activation()
        self.group_bn_3 = builder.groupbn(planes * self.expansion, fuse_relu=True, torch_channels_last=True, name_beta= "stage" + str(stage) + "_unit" + str(unit) + "_bn3_beta", name_gamma="stage" + str(stage) + "_unit" + str(unit) + "_bn3_gamma")
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.group_bn_1(out)

        out = self.conv2(out)
        out = self.group_bn_2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.group_bn_3(out, residual)

        return out

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = builder.conv7x7(3, 64, stride=2, name="conv0_weight")
        self.relu = builder.activation()
        self.group_bn1 = builder.groupbn(64, fuse_relu=True, torch_channels_last=True, name_beta= "bn0_beta", name_gamma="bn0_gamma")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0], 1, unit=1)
        self.layer2 = self._make_layer(builder, block, 128, layers[1], 2, unit=2, stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], 3, unit=3, stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], 4, unit=4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = builder.fc(512 * block.expansion, num_classes)


    def _make_layer(self, builder, block, planes, blocks, stage, unit=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(self.inplanes, planes * block.expansion,
                                    stride=stride, name="stage"+ str(stage) +"_unit"+ str(unit) + "_conv"+str(unit) + "sc_weight")
            
            dbn = builder.groupbn(planes * block.expansion, fuse_relu=False, name_beta= "stage" + str(stage) + "_unit" + str(unit) + "_bn_sc_beta", name_gamma="stage" + str(stage) + "_unit" + str(unit) + "_bn_sc_gamma")
            
            if dbn is not None: 
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, stage, unit))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.group_bn1 is not None:
            x = self.group_bn1(x)
        else:
            x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


resnet_configs = {
        'classic' : {
            'conv' : nn.Conv2d,
            'conv_init' : 'fan_out',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : lambda: nn.ReLU(inplace=True),
            },
        'fanin' : {
            'conv' : nn.Conv2d,
            'conv_init' : 'fan_in',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : lambda: nn.ReLU(inplace=True),
            },
        }

resnet_versions = {
        'resnet18' : {
            'net' : ResNet,
            'block' : BasicBlock,
            'layers' : [2, 2, 2, 2],
            'num_classes' : 1000,
            },
         'resnet34' : {
            'net' : ResNet,
            'block' : BasicBlock,
            'layers' : [3, 4, 6, 3],
            'num_classes' : 1000,
            },
         'resnet50' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'layers' : [3, 4, 6, 3],
            'num_classes' : 1000,
            },
        'resnet101' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'layers' : [3, 4, 23, 3],
            'num_classes' : 1000,
            },
        'resnet152' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'layers' : [3, 8, 36, 3],
            'num_classes' : 1000,
            },
        }

def build_resnet(version, config, model_state=None, local_rank=0, get_logs=False):
    version = resnet_versions[version]
    config = resnet_configs[config]

    builder = ResNetBuilder(version, config, local_rank=local_rank, get_logs=get_logs)
    print("Version: {}".format(version))
    print("Config: {}".format(config))
    model = version['net'](builder,
                           version['block'],
                           version['layers'],
                           version['num_classes'])

    return model

