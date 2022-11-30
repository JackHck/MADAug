from __future__ import print_function
# from __future__ import absolute_import
import torch

from torch import nn
from torch.nn import DataParallel

from .resnet import ResNet
from .wideresnet import WideResNet


def get_model(model_name='wresnet40_2', num_class=10, n_channel=3, use_cuda=True, data_parallel=False):
    name = model_name

    if name == 'resnet50':
        model = ResNet(dataset='imagenet', n_channel=n_channel, depth=50, num_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', n_channel=n_channel, depth=200, num_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)
    else:
        raise NameError('no model named, %s' % name)

    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        if use_cuda:
            model = model.cuda()
    return model

