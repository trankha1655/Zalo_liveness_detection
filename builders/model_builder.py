
# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
Author:   Ding Cheng(Deeachain)
File:     model_builder.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from model.inception_resnetv2 import Inception_ResNetv2
from model.mobilenetv2 import MobileNetv2
from model.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
import torch


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def build_model(model_name,
                num_classes, 
                checkpoint_path=None):
    if model_name == 'Inception_Resnetv2':
        model = Inception_ResNetv2(classes=num_classes)
    elif model_name == 'FCN_ResNet':
        model = FCN_ResNet(num_classes=num_classes, backbone=backbone, out_stride=out_stride, mult_grid=mult_grid)
    elif model_name == 'MobileNetv3_small':
        model = mobilenetv3_small(num_classes ==num_classes)
    elif model_name == 'MobileNetv3_large':
        model = mobilenetv3_large(num_classes ==num_classes)
    elif model_name == 'MobileNetv2':
        model = MobileNetv2(num_classes=num_classes)
    else:
        raise TypeError("No model name is", model_name)  
    
    

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        check_list = [i for i in checkpoint['model'].items()]
        # Read weights with multiple cards, and continue training with a single card this time
        if 'module.' in check_list[0][0]:
            new_stat_dict = {}
            for k, v in checkpoint['model'].items():
                new_stat_dict[k[7:]] = v
            model.load_state_dict(new_stat_dict, strict=True)
        # Read the training weight of a single card, and continue training with a single card this time
        else:
            model.load_state_dict(checkpoint['model'])
        

    return model


