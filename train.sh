#!/bin/bash

# Deeplabv3plus_res101  PSPNet_res101  DualSeg_res101  BiSeNet  BiSeNetV2  DDRNet
# FCN_ResNet  SegTrans

python train.py --model Inception_Resnetv2  --max_epochs 300 --val_epochs 10 --batch_size 2 --lr 0.001  --root ..
