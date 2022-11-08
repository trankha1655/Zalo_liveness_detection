#!/bin/bash

# Deeplabv3plus_res101  PSPNet_res101  DualSeg_res101  BiSeNet  BiSeNetV2  DDRNet
# FCN_ResNet  SegTrans

python train.py --model MobileNetv2  --max_epochs 150 --val_epochs 5 --batch_size 64 --lr 0.001  \
                    --root .. --resume ../Mobilenetv2_30epoch.pth --num_worker 4 
