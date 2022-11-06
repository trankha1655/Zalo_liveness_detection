#!/bin/bash

# Deeplabv3plus_res101  PSPNet_res101  DualSeg_res101  BiSeNet  BiSeNetV2  DDRNet
# FCN_ResNet  SegTrans
gdown 14bHUOzvPlylAG2IJDy35vmj35jAN_oM9
mv mobilenetv2.pth.tar ../mobilenetv2.pth.tar
python train.py --model MobileNetv2  --max_epochs 100 --val_epochs 10 --batch_size 64 --lr 0.001  --root ..
