#!/bin/bash

# Deeplabv3plus_res101  PSPNet_res101  DualSeg_res101  BiSeNet  BiSeNetV2  DDRNet
# FCN_ResNet  SegTrans
if [ -e ../mobilenetv2.pth.tar ]
then
    echo "exist pretrained!"
else
    echo "downloading pretrained!"
    gdown 14bHUOzvPlylAG2IJDy35vmj35jAN_oM9
    mv mobilenetv2.pth.tar ../mobilenetv2.pth.tar
fi
if [ -e ../Mobilenetv2_20epoch.pth ]
then
    echo "exist weight!"
else
    echo "downloading best_weight!"
    gdown 1xrOSUyIZvQK1UVEnf2ZnCDQkwJZaDlo_
    mv Mobilenetv2_20epoch.pth ../Mobilenetv2_20epoch.pth
fi
python train.py --model MobileNetv2  --max_epochs 100 --val_epochs 10 --batch_size 64 --lr 0.001  --root .. --resume ../Mobilenetv2_20epoch.pth
