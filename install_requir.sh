#!/bin/bash

if [ -e ../mobilenetv2.pth.tar ]
then
    echo "exist pretrained!"
else
    echo "downloading pretrained!"
    gdown 14bHUOzvPlylAG2IJDy35vmj35jAN_oM9
    mv mobilenetv2.pth.tar ../mobilenetv2.pth.tar
fi
if [ -e ../Mobilenetv2_30epoch.pth ]
then
    echo "exist weight!"
else
    echo "downloading best_weight!"
    gdown 1iPk9eKZjY3yQhpEVicLYkNddX3FAgnSd
    mv Mobilenetv2_30epoch.pth ../Mobilenetv2_30epoch.pth
fi

