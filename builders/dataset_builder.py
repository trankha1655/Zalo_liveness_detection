# _*_ coding: utf-8 _*_
"""
Time:     2020/11/30 17:02
Author:   Ding Cheng(Deeachain)
File:     dataset_builder.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import os
import pickle
import pandas as pd
from dataset.liveness import LivenessTrainDataSet, LivenessValDataSet, LivenessTestDataSet

def build_dataset_train(root, base_size, crop_size):
    data_dir = ''#os.path.join(root, dataset)
    train_data_list = os.path.join(root, 'datasets/train_list.txt')
 
    
    TrainDataSet = LivenessTrainDataSet(data_dir, train_data_list, base_size=base_size, crop_size=crop_size,
                                    ignore_label=0)   
    return TrainDataSet


def build_dataset_test(root, crop_size, gt=False)  :
    data_dir = '' 
    train_data_list = os.path.join(root, 'datasets/train_list.txt')
  
    test_data_list = os.path.join(root, 'datasets/test_list.txt')
    
                                
    
       
    if gt:
        test_data_list = os.path.join(root, 'test_list.txt')
        testdataset = LivenessValDataSet(data_dir, test_data_list, crop_size=crop_size,  ignore_label=0, RGB=RGB)
    else:
        test_data_list = os.path.join(root, 'test_list.txt')
        testdataset = LivenessTestDataSet(data_dir, test_data_list, crop_size=crop_size,  ignore_label=0, RGB=RGB)
    
    return testdataset
