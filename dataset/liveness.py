# _*_ coding: utf-8 _*_
"""
Time:     2021/3/12 10:28
Author:   Ding Cheng(Deeachain)
File:     cityscapes.py
Github:   https://github.com/Deeachain
"""
import os
import os.path as osp
import numpy as np
import cv2
from torch.utils import data
import torch
import pickle
from PIL import Image
from torchvision import transforms
from utils import image_transform as tr
#from .crop import *


class LivenessTrainDataSet(data.Dataset):
    """
       CityscapesTrainDataSet is employed to load train set
       Args:
        root: the Cityscapes dataset path,
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        list_path: cityscapes_train_list.txt, include partial path
        mean: bgr_mean (73.15835921, 82.90891754, 72.39239876)

    """

    def __init__(self, root='', list_path='', max_iters=None, base_size=720, crop_size=400, mean=(0., 0., 0.),
                 std=(1., 1., 1.), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.base_size = base_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.label_mapping =np.array( [[1,0], [0, 1]])

        for name in self.img_ids:
            img_file = os.path.join(root, name.split(',')[0])
            label = self.label_mapping[ int(name.split(',')[1]) ]
            try:
                name = img_file
            except:
                pass
            self.files.append({"img": img_file, "label": label, "name": name})

        print("length of train dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        #print(self.RGB)
        
        image = Image.open(datafiles["img"]).convert('RGB')
            
        
        label = datafiles["label"]
        size = np.asarray(image).shape
        #print(size)
        name = datafiles["name"]

        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            # tr.RandomRotate(180),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=0),
            #tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        sample = {'image': image}
        sampled = composed_transforms(sample)
        image = sampled['image']
        
        label = torch.from_numpy(label).float()
        #print(image.shape)

        return image, label, np.array(size), name


class LivenessValDataSet(data.Dataset):
    """
       CityscapesDataSet is employed to load val set
       Args:
        root: the Cityscapes dataset path,
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        list_path: cityscapes_val_list.txt, include partial path

    """

    def __init__(self, root='', list_path='', crop_size=400, mean=(0., 0., 0.), std=(1., 1., 1.),
                 ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
     
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.label_mapping =np.array( [[1,0], [0, 1]])

        for name in self.img_ids:
            img_file = os.path.join(root, name.split(',')[0])
            label = self.label_mapping[ int(name.split(',')[1]) ]
            try:
                name = img_file.split('/')[-1]
            except:
                pass
            self.files.append({"img": img_file, "label": label, "name": name})

        

        print("length of validation dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        image = Image.open(datafiles["img"]).convert('RGB')

        label = datafiles["label"]
        size = np.asarray(image).shape
        name = datafiles["name"]
        try:
            composed_transforms = transforms.Compose([
                tr.RandomScaleCrop( crop_size=self.crop_size, fill=0),
                #tr.FixScaleCrop(crop_size=self.crop_size),
                # tr.FixedResize(size=(1024,512)),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
        except:
            print(datafilesp['img'])
        sample = {'image': image}
        sampled = composed_transforms(sample)
        image, label = sampled['image']
        
        label = torch.from_numpy(label).float()
        

        return image, label, np.array(size), name


class LivenessTestDataSet(data.Dataset):
    """
       CityscapesDataSet is employed to load test set
       Args:
        root: the Cityscapes dataset path,
        list_path: cityscapes_test_list.txt, include partial path

    """

    def __init__(self, root='', list_path='', crop_size=400 , mean=(128, 128, 128), std=(128, 128, 128),
                 ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        

        for name in self.img_ids:
            img_file = os.path.join(root, name.split(',')[0])
            try:
                name = img_file.split('/')[-1]
            except:
                pass

            self.files.append({"img": img_file, "name": name})

        print("length of validation dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        image= Image.open(datafiles["img"])
            

        size = np.asarray(image).shape
        #print(size)
        name = datafiles["name"]
        composed_transforms = transforms.Compose([
            tr.RandomScaleCrop( crop_size=self.crop_size, fill=0),
            #tr.FixedResize( (1024, 768)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        sample = {'image': image}
        sampled = composed_transforms(sample)
        image = sampled['image']
        
        
        return image, np.array(size), name

