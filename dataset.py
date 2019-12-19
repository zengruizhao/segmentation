# coding=utf-8
"""
@File: dataset.py
@Time: 2019/12/14
@Author: Zengrui Zhao
""" 
from torchvision import transforms
from torch.utils.data import Dataset
import os.path as osp
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch


class Data(Dataset):
    def __init__(self, rootPth='/home/zzr/Data/multiNucle',
                 imgSize=(256, 256),
                 mode='train'):
        self.imgSize = imgSize
        self.mode = mode
        self.img = osp.join(rootPth, 'img')
        self.mask = osp.join(rootPth, 'mask')
        self.imgList = [osp.join(self.img, i) for i in os.listdir(self.img)]
        self.maskList = [osp.join(self.mask, i) for i in os.listdir(self.img)]
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item):
        img = Image.open(self.imgList[item]).resize(self.imgSize, Image.BILINEAR)
        mask = Image.open(self.maskList[item]).resize(self.imgSize, Image.NEAREST)
        if self.mode == 'train':
            # random flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            # random rotate
            if np.random.choice([1]):
                degree = np.random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                img = img.transpose(degree)
                mask = mask.transpose(degree)

        mask = np.array(mask)
        mask = mask[..., 2] - mask[..., 0]
        assert len(np.unique(mask)) == 2
        return self.toTensor(img), mask / 255

    def __len__(self):
        return len(self.imgList)


class DataTest(Data):
    def __init__(self, rootPth='/home/zzr/Data/multiNucle',
                 imgSize=(2048, 2048)):
        super().__init__()
        self.imgSize = imgSize
        self.imgList = [osp.join(rootPth, i) for i in os.listdir(rootPth)]

    def __getitem__(self, item):
        img = Image.open(self.imgList[item]).resize(self.imgSize, Image.BILINEAR)
        if np.array(img).shape[-1] == 4:
            img = Image.fromarray(np.array(img)[..., 0:-1])

        return transforms.ToTensor()(img), self.imgList[item].split('/')[-1]

    def __len__(self):
        return len(self.imgList)


if __name__ == '__main__':
    data = DataTest()
    print(len(data))
    for i in range(len(data)):
        img = data[i]
        print(img.shape)
