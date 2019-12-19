# coding=utf-8
"""
@File: inference.py
@Time: 2019/12/16
@Author: Zengrui Zhao
""" 
import torch
from src.dataset import DataTest
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import argparse
import tqdm
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batchSize', type=int, default=1)
    parse.add_argument('--numWorkers', type=int, default=10)
    parse.add_argument('--rootPth', type=str, default='/home/zzr/shared/data/XT_colorNorm')
    parse.add_argument('--modelPth', type=str, default='/home/zzr/Project/segmentation/model/final.pth')
    parse.add_argument('--savePth', type=str, default='/home/zzr/Data/multiNucle/linknet_dice_colorNorm')

    return parse.parse_args()


def inference(model, dataLoader, args):
    with torch.no_grad():
        for img, name in tqdm.tqdm(dataLoader):
            img = img.to(device)
            # output = F.softmax(model(img))
            output = F.sigmoid(model(img))
            for i in range(img.shape[0]):
                result = np.array(output[i, -1, ...].squeeze().cpu())
                # plt.imshow(result, cmap='jet')
                # plt.show()
            # break
                matplotlib.image.imsave(osp.join(args.savePth, name[i]), result)


def main():
    args = parseArgs()
    data = DataTest(rootPth=args.rootPth)
    dataLoader = DataLoader(data,
                            batch_size=args.batchSize,
                            shuffle=False,
                            pin_memory=False,
                            num_workers=args.numWorkers
                            )
    model = smp.Linknet(classes=1, encoder_name='se_resnext101_32x4d').to(device)
    model.load_state_dict(torch.load(args.modelPth))
    if not osp.exists(args.savePth):
        os.makedirs(args.savePth)
    inference(model, dataLoader, args)
    print('--Done--')


if __name__ == '__main__':
    main()