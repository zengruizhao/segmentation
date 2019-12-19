# coding=utf-8
"""
@File: testNumWorkers.py
@Time: 2019/12/16
@Author: Zengrui Zhao
""" 
import torch
import sys
import time
from torch.utils.data import DataLoader
sys.path.append('..')
from src.dataset import Data

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    for num_workers in range(1, 16):  # 遍历worker数
        kwargs = {'num_workers': num_workers, 'pin_memory': False} if use_cuda else {}
        train_set = Data(rootPth='/home/zzr/Data/multiNucle', mode='test')
        train_loader = DataLoader(train_set,
                                  batch_size=16,
                                  drop_last=False,
                                  shuffle=True,
                                  **kwargs)

        start = time.time()
        for epoch in range(1, 3):
            for batch_idx, (data, target) in enumerate(train_loader):  # 不断load
                pass

        end = time.time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))