# coding=utf-8
"""
@File: train.py
@Time: 2019/12/16
@Author: Zengrui Zhao
""" 
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.losses as smploss
from src.dataset import Data
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.logger import getLogger
import time
import os
import os.path as osp
import datetime
from tensorboardX import SummaryWriter
import sys


def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=200)
    parse.add_argument('--rootPth', type=str, default='/home/zzr/Data/multiNucle')
    parse.add_argument('--batchSizeTrain', type=int, default=16)
    parse.add_argument('--batchSizeTest', type=int, default=128)
    parse.add_argument('--numWorkers', type=int, default=8)
    parse.add_argument('--evalFrequency', type=int, default=1)
    parse.add_argument('--saveFrequency', type=int, default=1)
    parse.add_argument('--msgFrequency', type=int, default=10)
    parse.add_argument('--logDir', type=str, default='../log')
    parse.add_argument('--tensorboardDir', type=str, default='../tensorboard')
    parse.add_argument('--modelDir', type=str, default='../model')
    parse.add_argument('--scheduleStep', type=int, default=50)

    return parse.parse_args()


def main(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(logdir=args.subTensorboardDir)
    trainSet = Data(mode='train')
    trainLoader = DataLoader(trainSet,
                             batch_size=args.batchSizeTrain,
                             shuffle=True,
                             pin_memory=False,
                             drop_last=False,
                             num_workers=args.numWorkers)
    testSet = Data(mode='test')
    testLoader = DataLoader(testSet,
                            batch_size=args.batchSizeTest,
                            shuffle=False,
                            pin_memory=False,
                            num_workers=args.numWorkers)
    # net = smp.Unet(classes=2).to(device)
    net = smp.Linknet(classes=1,
                      activation='sigmoid',
                      encoder_name='se_resnext101_32x4d').to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = smploss.DiceLoss(eps=sys.float_info.min).to(device)
    # criterion = DiceLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=.1, momentum=.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduleStep, gamma=0.1)
    runningLoss = []
    st = stGloble = time.time()
    totalIter = len(trainLoader) * args.epoch
    iter = 0
    for epoch in range(args.epoch):
        if epoch != 0 and epoch % args.evalFrequency == 0:
            pass
        if epoch != 0 and epoch % args.saveFrequency == 0:
            modelName = osp.join(args.subModelDir, 'out_{}.pth'.format(epoch))
            state_dict = net.modules.state_dict() if hasattr(net, 'module') else net.state_dict()
            torch.save(state_dict, modelName)

        for img, mask in trainLoader:
            iter += 1
            img = img.to(device)
            mask = mask.to(device, dtype=torch.int64).unsqueeze(1)
            optimizer.zero_grad()
            outputs = net(img)
            # print(outputs.shape, mask.shape)
            # break
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            runningLoss.append(loss.item())
            if iter % args.msgFrequency == 0:
                # writer.add_images('img', img, iter)
                # writer.add_images('mask', mask.unsqueeze(1), iter)
                ed = time.time()
                spend = ed - st
                spendGloable = ed - stGloble
                st = ed
                eta = int((totalIter - iter) * (spendGloable / iter))
                spendGloable = str(datetime.timedelta(seconds=spendGloable))
                eta = str(datetime.timedelta(seconds=eta))
                avgLoss = np.mean(runningLoss)
                runningLoss = []
                lr = optimizer.param_groups[0]['lr']
                msg = '. '.join([
                    'epoch:{epoch}',
                    'iter/total_iter:{iter}/{totalIter}',
                    'lr:{lr:.5f}',
                    'loss:{loss:.4f}',
                    'spend/gloable_spend:{spend:.4f}/{gloable_spend}',
                    'eta:{eta}']).format(
                    epoch=epoch,
                    loss=avgLoss,
                    iter=iter,
                    totalIter=totalIter,
                    spend=spend,
                    gloable_spend=spendGloable,
                    lr=lr,
                    eta=eta)

                logger.info(msg)
                writer.add_scalar('loss', avgLoss, iter)
                writer.add_scalar('lr', lr, iter)

        scheduler.step()

    outName = osp.join(args.subModelDir, 'final.pth')
    torch.save(net.cpu().state_dict(), outName)


if __name__ == '__main__':
    args = parseArgs()
    uniqueName = time.strftime('%y%m%d-%H%M%S')
    args.subModelDir = osp.join(args.modelDir, uniqueName)
    args.subTensorboardDir = osp.join(args.tensorboardDir, uniqueName)
    for subDir in [args.logDir,
                   args.subModelDir,
                   args.subTensorboardDir]:
        if not osp.exists(subDir):
            os.makedirs(subDir)

    logFile = osp.join(args.logDir, uniqueName + '.log')
    logger = getLogger(logFile)
    for k, v in args.__dict__.items():
        logger.info(k)
        logger.info(v)

    main(args, logger)
