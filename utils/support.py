import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.HardNet.HarDMSEG import HarDMSEG
from utils.dataloader import get_loader,test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
from torchstat import stat


def train(model, loader, optimizer, loss_fn,trainsize):
    optimizer.zero_grad()
    epoch_loss = 0
    size_rates = [0.75, 1, 1.25]
    model.train()
    for i, (x, y) in enumerate(loader):
        for rate in size_rates:
            images = Variable(x).cuda()
            gts = Variable(y).cuda()
            trainsize = int(round(trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)    
            lateral_map_5=model(images)
            loss5 = loss_fn(lateral_map_5,gts)
            loss5.backward()
            optimizer.step()
            epoch_loss += loss5.item()
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn,trainsize):
    epoch_loss = 0
    size_rates = [0.75, 1, 1.25]
    model.train()
    for i, (x, y) in enumerate(loader):
        for rate in size_rates:
            images = Variable(x).cuda()
            gts = Variable(y).cuda()
            trainsize = int(round(trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)    
            lateral_map_5=model(images)
            loss5 = loss_fn(lateral_map_5,gts)
            epoch_loss += loss5.item()
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def test(model, path):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####
    
    model.eval()
    # image_root = '{}/images/'.format(data_path)
    # gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(data_path, 352)
    b=0.0
    for i in range(100):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        res  = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)
        
        loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        a =  '{:.4f}'.format(loss)
        a = float(a)
        b = b + a
        
    return b/100

