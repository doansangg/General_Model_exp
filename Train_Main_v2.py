import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib import HarDMSEG_v2
from utils.dataloader import get_loader,test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
from torchstat import stat
from Loss.loss import DiceBCELoss,DiceBCELogCoshLoss,bce_iou_loss
from utils.support import train,evaluate
import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    
    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')
    
    parser.add_argument('--batchsize', type=int,
                        default=32, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    
    parser.add_argument('--train_path', type=str,
                        default='path_data/train_kvasir.txt', help='path to train dataset')
    
    parser.add_argument('--test_path', type=str,
                        default='path_data/test_kvasir_100.txt' , help='path to testing Kvasir dataset')
    
    parser.add_argument('--train_save', type=str,
                        default='HarD-MSEG-best')
    
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = HarDMSEG_v2().cuda('1')
    device = torch.device('cuda')

    params = model.parameters()
    
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-4, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    print(optimizer)
    # image_root = '{}/images/'.format(opt.train_path)
    # gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(opt.train_path, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = opt.augmentation)
    val_loader = get_loader(opt.train_path, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = opt.augmentation)
    
    total_step = len(train_loader)
    loss_fn = bce_iou_loss()
    print("#"*20, "Start Training", "#"*20)
    checkpoint_path='snapshots/good_best_weights.pth'
    for epoch in range(1,opt.epoch):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn,opt.trainsize)
        valid_loss = evaluate(model, val_loader, loss_fn,opt.trainsize)
        scheduler.step(valid_loss)
        best_valid_loss = float('inf')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

        if (epoch+1) % 1 == 0:
            meandice = test(model,'path_data/test_kvasir.txt')
                #torch.save(model.state_dict(), save_path + 'dice_loss.pth' )
            print('meandice: ',meandice)

    # for epoch in range(1, opt.epoch):
    #     adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
    #     train(train_loader, model, optimizer, epoch, opt.test_path)
