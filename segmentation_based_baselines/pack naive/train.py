import argparse
import logging
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from models import UNet
from PIL import Image
from skimage.morphology import skeletonize
from skimage import measure

from tensorboardX import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from arguments import get_parser, update_dir_train, update_dir_test, update_dir_resume
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')

def train_net(net,
              args,
              epochs,
              batch_size,
              lr,
              device):

    train = BasicDataset(args,False)
    val = BasicDataset(args,True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    val_loader = DataLoader(val, batch_size=1, shuffle=False,  pin_memory=True, drop_last=False, num_workers=1)
    n_val = len(val)
    n_train = len(train) 
    # 
    writer = SummaryWriter('./records/tensorboard')
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # initial start epoch
    start_epoch = -1
    if args.resume:
        path_checkpoint = args.load_checkpoint # checkpont path
        checkpoint = torch.load(path_checkpoint) # load checkpoint
        net.load_state_dict(checkpoint['net']) # load parameter
        # net.load_state_dict(checkpoint) # load parameter
        optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
        start_epoch = checkpoint['epoch'] # set epoch
        global_step = (start_epoch + 1) * (n_train // (batch_size))
    else:
        global_step = 0

    # 
    criterion = nn.BCEWithLogitsLoss()
    train_loss = []
    valid_scores = []
    for epoch in range(start_epoch + 1, epochs):
        net.train()
        epoch_loss = 0
        # train_loss = []
        # valid_scores = []

        # np.load()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                masks = batch['mask']
                name = batch['name'][0]
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.float32)
                #
                masks_pred = net(imgs)
                loss = criterion(masks_pred,masks)
                pbar.set_postfix(**{'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])
                #
                global_step += 1
                writer.add_scalar('train', loss.item(), global_step)
                # save loss and plot
                train_loss.append(float(loss.item()))
                if (global_step % 10  == 0):
                    plt.figure()
                    plt.plot(train_loss)
                    plt.savefig('./records/loss_plt/plot/loss_plt_{}.jpg'.format(global_step))
                    np.save('./records/loss_plt/np/loss_plt_{}.npy'.format(global_step), train_loss)

                if (global_step % (n_train // (batch_size)) == 0):
                    # svae validation and plot
                    valid_score = eval_net(args,net,val_loader,device)
                    valid_scores.append(float(valid_score))
                    plt.figure()
                    plt.plot(valid_scores)
                    plt.savefig('./records/val_plt/plot/val_plt_{}.jpg'.format(global_step))
                    np.save('./records/val_plt/np/val_plt_{}.npy'.format(global_step), valid_scores)
                    # scheduler.step(val_score)
                    writer.add_scalar('valid', valid_score, global_step)
                    # save checkpoint
                    checkpoint = {
                        "net": net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch
                    }
                    if not os.path.isdir(args.checkpoints_dir):
                        os.mkdir(args.checkpoints_dir)
                    torch.save(checkpoint,
                               args.checkpoints_dir + 'naive_baseline_{}.pth'.format(epoch))
                    # torch.save(net.state_dict(),
                    #     args.checkpoints_dir + 'naive_baseline_{}.pth'.format(epoch))

def skeleton():
    print('Start skeletonization...')
    with open('./dataset/data_split.json','r') as jf:
        json_data = json.load(jf)['test']
    skel_list = [x+'.png' for x in json_data]
    with tqdm(total=len(skel_list), unit='img') as pbar:
        thr = 0.2
        for i,seg in enumerate(skel_list):
            seg_name = os.path.join('./records/test/segmentation',seg)
            img = np.array(Image.open(seg_name))[:,:,0] / 255
            # img = np.array(Image.open(seg_name)) / 255
            img = img / (np.max(img))
            # binarization
            img = (img > thr)
            # skeletonization
            seg_skeleton = skeletonize(img, method='lee')
            instances = measure.label(seg_skeleton / 255,background=0)
            indexs = np.unique(instances)[1:]
            # remove too short segments as outliers
            for index in indexs:
                instance_map = (instances == index)
                instance_points = np.where(instance_map==1)
                if len(instance_points[0]) < 30:
                    seg_skeleton[instance_points] = 0
            Image.fromarray(seg_skeleton).convert('RGB').save(os.path.join('./records/test/skeleton',seg))
            pbar.update()
    print('Finish skeletonization...')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.test:
        update_dir_test(args)
    else:
        if args.resume:
            update_dir_resume(args)
        else:
            update_dir_train(args)
    device = torch.device(args.device)
    # net = UNet(n_channels=4,n_classes=1,bilinear=True)
    net = UNet(n_channels=3, n_classes=1, bilinear=True)

    # test
    if args.test:
        # net.load_state_dict(
        #     torch.load(args.load_checkpoint, map_location='cpu')
        # )
        path_checkpoint = args.load_checkpoint  # checkpont path
        checkpoint = torch.load(path_checkpoint, map_location='cpu')  # load checkpoint
        net.load_state_dict(
            checkpoint["net"]
        )
    net.to(device=device)
    try:
        if not args.test:
            train_net(net=net,
                    args=args,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device)
        else:
            test = BasicDataset(args)
            test_loader = DataLoader(test, batch_size=1, shuffle=False,  pin_memory=True, drop_last=True)
            eval_net(args,net, test_loader, device)
            skeleton()

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
