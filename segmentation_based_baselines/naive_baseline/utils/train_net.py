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

import torch.nn.functional as F

def train_net(net,
              args,
              epochs,
              batch_size,
              lr,
              gamma,
              lr_steps,
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
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=gamma)
    # initial start epoch
    start_epoch = -1
    best_valid_socre = 0

    if args.resume:
        # load best valid score
        path_best_valid_score = args.checkpoints_dir + 'naive_baseline_best_valid_score.txt'
        with open(path_best_valid_score, 'r') as f:
            best_valid_socre = float(f.read())
        f.close()

        # load checkpoint
        path_checkpoint = args.load_checkpoint # checkpont path
        checkpoint = torch.load(path_checkpoint) # load checkpoint
        net.load_state_dict(checkpoint['net']) # load parameter
        # optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
        start_epoch = checkpoint['epoch'] # set epoch
        # lr_schedule.load_state_dict(checkpoint['lr_schedule'])
        global_step = (start_epoch + 1) * (n_train // batch_size + 1)

        if args.resume_epoch > 0:
            for i in range(args.resume_epoch):
                for j in range(args.iter_per_epoch):
                    optimizer.zero_grad()
                    optimizer.step()
                lr_schedule.step()

    else:
        global_step = 0
    #
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(start_epoch + 1, epochs):
        net.train()
        #lr_schedule.step()
        print('learning rate **********************:', optimizer.state_dict()['param_groups'][0]['lr'])
        epoch_loss = 0
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
                # save checkpoint
                # if (global_step % (n_train // (batch_size)) == 0):
                # if (epoch + 1) % 2 == 0:
                if (global_step % ((n_train // batch_size) + 1) == 0) and (epoch + 1) % 200 == 0:
                    # svae validation and plot
                    valid_score = eval_net(args,net,val_loader,device)
                    # scheduler.step(val_score)
                    writer.add_scalar('valid', valid_score, global_step)
                    # save checkpoint
                    checkpoint = {
                        "net": net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": epoch,
                        'lr_schedule': lr_schedule.state_dict()
                    }
                    if not os.path.isdir(args.checkpoints_dir):
                        os.mkdir(args.checkpoints_dir)
                    # save best model
                    if valid_score > best_valid_socre:
                        best_valid_socre = valid_score
                        with open(args.checkpoints_dir + 'naive_baseline_best_valid_score.txt', 'w') as f:
                            f.write(str(best_valid_socre))
                        f.close()
                        torch.save(checkpoint,
                                   args.checkpoints_dir + 'naive_baseline_best.pth')
                    # save checkpoint
                    torch.save(checkpoint,
                               args.checkpoints_dir + 'naive_baseline_{}.pth'.format(epoch))
                    # torch.save(net.state_dict(),
                    #     args.checkpoints_dir + 'naive_baseline_{}.pth'.format(epoch))
        lr_schedule.step()