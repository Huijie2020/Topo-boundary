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
from match import match_net
from models import UNet
from PIL import Image
from skimage.morphology import skeletonize
from skimage import measure

from tensorboardX import SummaryWriter
from utils.dataset import BasicDataset
from utils.unsupdataset import UnsupDataset
from utils.matchdataset import MatchDataset
from utils.pos_dataset import UnsupDataset_pos
from utils.neg_dataset import UnsupDataset_neg
from utils.grow_dataset import GrowDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from arguments import get_parser, update_dir_train, update_dir_test, update_dir_resume, update_dir_match
from utils.train_net import train_net
from utils.train_semi_net import train_semi_net

from itertools import cycle
from models.skeleton import soft_skel
from models.gabore_filter_bank import GaborFilters
from models.hog import HOGLayer
import torchvision.transforms.functional as TF
from torchvision import transforms
from losses.var_loss import var_loss
import cv2
import torchvision.transforms.functional_tensor as TFT
import torch.nn.functional as F

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')

def skeleton(args):
    print('Start skeletonization...')
    with open('./dataset/data_split.json','r') as jf:
        json_data = json.load(jf)['test']
    # with open('./records/test/100epoch_256_20000img.json','r') as jf:
    #     json_data = json.load(jf)['img_id']
    skel_list = [x+'.png' for x in json_data]
    with tqdm(total=len(skel_list), unit='img') as pbar:
        # thr = 0.2
        thr = args.skeleton_thr
        for i,seg in enumerate(skel_list):
            seg_name = os.path.join('./records/test/segmentation',seg)
            # seg_name = os.path.join('./records/test/segmentation_overlap',seg)
            # img = np.array(Image.open(seg_name))[:,:,0] / 255
            img = np.array(Image.open(seg_name)) / 255
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
                if len(instance_points[0]) < args.skeleton_ig_line:
                    seg_skeleton[instance_points] = 0
            Image.fromarray(seg_skeleton).convert('L').save(os.path.join('./records/test/skeleton',seg))
            pbar.update()
    print('Finish skeletonization...')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('---------------------------args--------------------------------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('---------------------------args--------------------------------')
    if args.test:
        update_dir_test(args)
    elif args.match:
        update_dir_match(args)
    else:
        if args.resume == True:
            update_dir_resume(args)
        else:
            update_dir_train(args)
    device = torch.device(args.device)
    # net = UNet(n_channels=4,n_classes=2,bilinear=True)
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
    if args.match:
        path_checkpoint = args.load_checkpoint  # checkpont path
        checkpoint = torch.load(path_checkpoint, map_location='cpu')  # load checkpoint
        net.load_state_dict(
            checkpoint["net"]
        )

    net.to(device=device)
    garbo_filter = GaborFilters(in_channels=1).to(device=device)
    hog = HOGLayer(nbins=12, pool=8).to(device=device)

    try:
        if (not args.test) and (not args.match):
            if args.semi == True:
                train_semi_net(net=net,
                               garbo_filter = garbo_filter,
                               hog = hog,
                               args=args,
                               epochs=args.epochs,
                               sup_batch_size=args.sup_batch_size,
                               unsup_batch_size=args.unsup_batch_size,
                               lr=args.lr,
                               gamma=args.gamma,
                               lr_steps=args.lr_steps,
                               device=device)
            else:
                train_net(net=net,
                        args=args,
                        epochs=args.epochs,
                        batch_size=args.sup_batch_size,
                        lr=args.lr,
                        gamma=args.gamma,
                        lr_steps=args.lr_steps,
                        device=device)
        elif args.match:
            match_data = MatchDataset(args)
            match_loader = DataLoader(match_data, batch_size=1, shuffle=False,  pin_memory=True, drop_last=False)
            match_net(args,net, hog, match_loader, device)
            skeleton(args)
        else:
            test = BasicDataset(args)
            test_loader = DataLoader(test, batch_size=1, shuffle=False,  pin_memory=True, drop_last=False)
            eval_net(args,net, test_loader, device)
            # skeleton(args)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
