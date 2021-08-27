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
from torch.utils.data import DataLoader
from torch.autograd import Variable
from arguments import get_parser, update_dir_train, update_dir_test, update_dir_resume, update_dir_match
from utils.train_net import train_net

from itertools import cycle
from models.skeleton import soft_skel
from models.gabore_filter_bank import GaborFilters
from models.hog import HOGLayer
import torchvision.transforms.functional as TF
from torchvision import transforms
from losses.var_loss import var_loss

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')

def normal_thr_output(net_unsup, thr_ske):
    net_unsup_min = net_unsup.min(-1)[0].min(-1)[0]
    net_unsup_max = net_unsup.max(-1)[0].max(-1)[0]
    net_unsup = (net_unsup - net_unsup_min[:, :, None, None]) / (net_unsup_max[:, :, None, None] - net_unsup_min[:, :, None, None])

    thr_mask = (net_unsup >= thr_ske).float()
    net_unsup = net_unsup * thr_mask

    return net_unsup

# flip and rotate back
def flip_rotate_back(net_unsup, flip_rotate, idx):
    # flip_rotate_1************** [tensor([0, 0]), tensor([0, 0]), tensor([180, 270])]
    net_unsup_idx = net_unsup[idx]
    angle = -int(flip_rotate[2][idx])
    net_unsup_idx = TF.rotate(net_unsup_idx, angle)

    whether_hori_flip = flip_rotate[0][idx]
    whether_ver_flip = flip_rotate[1][idx]

    transform_back = transforms.Compose([
        transforms.RandomVerticalFlip(whether_ver_flip),
        transforms.RandomHorizontalFlip(whether_hori_flip)
    ])

    net_unsup_idx = transform_back(net_unsup_idx)

    return net_unsup_idx

# # crop overlap region
# def crop_output(net_unsup, ul_position, idx, crop_h, crop_w):
#     h_position = int(ul_position[0][idx])
#     w_position = int(ul_position[1][idx])
#     overlap = net_unsup[:, h_position:h_position+crop_h, w_position:w_position+crop_w]
#
#     return overlap


def train_semi_net(net,
              garbo_filter,
              hog,
              args,
              epochs,
              sup_batch_size,
              unsup_batch_size,
              lr,
              gamma,
              lr_steps,
              device):

    sup_train = BasicDataset(args,False)
    unsup_pos_train = UnsupDataset_pos(args)
    unsup_neg_train = UnsupDataset_neg(args)
    val = BasicDataset(args,True)

    iter_per_epoch = args.iter_per_epoch
    n_train = len(unsup_pos_train)
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
        global_step = (start_epoch + 1) * (args.iter_per_epoch)

        # load optimizer and lr_scheduler
        for i in range(args.resume_epoch):
            for j in range(args.iter_per_epoch):
                optimizer.zero_grad()
                optimizer.step()
            lr_schedule.step()


    else:
        global_step = 0

    # train
    criterion_BCE = nn.BCEWithLogitsLoss()
    # criterion_var = var_loss()
    criterion_MSE = nn.MSELoss()

    for epoch in range(start_epoch + 1, epochs):
        # load supervised and unsupervised data
        sup_train_loader = DataLoader(sup_train, batch_size=sup_batch_size, shuffle=True, pin_memory=True, num_workers=4)
        unsup_train_pos_loader = DataLoader(unsup_pos_train, batch_size=unsup_batch_size, shuffle=True, pin_memory=True, num_workers=4)
        unsup_train_neg_loader = DataLoader(unsup_neg_train, batch_size=unsup_batch_size, shuffle=True, pin_memory=True, num_workers=4)
        train_loader = iter(zip(cycle(sup_train_loader), cycle(unsup_train_pos_loader), cycle(unsup_train_neg_loader)))
        tbar = tqdm(range(args.iter_per_epoch), desc=f'Epoch {epoch + 1}/{epochs}', ncols=135)

        print('learning rate **********************:', optimizer.state_dict()['param_groups'][0]['lr'])

        net.train()

        for batch_idx in tbar:
            # load data
            sup_l, unsup_pos_l, unsup_neg_l = next(train_loader)

            image_l, mask_l, name_l = sup_l['image'], sup_l['mask'], sup_l['name']
            image_l, mask_l = image_l.to(device=device, dtype=torch.float32), mask_l.to(device=device, dtype=torch.float32)
            image_l, mask_l = image_l.cuda(non_blocking=True), mask_l.cuda(non_blocking=True)

            # image_ul, mask_ul, overlap1_ul, overlap2_ul, overlap3_ul, overlap4_ul, flip_rotate_1, flip_rotate_2, flip_rotate_3, flip_rotate_4, name = \
            #     unsup_l['image'], unsup_l['mask'], unsup_l['overlap1_ul'], unsup_l['overlap2_ul'], unsup_l['overlap3_ul'], unsup_l['overlap4_ul'], \
            #     unsup_l['flip_rotate_1'], unsup_l['flip_rotate_2'], unsup_l['flip_rotate_3'], unsup_l['flip_rotate_4'], unsup_l['name']
            image_pos_ul, mask_pos_ul, overlap1_pos_ul, overlap2_pos_ul, flip_rotate_pos_1, flip_rotate_pos_2, name_pos = \
                unsup_pos_l['image'], unsup_pos_l['mask'], unsup_pos_l['overlap1_ul'], unsup_pos_l['overlap2_ul'], \
                unsup_pos_l['flip_rotate_1'], unsup_pos_l['flip_rotate_2'], unsup_pos_l['name']
            image_neg_ul, mask_neg_ul, overlap1_neg_ul, overlap2_neg_ul, flip_rotate_neg_1, flip_rotate_neg_2, name_neg = \
                unsup_neg_l['image'], unsup_neg_l['mask'], unsup_neg_l['overlap1_ul'], unsup_neg_l['overlap2_ul'], \
                unsup_neg_l['flip_rotate_1'], unsup_neg_l['flip_rotate_2'], unsup_neg_l['name']

            image_pos_ul, mask_pos_ul = image_pos_ul.to(device=device, dtype=torch.float32), mask_pos_ul.to(device=device, dtype=torch.float32)
            image_pos_ul, mask_pos_ul = image_pos_ul.cuda(non_blocking=True), mask_pos_ul.cuda(non_blocking=True)
            image_neg_ul, mask_neg_ul = image_neg_ul.to(device=device, dtype=torch.float32), mask_neg_ul.to(device=device, dtype=torch.float32)
            image_neg_ul, mask_neg_ul = image_neg_ul.cuda(non_blocking=True), mask_neg_ul.cuda(non_blocking=True)

            optimizer.zero_grad()

            if global_step < args.iter_start_unsup:
                # supervised loss (only)
                sup_pred = net(image_l)
                loss_sup = criterion_BCE(sup_pred, mask_l) * args.loss_sup_weight
                total_loss = loss_sup
                total_loss.backward()
                optimizer.step()

                global_step += 1
                writer.add_scalar('loss/sup', loss_sup.item(), global_step)
                writer.add_scalar('loss/unsup', 0, global_step)
                writer.add_scalar('loss/total', total_loss.item(), global_step)
                writer.add_scalar('train', total_loss.item(), global_step)
                tbar.set_postfix(**{'loss_sup': loss_sup.item(), 'loss_unsup': 0, 'loss_total': total_loss.item()})
            else:
                # supervised net and loss
                net_sup_pred = net(image_l)
                loss_sup = criterion_BCE(net_sup_pred, mask_l) * args.loss_sup_weight

                # unsupervised net
                # image_ul: [batch_size, 4, 3, H, W]
                image_pos_ul1 = image_pos_ul[:, 0, :, :, :] # [batch_size, 3, H, W]
                image_pos_ul2 = image_pos_ul[:, 1, :, :, :]
                image_neg_ul1 = image_neg_ul[:, 0, :, :, :] # [batch_size, 3, H, W]
                image_neg_ul2 = image_neg_ul[:, 1, :, :, :]


                # Unet, skeleton, gabor
                net_pos_unsup1_pre = net(image_pos_ul1) # [batch_size, 1, H, W]
                net_pos_unsup1_pre = normal_thr_output(net_pos_unsup1_pre, args.unsup_thr_ske_train) # [batch_size, 1, H, W]

                net_pos_unsup2_pre = net(image_pos_ul2)
                net_pos_unsup2_pre = normal_thr_output(net_pos_unsup2_pre, args.unsup_thr_ske_train)

                net_neg_unsup1_pre = net(image_neg_ul1) # [batch_size, 1, H, W]
                net_neg_unsup1_pre = normal_thr_output(net_neg_unsup1_pre, args.unsup_thr_ske_train) # [batch_size, 1, H, W]

                net_neg_unsup2_pre = net(image_neg_ul2)
                net_neg_unsup2_pre = normal_thr_output(net_neg_unsup2_pre, args.unsup_thr_ske_train)


                # get overlap part
                overlap_pos_unsup1_list = []
                overlap_pos_unsup2_list = []
                overlap_neg_unsup1_list = []
                overlap_neg_unsup2_list = []

                for idx in range(net_pos_unsup1_pre.size(0)):
                    net_pos_unsup1_back_idx = flip_rotate_back(net_pos_unsup1_pre, flip_rotate_pos_1, idx) # [1, 384, 384]
                    overlap_pos_unsup1_list.append(net_pos_unsup1_back_idx[:, int(overlap1_pos_ul[0][idx]):int(overlap1_pos_ul[0][idx])+args.unsup_crop_in_size, int(overlap1_pos_ul[1][idx]):int(overlap1_pos_ul[1][idx])+args.unsup_crop_in_size]) # [1, 256, 256]
                    net_pos_unsup2_back_idx = flip_rotate_back(net_pos_unsup2_pre, flip_rotate_pos_2, idx)
                    overlap_pos_unsup2_list.append(net_pos_unsup2_back_idx[:, int(overlap2_pos_ul[0][idx]):int(overlap2_pos_ul[0][idx])+args.unsup_crop_in_size, int(overlap2_pos_ul[1][idx]):int(overlap2_pos_ul[1][idx])+args.unsup_crop_in_size])

                for idx in range(net_neg_unsup1_pre.size(0)):
                    net_neg_unsup1_back_idx = flip_rotate_back(net_neg_unsup1_pre, flip_rotate_neg_1, idx) # [1, 384, 384]
                    overlap_neg_unsup1_list.append(net_neg_unsup1_back_idx[:, int(overlap1_neg_ul[0][idx]):int(overlap1_neg_ul[0][idx])+args.unsup_crop_in_size, int(overlap1_neg_ul[1][idx]):int(overlap1_neg_ul[1][idx])+args.unsup_crop_in_size]) # [1, 256, 256]
                    net_neg_unsup2_back_idx = flip_rotate_back(net_neg_unsup2_pre, flip_rotate_neg_2, idx)
                    overlap_neg_unsup2_list.append(net_neg_unsup2_back_idx[:, int(overlap2_neg_ul[0][idx]):int(overlap2_neg_ul[0][idx])+args.unsup_crop_in_size, int(overlap2_neg_ul[1][idx]):int(overlap2_neg_ul[1][idx])+args.unsup_crop_in_size])

                overlap_pos_unsup1 = torch.cat(overlap_pos_unsup1_list, 0).unsqueeze(1)  # [1, 1, 256, 256]
                overlap_neg_unsup1 = torch.cat(overlap_neg_unsup1_list, 0).unsqueeze(1)  # [1, 1, 256, 256]
                overlap_unsup1 = torch.cat((overlap_pos_unsup1, overlap_neg_unsup1), 0)  # [2, 1, 256, 256]
                hog_overlap_unsup1= hog(overlap_unsup1)                          # [2, 12, 32, 32]

                overlap_pos_unsup2 = torch.cat(overlap_pos_unsup2_list, 0).unsqueeze(1)  # [1, 1, 256, 256]
                overlap_neg_unsup2 = torch.cat(overlap_neg_unsup2_list, 0).unsqueeze(1)  # [1, 1, 256, 256]
                overlap_unsup2 = torch.cat((overlap_pos_unsup2, overlap_neg_unsup2), 0)  # [2, 1, 256, 256]
                hog_overlap_unsup2= hog(overlap_unsup2)

                # loss_unsup = criterion_var(overlap_unsup1, overlap_unsup2, overlap_unsup3, overlap_unsup4) * args.loss_unsup_var_weight
                loss_unsup = criterion_MSE(hog_overlap_unsup1, hog_overlap_unsup2) * args.loss_unsup_var_weight
                # print("loss_unsup************************", loss_unsup)
                total_loss = loss_sup + loss_unsup
                # print("total_loss************************", total_loss)

                total_loss.backward()
                optimizer.step()

                global_step += 1
                writer.add_scalar('loss/sup', loss_sup.item(), global_step)
                writer.add_scalar('loss/unsup', loss_unsup.item(), global_step)
                writer.add_scalar('loss/total', total_loss.item(), global_step)
                writer.add_scalar('train', total_loss.item(), global_step)
                tbar.set_postfix(**{'loss_sup': loss_sup.item(), 'loss_unsup': loss_unsup.item(), 'loss_total': total_loss.item(), 'fix': 0})

            # save checkpoints
            if ((global_step % (args.iter_per_epoch * args.save_per_epoch) == 0) and (epoch + 1) % args.save_per_epoch == 0) or ((global_step % args.iter_start_unsup == 0) and (epoch + 1 == 1)):
                checkpoint = {
                    "net": net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': lr_schedule.state_dict()
                }
                if not os.path.isdir(args.checkpoints_dir):
                    os.mkdir(args.checkpoints_dir)
                # save checkpoint
                torch.save(checkpoint,
                           args.checkpoints_dir + 'naive_baseline_{}.pth'.format(epoch))

        # validdation
        if ((global_step % (args.iter_per_epoch * args.save_per_epoch) == 0) and (epoch + 1) % args.save_per_epoch == 0) or ((global_step % args.iter_start_unsup == 0) and (epoch + 1 == 1)) :
        # if (global_step % 31968 == 0) :
            # save validation
            val_loader = DataLoader(val, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, num_workers=4)
            valid_score = eval_net(args, net, val_loader, device)
            writer.add_scalar('valid', valid_score, global_step)
            # save checkpoint
            checkpoint_best = {
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
                torch.save(checkpoint_best,
                           args.checkpoints_dir + 'naive_baseline_best.pth')

        lr_schedule.step()

def skeleton(args):
    print('Start skeletonization...')
    with open('./dataset/data_split.json','r') as jf:
        json_data = json.load(jf)['test']
    skel_list = [x+'.png' for x in json_data]
    with tqdm(total=len(skel_list), unit='img') as pbar:
        # thr = 0.2
        thr = args.skeleton_thr
        for i,seg in enumerate(skel_list):
            seg_name = os.path.join('./records/test/segmentation',seg)
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
    if args.test:
        update_dir_test(args)
    elif args.match:
        update_dir_match(args)
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
        else:
            test = BasicDataset(args)
            test_loader = DataLoader(test, batch_size=1, shuffle=False,  pin_memory=True, drop_last=False)
            eval_net(args,net, test_loader, device)
            skeleton(args)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
