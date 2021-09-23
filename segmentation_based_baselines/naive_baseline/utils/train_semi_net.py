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

def normal_thr_output(net_unsup, thr_ske):
    net_unsup_min = net_unsup.min(-1)[0].min(-1)[0]
    net_unsup_max = net_unsup.max(-1)[0].max(-1)[0]
    net_unsup = (net_unsup - net_unsup_min[:, :, None, None]) / (net_unsup_max[:, :, None, None] - net_unsup_min[:, :, None, None])

    thr_mask = (net_unsup >= thr_ske).float()
    net_unsup = net_unsup * thr_mask

    return net_unsup

# # flip and rotate back
# def flip_rotate_back(net_unsup, flip_rotate, idx):
#     # flip_rotate_1************** [tensor([0, 0]), tensor([0, 0]), tensor([180, 270])]
#     net_unsup_idx = net_unsup[idx]
#     angle = -int(flip_rotate[2][idx])
#     net_unsup_idx = TF.rotate(net_unsup_idx, angle)
#
#     whether_hori_flip = flip_rotate[0][idx]
#     whether_ver_flip = flip_rotate[1][idx]
#
#     transform_back = transforms.Compose([
#         transforms.RandomVerticalFlip(whether_ver_flip),
#         transforms.RandomHorizontalFlip(whether_hori_flip)
#     ])
#
#     net_unsup_idx = transform_back(net_unsup_idx)
#
#     return net_unsup_idx

def transform_back(net_unsup, flip_rotate):
    # flip_rotate_1************** [tensor([0, 0]), tensor([0, 0]), tensor([180, 270])]
    for idx in range(net_unsup.size(0)):
        net_unsup_idx = net_unsup[idx]
        angle = -int(flip_rotate[2][idx] / 90)
        # net_unsup_idx = TF.rotate(net_unsup_idx, angle)
        net_unsup_idx = torch.rot90(net_unsup_idx, angle, [1, 2])

        whether_hori_flip = flip_rotate[0][idx]
        whether_ver_flip = flip_rotate[1][idx]

        if whether_hori_flip:
            # net_unsup_idx = net_unsup_idx[:, :, ::-1]
            net_unsup_idx = TFT.hflip(net_unsup_idx)

        if whether_ver_flip:
            # net_unsup_idx = net_unsup_idx[:, ::-1, :]
            net_unsup_idx = TFT.vflip(net_unsup_idx)

        net_unsup[idx] = net_unsup_idx
    return net_unsup

def get_overlap(net_unsup, overlap_ul):
    output = []
    for idx in range(net_unsup.size(0)):
        output.append(net_unsup[idx, :, int(overlap_ul[0][idx]):int(overlap_ul[0][idx])+args.unsup_crop_in_size, int(overlap_ul[1][idx]):int(overlap_ul[1][idx])+args.unsup_crop_in_size])
    output = torch.stack(output, dim=0)
    return output

def logits_run(pos, hog_over_flatten, hog_unover_flatten):
    # mask_idx = (hog_unover_flatten.squeeze(-1).unsqueeze(0) != hog_over_flatten).float()  # [n, b]
    # print('\nmask_idx 111111111111111111111111111111110\n', mask_idx)
    # print('\nmask_idx shape 111111111111111111111111111111110\n', mask_idx.shape)
    neg_idx = (hog_over_flatten * hog_unover_flatten) / args.temp  # [n, 1]
    neg_idx = torch.cat([pos, neg_idx], 1)  # [n, 1+1]
    # mask_idx = torch.cat([torch.ones(mask_idx.size(0), 1).float().cuda(), mask_idx], 1)  # [n, 1+b]
    # print('\nmask_idx 111111111111111111111111111111112\n', mask_idx)
    # print('\nmask_idx shape 111111111111111111111111111111112\n', mask_idx.shape)
    neg_max = torch.max(neg_idx, 1, keepdim=True)[0]  # [n, 1]
    logits_neg_idx = (torch.exp(neg_idx - neg_max)).sum(-1)  # [n, ]
    # # print('\n hog_over_flatten 11111111111111111111111\n',hog_over_flatten)
    # # print('\n hog_over_flatten size 11111111111111111111111\n',hog_over_flatten.shape)
    # print('\nneg_idx 222222222222222222222222222222222\n',neg_idx)
    # # print('\nneg_idx shape222222222222222222222222222222222\n',neg_idx.shape)
    # print('\n neg_max 33333333333333333333333333333333\n',neg_max)
    # print('\n neg_idx - neg_max 4444444444444444444444\n',neg_idx - neg_max)
    # print('\n torch.exp(neg_idx - neg_max) 5555555555555555555555555\n',torch.exp(neg_idx - neg_max))
    # print('\n logits_neg_idx 6666666666666666666666666666\n',logits_neg_idx)

    return logits_neg_idx, neg_max

def moco(pos1, pos2, neg):
    # positive logits: Nx1
	l_pos = torch.einsum('nc,nc->n', [pos1, pos2]).unsqueeze(-1)
	# negative logits: NxK
	l_neg = torch.einsum('nc,ck->nk', [pos1, neg.T.detach()])

	# logits: Nx(1+K)
	logits = torch.cat([l_pos, l_neg], dim=1)

	# apply temperature
	logits /= args.temp

	# labels: positive key indicators
	labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

	return logits, labels


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
    unsup_train = UnsupDataset(args)
    val = BasicDataset(args,True)

    iter_per_epoch = args.iter_per_epoch
    n_train = len(unsup_train)
    #
    writer = SummaryWriter('./records/tensorboard')
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=gamma)
    # initial start epoch
    start_epoch = -1
    best_valid_socre = 0

    if args.resume == True:
        # load best valid score
        path_best_valid_score = args.checkpoints_dir + 'naive_baseline_best_valid_score.txt'
        with open(path_best_valid_score, 'r') as f:
            best_valid_socre = float(f.read())
        f.close()

        # load checkpoint
        path_checkpoint = args.load_checkpoint # checkpont path
        checkpoint = torch.load(path_checkpoint) # load checkpoint
        net.load_state_dict(checkpoint['net']) # load parameter

        if args.whether_fintune:
            global_step = 0
        else:
            # optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
            start_epoch = checkpoint['epoch'] # set epoch
            # lr_schedule.load_state_dict(checkpoint['lr_schedule'])
            global_step = (start_epoch + 1) * (args.iter_per_epoch)

        # load optimizer and lr_scheduler
        if args.resume_epoch > 0:
            for i in range(args.resume_epoch):
                for j in range(args.iter_per_epoch):
                    optimizer.zero_grad()
                    optimizer.step()
                lr_schedule.step()


    else:
        global_step = 0

    # train
    criterion_BCE = nn.BCEWithLogitsLoss()
    criterion_BCEwo = nn.CrossEntropyLoss()
    # criterion_var = var_loss()
    criterion_MSE = nn.MSELoss()

    sup_train_loader = DataLoader(sup_train, batch_size=sup_batch_size, shuffle=True, pin_memory=True, num_workers=4)
    unsup_train_loader = DataLoader(unsup_train, batch_size=unsup_batch_size, shuffle=True, pin_memory=True,
                                    num_workers=4)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, num_workers=4)
    sup_train_loader_iter = iter(sup_train_loader)
    unsup_train_loader_iter = iter(unsup_train_loader)

    for epoch in range(start_epoch + 1, epochs):
        # load supervised and unsupervised data
        tbar = tqdm(range(args.iter_per_epoch), desc=f'Epoch {epoch + 1}/{epochs}', ncols=135)
        print('learning rate **********************:', optimizer.state_dict()['param_groups'][0]['lr'])
        net.train()

        for batch_idx in tbar:
            # load data
            try:
                sup_l = next(sup_train_loader_iter)
            except:
                sup_train_loader_iter = iter(sup_train_loader)
                sup_l = next(sup_train_loader_iter)

            try:
                unsup_l = next(unsup_train_loader_iter)
            except:
                unsup_train_loader_iter = iter(unsup_train_loader)
                unsup_l = next(unsup_train_loader_iter)

            image_l, mask_l, name_l = sup_l['image'], sup_l['mask'], sup_l['name']
            image_l, mask_l = image_l.to(device=device, dtype=torch.float32), mask_l.to(device=device, dtype=torch.float32)


            image_ul, mask_ul, overlap1_ul, overlap2_ul, flip_rotate_1, flip_rotate_2, image_unover, mask_unover, unover_ul, flip_rotate_unover, name = unsup_l['image'], unsup_l['mask'], unsup_l['overlap1_ul'], unsup_l['overlap2_ul'], unsup_l['flip_rotate_1'], unsup_l['flip_rotate_2'], unsup_l['image_unover'], unsup_l['unover_ul'], unsup_l['unover_ul'], unsup_l['flip_rotate_unover'], unsup_l['name']
            image_ul, mask_ul, image_unover = image_ul.to(device=device, dtype=torch.float32), mask_ul.to(device=device, dtype=torch.float32), image_unover.to(device=device, dtype=torch.float32)

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
                image_ul1 = image_ul[:, 0, :, :, :] # [batch_size, 3, H, W]
                image_ul2 = image_ul[:, 1, :, :, :]

                net_unsup1_pre = torch.sigmoid(net(image_ul1))  # [batch_size, 1, H, W]
                net_unsup2_pre = torch.sigmoid(net(image_ul2))

                net_unsup1_pre = transform_back(net_unsup1_pre, flip_rotate_1)  # [1, 384, 384]
                net_unsup2_pre = transform_back(net_unsup2_pre, flip_rotate_2)

                overlap_unsup1 = get_overlap(net_unsup1_pre, overlap1_ul)
                overlap_unsup2 = get_overlap(net_unsup2_pre, overlap2_ul)

                net_unover_pre = torch.sigmoid(net(image_unover))
                net_unover_pre = transform_back(net_unover_pre, flip_rotate_unover) # [2, 1, 256, 256]

                hog_overlap_unsup1= hog(overlap_unsup1) # [2, 12, 32, 32]
                hog_overlap_unsup2= hog(overlap_unsup2) # [2, 12, 32, 32]
                hog_unover= hog(net_unover_pre) # [2, 12, 32, 32]

                b, c, h, w = hog_overlap_unsup1.size()
                hog_overlap_unsup1_flatten = hog_overlap_unsup1.view(-1, 1) # [bchw, 1]
                # print("\n hog_overlap_unsup1_flatten 0000000000000000000000000000\n", hog_overlap_unsup1_flatten)
                # print("\n hog_overlap_unsup1_flatten shape 0000000000000000000000000000\n", hog_overlap_unsup1_flatten.shape)

                hog_overlap_unsup2_flatten = hog_overlap_unsup2.view(-1, 1)
                # print("\n hog_overlap_unsup2_flatten 0000000000000000000000000000\n", hog_overlap_unsup2_flatten)

                hog_unover_flatten = hog_unover.view(-1, 1)
                # print("\n hog_unover_flatten 0000000000000000000000000000\n", hog_unover_flatten)
                # print("\n hog_unover_flatten shape 0000000000000000000000000000\n", hog_unover_flatten.shape)

                # # moco
                # logits1, labels1 = moco(hog_overlap_unsup1_flatten, hog_overlap_unsup2_flatten, hog_unover_flatten)
                # loss1 = criterion_BCEwo(logits1, labels1)
                #
                # logits2, labels2 = moco(hog_overlap_unsup2_flatten, hog_overlap_unsup1_flatten, hog_unover_flatten)
                # loss2 = criterion_BCEwo(logits2, labels2)

                # # cosine similarity
                # eps = 1e-8
                # pos_simi = F.cosine_similarity(hog_overlap_unsup1_flatten, hog_overlap_unsup2_flatten, dim=0) / 2 / args.temp
                # logit_pos = torch.exp(pos_simi)
                # # print('\n pos_simi, logit_pos, 000000000000000000 \n', pos_simi, logit_pos)
                #
                # neg1_simi = F.cosine_similarity(hog_overlap_unsup1_flatten, hog_unover_flatten, dim=0) / 2 / args.temp
                # logit_neg1 = torch.exp(neg1_simi)
                # loss1 = -torch.log(logit_pos / ((logit_pos + logit_neg1) + eps))
                # # print('\n neg1_simi, logit_neg1, loss1 000000000000001\n', neg1_simi, logit_neg1, loss1)
                #
                # neg2_simi = F.cosine_similarity(hog_overlap_unsup2_flatten, hog_unover_flatten, dim=0) / 2 / args.temp
                # logit_neg2 = torch.exp(neg2_simi)
                # loss2 = -torch.log(logit_pos / ((logit_pos + logit_neg2) + eps))
                # # print('\n neg2_simi, logit_neg2, loss2 000000000000002\n', neg2_simi, logit_neg2, loss2)

                # context-aware contrastive loss
                eps = 1e-8
                # positive similarity
                pos1 = (hog_overlap_unsup1_flatten * hog_overlap_unsup2_flatten.detach()).sum(-1, keepdim=True) / args.temp  # [n, 1]
                # print('hog_overlap_unsup1_flatten * hog_overlap_unsup2_flatten.detach() 888888888888888888888888888', (hog_overlap_unsup1_flatten * hog_overlap_unsup2_flatten.detach()))
                pos2 = (hog_overlap_unsup1_flatten.detach() * hog_overlap_unsup2_flatten).sum(-1, keepdim=True) / args.temp  # [n, 1]

                # negative overlap1
                # logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(logits_run, pos1, hog_overlap_unsup1_flatten, hog_unover_flatten)
                logits1_neg_idx, neg_max1 = logits_run(pos1, hog_overlap_unsup1_flatten, hog_unover_flatten)

                logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_neg_idx + eps)
                # print('\n logits1 7777777777777777777777777777777777777770\n',logits1)
                # print('\n pos1 - neg_max1 7777777777777777777777777777777777777770\n',pos1 - neg_max1)
                # print('\n torch.exp(pos1 - neg_max1).squeeze(-1) 7777777777777777777777777777777777777770\n',torch.exp(pos1 - neg_max1).squeeze(-1))
                loss1 = -torch.log(logits1 + eps)
                # print('\n loss1 7777777777777777777777777777777777777770\n',loss1)
                loss1 = loss1.sum()/(loss1.numel() + 1e-12)
                # print('\n loss1 7777777777777777777777777777777777777770\n',loss1)

                # negative overlap2
                # logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(logits_run, pos2,
                #                                                               hog_overlap_unsup2_flatten,
                #                                                               hog_unover_flatten)
                logits2_neg_idx, neg_max2 = logits_run(pos2, hog_overlap_unsup2_flatten, hog_unover_flatten)

                logits2 = torch.exp(pos2 - neg_max2).squeeze(-1) / (logits2_neg_idx + eps)
                # print('\n logits2 7777777777777777777777777777777777777771\n',logits2)
                # print('\n pos2 - neg_max2 7777777777777777777777777777777777777770\n', pos2 - neg_max2)
                # print('\n torch.exp(pos2 - neg_max2).squeeze(-1) 7777777777777777777777777777777777777770\n', torch.exp(pos2 - neg_max2).squeeze(-1))
                loss2 = -torch.log(logits2 + eps)
                # print('\n loss2 7777777777777777777777777777777777777771\n', loss2)
                loss2 = loss2.sum() / (loss2.numel() + 1e-12)
                # print('\n loss2 7777777777777777777777777777777777777771\n', loss2)

                loss_unsup = (loss1 + loss2) * args.loss_unsup_var_weight
                total_loss = loss_sup + loss_unsup
                # print("\n loss1, loss2, loss_unsup 777777777777777777777777777777772\n", loss1, loss2, loss_unsup)

                total_loss.backward()
                optimizer.step()

                global_step += 1
                writer.add_scalar('loss/sup', loss_sup.item(), global_step)
                writer.add_scalar('loss/unsup', loss_unsup.item(), global_step)
                writer.add_scalar('loss/total', total_loss.item(), global_step)
                writer.add_scalar('train', total_loss.item(), global_step)
                tbar.set_postfix(**{'loss_sup': loss_sup.item(), 'loss_unsup': loss_unsup.item(), 'loss_total': total_loss.item(), 'fix': 0})

            # save checkpoints
            # if ((global_step % (args.iter_per_epoch * args.save_per_epoch) == 0) and (epoch + 1) % args.save_per_epoch == 0) or ((global_step % args.iter_start_unsup == 0) and (epoch + 1 == 1)):
            if ((global_step % (args.iter_per_epoch * args.save_per_epoch) == 0) and (epoch + 1) % args.save_per_epoch == 0):
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
        # if ((global_step % (args.iter_per_epoch * args.save_per_epoch) == 0) and (epoch + 1) % args.save_per_epoch == 0) or ((global_step % args.iter_start_unsup == 0) and (epoch + 1 == 1)) :
        if ((global_step % (args.iter_per_epoch * args.save_per_epoch) == 0) and (epoch + 1) % args.save_per_epoch == 0):
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