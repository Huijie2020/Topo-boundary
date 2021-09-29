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
from utils.unsupdataset_contr import UnsupDataset
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
from models.projector import projection
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

def get_overlap(net_unsup, overlap_ul, crop_size):
    output = []
    for idx in range(net_unsup.size(0)):
        output.append(net_unsup[idx, :, int(overlap_ul[0][idx]):int(overlap_ul[0][idx])+crop_size, int(overlap_ul[1][idx]):int(overlap_ul[1][idx])+crop_size])
    output = torch.stack(output, dim=0)
    return output

def mask_repeat(pre_mask):
    pre_mask = pre_mask.squeeze(1) #[2, 1, 32, 32] ->[2, 32, 32]
    output = []
    for idx in range(pre_mask.size(0)):
        output.append(pre_mask[idx].unsqueeze(0).repeat(12, 1, 1))
    output = torch.stack(output, dim=0)
    return output


def generate_mask(binary_base, binary_compare, pooler):
    mask = pooler(torch.abs(binary_base - binary_compare)) #detach
    mask = (mask > args.hog_thr).float()
    mask = mask_repeat(mask)
    mask_flatten = mask.view(-1, 1) # [n, 1]
    return mask_flatten

def logits_run(pos, hog_over_flatten, hog_neg, mask):
    neg_idx = (hog_over_flatten * hog_neg) / args.temp  # [n, 4]
    neg_idx = torch.cat([pos, neg_idx], 1)  # [n, 1+4]
    mask_idx = torch.cat([torch.ones(mask.size(0), 1).float().cuda(), mask], 1)  # [n, 1+b]
    neg_max = torch.max(neg_idx, 1, keepdim=True)[0]  # [n, 1]
    logits_neg_idx = (torch.exp(neg_idx - neg_max) * mask_idx).sum(-1)  # [n, ]
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
              projector,
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
            optimizer.load_state_dict(checkpoint['optimizer']) # load optimizer
            start_epoch = checkpoint['epoch'] # set epoch
            lr_schedule.load_state_dict(checkpoint['lr_schedule'])
            global_step = (start_epoch + 1) * (args.iter_per_epoch)

        # # load optimizer and lr_scheduler
        # if args.resume_epoch > 0:
        #     for i in range(args.resume_epoch):
        #         for j in range(args.iter_per_epoch):
        #             optimizer.zero_grad()
        #             optimizer.step()
        #         lr_schedule.step()

        # val_loader = DataLoader(val, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, num_workers=1)
        # valid_score = eval_net(args, net, val_loader, device)
        # writer.add_scalar('valid', valid_score, global_step)
        # # save checkpoint
        # checkpoint_best = {
        #     "net": net.state_dict(),
        #     # 'optimizer': optimizer.state_dict(),
        #     "epoch": start_epoch
        #     # 'lr_schedule': lr_schedule.state_dict()
        # }
        # if not os.path.isdir(args.checkpoints_dir):
        #     os.mkdir(args.checkpoints_dir)
        # # save best model
        # if valid_score > best_valid_socre:
        #     best_valid_socre = valid_score
        #     with open(args.checkpoints_dir + 'naive_baseline_best_valid_score.txt', 'w') as f:
        #         f.write(str(best_valid_socre))
        #     f.close()
        #     torch.save(checkpoint_best,
        #                args.checkpoints_dir + 'naive_baseline_best.pth')
    else:
        global_step = 0

    # train
    criterion_BCE = nn.BCEWithLogitsLoss()
    criterion_BCEwo = nn.CrossEntropyLoss()
    criterion_MSE = nn.MSELoss()
    pooler = nn.AvgPool2d((8, 8), stride=(8, 8), padding=0, ceil_mode=False, count_include_pad=True)

    sup_train_loader = DataLoader(sup_train, batch_size=sup_batch_size, shuffle=True, pin_memory=True, num_workers=4)
    unsup_train_loader = DataLoader(unsup_train, batch_size=unsup_batch_size, shuffle=True, pin_memory=True,
                                    num_workers=4)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, num_workers=1)
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

                # unsupervised net for feature prediction
                # image_ul: [batch_size, 4, 3, H, W]
                image_ul1 = image_ul[:, 0, :, :, :] # [batch_size, 3, H, W]
                image_ul2 = image_ul[:, 1, :, :, :]

                net_unsup1_pre = torch.sigmoid(net(image_ul1))  # [batch_size, 1, H, W]
                net_unsup2_pre = torch.sigmoid(net(image_ul2))

                net_unsup1_pre = transform_back(net_unsup1_pre, flip_rotate_1)  # [batich_size, 1, 384, 384]
                net_unsup2_pre = transform_back(net_unsup2_pre, flip_rotate_2)

                overlap_unsup1 = get_overlap(net_unsup1_pre, overlap1_ul, args.unsup_crop_in_size)
                overlap_unsup2 = get_overlap(net_unsup2_pre, overlap2_ul, args.unsup_crop_in_size)

                net_unover_pre = torch.sigmoid(net(image_unover))
                net_unover_pre = transform_back(net_unover_pre, flip_rotate_unover)  # [2, 1, 256, 256]
                net_unover = get_overlap(net_unover_pre, unover_ul, args.unsup_unover_in_size)  # [2, 1, 256, 256]

                # generate binary map for mask
                overlap_unsup1_binary = (overlap_unsup1 > args.mask_thr).float()
                overlap_unsup2_binary = (overlap_unsup2 > args.mask_thr).float()
                unvoer_binary = (net_unover > args.mask_thr).float()

                # # negative MSE
                # # generate mask for negative sample
                # mask1 = generate_mask(overlap_unsup1_binary, unvoer_binary, pooler)
                # mask2 = generate_mask(overlap_unsup2_binary, unvoer_binary, pooler)

                # generate binary maps of negative sameple from existing samples
                # overlap_unsup1_reverse_binary = torch.zeros(overlap_unsup1_binary.shape).float().cuda()
                # overlap_unsup1_reverse_binary[0] = overlap_unsup1_binary[1]
                # overlap_unsup1_reverse_binary[1] = overlap_unsup1_binary[0]
                #
                # overlap_unsup2_reverse_binary = torch.zeros(overlap_unsup2_binary.shape).float().cuda()
                # overlap_unsup2_reverse_binary[0] = overlap_unsup2_binary[1]
                # overlap_unsup2_reverse_binary[1] = overlap_unsup2_binary[0]

                unover_reverse_binary = torch.zeros(unvoer_binary.shape).float().cuda()
                unover_reverse_binary[0] = unvoer_binary[1]
                unover_reverse_binary[1] = unvoer_binary[0]

                # generate masks for negative samples
                mask1 = []
                mask1.append(generate_mask(overlap_unsup1_binary, unvoer_binary, pooler)) # mask1_unvoer_flatten
                mask1.append(generate_mask(overlap_unsup1_binary, unover_reverse_binary, pooler)) # mask1_reverseunvoer_flatten
                # mask1.append(generate_mask(overlap_unsup1_binary, overlap_unsup1_reverse_binary, pooler)) # mask1_reverse1_flatten
                # mask1.append(generate_mask(overlap_unsup1_binary, overlap_unsup2_reverse_binary, pooler)) # mask1_reverse2_flatten
                mask1 = torch.cat(mask1, dim=1)
                mask1_detach = mask1.detach()

                mask2 = []
                mask2.append(generate_mask(overlap_unsup2_binary, unvoer_binary, pooler)) # mask2_unvoer_flatten
                mask2.append(generate_mask(overlap_unsup2_binary, unover_reverse_binary, pooler)) # mask2_reverseunvoer_flatten
                # mask2.append(generate_mask(overlap_unsup2_binary, overlap_unsup1_reverse_binary, pooler)) # mask2_reverse1_flatten
                # mask2.append(generate_mask(overlap_unsup2_binary, overlap_unsup2_reverse_binary, pooler)) # mask2_reverse2_flatten
                mask2 = torch.cat(mask2, dim=1) #[n, 4]
                mask2_detach = mask2.detach()

                # generate hog layers
                hog_overlap_unsup1= hog(overlap_unsup1) # [2, 12, 32, 32]
                hog_overlap_unsup2= hog(overlap_unsup2) # [2, 12, 32, 32]
                hog_unover= hog(net_unover) # [2, 12, 32, 32]

                # go through projector
                hog_overlap_unsup1 = projector(hog_overlap_unsup1)  # [2, 12, 32, 32]
                hog_overlap_unsup2 = projector(hog_overlap_unsup2)  # [2, 12, 32, 32]
                hog_unover = projector(hog_unover)  # [2, 12, 32, 32]


                # generate hog layers of negative sameple from existing samples
                # hog_overlap_unsup1_reverse = torch.zeros(hog_overlap_unsup1.shape).float().cuda()
                # hog_overlap_unsup1_reverse[0] = hog_overlap_unsup1[1]
                # hog_overlap_unsup1_reverse[1] = hog_overlap_unsup1[0]  # [2, 12, 32, 32]
                #
                # hog_overlap_unsup2_reverse = torch.zeros(hog_overlap_unsup2.shape).float().cuda()
                # hog_overlap_unsup2_reverse[0] = hog_overlap_unsup2[1]
                # hog_overlap_unsup2_reverse[1] = hog_overlap_unsup2[0]  # [2, 12, 32, 32]

                hog_unover_reverse = torch.zeros(hog_unover.shape).float().cuda()
                hog_unover_reverse[0] = hog_unover[1]
                hog_unover_reverse[1] = hog_unover[0]  # [2, 12, 32, 32]

                # generate hog for negative samples
                hog_overlap_unsup1_flatten = hog_overlap_unsup1.view(-1, 1) # [bchw, 1]
                hog_overlap_unsup2_flatten = hog_overlap_unsup2.view(-1, 1)
                hog_unover_flatten = hog_unover.view(-1, 1)
                # hog_overlap_unsup1_reverse_flatten = hog_overlap_unsup1_reverse.view(-1, 1)
                # hog_overlap_unsup2_reverse_flatten = hog_overlap_unsup2_reverse.view(-1, 1)
                hog_unover_reverse_flatten = hog_unover_reverse.view(-1, 1)

                hog_neg = []
                hog_neg.append(hog_unover_flatten)
                hog_neg.append(hog_unover_reverse_flatten)
                # hog_neg.append(hog_overlap_unsup1_reverse_flatten)
                # hog_neg.append(hog_overlap_unsup2_reverse_flatten)
                hog_neg = torch.cat(hog_neg, dim=1) #[n, 4]

                # context-aware contrastive loss
                eps = 1e-8
                # positive similarity
                pos1 = (hog_overlap_unsup1_flatten * hog_overlap_unsup2_flatten.detach()).sum(-1, keepdim=True) / args.temp  # [n, 1]
                pos2 = (hog_overlap_unsup1_flatten.detach() * hog_overlap_unsup2_flatten).sum(-1, keepdim=True) / args.temp  # [n, 1]
                # pos = (hog_overlap_unsup1_flatten * hog_overlap_unsup2_flatten).sum(-1, keepdim=True) / args.temp  # [n, 1]

                # negative overlap1
                logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(logits_run, pos1, hog_overlap_unsup1_flatten, hog_neg, mask1_detach)
                # logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(logits_run, pos, hog_overlap_unsup1_flatten, hog_neg, mask1_detach)
                logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_neg_idx + eps)
                # logits1 = torch.exp(pos - neg_max1).squeeze(-1) / (logits1_neg_idx + eps)
                loss1 = -torch.log(logits1 + eps)
                loss1 = loss1.sum()/(loss1.numel() + 1e-12)

                # negative overlap2
                logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(logits_run, pos2, hog_overlap_unsup2_flatten, hog_neg, mask2_detach)
                # logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(logits_run, pos, hog_overlap_unsup2_flatten, hog_neg, mask2_detach)
                logits2 = torch.exp(pos2 - neg_max2).squeeze(-1) / (logits2_neg_idx + eps)
                # logits2 = torch.exp(pos - neg_max2).squeeze(-1) / (logits2_neg_idx + eps)
                loss2 = -torch.log(logits2 + eps)
                loss2 = loss2.sum() / (loss2.numel() + 1e-12)

                # loss_pos = criterion_MSE(hog_overlap_unsup1, hog_overlap_unsup2)
                # loss_neg1 = criterion_MSE(hog_overlap_unsup1 * mask1, hog_unover * mask1)
                # loss_neg2 = criterion_MSE(hog_overlap_unsup2 * mask2, hog_unover * mask2)
                # loss_neg = 0.5 * (loss_neg1 + loss_neg2) * args.loss_unsup_neg_weight

                # loss_unsup = (loss_pos - loss_neg) * args.loss_unsup_weight
                loss_unsup = (loss1 + loss2) * args.loss_unsup_weight
                total_loss = loss_sup + loss_unsup

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

                total_loss.backward()
                optimizer.step()

                global_step += 1
                writer.add_scalar('loss/sup', loss_sup.item(), global_step)
                writer.add_scalar('loss/unsup', loss_unsup.item(), global_step)
                # writer.add_scalar('loss/pos', loss_pos.item(), global_step)
                # writer.add_scalar('loss/neg', loss_neg.item(), global_step)
                writer.add_scalar('loss/total', total_loss.item(), global_step)
                writer.add_scalar('train', total_loss.item(), global_step)
                # tbar.set_postfix(**{'sup': loss_sup.item(), 'pos': loss_pos.item(),'neg': loss_neg.item(), 'unsup': loss_unsup.item(), 'total': total_loss.item()})
                tbar.set_postfix(**{'sup': loss_sup.item(), 'unsup': loss_unsup.item(), 'total': total_loss.item()})

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
    projector = projection(in_dim=12, out_dim=12).to(device=device)

    try:
        if (not args.test) and (not args.match):
            if args.semi == True:
                train_semi_net(net=net,
                               projector = projector,
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
            # skeleton(args)
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
