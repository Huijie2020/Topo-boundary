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
from models.projector import projection, prediction
import torchvision.transforms.functional as TF
from torchvision import transforms
from losses.var_loss import var_loss
from losses.sim_loss import D
import cv2
import torchvision.transforms.functional_tensor as TFT

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

def transform_back(net_unsup, flip_rotate):
    for idx in range(net_unsup.size(0)):
        net_unsup_idx = net_unsup[idx]
        angle = -int(flip_rotate[2][idx] / 90)
        net_unsup_idx = torch.rot90(net_unsup_idx, angle, [1, 2])

        whether_hori_flip = flip_rotate[0][idx]
        whether_ver_flip = flip_rotate[1][idx]

        if whether_hori_flip:
            net_unsup_idx = TFT.hflip(net_unsup_idx)

        if whether_ver_flip:
            net_unsup_idx = TFT.vflip(net_unsup_idx)

        net_unsup[idx] = net_unsup_idx
    return net_unsup

def get_overlap(net_unsup, overlap_ul):
    output = []
    for idx in range(net_unsup.size(0)):
        output.append(net_unsup[idx, :, int(overlap_ul[0][idx]):int(overlap_ul[0][idx])+args.unsup_crop_in_size, int(overlap_ul[1][idx]):int(overlap_ul[1][idx])+args.unsup_crop_in_size])
    output = torch.stack(output, dim=0)
    return output

def train_semi_net(net,
              projector,
              prediction,
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
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(),lr=lr)
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

            # sup_l, unsup_l = next(train_loader)

            image_l, mask_l, name_l = sup_l['image'], sup_l['mask'], sup_l['name']
            image_l, mask_l = image_l.to(device=device, dtype=torch.float32), mask_l.to(device=device, dtype=torch.float32)

            image_ul, mask_ul, overlap1_ul, overlap2_ul, flip_rotate_1, flip_rotate_2, name = \
                unsup_l['image'], unsup_l['mask'], unsup_l['overlap1_ul'], unsup_l['overlap2_ul'], \
                unsup_l['flip_rotate_1'], unsup_l['flip_rotate_2'], unsup_l['name']
            image_ul, mask_ul = image_ul.to(device=device, dtype=torch.float32), mask_ul.to(device=device, dtype=torch.float32)

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

                # net_unsup2_pre = net(image_ul2)
                # net_unsup2_pre = torch.sigmoid(net_unsup2_pre)
                # net_unsup2_pre = normal_thr_output(net_unsup2_pre, args.unsup_thr_ske_train)
                # net_unsup2_pre = soft_skel(net_unsup2_pre, 10)
                # net_unsup2_pre = garbo_filter(net_unsup2_pre)

                # net_unsup3_pre = net(image_ul3)
                # net_unsup3_pre = normal_thr_output(net_unsup3_pre, args.unsup_thr_ske_train)
                # net_unsup3_pre = soft_skel(net_unsup3_pre, 10)
                # net_unsup3_pre = garbo_filter(net_unsup3_pre)

                # net_unsup4_pre = net(image_ul4)
                # net_unsup4_pre = normal_thr_output(net_unsup4_pre, args.unsup_thr_ske_train)
                # net_unsup4_pre = soft_skel(net_unsup4_pre, 10)
                # net_unsup4_pre = garbo_filter(net_unsup4_pre)

                net_unsup1_pre = net(image_ul1)  # [batch_size, 1, H, W]
                net_unsup2_pre = net(image_ul2)

                net_unsup1_pre = normal_thr_output(net_unsup1_pre, args.unsup_thr_ske_train)  # [batch_size, 1, H, W]
                net_unsup2_pre = normal_thr_output(net_unsup2_pre, args.unsup_thr_ske_train)

                net_unsup1_pre = transform_back(net_unsup1_pre, flip_rotate_1)  # [1, 384, 384]
                net_unsup2_pre = transform_back(net_unsup2_pre, flip_rotate_2)

                overlap_unsup1 = get_overlap(net_unsup1_pre, overlap1_ul)
                overlap_unsup2 = get_overlap(net_unsup2_pre, overlap2_ul)

                hog_overlap_unsup1 = hog(overlap_unsup1)                          # [2, 12, 32, 32]
                hog_overlap_unsup2 = hog(overlap_unsup2)

                # MSE stop gradient
                # hog_overlap_unsup1_de = hog_overlap_unsup1.detach()
                # hog_overlap_unsup2_de = hog_overlap_unsup2.detach()
                #
                # loss_unsup = (criterion_MSE(hog_overlap_unsup1, hog_overlap_unsup2_de) / 2 + criterion_MSE(hog_overlap_unsup2, hog_overlap_unsup1_de) / 2) * args.loss_unsup_var_weight

                #Sim stop gradient
                # hog_overlap_unsup1_projector = projector(hog_overlap_unsup1)                          # [2, 12, 32, 32]
                # hog_overlap_unsup2_projector = projector(hog_overlap_unsup2)
                # hog_overlap_unsup1_prediction = prediction(hog_overlap_unsup1_projector)                          # [2, 12, 32, 32]
                # hog_overlap_unsup2_prediction = prediction(hog_overlap_unsup2_projector)
                #
                # hog_overlap_unsup1_projector = torch.flatten(hog_overlap_unsup1_projector, 1) # [b, chw]
                # hog_overlap_unsup2_projector = torch.flatten(hog_overlap_unsup2_projector, 1)
                # hog_overlap_unsup1_prediction = torch.flatten(hog_overlap_unsup1_prediction, 1)
                # hog_overlap_unsup2_prediction = torch.flatten(hog_overlap_unsup2_prediction, 1)

                # MSE stop gradient only with prediction
                hog_overlap_unsup1_prediction = prediction(hog_overlap_unsup1)                          # [2, 12, 32, 32]
                hog_overlap_unsup2_prediction = prediction(hog_overlap_unsup2)

                hog_overlap_unsup1_de = hog_overlap_unsup1_prediction.detach()
                hog_overlap_unsup2_de = hog_overlap_unsup2_prediction.detach()

                loss_unsup = (criterion_MSE(hog_overlap_unsup1_prediction, hog_overlap_unsup2_de) / 2 + criterion_MSE(hog_overlap_unsup2_prediction, hog_overlap_unsup1_de) / 2) * args.loss_unsup_var_weight

                # loss_unsup = (D(hog_overlap_unsup1_prediction, hog_overlap_unsup2_projector, args.sim_versiion) / 2 + D(hog_overlap_unsup2_prediction, hog_overlap_unsup1_projector, args.sim_versiion) / 2) * args.loss_unsup_var_weight
                total_loss = loss_sup + loss_unsup

                total_loss.backward()
                optimizer.step()

                global_step += 1
                writer.add_scalar('loss/sup', loss_sup.item(), global_step)
                writer.add_scalar('loss/unsup', loss_unsup.item(), global_step)
                writer.add_scalar('loss/total', total_loss.item(), global_step)
                writer.add_scalar('train', total_loss.item(), global_step)
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
        path_checkpoint = args.load_checkpoint  # checkpoint path
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
    projector = projection(in_dim=12, hidden_dim=12, out_dim=12).to(device=device)
    prediction = prediction(in_dim=12, hidden_dim=12, out_dim=12).to(device=device)

    try:
        if (not args.test) and (not args.match):
            if args.semi == True:
                train_semi_net(net=net,
                               projector=projector,
                               prediction=prediction,
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
            skeleton(args)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
