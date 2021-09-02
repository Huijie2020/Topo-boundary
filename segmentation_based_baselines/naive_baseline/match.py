import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
from sklearn.metrics import average_precision_score
from losses.dice_loss import dice_coeff
from models.skeleton import soft_skel
from models.gabore_filter_bank import GaborFilters
import json

import torch.nn as nn

def match_net(args,net, hog, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    epochs = args.epochs
    if args.tta == True:
        net.eval()
        n_val = len(loader)  # the number of batch
        criterion_MSE = nn.MSELoss()
        image_name = []
        image_mse = []
        overlap_img_ul_list = []
        for epoch in range(0, epochs):
            with tqdm(total=n_val, desc=f'Match Epoch {epoch + 1}/{epochs}', unit='img', leave=False) as pbar:
                for batch in loader:
                    # img [b, i, n, c, h, w], [b, 4, n, 3, 1300, 1300]
                    img, mask, overlap1_ul, overlap2_ul, overlap3_ul, overlap4_ul, image1_ul, image2_ul, image3_ul, image4_ul, overlap_img_ul, name = \
                        batch['image'], batch['mask'], batch['overlap1_ul'], batch['overlap2_ul'], batch['overlap3_ul'], batch['overlap4_ul'], \
                        batch['image1_ul'], batch['image2_ul'], batch['image3_ul'], batch['image4_ul'], batch['overlap_img_ul'], batch['name']

                    overlap_img_ul_0 = overlap_img_ul[0].detach().cpu().numpy()
                    overlap_img_ul_1 = overlap_img_ul[1].detach().cpu().numpy()
                    overlap_img_ul = [overlap_img_ul_0[0], overlap_img_ul_1[0]]
                    with open('./records/test/overlap_img_ul.txt', 'a') as name_file:
                        name_file.write(str(overlap_img_ul))
                        name_file.write('\n')
                    overlap_img_ul_list.append(overlap_img_ul)

                    # img, mask, name = batch['image'], batch['mask'],batch['name']
                    overlap_ul = [overlap1_ul, overlap2_ul, overlap3_ul, overlap4_ul]
                    img = img.to(device=device, dtype=torch.float32)
                    mask = mask.to(device=device, dtype=torch.float32)

                    # name_epoch = name[0] + '_epoch' + str(epoch+1)
                    name_epoch = name[0] + '_' + str(epoch+1)
                    image_name.append(name_epoch)
                    with open('./records/test/name.txt', 'a') as name_file:
                        name_file.write(name_epoch)
                        name_file.write('\n')

                    b, i, n, c, h, w = img.shape
                    overlap_pre_list = []
                    overlap_save_list = []
                    for idx_img in range(i):
                        img_idx = img[:, idx_img, :, :, :, :] # [b, idx_img, 4, 3, 1300, 1300] -> [b, 4, 3, 1300, 1300]
                        gt_idx = mask[:, idx_img, :, :, :] # [b, i, c, h, w]  -> [b, 1, 1300, 1300]
                        mask_pred_list = []
                        mask_save_list = []
                        for idx_tta in range(n):
                            rotated_img = img_idx[:, idx_tta, :, :, :] # [b, idx_tta, 3, 1300, 1300] -> [b, 3, 1300, 1300]
                            if idx_tta == 0:
                                # save 256*256 image
                                img_save = rotated_img.squeeze(0) #[b, 3, 1300, 1300] -> [3, 1300, 1300]
                                img_save = img_save[:, int(overlap_ul[idx_img][0]):int(overlap_ul[idx_img][0]) + args.unsup_crop_in_size,int(overlap_ul[idx_img][1]):int(overlap_ul[idx_img][1]) + args.unsup_crop_in_size]
                                img_save = img_save.cpu().detach().numpy()
                                img_save = img_save.transpose((1, 2, 0))  # [c, h, w]->[h,w,c]
                                img_save = img_save * 255
                                img_save = img_save.astype(np.uint8)
                                # print('img_save shape###################################', img_save.shape)
                                Image.fromarray(img_save) \
                                    .convert('RGB').save(os.path.join('./records/test/match_image', name_epoch + '.png'))

                                # save mask
                                gt_save = gt_idx.squeeze(0).squeeze(0) #[b, 1, 1300, 1300] -> [1300, 1300]
                                gt_save = gt_save[int(overlap_ul[idx_img][0]):int(overlap_ul[idx_img][0]) + args.unsup_crop_in_size,int(overlap_ul[idx_img][1]):int(overlap_ul[idx_img][1]) + args.unsup_crop_in_size]
                                gt_save = gt_save.cpu().detach().numpy()
                                gt_save = gt_save * 255
                                gt_save = gt_save.astype(np.uint8)
                                Image.fromarray(gt_save) \
                                    .convert('L').save(os.path.join('./records/test/match_gt', name_epoch + '.png'))

                            # rotated_img = torch.squeeze(rotated_img, axis=1) # [b, 3, 1300, 1300] -> [b, 3, 1300, 1300]
                            # rotated_img = rotated_img.to(device=device, dtype=torch.float32)

                            with torch.no_grad():
                                rotated_mask_pred = net(rotated_img)   #[b, 1, 1300, 1300]
                                rotated_mask_save = torch.sigmoid(rotated_mask_pred).squeeze(0).squeeze(0)  #[b, 1, 1300, 1300] -> [1300, 1300]
                                if idx_tta == 0:
                                    mask_pred_list.append(rotated_mask_pred)
                                    mask_save_list.append(rotated_mask_save)
                                else:
                                    back_mask_pred = torch.rot90(rotated_mask_pred, 4-idx_tta, (2, 3))
                                    back_mask_save = torch.rot90(rotated_mask_save, 4-idx_tta, (0, 1))
                                    mask_pred_list.append(back_mask_pred)
                                    mask_save_list.append(back_mask_save)

                        mask_pred = torch.mean(torch.stack(mask_pred_list), 0) # [1, 1, 1300, 1300]
                        mask_save = torch.mean(torch.stack(mask_save_list), 0) # [1300, 1300]

                        mask_pred = torch.sigmoid(mask_pred) # [1, 1, 1300, 1300]
                        mask_overlap_pre = mask_pred[:, :, int(overlap_ul[idx_img][0]):int(overlap_ul[idx_img][0]) + args.unsup_crop_in_size,int(overlap_ul[idx_img][1]):int(overlap_ul[idx_img][1]) + args.unsup_crop_in_size]
                        overlap_pre_list.append(mask_overlap_pre) # [4, 1, 1, 256, 256]

                        mask_overlap_save = mask_save[int(overlap_ul[idx_img][0]):int(overlap_ul[idx_img][0]) + args.unsup_crop_in_size,int(overlap_ul[idx_img][1]):int(overlap_ul[idx_img][1]) + args.unsup_crop_in_size] #[256, 256]
                        overlap_save_list.append(mask_overlap_save) #[4, 256, 256]

                        # # save seperate patch
                        # mask_save_seg = mask_save.cpu().detach().numpy()
                        # mask_max = np.max(mask_save_seg)
                        # mask_min = np.min(mask_save_seg)
                        # mask_save_seg = (mask_save_seg - mask_min) / (mask_max - mask_min)
                        # Image.fromarray(mask_save_seg * 255) \
                        #     .convert('L').save(os.path.join('./records/test/segmentation_seperate', name_epoch + '_idx' + str(idx_img) + '.png'))

                    # save overlap
                    overlap_save = torch.mean(torch.stack(overlap_save_list), 0)

                    overlap_save_seg = overlap_save.cpu().detach().numpy()
                    overlap_max = np.max(overlap_save_seg)
                    overlap_min = np.min(overlap_save_seg)
                    overlap_save_seg = (overlap_save_seg - overlap_min) / (overlap_max - overlap_min)
                    Image.fromarray(overlap_save_seg * 255) \
                        .convert('L').save(os.path.join('./records/test/segmentation_overlap',
                                                        name_epoch + '.png'))

                    # overlap hog
                    hog_overlap_list = []
                    for idx_overlap in range(4):
                        overlap_idx = overlap_pre_list[idx_overlap] #  [1, 1, 256, 256]
                        hog_overlap_idx = hog(overlap_idx) # [1, 12, 32, 32]
                        hog_overlap_list.append(hog_overlap_idx)

                    # MSE
                    mse_01 = criterion_MSE(hog_overlap_list[0], hog_overlap_list[1])
                    mse_02 = criterion_MSE(hog_overlap_list[0], hog_overlap_list[2])
                    mse_03 = criterion_MSE(hog_overlap_list[0], hog_overlap_list[3])
                    mse_12 = criterion_MSE(hog_overlap_list[1], hog_overlap_list[2])
                    mse_13 = criterion_MSE(hog_overlap_list[1], hog_overlap_list[3])
                    mse_23 = criterion_MSE(hog_overlap_list[2], hog_overlap_list[3])
                    mse = (mse_01 + mse_02 + mse_03 + mse_12 + mse_13 + mse_23)/6
                    mse_np = mse.cpu().detach().item()
                    with open('./records/test/mse.txt', 'a') as the_file:
                        the_file.write(str(mse_np))
                        the_file.write('\n')
                    image_mse.append(mse_np)

                    pbar.update()
        with open('./records/test/100epoch_256_20000img.json','w') as jf:
            # with open('./scripts/data_split_100val.json','w') as jf:
            json.dump({'img_id': image_name[:len(image_name)],
                       'mse': image_mse[:len(image_mse)],
                       'overlap_ul':overlap_img_ul_list
                       }, jf)

    if args.tta == False:
        net.eval()
        n_val = len(loader)  # the number of batch
        tot = 0
        with tqdm(total=n_val, desc='Validation' if not args.test else 'Testing', unit='img', leave=False) as pbar:
            for batch in loader:
                # img [b, 3, 1300, 1300]
                img, mask, name = batch['image'], batch['mask'], batch['name']
                img = img.to(device=device, dtype=torch.float32)
                mask = mask.to(device=device, dtype=torch.float32)
                # crop_imgs = crop(img).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    # mask_preds= []
                    # for crop_img in crop_imgs:
                    #     mask_pred = net(crop_img)
                    #     mask_preds.append(mask_pred)
                    # mask_pred = merge(img, mask_preds).to(device=device, dtype=torch.float32)
                    mask_pred = net(img)
                    # mask_save = torch.sigmoid(mask_pred).squeeze(0).squeeze(0).cpu().detach().numpy()
                    mask_save = torch.sigmoid(mask_pred).squeeze(0).squeeze(0)
                    if not args.test:
                        mask_save = mask_save.cpu().detach().numpy()
                        mask_max = np.max(mask_save)
                        mask_min = np.min(mask_save)
                        mask_save = (mask_save - mask_min) / (mask_max - mask_min)
                        Image.fromarray(mask_save * 255) \
                            .convert('L').save(os.path.join('./records/valid/segmentation', name[0] + '.png'))
                        pred = torch.sigmoid(mask_pred)
                        pred = (pred > 0.1).float()
                        tot += dice_coeff(pred, mask, args).item()
                    else:
                        if args.thr0 == True:
                            mask_save = mask_save.cpu().detach().numpy()
                            mask_max = np.max(mask_save)
                            mask_min = np.min(mask_save)
                            mask_save = (mask_save - mask_min) / (mask_max - mask_min)
                            Image.fromarray(mask_save * 255) \
                                .convert('L').save(os.path.join('./records/test/segmentation', name[0] + '.png'))
                        else:
                            mask_max = torch.max(mask_save)
                            mask_min = torch.min(mask_save)
                            mask_save = (mask_save - mask_min) / (mask_max - mask_min)
                            no_tta_thr = args.no_tta_thr
                            thr_mask = (mask_save >= no_tta_thr).float()
                            mask_save = mask_save * thr_mask


                            mask_save_seg = mask_save.cpu().detach().numpy()
                            Image.fromarray(mask_save_seg * 255) \
                                .convert('L').save(os.path.join('./records/test/segmentation', name[0] + '.png'))

                pbar.update()
        return tot / n_val



