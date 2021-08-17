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


def eval_net(args,net,loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    epochs = args.epochs
    if args.tta == True:
        net.eval()
        n_val = len(loader)  # the number of batch
        tot = 0
        for epoch in range(0, epochs):
            with tqdm(total=n_val, desc=f'Match Epoch {epoch + 1}/{epochs}', unit='img', leave=False) as pbar:
                for batch in loader:
                    # img [b, i, n, c, h, w], [b, 4, n, 3, 1300, 1300]
                    img, mask, overlap1_ul, overlap2_ul, overlap3_ul, overlap4_ul, image1_ul, image2_ul, image3_ul, image4_ul, name = \
                        batch['image'], batch['mask'], batch['overlap1_ul'], batch['overlap2_ul'], batch['overlap3_ul'], batch['overlap4_ul'], \
                        batch['image1_ul'], batch['image2_ul'], batch['image3_ul'], batch['image4_ul'], batch['name']
                    # img, mask, name = batch['image'], batch['mask'],batch['name']
                    img = img.to(device=device, dtype=torch.float32)
                    mask = mask.to(device=device, dtype=torch.float32)

                    b, i, n, c, h, w = img.shape
                    mask_pred_list = []
                    mask_save_list = []
                    for idx in range(n):
                        rotated_img = img[:, idx, :, :, :] # [b, idx, 3, 1300, 1300] -> [b, 3, 1300, 1300]
                        rotated_img = torch.squeeze(rotated_img, axis=1) # [b, 3, 1300, 1300] -> [b, 3, 1300, 1300]
                        # rotated_img = rotated_img.to(device=device, dtype=torch.float32)

                        with torch.no_grad():
                            rotated_mask_pred = net(rotated_img)   #[b, 1, 1300, 1300]
                            # rotated_mask_save = torch.sigmoid(rotated_mask_pred).squeeze(0).squeeze(0).cpu().detach().numpy()
                            rotated_mask_save = torch.sigmoid(rotated_mask_pred).squeeze(0).squeeze(0)  #[b, 1, 1300, 1300] -> [1300, 1300]
                            if idx == 0:
                                mask_pred_list.append(rotated_mask_pred)
                                mask_save_list.append(rotated_mask_save)
                            else:
                                back_mask_pred = torch.rot90(rotated_mask_pred, 4-idx, (2, 3))
                                back_mask_save = torch.rot90(rotated_mask_save, 4-idx, (0, 1))
                                mask_pred_list.append(back_mask_pred)
                                mask_save_list.append(back_mask_save)
                    pbar.update()

                    mask_pred = torch.mean(torch.stack(mask_pred_list), 0)
                    # mask_save = np.mean(np.stack(mask_save_list), 0)
                    mask_save = torch.mean(torch.stack(mask_save_list), 0)

                    if not args.test:
                        mask_save = mask_save.cpu().detach().numpy()
                        mask_max = np.max(mask_save)
                        mask_min = np.min(mask_save)
                        mask_save = (mask_save - mask_min)/(mask_max - mask_min)
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
                            tta_thr = args.tta_thr
                            thr_mask = (mask_save >= tta_thr).float()
                            mask_save = mask_save * thr_mask

                            mask_save_seg = mask_save.cpu().detach().numpy()
                            Image.fromarray(mask_save_seg * 255) \
                                        .convert('L').save(os.path.join('./records/test/segmentation', name[0] + '.png'))
        return tot / n_val

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



