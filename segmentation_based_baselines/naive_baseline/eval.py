import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import os 
import numpy as np
from sklearn.metrics import average_precision_score
from dice_loss import dice_coeff

# def crop(data):
#     # height = data.size(2)
#     # width = data.size(3)
#     height = data.size(3)  #[b, c, w, h]
#     width = data.size(2)
#     h = height // 2
#     w = width // 2
#     crop_data = []
#     # for i in range(0, height, h):
#     #     for j in range(0, width, w):
#     #         crop_data.append(data[:,:, i:i+w, j:j+h])
#     for i in range(0, width, w):
#         for j in range(0, height, h):
#             crop_data.append(data[:,:, i:i+w, j:j+h])
#     return crop_data
#
#
# def merge(data, pred_masks):
#     pred_mask = torch.zeros(data.size(0), 1, data.size(2), data.size(3))
#     height = data.size(3)
#     width = data.size(2)
#     h = height // 2
#     w = width // 2
#     count = 0
#     # for i in range(0, height, h):
#     #     for j in range(0, width, w):
#     #         pred_mask[:, :, i:i + h, j:j + h] = pred_masks[count]
#     #         count += 1
#     for i in range(0, width, w):
#         for j in range(0, height, h):
#             pred_mask[:, :, i:i + w, j:j + h] = pred_masks[count]
#             count += 1
#     return pred_mask

def eval_net(args,net,loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0
    with tqdm(total=n_val, desc='Validation' if not args.test else 'Testing', unit='img', leave=False) as pbar:
        for batch in loader:
            # img [b, 3, 1300, 1300]
            img, mask, name = batch['image'], batch['mask'],batch['name']
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
                mask_save = torch.sigmoid(mask_pred).squeeze(0).squeeze(0).cpu().detach().numpy()
                #
                if not args.test:
                    Image.fromarray(mask_save/np.max(mask_save) *255)\
                        .convert('RGB').save(os.path.join('./records/valid/segmentation',name[0]+'.png'))                
                    pred = torch.sigmoid(mask_pred)
                    pred = (pred > 0.1).float()
                    tot += dice_coeff(pred, mask, args).item()
                else:
                    mask_max = np.max(mask_save)
                    # mask_save[(mask_save / mask_max) < 0.306] = 0
                    # Image.fromarray(mask_save / mask_max * 255) \
                    #     .convert('L').save(os.path.join('./records/test/segmentation', name[0] + '.png'))

                    # Image.fromarray(mask_save / mask_max * 255)\
                    #     .convert('L').save(os.path.join('./records/test/segmentation',name[0]+'.png'))

                    # Image.fromarray(mask_save /np.max(mask_save)*255)\
                    #     .convert('RGB').save(os.path.join('./records/test/segmentation',name[0]+'.png'))
            pbar.update()
    return tot / n_val
