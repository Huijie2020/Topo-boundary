import os
# import cv2
import numpy as np
import skimage.io as io
from tqdm import tqdm
import matplotlib.pyplot as plt
from roc_utils import *
import sklearn.metrics as skm
import torch
from ignite.contrib.metrics import ROC_AUC


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path1 = "/mnt/git/Topo-boundary/dataset/prediction/spacenet_gt_binary"  # Dir of Ground Truth
path2 = "/mnt/git/Topo-boundary/dataset/prediction/1data_exp1.5.4_unsupeweight0.1_hog_contrtemp0.07_maskthr0.5_projector_ta_tta_hz_1600_iter5"  # Dir of predict map


sample1 = os.listdir(path1)
AUC = []
# roc_auc = ROC_AUC()
# roc_auc.attach(default_evaluator, 'roc_auc')
# gt = np.array([0],)
# gt = torch.tensor(gt, device=device).float()

# prediction = np.array([0],)
# prediction = torch.tensor(prediction, device=device).float()

for name in tqdm(sample1):
    mask1 = io.imread(os.path.join(path1, name))
    # mask1 = torch.tensor(mask1, device=device).float()
    mask1 = mask1 / 255
    mask1 = mask1.flatten()
    # gt = torch.cat([gt, mask1])

    mask2 = io.imread(os.path.join(path2, name))
    # mask2 = torch.tensor(mask2, device=device).float()
    mask2 = mask2 / 255.0
    mask2 = mask2.flatten()
    # prediction = torch.cat([prediction, mask2])

    # state = default_evaluator.run([[mask2, mask1]])
    auc = skm.roc_auc_score(y_true=mask1, y_score=mask2)
    AUC.append(auc)
    # AUC.append(state.metrics['roc_auc'])

AUC_mean = sum(AUC)/len(AUC)

print('\n{}\n'.format(os.path.basename(path2)))
print("1. auc (sum(AUC)/len(AUC)): {}".format(AUC_mean), '\n')
# print("1. auc all images: {}".format(auc), '\n')


