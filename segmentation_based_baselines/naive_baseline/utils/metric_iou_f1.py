# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:14:52 2018
1.split pic(mask/images)
2.random split pic for train/test
@author: zetn
"""

'''
------------------------1.split to mask/images--------------------
import os
import shutil
path_img='train'
ls = os.listdir(path_img)
print(len(ls))
for i in ls:
    if i.find('mask')!=-1:      #cannot find key words, then return -1,else return the index position
       shutil.move(path_img+'/'+i,"data/train2/images/"+i)
'''

'''
------------------------2.split to train/test(mask&&images)--------------------
#reference: https://blog.csdn.net/kenwengqie2235/article/details/81509714
import os, sys
import random
import shutil


def copyFile(fileDir):
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir, 1226)
    #print(sample)
    for name in sample:
        shutil.move(fileDir+'/' + name, tarDir+'/' + name)
        cor_mask_name = name[0:-7]+'mask.png'
        shutil.move(path_masks+'/' + cor_mask_name, tar_masks+'/' + cor_mask_name)
        #print(cor_mask_name)


if __name__ == '__main__':
    # open /textiles
    path = "data/membrane/train/images/"
    path_masks = "data/membrane/train/masks/"
    ls = os.listdir(path)
    print(len(ls))
    tarDir = "data/membrane/test/images/"
    tar_masks = "data/membrane/test/masks/"
    copyFile(path)
'''

'''
#------------------------3.get 8 Bit test masks--------------------
import os
import cv2
path = "data/membrane/train/masks/"
ls = os.listdir(path)
#i = 0
for name in ls:
    img = cv2.imread(os.path.join(path, name))
    img1 = img[:, :, 0]
    #cv2.imwrite("data/membrane/train/mask8/%d.png"%i, img1)
    cv2.imwrite("data/membrane/train/mask8/%s"%name, img1)
    #i = i+1
'''
# ------------------------4.metrics of pre and GT--------------------
import os
import cv2
import numpy as np
import skimage.io as io
from tqdm import tqdm

# path1 = "data/membrane/test/mask_8bit"

path1 = "/mnt/git/Topo-boundary/dataset/prediction/spacenet_gt_binary"  # Dir of Ground Truth
path2 = "/mnt/git/Topo-boundary/dataset/prediction/1data_exp1.5.4_contr_ta_tta_nohog_nomask_projector_unsupweight0.1_temp0.07_lr0.0005_repeat_hz_epoch80"  # Dir of predict map
# path1 = "/mnt/git/Topo-boundary/dataset/prediction/spacenet_gt_binary_test(431)"  # Dir of Ground Truth
# path2 = "/mnt/git/Topo-boundary/dataset/prediction/c3_1percent"  # Dir of predict map

sample1 = os.listdir(path1)
Iou_all = []  # Iou for each test images
f1_all = []
precision_all = []
recall_all = []
TP = 0
FP = 0
FN = 0
sum_fenmu = 0
for name in tqdm(sample1):
    mask1 = io.imread(os.path.join(path1, name))
    mask1 = mask1 / 255
    mask1 = mask1.flatten()

    # name1 = name[0:-8]+'sat.jpg'
    # mask2 = io.imread(os.path.join(path2, name1))
    mask2 = io.imread(os.path.join(path2, name))
    # print('mask1.shape',mask1.shape)
    # print('mask2.shape',mask2.shape)
    mask2 = mask2 / 255.0
    mask2[mask2 >= 0.1] = 1
    mask2[mask2 < 0.1] = 0
    mask2 = mask2.flatten()

    tp = np.dot(mask1, mask2)
    TP = TP + tp

    fp = mask2.sum() - tp
    FP = FP + fp

    fn = mask1.sum() - tp
    FN = FN + fn


    if (tp + fn) == 0:
        precision = 0
    else:
        precision = tp / (tp + fn)

    if (tp + fp) == 0:
        recall = 0
    else:
        recall = tp / (tp + fp)

    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall)/(precision + recall)


    f1_all.append(f1)
    precision_all.append(precision)
    recall_all.append(recall)

    # fenmu = mask1.sum() + mask2.sum()-tp
    fenmu = mask1.sum() + mask2.sum() - tp
    sum_fenmu = sum_fenmu + fenmu
    # element_wise = np.multiply(mask1, mask2)
    Iou_all.append(tp / fenmu)
    # if(tp / fenmu == 0.0):
    # print(name)

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall)/(Precision + Recall)

precision_mean = sum(precision_all)/len(precision_all)
recall_mean = sum(recall_all)/len(recall_all)
f1_mean = 2 * (precision_mean * recall_mean)/(precision_mean + recall_mean)

print('\n{}\n'.format(os.path.basename(path2)))
print("1. mIoU (sum(Iou_all)/len(Iou_all)): {}".format(sum(Iou_all)/len(Iou_all)), '\n')
print("2. active mIoU (TP / sum_fenmu): {}".format(TP / sum_fenmu), '\n')
print("3. f1 (each image and average): {}".format(sum(f1_all)/len(f1_all)), '\n')
print("4. precision_mean (each image and average): {}".format(precision_mean), '\n')
print("5. recall_mean for (each image and average): {}".format(recall_mean), '\n')
print("6. f1_mean from precision_mean and recall_mean: {}".format(f1_mean), '\n')
print("7. F1 over all iamges: {}".format(F1), '\n')
print("8. Precision over all iamges: {}".format(Precision), '\n')
print("9. Recall over all iamges: {}".format(Recall), '\n')

print("3. f1 (each image and average): {}".format(sum(f1_all)/len(f1_all)), '\n')
print("1. mIoU (sum(Iou_all)/len(Iou_all)): {}".format(sum(Iou_all)/len(Iou_all)), '\n')


# print(sum(Iou_all)/len(Iou_all))
# print(TP / sum_fenmu)  # active IoU
# print(TP / (TP + FN))  # recall
# print(TP / (TP + FP))  # precision