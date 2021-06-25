import argparse
import os
import sys
import numpy as np
import cv2
from scipy.ndimage.morphology import *
import glob
import math
import time
from osgeo import gdal
from tqdm import tqdm
from PIL import Image

# source_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/test_best_skele2binary/iter1/1998gt/thr0/skeleton'
# tar_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/test_best_skele2binary/iter1/1998gt/thr0/skeleton2binary'

source_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/exp1.1_best_thrbinary_iter/best_binary_iter2/1998gt_thr0.306_pre_from_iter1/segmentation_thr0'
tar_dir = '/mnt/git/Topo-boundary/conn_experiment/expriment/06152021_unet_baseline_0.01data_24batch_0.01lr_25500epoch/exp1.1_best_thrbinary_iter/best_binary_iter2/1998gt_thr0.306_pre_from_iter1/binary_thr0.306'

# # skeleton to gaussian or binary
# progress_bar = tqdm(glob.glob(source_dir + '/*.png'), ncols=150)
# # x = glob.glob(source_dir + '/*.png')
# for file_ in progress_bar:
#     name = file_.split('/')[-1]
#     input_name = os.path.join(source_dir, name)
#     gt_dataset = gdal.Open(input_name, gdal.GA_ReadOnly)
#     if not gt_dataset:
#         continue
#     gt_array = gt_dataset.GetRasterBand(1).ReadAsArray()
#
#     distance_array = distance_transform_edt(1 - (gt_array / 255))
#     std = 15
#     distance_array = np.exp(-0.5 * (distance_array * distance_array) / (std * std))
#     # distance_array = distance_array >= 0.76
#     # distance_array[distance_array < 0.7] = 0
#     # distance_array[distance_array >= 0.7] = 1
#     distance_array *= 255
#     out_png_file = os.path.join(tar_dir, name)
#     cv2.imwrite(out_png_file, distance_array)

# confidence map to binary
progress_bar = tqdm(glob.glob(source_dir + '/*.png'), ncols=150)
for file_ in progress_bar:
    name = file_.split('/')[-1]
    input_name = os.path.join(source_dir, name)
    img = cv2.imread(input_name, cv2.IMREAD_UNCHANGED)
    img = img/255
    img[img < 0.306] = 0
    img[img >= 0.306] = 1
    img *= 255
    out_png_file = os.path.join(tar_dir, name)
    cv2.imwrite(out_png_file, img)