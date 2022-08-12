from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import json
import os
import torchvision.transforms.functional as TF
import random
import torchvision.transforms.functional as TF
from torchvision import transforms

class CalCtrDataset(Dataset):
    def __init__(self, args):
        self.imgs_dir = args.image_dir
        self.masks_dir = args.mask_dir
        self.semi_crop_in_size = args.unsup_crop_in_size
        self.semi_crop_out_size = args.unsup_crop_out_size
        self.semi_unover_in_size = args.unsup_unover_in_size
        self.semi_unover_out_size = args.unsup_unover_out_size
        self.unover_ul_xy = args.unover_ul_xy
        self.data_augumentation = args.data_augumentation
        with open('./dataset/data_split.json', 'r') as jf:
            data_load = json.load(jf)
        # self.ids = data_load['train'] + data_load['pretrain']
        self.ids = data_load['train_unsup']
        print('=================')
        print('Training mode: Training unsupervised data length {}.'.format(len(self.ids)))

        brightness = (1, 10)
        contrast = (1, 10)
        saturation = (1, 10)
        hue = (-0.5, 0.5)
        self.train_transform = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, hue)
        ])

    def __len__(self):
        return len(self.ids)

    def out_crop(self, image, mask, w, h, x_in_ul, y_in_ul, crop_in_h, crop_in_w, crop_out_h, crop_out_w):
        # upper left location
        x_out_ul = np.random.randint(max(x_in_ul - (crop_out_w - crop_in_w), 0), min(x_in_ul, w - crop_out_w))
        y_out_ul = np.random.randint(max(y_in_ul - (crop_out_h - crop_in_h), 0), min(y_in_ul, h - crop_out_h))

        # outside crop image
        image_crop = image.crop((x_out_ul, y_out_ul, x_out_ul + crop_out_w, y_out_ul + crop_out_h))
        mask_crop = mask.crop((x_out_ul, y_out_ul, x_out_ul + crop_out_w, y_out_ul + crop_out_h))

        # overlap location
        overlap_ul = [y_in_ul - y_out_ul, x_in_ul - x_out_ul]

        return image_crop, mask_crop, overlap_ul

    def unoverlap_crop(self, image, mask, w, h, x_in_ul, y_in_ul, crop_in_w, crop_out_h, unover_out_h, unover_out_w,
                       semi_unover_in_size, unover_ul_xy):
        max_iter = 50
        k = 0
        while k < max_iter:
            x_unover_ul = np.random.randint(0, w - unover_out_w)
            y_unover_ul = np.random.randint(0, h - unover_out_h)

            if (abs((x_unover_ul + unover_ul_xy) - x_in_ul) - semi_unover_in_size) > 0 or (
                    abs((y_unover_ul + unover_ul_xy) - y_in_ul) - semi_unover_in_size) > 0:
                break

            k += 1

        if (abs((x_unover_ul + unover_ul_xy) - x_in_ul) - semi_unover_in_size) < 0 and (
                abs((y_unover_ul + unover_ul_xy) - y_in_ul) - semi_unover_in_size) < 0:
            if (x_in_ul - unover_out_w) > 0:
                x_unover_ul = np.random.randint(0, x_in_ul - unover_out_w)
                y_unover_ul = np.random.randint(0, h - unover_out_h)
            else:
                x_unover_ul = np.random.randint(x_in_ul + crop_in_w, w - unover_out_w)
                y_unover_ul = np.random.randint(0, h - unover_out_h)

        image_crop = image.crop((x_unover_ul, y_unover_ul, x_unover_ul + unover_out_w, y_unover_ul + unover_out_h))
        mask_crop = mask.crop((x_unover_ul, y_unover_ul, x_unover_ul + unover_out_w, y_unover_ul + unover_out_h))

        unover_ul = [unover_ul_xy, unover_ul_xy]

        return image_crop, mask_crop, unover_ul

    def overlap_crop(self, image, mask, semi_crop_in_size, semi_crop_out_size, semi_unover_out_size,
                     semi_unover_in_size, unover_ul_xy):
        w, h = image.size

        # inside crop h,w
        crop_in_h = semi_crop_in_size
        crop_in_w = semi_crop_in_size

        # outside crop h,w
        crop_out_h = semi_crop_out_size
        crop_out_w = semi_crop_out_size

        # inside crop upperleft location
        x_in_ul = np.random.randint(1, w - crop_in_w - 1)
        y_in_ul = np.random.randint(1, h - crop_in_h - 1)

        # unoverlap crop h, w
        unover_out_h = semi_unover_out_size
        unover_out_w = semi_unover_out_size

        # outside crop image1 upperleft location
        image1, mask1, overlap1_ul = self.out_crop(image, mask, w, h, x_in_ul, y_in_ul, crop_in_h, crop_in_w,
                                                   crop_out_h, crop_out_w)

        # outside crop image2 upperleft location
        image2, mask2, overlap2_ul = self.out_crop(image, mask, w, h, x_in_ul, y_in_ul, crop_in_h, crop_in_w,
                                                   crop_out_h, crop_out_w)

        image_unover_all, mask_unover_all, unover_ul_all = [], [], []
        for i in range(10):
            image_unover, mask_unover, unover_ul = self.unoverlap_crop(image, mask, w, h, x_in_ul, y_in_ul, crop_in_w,
                                                                       crop_out_h, unover_out_h, unover_out_w,
                                                                       semi_unover_in_size, unover_ul_xy)
            image_unover_all.append(image_unover)
            mask_unover_all.append(mask_unover)
            unover_ul_all.append(unover_ul)

        return image1, mask1, overlap1_ul, image2, mask2, overlap2_ul, image_unover_all, mask_unover_all, unover_ul_all

    def data_noaug(self, image, mask):
        whether_hori_flip = 0
        whether_ver_flip = 0
        rotate_angle = 0

        image_nd = np.array(image)
        mask_nd = np.array(mask)

        flip_rotate = [whether_hori_flip, whether_ver_flip, rotate_angle]

        return image_nd, mask_nd, flip_rotate

    @classmethod
    def preprocess(cls, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img_trans = img.transpose((2, 0, 1))  # -> [c, h, w]
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        # img_file = os.path.join(self.imgs_dir, idx + '.tiff')
        img_file = os.path.join(self.imgs_dir, idx + '.png')
        mask_file = os.path.join(self.masks_dir, idx + '.png')
        img = Image.open(img_file)
        mask = Image.open(mask_file)

        image1, mask1, overlap1_ul, image2, mask2, overlap2_ul, image_unover_all, mask_unover_all, unover_ul_all = self.overlap_crop(
            img, mask, self.semi_crop_in_size, self.semi_crop_out_size, self.semi_unover_out_size,
            self.semi_unover_in_size, self.unover_ul_xy)
        image1, mask1, flip_rotate_1 = self.data_noaug(image1, mask1)
        image2, mask2, flip_rotate_2 = self.data_noaug(image2, mask2)

        image_unover, mask_unover, flip_rotate_unover = [], [], []
        for i in range(len(image_unover_all)):
            image_unover_i, mask_unover_i = image_unover_all[i], mask_unover_all[i]
            image_unover_i, mask_unover_i, flip_rotate_unover_i = self.data_noaug(image_unover_i, mask_unover_i)
            image_unover_i, mask_unover_i = self.preprocess(image_unover_i), self.preprocess(mask_unover_i)
            image_unover.append(image_unover_i)
            mask_unover.append(mask_unover_i)
            flip_rotate_unover.append(flip_rotate_unover_i)

        image1 = self.preprocess(image1)
        mask1 = self.preprocess(mask1)
        image2 = self.preprocess(image2)
        mask2 = self.preprocess(mask2)

        images = np.stack([image1, image2])
        masks = np.stack([mask1, mask2])
        image_unover = np.stack(image_unover)
        mask_unover = np.stack(mask_unover)

        return {
            'image': torch.from_numpy(images).type(torch.FloatTensor),  # [n, c, h, w]
            'mask': torch.from_numpy(masks).type(torch.FloatTensor),
            'image_unover': torch.from_numpy(image_unover).type(torch.FloatTensor),  # [c, h, w] [n=10, c, h, w]
            'mask_unover': torch.from_numpy(mask_unover).type(torch.FloatTensor),
            'overlap1_ul': overlap1_ul,
            'overlap2_ul': overlap2_ul,
            'unover_ul': unover_ul_all,
            'flip_rotate_1': flip_rotate_1,
            'flip_rotate_2': flip_rotate_2,
            'flip_rotate_unover': flip_rotate_unover,
            'name': idx
        }