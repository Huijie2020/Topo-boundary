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

class UnsupDataset(Dataset):
    def __init__(self, args):
        self.imgs_dir = args.image_dir
        self.masks_dir = args.mask_dir
        self.semi_crop_in_size = args.unsup_crop_in_size
        self.semi_crop_out_size = args.unsup_crop_out_size
        with open('./dataset/data_split.json', 'r') as jf:
            data_load = json.load(jf)
        # self.ids = data_load['train'] + data_load['pretrain']
        self.ids = data_load['train_unsup']
        print('=================')
        print('Training mode: Training unsupervised data length {}.'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def out_crop(self, image_nd, mask_nd, w, h, x_in_ul, y_in_ul, crop_in_h, crop_in_w, crop_out_h, crop_out_w):
        # upper left location
        x_out_ul = np.random.randint(max(x_in_ul - (crop_out_w - crop_in_w), 0), min(x_in_ul, w - crop_out_w))
        y_out_ul = np.random.randint(max(y_in_ul - (crop_out_h - crop_in_h), 0), min(y_in_ul, h - crop_out_h))

        # outside crop image
        image_nd = image_nd[y_out_ul: y_out_ul + crop_out_h, x_out_ul: x_out_ul + crop_out_w, :]
        mask_nd = mask_nd[y_out_ul: y_out_ul + crop_out_h, x_out_ul: x_out_ul + crop_out_w]
        image_crop = Image.fromarray(image_nd)
        mask_crop = Image.fromarray(mask_nd)

        # overlap location
        overlap_ul = [y_in_ul - y_out_ul, x_in_ul - x_out_ul]
        image_ul = [y_out_ul, x_out_ul]

        return image_crop, mask_crop, image_ul, overlap_ul

    def overlap_crop(self, image, mask, semi_crop_in_size, semi_crop_out_size):
        w, h = image.size
        # newW, newH = int(w), int(h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # image = image.resize((newW, newH))
        # mask = mask.resize((newW, newH))

        # inside crop h,w
        crop_in_h = semi_crop_in_size
        crop_in_w = semi_crop_in_size

        # outside crop h,w
        crop_out_h = semi_crop_out_size
        crop_out_w = semi_crop_out_size

        image_nd = np.array(image)
        mask_nd = np.array(mask)

        # inside crop upperleft location
        x_in_ul = np.random.randint(1, w - crop_in_w - 1)
        y_in_ul = np.random.randint(1, h - crop_in_h - 1)
        # # inside crop image
        # image_in_nd = image_nd[y_in_ul: y_in_ul + crop_in_h, x_in_ul: x_in_ul + crop_in_w, :]
        # mask_in_nd = mask_nd[y_in_ul: y_in_ul + crop_in_h, x_in_ul: x_in_ul + crop_in_w]
        # image_in = Image.fromarray(image_in_nd)
        # mask_in = Image.fromarray(mask_in_nd)

        # outside crop image1 upperleft location
        image1, mask1, image1_ul, overlap1_ul = self.out_crop(image_nd, mask_nd, w, h, x_in_ul, y_in_ul,
                                                   crop_in_h, crop_in_w, crop_out_h, crop_out_w)

        # outside crop image2 upperleft location
        image2, mask2, image2_ul, overlap2_ul = self.out_crop(image_nd, mask_nd, w, h, x_in_ul, y_in_ul,
                                                             crop_in_h, crop_in_w, crop_out_h, crop_out_w)

        # outside crop image3 upperleft location
        image3, mask3, image3_ul, overlap3_ul = self.out_crop(image_nd, mask_nd, w, h, x_in_ul, y_in_ul,
                                                             crop_in_h,crop_in_w, crop_out_h, crop_out_w)

        # outside crop image4 upperleft location
        image4, mask4, image4_ul, overlap4_ul = self.out_crop(image_nd, mask_nd, w, h, x_in_ul, y_in_ul,
                                                   crop_in_h, crop_in_w, crop_out_h, crop_out_w)

        return image1, mask1, image1_ul, overlap1_ul, image2, mask2, image2_ul, overlap2_ul, image3, mask3, image3_ul, overlap3_ul, image4, mask4, image4_ul, overlap4_ul

    # rotation image [0, 90, 180, 270]
    def rotate_img(self, image, mask):

        image_nd_list = []
        image_nd_0 = np.array(image)  # --> (h, w, c)
        mask_nd = np.array(mask)
        image_nd_list.append(image_nd_0)

        # rotation
        angle_list = [90, 180, 270]
        for angle in angle_list:
            image_rotate = TF.rotate(image, angle)
            image_rotate_nd = np.array(image_rotate)
            image_nd_list.append(image_rotate_nd)

        img_nd = np.stack(image_nd_list, axis=0)  # --> (n, h, w, c)
        return img_nd, mask_nd

    # change to array
    def toarray(self, image, mask):

        image_nd = np.array(image)
        mask_nd = np.array(mask)

        return image_nd, mask_nd

    @classmethod
    def preprocess(cls, img_nd, whether_mask, whether_test, whether_valid, whether_tta):
        # normalize image
        def trans(img):
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            img_trans = img.transpose((2, 0, 1))  # -> [c, h, w]
            if img_trans.max() > 1:
                img_trans = img_trans / 255
            return img_trans

        if whether_tta:
            # if whether_test or whether_valid:
            if whether_mask:
                img_trans = trans(img_nd)
            else:
                img_trans = img_nd.transpose((0, 3, 1, 2))  # (n, h, w, c) --> (n, c, h, w)
                if img_trans.max() > 1:
                    img_trans = img_trans / 255
            # else:
            #     img_trans = trans(img_nd)
        else:
            img_trans = trans(img_nd)
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        # img_file = os.path.join(self.imgs_dir, idx + '.tiff')
        img_file = os.path.join(self.imgs_dir, idx + '.png')
        mask_file = os.path.join(self.masks_dir, idx + '.png')
        img = Image.open(img_file)
        mask = Image.open(mask_file)

        # random crop image and traning data augumentation
        image1, mask1, image1_ul, overlap1_ul, image2, mask2, image2_ul, overlap2_ul, image3, mask3, image3_ul, overlap3_ul, image4, mask4, image4_ul, overlap4_ul = self.overlap_crop(img, mask, self.semi_crop_in_size, self.semi_crop_out_size)
        # rotate test image and save as list
        if self.tta:
            image1, mask1 = self.rotate_img(image1, mask1)
            image2, mask2 = self.rotate_img(image2, mask2)
            image3, mask3 = self.rotate_img(image3, mask3)
            image4, mask4 = self.rotate_img(image4, mask4)
        else:
            image1, mask1 = self.toarray(image1, mask1)
            image2, mask2 = self.toarray(image2, mask2)
            image3, mask3 = self.toarray(image3, mask3)
            image4, mask4 = self.toarray(image4, mask4)


        image1 = self.preprocess(image1) # (n, h, w, c) --> (n, c, h, w)
        mask1 = self.preprocess(mask1)
        image2 = self.preprocess(image2)
        mask2 = self.preprocess(mask2)
        image3 = self.preprocess(image3)
        mask3 = self.preprocess(mask3)
        image4 = self.preprocess(image4)
        mask4 = self.preprocess(mask4)

        images =  np.stack([image1, image2, image3, image4], axis=0) # (n, c, h, w) -> (i, n, c, h, w)
        masks = np.stack([mask1, mask2, mask3, mask4], axis=0) # [c, h, w] -> [i, c, h, w]

        return {
            'image': torch.from_numpy(images).type(torch.FloatTensor), # (i, n, c, h, w)
            'mask': torch.from_numpy(masks).type(torch.FloatTensor), # [i, c, h, w]
            'overlap1_ul': overlap1_ul,
            'overlap2_ul': overlap2_ul,
            'overlap3_ul': overlap3_ul,
            'overlap4_ul': overlap4_ul,
            'image1_ul': image1_ul,
            'image2_ul': image2_ul,
            'image3_ul': image3_ul,
            'image4_ul': image4_ul,
            'name':idx
        }