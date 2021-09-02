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
import cv2

class UnsupDataset(Dataset):
    def __init__(self, args):
        self.imgs_dir = args.image_dir
        self.masks_dir = args.mask_dir
        self.semi_crop_in_size = args.unsup_crop_in_size
        self.semi_crop_out_size = args.unsup_crop_out_size
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

    def overlap_crop(self, image, mask, semi_crop_in_size, semi_crop_out_size):
        w, h = image.size

        # inside crop h,w
        crop_in_h = semi_crop_in_size
        crop_in_w = semi_crop_in_size

        # outside crop h,w
        crop_out_h = semi_crop_out_size
        crop_out_w = semi_crop_out_size

        # Pu: remove the conversion between numpy and Image
        # image_nd = np.array(image)
        # mask_nd = np.array(mask)

        # inside crop upperleft location
        x_in_ul = np.random.randint(1, w - crop_in_w - 1)
        y_in_ul = np.random.randint(1, h - crop_in_h - 1)
        # # inside crop image
        # image_in_nd = image_nd[y_in_ul: y_in_ul + crop_in_h, x_in_ul: x_in_ul + crop_in_w, :]
        # mask_in_nd = mask_nd[y_in_ul: y_in_ul + crop_in_h, x_in_ul: x_in_ul + crop_in_w]
        # image_in = Image.fromarray(image_in_nd)
        # mask_in = Image.fromarray(mask_in_nd)

        # outside crop image1 upperleft location
        image1, mask1, overlap1_ul = self.out_crop(image, mask, w, h, x_in_ul, y_in_ul,
                                                   crop_in_h, crop_in_w, crop_out_h, crop_out_w)

        # outside crop image2 upperleft location
        image2, mask2, overlap2_ul = self.out_crop(image, mask, w, h, x_in_ul, y_in_ul,
                                                             crop_in_h, crop_in_w, crop_out_h, crop_out_w)

        # # outside crop image3 upperleft location
        # image3, mask3, overlap3_ul = self.out_crop(image_nd, mask_nd, w, h, x_in_ul, y_in_ul,
        #                                                      crop_in_h,crop_in_w, crop_out_h, crop_out_w)
        #
        # # outside crop image4 upperleft location
        # image4, mask4, overlap4_ul = self.out_crop(image_nd, mask_nd, w, h, x_in_ul, y_in_ul,
        #                                            crop_in_h, crop_in_w, crop_out_h, crop_out_w)

        # return image1, mask1, overlap1_ul, image2, mask2, overlap2_ul, image3, mask3, overlap3_ul, image4, mask4, overlap4_ul
        return image1, mask1, overlap1_ul, image2, mask2, overlap2_ul


    def data_augu(self, image, mask):
        # Pu: this is awkward.
        whether_hori_flip = random.choice([0, 1])
        whether_ver_flip = random.choice([0, 1])
        whether_colorjitter = random.choice([0, 1])
        rotate_angle = random.choice([0, 1, 2, 3])

        if whether_colorjitter > 0:
            image = self.train_transform(image)

        if whether_hori_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if whether_ver_flip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        image_nd = np.array(image)
        mask_nd = np.array(mask)

        image_nd = np.rot90(image_nd, k=rotate_angle, axes=(0, 1)).copy()
        mask_nd = np.rot90(mask_nd, k=rotate_angle, axes=(0, 1)).copy()

        flip_rotate = [whether_hori_flip, whether_ver_flip, rotate_angle * 90]

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

        # random crop image and traning data augumentation
        if self.data_augumentation:
            # image1, mask1, overlap1_ul, image2, mask2, overlap2_ul, image3, mask3, overlap3_ul, image4, mask4, overlap4_ul = self.overlap_crop(img, mask, self.semi_crop_in_size, self.semi_crop_out_size)
            image1, mask1, overlap1_ul, image2, mask2, overlap2_ul = self.overlap_crop(img, mask, self.semi_crop_in_size, self.semi_crop_out_size)
            # # save middle image
            # image1_save = np.array(image1).astype(np.uint8)
            # image1_save = cv2.rectangle(image1_save,
            #                             (overlap1_ul[1], overlap1_ul[0]),
            #                             (overlap1_ul[1] + self.semi_crop_in_size,overlap1_ul[0]+ self.semi_crop_in_size),
            #                             (255, 0, 0), 2)
            # image1_save_path = os.path.join('/mnt/git/Topo-boundary/segmentation_based_baselines/naive_baseline/checkpoints/middle_image', idx + '_01.png')
            # cv2.imwrite(image1_save_path, image1_save)
            #
            # image2_save = np.array(image2).astype(np.uint8)
            # image2_save = cv2.rectangle(image2_save,
            #                             (overlap2_ul[1], overlap2_ul[0]),
            #                             (overlap2_ul[1] + self.semi_crop_in_size, overlap2_ul[0] + self.semi_crop_in_size),
            #                             (255, 0, 0), 2)
            # image2_save_path = os.path.join('/mnt/git/Topo-boundary/segmentation_based_baselines/naive_baseline/checkpoints/middle_image',idx + '_02.png')
            # cv2.imwrite(image2_save_path, image2_save)

            image1, mask1, flip_rotate_1 = self.data_augu(image1, mask1)
            image2, mask2, flip_rotate_2 = self.data_augu(image2, mask2)
            # image3, mask3, flip_rotate_3 = self.data_augu(image3, mask3)
            # image4, mask4, flip_rotate_4 = self.data_augu(image4, mask4)
        else:
            # image1, mask1, overlap1_ul, image2, mask2, overlap2_ul, image3, mask3, overlap3_ul, image4, mask4, overlap4_ul = self.overlap_crop(img, mask, self.semi_crop_in_size, self.semi_crop_out_size)
            image1, mask1, overlap1_ul, image2, mask2, overlap2_ul = self.overlap_crop(img, mask, self.semi_crop_in_size, self.semi_crop_out_size)
            image1 = np.array(image1)
            mask1 = np.array(mask1)
            image2 = np.array(image2)
            mask2 = np.array(mask2)
            # image3 = np.array(image3)
            # mask3 = np.array(mask3)
            # image4 = np.array(image4)
            # mask4 = np.array(mask4)
            # flip_rotate_1, flip_rotate_2, flip_rotate_3, flip_rotate_4 = None, None, None, None
            flip_rotate_1, flip_rotate_2 = None, None

        image1 = self.preprocess(image1)
        mask1 = self.preprocess(mask1)
        image2 = self.preprocess(image2)
        mask2 = self.preprocess(mask2)
        # image3 = self.preprocess(image3)
        # mask3 = self.preprocess(mask3)
        # image4 = self.preprocess(image4)
        # mask4 = self.preprocess(mask4)

        # images =  np.stack([image1, image2, image3, image4])
        # masks = np.stack([mask1, mask2, mask3, mask4])
        images =  np.stack([image1, image2])
        masks = np.stack([mask1, mask2])

        return {
            'image': torch.from_numpy(images).type(torch.FloatTensor), #[n, c, h, w]
            'mask': torch.from_numpy(masks).type(torch.FloatTensor),
            'overlap1_ul': overlap1_ul,
            'overlap2_ul': overlap2_ul,
            # 'overlap3_ul': overlap3_ul,
            # 'overlap4_ul': overlap4_ul,
            'flip_rotate_1': flip_rotate_1,
            'flip_rotate_2': flip_rotate_2,
            # 'flip_rotate_3': flip_rotate_3,
            # 'flip_rotate_4': flip_rotate_4,
            'name':idx
        }