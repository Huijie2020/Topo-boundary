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


class BasicDataset(Dataset):
    def __init__(self, args, valid=False):

        self.valid = valid
        self.imgs_dir = args.image_dir
        self.masks_dir = args.mask_dir
        self.threshold = args.thresh
        self.crop_size = args.sup_crop_size
        self.test = args.test
        self.tta = args.tta
        self.data_augumentation = args.data_augumentation
        with open('./dataset/data_split.json', 'r') as jf:
            data_load = json.load(jf)
        if args.test:
            self.ids = data_load['test']
            print('Testing mode. Data length {}.'.format(len(self.ids)))
        else:
            if valid:
                self.ids = data_load['valid']
                print('Validation length {}.'.format(len(self.ids)))
                print('=================')
            else:
                # self.ids = data_load['train'] + data_load['pretrain']
                self.ids = data_load['train_sup']
                print('=================')
                print('Training mode: Training supervised data length {}.'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    # random crop image and gt
    def random_crop(self, image, mask, size):
        w, h = image.size
        newW, newH = int(w), int(h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        image = image.resize((newW, newH))
        mask = mask.resize((newW, newH))
        crop_h = size
        crop_w = size

        start_x = np.random.randint(0, w - crop_w)
        start_y = np.random.randint(0, h - crop_h)

        image_nd = np.array(image)
        mask_nd = np.array(mask)

        image_nd = image_nd[start_y: start_y + crop_h, start_x: start_x + crop_w, :]
        mask_nd = mask_nd[start_y: start_y + crop_h, start_x: start_x + crop_w]

        img_crop = Image.fromarray(image_nd)
        mask_crop = Image.fromarray(mask_nd)

        return img_crop, mask_crop

    def data_augu(self, image, mask):
        whether_hori_flip = random.choice([0, 1])
        whether_ver_flip = random.choice([0, 1])
        whether_colorjitter = random.choice([0, 1])
        rotate_angle = random.choice([0, 90, 180, 270])

        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(whether_hori_flip),
            transforms.RandomVerticalFlip(whether_ver_flip)
        ])

        if whether_colorjitter == 0:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(whether_hori_flip),
                transforms.RandomVerticalFlip(whether_ver_flip)
            ])
        else:
            brightness = (1, 10)
            contrast = (1, 10)
            saturation = (1, 10)
            hue = (-0.5, 0.5)
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(whether_hori_flip),
                transforms.RandomVerticalFlip(whether_ver_flip),
                transforms.ColorJitter(brightness, contrast, saturation, hue)
            ])

        image_trans = train_transform(image)
        image_trans = TF.rotate(image_trans, rotate_angle)
        image_nd = np.array(image_trans)

        mask_trans = test_transform(mask)
        mask_trans = TF.rotate(mask_trans, rotate_angle)
        mask_nd = np.array(mask_trans)

        return image_nd, mask_nd

    # rotation image [0, 90, 180, 270]
    def rotate_img(self, image, mask):
        # resize image and mask
        w, h = image.size
        newW, newH = int(w), int(h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        image = image.resize((newW, newH))
        mask = mask.resize((newW, newH))

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
        w, h = image.size
        newW, newH = int(w), int(h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        image = image.resize((newW, newH))
        mask = mask.resize((newW, newH))

        image_nd = np.array(image)
        mask_nd = np.array(mask)

        return image_nd, mask_nd

    @classmethod
    def preprocess(cls, img_nd, whether_mask, threshold, whether_test, whether_valid, whether_tta):
        # w, h = pil_img.size
        # newW, newH = int(w), int(h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))
        # img_nd = np.array(pil_img)
        # if whether_mask:
        #     # img_nd = img_nd[:,:,0]
        #     img_nd = img_nd / 255.0
        #     # img_nd[img_nd < threshold] = 0
        #     # img_nd[img_nd >= threshold] = 1

        # normalize image
        def trans(img):
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            img_trans = img.transpose((2, 0, 1))  # -> [c, h, w]
            if img_trans.max() > 1:
                img_trans = img_trans / 255
            return img_trans

        if whether_tta:
            if whether_test or whether_valid:
                if whether_mask:
                    img_trans = trans(img_nd)
                else:
                    img_trans = img_nd.transpose((0, 3, 1, 2))  # (n, h, w, c) --> (n, c, h, w)
                    if img_trans.max() > 1:
                        img_trans = img_trans / 255
            else:
                img_trans = trans(img_nd)
        else:
            img_trans = trans(img_nd)
        return img_trans

        # if len(img_nd.shape) == 2:
        #     img_nd = np.expand_dims(img_nd, axis=2)

        # img_trans = img_nd.transpose((2, 0, 1)) # -> [c, h, w]
        # if img_trans.max() > 1:
        #     img_trans = img_trans / 255
        # return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        # img_file = os.path.join(self.imgs_dir, idx + '.tiff')
        img_file = os.path.join(self.imgs_dir, idx + '.png')
        mask_file = os.path.join(self.masks_dir, idx + '.png')
        img = Image.open(img_file)
        mask = Image.open(mask_file)

        # random crop image and traning data argumentation
        if (not self.test) and (not self.valid):
            if self.data_augumentation:
                img, mask = self.random_crop(img, mask, self.crop_size)
                img, mask = self.data_augu(img, mask)
            else:
                img, mask = self.random_crop(img, mask, self.crop_size)
                img = np.array(img)
                mask = np.array(mask)

        # # np.array test and valid
        # if (self.test) or (self.valid):
        #     img, mask = self.toarray(img, mask)

        # rotate test image and save as list
        if (self.test) or (self.valid):
            if self.tta:
                img, mask = self.rotate_img(img, mask)
            else:
                img, mask = self.toarray(img, mask)

        img = self.preprocess(img, False, self.threshold, self.test, self.valid, self.tta)
        mask = self.preprocess(mask, True, self.threshold, self.test, self.valid, self.tta)

        # if self.test or self.valid:
        #     sub_imgs = self.crop(img, self.crop_size)
        #     img = np.stack(sub_imgs, axis=0) #[c, w, h] -> [n, c, h, w]

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),  # [n, c, h, w]
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': idx
        }