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
    def __init__(self,args, valid=False):
        
        self.valid = valid
        self.imgs_dir = args.image_dir
        self.masks_dir = args.mask_dir
        self.threshold = args.thresh
        self.crop_size = args.sup_crop_size
        self.test = args.test
        self.tta = args.tta
        self.data_augumentation = args.data_augumentation
        with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_99unsup_100val_431test.json','r') as sup_file:
            data_sup20 = json.load(sup_file)
        self.sup20 = data_sup20['train_sup']
        with open('./dataset/data_split.json','r') as jf:
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

        brightness = (1, 10)
        contrast = (1, 10)
        saturation = (1, 10)
        hue = (-0.5, 0.5)
        self.train_transform = transforms.Compose([
                transforms.ColorJitter(brightness, contrast, saturation, hue)
            ])
        

    def __len__(self):
        return len(self.ids)

    # random crop image and gt
    def random_crop(self, image, mask, size):
        # Pu: remove those resizing code.
        w, h = image.size
        crop_h = size
        crop_w = size

        start_x = np.random.randint(0, w - crop_w)
        start_y = np.random.randint(0, h - crop_h)

        # Pu: you may just use image.crop((start_x, start_y. start_x + crop_w, start_y + crop_h))
        # image_nd = np.array(image)
        # mask_nd = np.array(mask)
        #
        # image_nd = image_nd[start_y: start_y + crop_h, start_x: start_x + crop_w,  :]
        # mask_nd = mask_nd[start_y: start_y + crop_h, start_x: start_x + crop_w]
        #
        # img_crop = Image.fromarray(image_nd)
        # mask_crop = Image.fromarray(mask_nd)

        img_crop = image.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))
        mask_crop = mask.crop((start_x, start_y, start_x + crop_w, start_y + crop_h))

        return img_crop, mask_crop

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

        return image_nd, mask_nd


    # rotation image [0, 90, 180, 270]
    def rotate_img(self, image, mask):
        # Pu: remove those resizing code
        image_nd_list = []
        image_nd_0 = np.array(image)  # --> (h, w, c)
        mask_nd = np.array(mask)
        image_nd_list.append(image_nd_0)

        # rotation
        # Pu: it might be better to use np.rot90(img, k=1, axes=(0, 1)) k=1 for 90 degrees.
        for i in range(1, 4):
            image_rotate_nd = np.rot90(image_nd_0, k=i, axes=(0, 1)).copy()
            image_nd_list.append(image_rotate_nd)

        img_nd = np.stack(image_nd_list, axis=0)  # --> (n, h, w, c)
        return img_nd, mask_nd

    # change to array
    def toarray(self, image, mask):
        # Pu: remove those resizing code
        image_nd = np.array(image)
        mask_nd = np.array(mask)
        return image_nd, mask_nd

    @classmethod
    def preprocess(cls, img_nd, whether_mask, threshold, whether_test, whether_valid, whether_tta):
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

    def __getitem__(self, i):
        idx = self.ids[i]
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
            # if idx in self.sup20:
            #     if self.data_augumentation:
            #         img, mask = self.random_crop(img, mask, self.crop_size)
            #         img, mask = self.data_augu(img, mask)
            #     else:
            #         img, mask = self.random_crop(img, mask, self.crop_size)
            #         img = np.array(img)
            #         mask = np.array(mask)
            # else:
            #     if self.data_augumentation:
            #         img, mask = self.data_augu(img, mask)
            #     else:
            #         img = np.array(img)
            #         mask = np.array(mask)

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
            'image': torch.from_numpy(img).type(torch.FloatTensor), #[n, c, h, w]
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': idx
        }