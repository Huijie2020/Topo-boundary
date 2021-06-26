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

class BasicDataset(Dataset):
    def __init__(self,args, valid=False):
        
        self.valid = valid
        self.imgs_dir = args.image_dir
        self.masks_dir = args.mask_dir
        self.threshold = args.thresh
        self.crop_size = args.crop_size
        self.test = args.test
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
                self.ids = data_load['train'] + data_load['pretrain']
                print('=================')
                print('Training mode: Training length {}.'.format(len(self.ids)))
        
        

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

        image_nd = image_nd[start_x: start_x + crop_w, start_y: start_y + crop_h, :]
        mask_nd = mask_nd[start_x: start_x + crop_w, start_y: start_y + crop_h]

        return image_nd, mask_nd

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
    def preprocess(cls, img_nd, whether_mask, threshold):
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
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        #
        img_trans = img_nd.transpose((2, 0, 1)) # -> [c, h, w]
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

        # random crop image
        if (not self.test) and (not self.valid):
            img, mask = self.random_crop(img, mask, self.crop_size)

        # np.array test and valid
        if (self.test) or (self.valid):
            img, mask = self.toarray(img, mask)

        img = self.preprocess(img, False, self.threshold)
        mask = self.preprocess(mask,True, self.threshold)

        # if self.test or self.valid:
        #     sub_imgs = self.crop(img, self.crop_size)
        #     img = np.stack(sub_imgs, axis=0) #[c, w, h] -> [n, c, h, w]
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor), #[b, n, c, h, w]
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name':idx
        }