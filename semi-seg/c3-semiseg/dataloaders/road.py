import sys
# sys.path.append('/home/ma-user/work/pytorch_segmentation-master')
from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import os
import json

ignore_label = -100
from dataloaders.transforms import *

ID_TO_TRAINID = { 0: ignore_label, 1: 0}

class RoadDataset(BaseDataSet):
    def __init__(self, mode='fine', n_sup=-1, split_seed=1, end_percent=0.4,
                 color_jitter=1, random_grayscale=0.2, gaussian_blur=11, to_bgr=False, rotate=False,
                 autoaug=False, high=512, scale_list=[1.0], **kwargs):
        self.num_classes = 2
        self.mode = mode
        self.palette = palette.ADE20K_palette
        self.id_to_trainId = ID_TO_TRAINID
        self.n_sup = n_sup  #
        self.split_seed = split_seed

        self.start_percent = 0.2
        self.end_percent = end_percent
        self.use_cb_threshold = False
        self.curr_epoch = 0
        if rotate == True:
            rotation_degree = 10
        else:
            rotation_degree = 0
        self.rotation_degree = rotation_degree
        self.color_jitter = color_jitter
        self.random_grayscale = random_grayscale
        self.gaussian_blur = gaussian_blur
        self.to_bgr = to_bgr
        self.split = kwargs['split']
        self.autoaug = autoaug
        self.high = high
        self.scale_list = scale_list
        self.crop_size = kwargs['crop_size']
        self._set_transform()
        self.filter_front_camera = True
        super(RoadDataset, self).__init__(to_bgr=to_bgr, rotate=rotate, **kwargs)

    def _set_transform(self):
        if self.split == 'train':
            self.transform = build_transform(
                self.rotation_degree,
                self.color_jitter,
                self.random_grayscale, self.gaussian_blur,
                crop=self.crop_size,
                to_bgr=self.to_bgr,
                flip=True, high=self.high,
                scales=self.scale_list, pad_if_need=True)
        else:
            self.transform = build_transform(
                rotation=0,
                color_jitter=0,
                random_grayscale=0,
                gaussian_blur=0,
                to_bgr=self.to_bgr,
                flip=False,
                high=self.high, pad_if_need=True)

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def _set_files(self):
        image_path = os.path.join(self.root, 'images')
        label_path = os.path.join(self.root, 'gt')

        with open(os.path.join(self.root, 'data_split.json'), 'r') as jf:
            data_split = json.load(jf)
        jf.close()

        train_sup = data_split['train_sup']
        valid = data_split['valid']
        test = data_split['test']

        image_paths, label_paths = [], []

        if self.split == 'train':
            for id in train_sup:
                image_paths.append(os.path.join(image_path, id + '.png'))
                label_paths.append(os.path.join(label_path, id + '.png'))
        if self.split == 'val':
            for id in valid:
                image_paths.append(os.path.join(image_path, id + '.png'))
                label_paths.append(os.path.join(label_path, id + '.png'))
        if self.split == 'test':
            for id in test:
                image_paths.append(os.path.join(image_path, id + '.png'))
                label_paths.append(os.path.join(label_path, id + '.png'))

        assert len(label_paths) == len(image_paths)
        self.files = list(zip(image_paths, label_paths))
        print('Load {} SpaceNet data as {} labeled set.'.format(len(self.files), self.split))

    def _augmentation(self, image, label):
        image, label = self.transform(image, label)
        return image, label

    def _load_data(self, index):
        while True:
            image_path, label_path = self.files[index]
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
            label = np.array(Image.open(label_path)) / 255
            label = np.asarray(label, dtype=np.int32)
            index = min(index - 1, 0)
            if (label != 255).any():
                break
        return image, label, image_id

class PseudoRoadDataset(RoadDataset):
    def __init__(self, mode='fine', n_sup=-1, split_seed=1,
                 end_percent=0.4, autoaug=False, labeled_part=False, high=720,
                 scale_list=[0.75, 1.0, 1.25], **kwargs):
        self.num_classes = 2
        self.mode = mode
        self.palette = palette.CityScpates_palette
        self.id_to_trainId = ID_TO_TRAINID
        self.n_sup = n_sup
        self.split_seed = split_seed

        self.start_percent = 0.2
        self.end_percent = end_percent
        self.use_cb_threshold = False
        self.curr_epoch = 0
        self.cls_prob_list = [[] for _ in range(2)]
        self.autoaug = autoaug
        self.high = high
        self.crop_size = kwargs['crop_size']
        self.scale_list = scale_list
        self.labeled_part = labeled_part  # if true, load labeled set, if false load unlabeled set
        self.init_cb_threshold()
        super(PseudoRoadDataset, self).__init__(mode=mode, n_sup=n_sup, split_seed=split_seed,
                                                  end_percent=end_percent, high=high, **kwargs)

    def _set_transform(self):
        '''
        strong contain jittering, grayscale and gaussian
        :return: self.transform = [share_trans, strong_trans, weak_trans]
        '''

        if not self.autoaug:
            self.transform = build_weak_strong_transform(
                self.rotation_degree,
                self.color_jitter,
                self.random_grayscale,
                self.gaussian_blur,
                crop=self.crop_size,
                to_bgr=self.to_bgr,
                high=self.high,
                scales=self.scale_list,
                pad_if_need=True
            )
        else:
            self.transform = build_autoaug_transform(
                self.rotation_degree,
                self.color_jitter,
                self.random_grayscale,
                self.gaussian_blur,
                crop=self.crop_size,
                to_bgr=self.to_bgr,
                high=self.high,
                scales=self.scale_list,
                pad_if_need=True
            )

    def _set_files(self):
        image_path = os.path.join(self.root, 'images')
        label_path = os.path.join(self.root, 'gt')

        with open(os.path.join(self.root, 'data_split.json'), 'r') as jf:
            data_split = json.load(jf)
        jf.close()

        train_unsup = data_split['train_unsup']

        image_paths, label_paths = [], []

        if self.split == 'train':
            for id in train_unsup:
                image_paths.append(os.path.join(image_path, id + '.png'))
                label_paths.append(os.path.join(label_path, id + '.png'))
        assert len(label_paths) == len(image_paths)
        self.files = list(zip(image_paths, label_paths))
        print('Load {} SpaceNet data as unlabeled part.'.format(len(self.files)))

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.array(Image.open(label_path)) / 255
        label = np.asarray(label, dtype=np.int32)
        return image, label, image_id

    def _augmentation(self, image, label):
        image, label = self.transform[0](image, label)
        weak, _ = self.transform[2](image.copy(), None)
        strong, _ = self.transform[1](image.copy(), None)
        return weak, strong, label

    def __getitem__(self, index):
        while True:
            image, label, image_id = self._load_data(index)
            aug_data = self._augmentation(image, label)
            weak, strong, label = aug_data
            weak = Image.fromarray(np.uint8(weak))
            strong = Image.fromarray(np.uint8(strong))
            label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
            index = min(index - 1, 0)
            if (label != 255).any():
                break
        return self.normalize(self.to_tensor(weak)), self.normalize(self.to_tensor(strong)), label, image_id

    def init_cb_threshold(self, prob=0.9):
        self.cls_conf_threshold = [prob] * 2  # [prob] * 19

    def update_ema_cb_threshold(self, prob, ema=0.95, gamma=0.0):

        probability, predict = torch.max(prob.data, 1)
        probability = probability.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()
        curr_prob_list = [[] for _ in range(2)]  #
        curr_cls_conf_threshold = [0] * 2
        for cls in range(2):
            if (cls == predict).any():
                # todo:  subsample curr_prob_list[cls].extend(probability[cls == predict].tolist()[::4])
                curr_prob_list[cls].extend(probability[cls == predict].tolist())
        curr_total_num = [len(cls_prob) for cls_prob in curr_prob_list]
        curr_prob_list = [sorted(cls_prob, reverse=True) for cls_prob in curr_prob_list]
        for cls in range(2):
            if curr_total_num[cls] > 0:
                try:

                    if self.cls_conf_threshold[cls] == 0:
                        curr_cls_conf_threshold[cls] = curr_prob_list[cls] \
                            [int(curr_total_num[cls] * self.end_percent)]
                        self.cls_conf_threshold[cls] = curr_cls_conf_threshold[cls]
                    else:
                        curr_cls_conf_threshold[cls] = curr_prob_list[cls] \
                            [int(curr_total_num[cls] * self.end_percent * self.cls_conf_threshold[cls] ** gamma)]
                        self.cls_conf_threshold[cls] = self.cls_conf_threshold[cls] * ema + (1 - ema) * \
                                                       curr_cls_conf_threshold[cls]
                except:
                    print('bug in ema bc threshold func!')

    def get_cb_threshould_mask(self, prob):
        # prob = torch.nn.functional.softmax(prob.clone(), 1)
        prob, pred = torch.max(prob, 1)

        conf_mask = torch.zeros_like(pred).float()
        binary_mask = torch.zeros_like(pred)
        for cls_id in range(2):
            conf_mask[pred == cls_id] = self.cls_conf_threshold[cls_id]
        binary_mask[prob >= conf_mask] = 1
        return binary_mask[:, None, :, :]


class PseudoRoad(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 mode='fine', val=False,
                 shuffle=False, num_samples=1000, flip=False, rotate=False, blur=False, augment=False, val_split=None,
                 return_id=False,
                 distributed=False, world_size=1, rank=-1, pretrain_from='imagenet', split_seed=1,
                 n_sup=-1, end_percent=0.4, color_jitter=1, random_grayscale=0.2, gaussian_blur=11,
                 generation=False, autoaug=False, labeled_part=False, high=720, scale_list=[0.75, 1.0, 1.25],
                 siam_weak=False, siam_strong=False
                 ):
        if pretrain_from == 'imagenet':
            # self.MEAN = [0.28689529, 0.32513294, 0.28389176]
            # self.STD = [0.17613647, 0.18099176, 0.17772235]
            to_bgr = False
            self.MEAN = [0.485, 0.456, 0.406]
            self.STD = [0.229, 0.224, 0.225]
        else:
            to_bgr = True
            self.MEAN = [73.15835921, 82.90891754, 72.39239876]
            self.STD = [1., 1., 1.]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'to_bgr': to_bgr,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'split_seed': split_seed,
            'n_sup': n_sup,

            'end_percent': end_percent,
            "color_jitter": color_jitter,
            "random_grayscale": random_grayscale,
            "gaussian_blur": gaussian_blur,
            "autoaug": autoaug,
            "labeled_part": labeled_part,
            "high": high,
            "scale_list": scale_list
        }

        self.dataset = PseudoRoadDataset(mode=mode, **kwargs)
        super(PseudoRoad, self).__init__(self.dataset, batch_size, shuffle, num_workers,
                                            val_split, distributed, world_size, rank, num_samples)


class Road(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 mode='fine', val=False,
                 shuffle=False, num_samples=1000, flip=False, rotate=False, blur=False, augment=False, val_split=None,
                 return_id=False,
                 distributed=False, world_size=1, rank=-1, pretrain_from='imagenet', split_seed=1, n_sup=-1,
                 end_percent=0.4, color_jitter=1, random_grayscale=0.2, gaussian_blur=11, high=720,
                 scale_list=[0.75, 1.0, 1.25]):
        if pretrain_from == 'imagenet':

            to_bgr = False
            self.MEAN = [0.485, 0.456, 0.406]
            self.STD = [0.229, 0.224, 0.225]
        else:
            to_bgr = True
            self.MEAN = [73.15835921, 82.90891754, 72.39239876]
            self.STD = [1., 1., 1.]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'to_bgr': to_bgr,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'split_seed': split_seed,
            'n_sup': n_sup,
            'end_percent': end_percent,
            "color_jitter": color_jitter,
            "random_grayscale": random_grayscale,
            "gaussian_blur": gaussian_blur,
            "high": high,
            "scale_list": scale_list
        }

        self.dataset = RoadDataset(mode=mode, **kwargs)
        super(Road, self).__init__(self.dataset, batch_size, shuffle, num_workers,
                                      val_split, distributed, world_size, rank, num_samples)


if __name__ == '__main__':
    pass
