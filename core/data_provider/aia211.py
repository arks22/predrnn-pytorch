from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
import numpy as np
from core.utils import preprocess

class Norm(object):
    def __init__(self, max_val=255.):
        self.max_val = max_val

    def __call__(self, sample):
        video_x = sample
        new_video_x = video_x / self.max_val
        return new_video_x


class ToTensor(object):
    def __call__(self, sample):
        video_x = sample
        video_x = video_x.transpose((0, 3, 1, 2))
        video_x = np.array(video_x)
        return torch.from_numpy(video_x).float()


class aia211(Dataset):
    def __init__(self, configs, data_train_path, data_test_path, mode, transform=None):
        self.transform = transform
        self.mode = mode
        self.configs = configs
        self.patch_size = configs.patch_size
        self.img_width = configs.img_width
        self.img_height = configs.img_height
        self.img_channel = configs.img_channel
        if self.mode == 'train':
            print('Loading train dataset')
            self.path = data_train_path
            self.data = np.load(self.path)
            print('{} data * {} frames * {} px * {} px * {} channel'.format(self.data.shape[1], self.data.shape[0], self.data.shape[2], self.data.shape[3], self.data.shape[4] ))
        else:
            print('Loading test dataset')
            self.path = data_test_path
            self.data = np.load(self.path)
            print('{} data * {} frames * {} px * {} px * {} channel'.format(self.data.shape[1], self.data.shape[0], self.data.shape[2], self.data.shape[3], self.data.shape[4] ))

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        #print(self.index[3])
        sample = self.data[:, idx, :]

        if self.transform:
            sample = preprocess.reshape_patch(sample, self.patch_size)
            sample = self.transform(sample)
        return sample
