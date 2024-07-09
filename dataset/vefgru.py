# -*- coding: utf-8 -*-
# @Time    : 2024/3/19
# @Author  : White Jiang

import sys
import torch
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image, ImageFile

from dataset.transform import encode_onehot
from dataset.transform import train_transform, test_transform


def load_data(root, batch_size, num_workers):
    Vegfru.init(root)
    test_dataset = Vegfru(root, 'test', test_transform())
    train_dataset = Vegfru(root, 'train', train_transform())
    base_dataset = Vegfru(root, 'train', test_transform())
    print(len(test_dataset))
    print(len(train_dataset))

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    base_dataloader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_dataloader, train_dataloader, base_dataloader


class Vegfru(Dataset):

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = Vegfru.TRAIN_DATA
            self.targets = Vegfru.TRAIN_TARGETS
        elif mode == 'test':
            self.data = Vegfru.TEST_DATA
            self.targets = Vegfru.TEST_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    @staticmethod
    def init(root):
        # This file use train data (do not combine train and val as a new train dataset).
        images_train = pd.read_csv(os.path.join(root, 'vegfru_list/vegfru_train.txt'), sep=' ',
                                   names=['filepath', 'target'])
        images_val = pd.read_csv(os.path.join(root, 'vegfru_list/vegfru_val.txt'), sep=' ',
                                 names=['filepath', 'target'])
        images_test = pd.read_csv(os.path.join(root, 'vegfru_list/vegfru_test.txt'), sep=' ',
                                  names=['filepath', 'target'])
        train_images = []
        test_images = []
        for i in range(len(images_train)):
            train_images.append(images_train['filepath'][i])
        for i in range(len(images_test)):
            test_images.append(images_test['filepath'][i])
        # print(images_train[:10])
        label_list_train = []
        img_id_train = []
        for i in range(len(images_train)):
            label_list_train.append(int(images_train['target'][i]) + 1)
            img_id_train.append(i + 1)

        # print(label_list_train[:10])
        images_train = []
        # print(len(train_images))
        for i in range(len(train_images)):
            images_train.append([img_id_train[i], train_images[i], label_list_train[i]])
        # print(images_train[:10])
        images_train = pd.DataFrame(images_train, columns=['img_id', 'filepath', 'target'])

        k = len(train_images)
        label_list_test = []
        img_id_test = []
        for i in range(len(test_images)):
            label_list_test.append(int(images_test['target'][i]) + 1)
            img_id_test.append(k + i + 1)
        images_test = []
        for i in range(len(test_images)):
            images_test.append([img_id_test[i], test_images[i], label_list_test[i]])
        # print(images_test[:10])
        images_test = pd.DataFrame(images_test, columns=['img_id', 'filepath', 'target'])

        train_data = images_train
        test_data = images_test

        # Split dataset
        Vegfru.TEST_DATA = test_data['filepath'].to_numpy()
        Vegfru.TEST_TARGETS = encode_onehot((test_data['target'] - 1).tolist(), 292)

        Vegfru.TRAIN_DATA = train_data['filepath'].to_numpy()
        Vegfru.TRAIN_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 292)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')
        except:
            img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGBA').convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx], idx
