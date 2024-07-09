# -*- coding: utf-8 -*-
# @Time    : 2024/3/13
# @Author  : White Jiang
import torch
import numpy as np

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
    STANFORD_CAR.init(root)
    test_dataset = STANFORD_CAR(root, 'query', test_transform())
    train_dataset = STANFORD_CAR(root, 'train', train_transform())
    base_dataset = STANFORD_CAR(root, 'train', test_transform())

    print("query dataset:", len(test_dataset))
    print("train dataset:", len(train_dataset))

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
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return test_dataloader, train_dataloader, base_dataloader


class STANFORD_CAR(Dataset):

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            # 路径
            self.data = STANFORD_CAR.TRAIN_DATA
            # 对应的标签
            self.targets = STANFORD_CAR.TRAIN_TARGETS
        elif mode == 'test':
            self.data = STANFORD_CAR.TEST_DATA
            self.targets = STANFORD_CAR.TEST_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    @staticmethod
    def init(root):
        images_train = pd.read_csv(os.path.join(root, 'train.txt'), sep=' ', names=['filepath', 'target'])
        images_test = pd.read_csv(os.path.join(root, 'test.txt'), sep=' ', names=['filepath', 'target'])
        train_images = []
        label_list_train = []
        img_id_train = []
        for i in range(len(images_train)):
            train_images.append(images_train['filepath'][i])
            label_list_train.append(int(images_train['target'][i]))
            img_id_train.append(i + 1)

        k = len(train_images)
        test_images = []
        label_list_test = []
        img_id_test = []
        for i in range(len(images_test)):
            test_images.append(images_test['filepath'][i])
            label_list_test.append(int(images_test['target'][i]))
            img_id_test.append(k + i + 1)

        images_train = []
        for i in range(len(train_images)):
            images_train.append([img_id_train[i], 'cars_train/' + train_images[i], label_list_train[i]])
        images_train = pd.DataFrame(images_train, columns=['img_id', 'filepath', 'target'])

        images_test = []
        for i in range(len(test_images)):
            images_test.append([img_id_test[i], 'cars_test/' + test_images[i], label_list_test[i]])
        images_test = pd.DataFrame(images_test, columns=['img_id', 'filepath', 'target'])

        train_data = images_train
        test_data = images_test

        # Split dataset
        STANFORD_CAR.TEST_DATA = test_data['filepath'].to_numpy()
        STANFORD_CAR.TEST_TARGETS = encode_onehot((test_data['target'] - 1).tolist(), 196)

        STANFORD_CAR.TRAIN_DATA = train_data['filepath'].to_numpy()
        STANFORD_CAR.TRAIN_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 196)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx], idx
