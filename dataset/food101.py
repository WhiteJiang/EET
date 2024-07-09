# -*- coding: utf-8 -*-
# @Time    : 2024/3/21
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
    Food101.init(root)
    test_dataset = Food101(root, 'test', test_transform())
    train_dataset = Food101(root, 'train', train_transform())
    base_dataset = Food101(root, 'train', test_transform())
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


class Food101(Dataset):
    base_folder = 'images/'
    filename = 'food-101.tar.gz'

    def __init__(self, root, mode, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader

        if mode == 'train':
            self.data = Food101.TRAIN_DATA
            self.targets = Food101.TRAIN_TARGETS
        elif mode == 'test':
            self.data = Food101.TEST_DATA
            self.targets = Food101.TEST_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    @staticmethod
    def init(root):
        image_class_labels = pd.read_csv(os.path.join(root, 'meta/labels.txt'), names=['target'])
        # print(image_class_labels)
        d = {}
        for i in range(len(image_class_labels)):
            image_class_labels['target'][i] = image_class_labels['target'][i].replace(' ', '_')
            image_class_labels['target'][i] = image_class_labels['target'][i].lower()
            d[image_class_labels['target'][i]] = i + 1

        images_train = pd.read_csv(os.path.join(root, 'meta/train.txt'), names=['filepath'])
        images_test = pd.read_csv(os.path.join(root, 'meta/test.txt'), names=['filepath'])
        train_images = []
        test_images = []
        for i in range(len(images_train)):
            train_images.append(images_train['filepath'][i] + '.jpg')
        for i in range(len(images_test)):
            test_images.append(images_test['filepath'][i] + '.jpg')
        # print(images_train[:10])
        label_list_train = []
        img_id_train = []
        for i in range(len(train_images)):
            label = train_images[i].split('/')[0]
            label_list_train.append(d[label])
            img_id_train.append(i + 1)

        # print(label_list_train[:10])
        images_train = []
        for i in range(len(train_images)):
            images_train.append([img_id_train[i], 'images/' + train_images[i], label_list_train[i]])
        # images_train = images_train.merge(label_list_train, on='target')
        # images_train = images_train.merge(img_id_train, on='img_id')
        images_train = pd.DataFrame(images_train, columns=['img_id', 'filepath', 'target'])
        k = len(train_images)
        label_list_test = []
        img_id_test = []
        for i in range(len(test_images)):
            label = test_images[i].split('/')[0]
            label_list_test.append(d[label])
            img_id_test.append(k + i + 1)
        images_test = []
        for i in range(len(test_images)):
            images_test.append([img_id_test[i], 'images/' + test_images[i], label_list_test[i]])
        images_test = pd.DataFrame(images_test, columns=['img_id', 'filepath', 'target'])
        train_data = images_train
        test_data = images_test

        # Split dataset
        Food101.TEST_DATA = test_data['filepath'].to_numpy()
        Food101.TEST_TARGETS = encode_onehot((test_data['target'] - 1).tolist(), 101)

        Food101.TRAIN_DATA = train_data['filepath'].to_numpy()
        Food101.TRAIN_TARGETS = encode_onehot((train_data['target'] - 1).tolist(), 101)


    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx], idx

# def main():
#     query_dataloader, train_dataloader, retrieval_dataloader=load_data('/dataset/food101/', 16, 4)

# main()
