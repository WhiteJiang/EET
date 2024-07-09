from model.vit import VisionTransformer, checkpoint_filter_fn, VisionTransformerTeacher
import torch
import argparse
from dataset import cub, nabirds, aircraft, car, vegfru, food101
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from train_model import train_model
import numpy as np
import random
import os
from utils import WarmupCosineSchedule
from loss import DistillKL, HashDistill


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def start_train(args, code_length, w=0.0):
    dataloader = {}

    if args.dataset == 'cub_bird':
        classes = 200
        data_dir = '/path/to/CUB_200_2011/'
        test_dataloader, train_dataloader, base_dataloader = \
            cub.load_data(root=data_dir, batch_size=args.batch_size, num_workers=8)
    elif args.dataset == 'car':
        classes = 196
        data_dir = '/path/to/Stanford_Cars/'
        test_dataloader, train_dataloader, base_dataloader = \
            car.load_data(root=data_dir, batch_size=args.batch_size, num_workers=8)
    elif args.dataset == 'vegfru':
        classes = 292
        data_dir = '/path/to/vegfru/'
        test_dataloader, train_dataloader, base_dataloader = \
            vegfru.load_data(root=data_dir, batch_size=args.batch_size, num_workers=8)
    elif args.dataset == 'food101':
        classes = 101
        data_dir = '/path/to/food-101/'
        test_dataloader, train_dataloader, base_dataloader = \
            food101.load_data(root=data_dir, batch_size=args.batch_size, num_workers=8)
    elif args.dataset == 'nabirds':
        classes = 555
        data_dir = '/path/to/nabirds/'
        test_dataloader, train_dataloader, base_dataloader = \
            nabirds.load_data(root=data_dir, batch_size=args.batch_size, num_workers=8)
    else:
        print('undefined dataset ! ')

    dataloader['train'] = train_dataloader
    dataloader['val'] = test_dataloader
    dataloader['base'] = base_dataloader

    # model vit-small
    if 'vit-small' in args.model_name:
        model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                  qkv_bias=True, num_classes=classes, hash_code=code_length)
        model_path = "./deit_small_patch16_224-cd65a155.pth"
        model_teacher = VisionTransformerTeacher(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                                 qkv_bias=True, num_classes=classes, hash_code=code_length)
    elif 'vit-base' in args.model_name:
        # model vit-base
        model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                  qkv_bias=True, num_classes=classes, hash_code=code_length)
        model_path = "./deit_base_patch16_224-b5f2ef4d.pth"
        model_teacher = VisionTransformerTeacher(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                                 qkv_bias=True, num_classes=classes, hash_code=code_length)
    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint_filter_fn(checkpoint, model)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    _, _ = model_teacher.load_state_dict(ckpt, strict=False)
    print('# missing keys=', missing_keys)
    print('# unexpected keys=', unexpected_keys)
    print('successfully loaded from pre-trained weights:', model_path)
    model = model.cuda()
    t_total = args.epoch * len(dataloader['train'])
    # for teacher
    model_teacher = model_teacher.cuda()
    criterion_hash_kd = HashDistill()
    optimizer_teacher = optim.SGD(model_teacher.parameters(), lr=args.lr, momentum=0.9)
    exp_lr_scheduler_teacher = WarmupCosineSchedule(optimizer_teacher, warmup_steps=500, t_total=t_total)

    # loss and opt
    criterion = nn.CrossEntropyLoss()
    criterion_hash = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    exp_lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=t_total)

    log_file = open(args.model_name + '_' + args.dataset + '_' + str(code_length) + '_' + str(w) + '.log', 'a')
    log_file.write(str(args))
    log_file.write('\n')
    print('training start ...')

    train_model(model, dataloader, criterion, criterion_hash, optimizer, exp_lr_scheduler, args.epoch,
                code_length, classes, log_file, model_teacher, criterion_hash_kd, exp_lr_scheduler_teacher,
                optimizer_teacher, w, args.model_name, args.dataset)
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--random-seed', default=42, type=int, help='random seed')

    parser.add_argument('--dataset', default='cub_bird', type=str,
                        choices=['cub_bird', 'car', 'vegfru', 'food101', 'nabirds'],
                        help='name of the dataset (cub_bird or stanford_dog or aircraft or vegfru)')

    parser.add_argument('--code-length', default=16, type=int, help='length of hash codes')

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

    parser.add_argument('--batch-size', default=64, type=int, help='batch size')

    parser.add_argument('--epoch', default=90, type=int, help='total training epochs')

    parser.add_argument('--gpu-ids', default='0', help='gpu id list(eg: 0,1,2...)')
    parser.add_argument('--w', default=4, type=int, help='the number of mask token')

    parser.add_argument('--model-name', default='vit-small', type=str, help='model name (resnet18 or alexnet)')

    args = parser.parse_args()

    set_seed(args.random_seed)

    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    start_train(args, args.code_length, args.w)
