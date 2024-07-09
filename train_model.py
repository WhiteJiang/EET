# -*- coding: utf-8 -*-
# @Time    : 2024/3/9
# @Author  : White Jiang
import torch
import torch.nn as nn
import random
from utils import calc_train_codes, calc_map_k


def eval_turn(model, data_loader):
    model.eval()
    with torch.no_grad():
        for batch_cnt_val, (inputs, labels, _) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs, outputs_codes = model(inputs)

            if batch_cnt_val == 0:
                ground = labels
                pred_out = outputs_codes
            else:
                ground = torch.cat((ground, labels))
                pred_out = torch.cat((pred_out, outputs_codes))

    return ground, pred_out


def train_model(model, dataloader, criterion, criterion_hash, optimizer, scheduler, num_epochs, bits, classes,
                log_file, model_teacher=None, criterion_hash_kd=None, scheduler_teacher=None,
                optimizer_teacher=None, w=0, model_name='vit-small', data_name='cub'):
    train_codes = calc_train_codes(dataloader, bits, classes)
    best_map = 0
    for epoch in range(num_epochs):

        model.train()
        model_teacher.train()
        ce_loss = 0.0
        ce_loss_teacher = 0.0
        total_item_len = 0
        for batch_cnt, (inputs, labels, item) in enumerate(dataloader['train']):
            codes = torch.tensor(train_codes[item, :]).float().cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            optimizer_teacher.zero_grad()
            outputs_class, outputs_codes = model(inputs)

            outputs_class_teacher, outputs_codes_teacher, attn = model_teacher(inputs)

            # for mask image
            _, max_inx = torch.topk(attn, dim=1, k=w)
            B, _ = attn.size()
            mask_all = torch.ones(B, 1, 224, 224).cuda()
            for j in range(B):
                for i in range(w):
                    x0 = (max_inx[j, i] % 14) * 16
                    x1 = (max_inx[j, i] % 14 + 1) * 16
                    y0 = (max_inx[j, i] // 14) * 16
                    y1 = (max_inx[j, i] // 14 + 1) * 16
                    mask_all[j, :, x0:x1, y0:y1] = 0
            hide_images = inputs * mask_all
            outputs_class_hide, _ = model(hide_images)
            # ------------------------------------------------------------
            loss_class = criterion(outputs_class, labels)
            loss_class_hide = criterion(outputs_class_hide, labels)
            loss_codes = criterion_hash(outputs_codes, codes)
            loss_hash_kd = criterion_hash_kd(outputs_codes, outputs_codes_teacher)
            loss = 0.1 * (loss_class + loss_class_hide) + loss_codes + 1.0 * loss_hash_kd

            # ------------------------------------------------------------
            loss_class_teacher = criterion(outputs_class_teacher, labels)
            loss_codes_teacher = criterion_hash(outputs_codes_teacher, codes)
            loss_teacher = loss_class_teacher + loss_codes_teacher
            # ------------------------------------------------------------
            loss.backward()
            loss_teacher.backward()
            # ------------------------------------------------------------
            optimizer.step()
            optimizer_teacher.step()
            # ------------------------------------------------------------
            ce_loss += loss.item() * inputs.size(0)
            ce_loss_teacher += loss_teacher.item() * inputs.size(0)
            total_item_len += inputs.size(0)
            # ------------------------------------------------------------
            scheduler.step()
            scheduler_teacher.step()

        epoch_loss = ce_loss / total_item_len
        epoch_loss_teacher = ce_loss_teacher / total_item_len
        # scheduler.step()

        if (epoch + 1) % 1 == 0:
            labels_onehot_q, code_q = eval_turn(model, dataloader['val'])
            labels_onehot_q_teacher, code_q_teacher = eval_turn(model_teacher, dataloader['val'])
            labels_onehot_d, code_d = eval_turn(model, dataloader['base'])

            map_1 = calc_map_k(torch.sign(code_q), torch.tensor(train_codes).float().cuda(), labels_onehot_q,
                               labels_onehot_d)

            map_1_teacher = calc_map_k(torch.sign(code_q_teacher), torch.tensor(train_codes).float().cuda(),
                                       labels_onehot_q,
                                       labels_onehot_d)

            print('epoch:{}:  loss:{:.4f},  MAP:{:.4f}, loss_teacher:{:.4f},  MAP_teacher:{:.4f}'
                  .format(epoch + 1, epoch_loss, map_1, epoch_loss_teacher, map_1_teacher))
            log_file.write('epoch:{}:  loss:{:.4f},  MAP:{:.4f}, loss_teacher:{:.4f},  MAP_teacher:{:.4f}'
                           .format(epoch + 1, epoch_loss, map_1, epoch_loss_teacher, map_1_teacher) + '\n')