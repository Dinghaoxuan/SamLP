import logging
import os
import sys
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import Focal_loss, DiceLoss, DiceLoss_softmax
from torchvision import transforms
from icecream import ic
import cv2

# from datasets.dataset_ufpr_cls2 import UFPR_ALPR_Dataset, RandomGenerator
from datasets.dataset_ufpr_sam import UFPR_ALPR_Dataset, SamTransform, collater

def calc_loss(outputs_logits, low_res_label_batch, dice_loss, dice_weight:float=0.8):
    loss_dice = dice_loss(outputs_logits, low_res_label_batch, softmax=True)
    return loss_dice


def trainer_UFPR(args, predictor, log_path, multimask_output):
    logging.basicConfig(filename=os.path.join(log_path,'log.txt'), level=logging.INFO, 
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    trainset = UFPR_ALPR_Dataset(root=args.root_path, split='training', transform=SamTransform(1024))
    print("The length of train set is: {}".format(len(trainset)))

    curr_epoch = 0
    if args.lora_ckpt != None:
        curr_epoch = os.path.split(args.lora_ckpt)[1]
        curr_epoch = int(curr_epoch.replace('epoch_','').replace('.pth',''))
    

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collater, pin_memory=True, worker_init_fn=worker_init_fn)

    predictor.model.train()

    dice_loss = DiceLoss_softmax()

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, predictor.model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, predictor.model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)
    
    writer = SummaryWriter(os.path.join(log_path, 'log'))
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))


    iterator = tqdm(range(curr_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            outputs = predictor.forward(image_batch, multimask_output)

            loss_level0 = calc_loss(outputs[:,0,:,:], low_res_label_batch, dice_loss, args.dice_param)
            loss_level1 = calc_loss(outputs[:,1,:,:], low_res_label_batch, dice_loss, args.dice_param)
            loss_level2 = calc_loss(outputs[:,2,:,:], low_res_label_batch, dice_loss, args.dice_param)

            loss = 1/3 * (loss_level0 + loss_level1 + loss_level2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)


            logging.info('iteration %d : loss : %f, loss_mse : %f, loss_dice : %f, max_label : %f' % (iter_num, loss.item(), loss_mse.item(), loss_dice.item(), torch.max(outputs).item()))

            if iter_num % 20 == 0:
                image = image_batch[1, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/image', image, iter_num)
                writer.add_image('train/pred_level0', (outputs[1, 0, ...]>0).unsqueeze(0) * 50, iter_num)
                writer.add_image('train/pred_level1', (outputs[1, 1, ...]>0).unsqueeze(0) * 50, iter_num)
                writer.add_image('train/pred_level2', (outputs[1, 2, ...]>0).unsqueeze(0) * 50, iter_num)
                labs = low_res_label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/gt', labs, iter_num)

        save_interval = 1
        if (epoch_num + 1) % save_interval == 0:
            save_model_path = os.path.join(log_path, 'epoch_'+str(epoch_num+1)+'.pth')
            try:
                predictor.model.save_lora_parameters(save_model_path)
            except:
                predictor.model.module.save_lora_parameters(save_model_path)
            logging.info('save model to {}'.format(save_model_path))

        if epoch_num >= max_epoch - 1:
            save_model_path = os.path.join(log_path, 'epoch_'+str(epoch_num+1)+'.pth')
            try:
                predictor.model.save_lora_parameters(save_model_path)
            except:
                predictor.model.module.save_lora_parameters(save_model_path)
            logging.info('save model to {}'.format(save_model_path))
            iterator.close()
            break

    writer.close()
    return 'Training Finished!'

