import argparse
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lora_predictor import LoRA_SamPredictor

from importlib import import_module

from segment_anything import sam_model_registry
from trainer import trainer_UFPR
from icecream import ic
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/media/disk1/yxding/dhx/Dataset/UFPR-ALPR/', help='root dir for data')
parser.add_argument('--output', type=str, default='./exp/')
parser.add_argument('--dataset', type=str, default='UFPR')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.005)
parser.add_argument('--img_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vit_name', type=str, default='vit_b')
parser.add_argument('--ckpt', type=str, default='./checkpoints/sam_vit_b_01ec64.pth')
parser.add_argument('--lora_ckpt', type=str, default=None)
parser.add_argument('--rank', type=int, default=2)
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--warmup_period', type=int, default=250)
parser.add_argument('--AdamW', action='store_true')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder_mask_decoder')
parser.add_argument('--dice_param', type=float, default=0.8)
args = parser.parse_args()

if __name__=='__main__':
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'UFPR': {
            'root_path': args.root_path,
            'num_classes': args.num_classes,
        }
    }

    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    log_path = os.path.join(args.output, "{}".format(args.exp))
    time_str = time.strftime('_%Y-%m-%d-%H:%M:%S', time.localtime())
    log_path = log_path +  time_str
    log_path = log_path + '_' + args.vit_name
    log_path = log_path + '_' + str(args.module)
    log_path = log_path + '_cls' + str(args.num_classes)
    log_path = log_path + '_epo' + str(args.max_epochs)
    log_path = log_path + '_bs' + str(args.batch_size) 
    log_path = log_path + '_lr' + str(args.base_lr)
    log_path = log_path + '_seed' + str(args.seed)
    log_path = log_path + '_rank' + str(args.rank)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    sam = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    predictor = LoRA_SamPredictor(net)

    if args.lora_ckpt is not None:
        predictor.model.load_lora_parameters(args.lora_ckpt)

    multimask_output = True

    trainer = {'UFPR': trainer_UFPR}
    trainer[dataset_name](args, predictor, log_path, multimask_output)


    







