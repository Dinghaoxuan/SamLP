import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import logging
import numpy as np
import argparse
import random
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from importlib import import_module
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from datasets.dataset_ufpr_sam import UFPR_ALPR_Dataset, UFPR_ALPR_Dataset, SamTransform, SamTransformTest, collater
from lora_predictor import LoRA_SamPredictor
import cv2
from icecream import ic
from tqdm import tqdm
from scipy.ndimage.interpolation import zoom
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

def ap(tp, conf, count):
    tp = np.array(tp)
    conf = np.array(conf)
    i = np.argsort(-conf)
    tp, conf = tp[i], conf[i]
    n_gt = count
    fpc = (1-tp[i]).cumsum()
    tpc = (tp[i]).cumsum()
    recall_curve = tpc / (n_gt + 1e-16)
    precision_curve = tpc / (tpc + fpc)

    ap = compute_ap(precision_curve, recall_curve)
    return ap

def compute_ap(precision, recall):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def iou(a,b):

    left1,top1,right1,down1 = a[0], a[1], a[2], a[3]
    left2,top2,right2,down2 = b[0], b[1], b[2], b[3]
    
    area1 = (right1-left1)*(top1-down1)
    area2 = (right2-left2)*(top2-down2)
    area_sum = area1+area2
    
    left = max(left1,left2)
    right = min(right1,right2)
    top = max(top1,top2)
    bottom = min(down1,down2)

    if left>=right or top>=bottom:
        return 0
    else:
        inter = (right-left)*(top-bottom)
        return inter/(area_sum-inter)


def mask2bbox(mask, is_gt):
    # pred: w, h  |  label: w, h
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy().astype(np.uint8)
    elif isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=3)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes_list = []
    max_w, max_h = 0, 0

    for cont in contours:
        x1, y1, w, h = cv2.boundingRect(cont)
        x2, y2 = x1+w, y1+h
        bboxes_list.append([x1, y1, x2, y2])

    return bboxes_list

TP = 0
FP = 0
FN = 0
tp_list = []
conf_list = []
gt_count = 0
pred_count = 0

def evaluation(pred, label, mask_iou):
    global TP
    global FP
    global FN
    global tp_list
    global conf_list
    global gt_count
    global pred_count

    pred_bboxes = mask2bbox(pred, False)
    label_bboxes = mask2bbox(label, True)

    gt_count += len(label_bboxes)
    pred_count += len(pred_bboxes)

    if len(pred_bboxes) == 0:
        FN += 1
    else:
        for gt in label_bboxes:
            is_true = False
            for pred in pred_bboxes:
                # print(iou(pred, gt))
                if iou(pred, gt) >= 0.5:
                    is_true = True
            if is_true:
                TP += 1
                tp_list.append(1.0)
                conf_list.append(mask_iou.item())
            else:
                FP += 1
                tp_list.append(0.0)
                conf_list.append(mask_iou.item())
    return pred_bboxes, label_bboxes
            

def inference(args, multimask_output, predictor, test_save_path):
    # testset = UFPR_ALPR_Dataset(root=args.root_path, split='testing', transform=transforms.Compose([Resizer([args.img_size, args.img_size])]))
    testset = UFPR_ALPR_Dataset(root=args.root_path, split='testing', transform=SamTransformTest(1024))

    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collater, pin_memory=True)
    # trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collater, pin_memory=True, worker_init_fn=worker_init_fn)

    logging.info(f'{len(testloader)} test iterations per epoch')
    predictor.model.eval()

    for i_batch, sample_batch in tqdm(enumerate(testloader)):
        # print(sample_batch.keys())

        with torch.no_grad():
            image, label = sample_batch['image'].cuda(), sample_batch['label'].cuda()

            show_image = image.squeeze(0) * predictor.pixel_std.cuda() + predictor.pixel_mean.cuda()
            # h, w = image.shape[2], image.shape[3]
            label = label.unsqueeze(0).unsqueeze(1)

            label = predictor.model.sam.postprocess_masks(label, predictor.input_size, predictor.original_size).squeeze().detach().cpu().numpy()

            masks, iou_predictions, low_res_masks = predictor.forward_test(image, multimask_output)
            bset_idx = torch.argmax(iou_predictions)
            masks = masks.squeeze()
            iou_predictions = iou_predictions.squeeze()
            best_idx = torch.argmax(iou_predictions)
            # masks = masks[best_idx]
            mask_iou = iou_predictions[best_idx]
            # print(iou_predictions.shape)
            # raise
            mask = masks[bset_idx].squeeze().detach().cpu().numpy()

            min_area = 2500
            mask, _ = remove_small_regions(mask, min_area, 'islands')
            mask, _ = remove_small_regions(mask, min_area, 'holes')

            pred_bboxes, label_bboxes = evaluation(mask, label, mask_iou)

            image_np = predictor.model.sam.postprocess_masks(show_image.clone().unsqueeze(0), predictor.input_size, predictor.original_size).squeeze().permute(1,2,0).detach().cpu().numpy()
            # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            show_mask = np.expand_dims(mask.copy(), axis=2).astype(np.uint8)
            show_mask = cv2.cvtColor(show_mask, cv2.COLOR_GRAY2BGR)
            results = cv2.addWeighted(image_np, 1.0, show_mask*255, 0.5, 0, 0, cv2.CV_32F)

            if label_bboxes is not None:
                for gt in label_bboxes:
                    cv2.rectangle(results, (int(gt[0]),int(gt[1])), (int(gt[2]),int(gt[3])), color=(255,0,0), thickness=2)
            if pred_bboxes is not None:
                for pred in pred_bboxes:
                    cv2.rectangle(results, (int(pred[0]),int(pred[1])), (int(pred[2]),int(pred[3])), color=(0,255,0), thickness=2)

            cv2.imwrite(os.path.join(test_save_path, '{}.png'.format(i_batch)), results)

    P = TP / (pred_count + 1e-16)
    R = TP / (gt_count + 1e-16)
    F1 = 2 * P * R / (P + R + 1e-16)
    AP50 = ap(tp_list, conf_list, gt_count)

    print('P: {:.4f}\t'.format(P),
      'R: {:.4f}\t'.format(R),
      'F1: {:.4f}\t'.format(F1),
      'AP50: {:.4f}\t'.format(AP50)) 
    # return  P, R, F1, AP50  


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/media/disk1/yxding/dhx/Dataset/UFPR-ALPR/')
    parser.add_argument('--dataset', type=str, default='UFPR')
    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/sam_vit_b_01ec64.pth')
    parser.add_argument('--lora_ckpt', type=str, 
                        default="/media/disk1/yxding/dhx/Project/LP_SAM/LoRA_LP/exp/refine/UFPR_1024_2023-08-14-12:47:59_vit_b_sam_lora_image_encoder_mask_decoder_cls1_epo160_bs1_lr0.0005_seed0/epoch_90.pth")
    parser.add_argument('--vit_name', type=str, default='vit_b')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder_mask_decoder')

    args = parser.parse_args()

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

    load_ckpt_path = args.lora_ckpt
    output_dir = os.path.join(os.path.split(load_ckpt_path)[0], 'predictions_predictor')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sam = sam_model_registry[args.vit_name](checkpoint=args.ckpt)

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    predictor = LoRA_SamPredictor(net)

    assert args.lora_ckpt is not None
    predictor.model.load_lora_parameters(args.lora_ckpt)

    multimask_output = True
    print(os.path.split(load_ckpt_path)[1])
    inference(args, multimask_output, predictor, output_dir)


    