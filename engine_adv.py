import math
import sys
import time
import torch
import utils

import torchvision.models.detection.mask_rcnn
from torch.utils.tensorboard import SummaryWriter

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import torch.nn.functional as F

def apply_grl(grl,student_fea,teacher_fea):
    grl_student_fea=grl(student_fea)
    student_teacher_fea=torch.cat((grl_student_fea,teacher_fea),0)
    return student_teacher_fea

def dc_loss(outputs):
    targets=torch.zeros_like(outputs)
    dc_batch=targets.size()[0]
    targets[int(dc_batch/2):dc_batch+1,:]=1
    return F.binary_cross_entropy_with_logits(outputs,targets)

def train_one_epoch_adv(teacher_model,model,da_Fnet0,da_Fnet1,da_Fnet2,da_Fnet3,grl,optimizer, data_loader, device, epoch, print_freq):
    teacher_model.eval()
    model.train()
    da_Fnet0.train()
    da_Fnet1.train()
    da_Fnet2.train()
    da_Fnet3.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)  # this is different
    #     lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, edited
    #     total_iters=warmup_iters)
    for rgb_gateds ,images, targets in metric_logger.log_every(data_loader, print_freq, header):
        rgb_gateds = list(rgb_gated.to(device) for rgb_gated in rgb_gateds)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            _,features_teacher=teacher_model(rgb_gateds)
        images = list(image.to(device) for image in images)

        loss_dict,features_student = model(images, targets)

        dc_outputs0=da_Fnet0(apply_grl(grl,features_student['0'],features_teacher['0']))
        dc_outputs1=da_Fnet1(apply_grl(grl,features_student['1'],features_teacher['1']))
        dc_outputs2=da_Fnet2(apply_grl(grl,features_student['2'],features_teacher['2']))
        dc_outputs3=da_Fnet3(apply_grl(grl,features_student['3'],features_teacher['3']))

        adv_loss0=dc_loss(dc_outputs0)
        adv_loss1=dc_loss(dc_outputs1)
        adv_loss2=dc_loss(dc_outputs2)
        adv_loss3=dc_loss(dc_outputs3)

        loss_dict['adv_loss']=(adv_loss0+adv_loss1+adv_loss2+adv_loss3)/4

        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger
