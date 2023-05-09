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

def train_one_epoch_mse(teacher_model,model,alpha,optimizer,data_loader, device, epoch, print_freq):
    teacher_model.eval()
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    used_features=['0', '1', '2', '3']
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)  # this is different
        #lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, edited
        #total_iters=warmup_iters)
    for rgb_gateds ,images, targets in metric_logger.log_every(data_loader, print_freq, header):
        rgb_gateds = list(rgb_gated.to(device) for rgb_gated in rgb_gateds)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            _,features_teacher=teacher_model(rgb_gateds)
        images = list(image.to(device) for image in images)
        loss_dict,features_student = model(images, targets)
        fea_mse_loss=0
        for fea in used_features:
            fea_mse_loss=fea_mse_loss+F.mse_loss(features_student[fea],features_teacher[fea])
        loss_dict['kd_mse_features']=alpha*fea_mse_loss/len(used_features)
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
