import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import cv2
import numpy as np
import utils
from torch.utils.data import Dataset
from engine import train_one_epoch, evaluate
from torchvision.transforms import functional as F
import torch.nn.functional as Fn
import transforms as T
from PIL import Image
from faster_rcnn import fasterrcnn_resnet50_fpn

data_root = "data/"

class SeeingThroughFog(Dataset):
    def __init__(self, imgs_list,only_RGB=False):
        f_imgs=open(data_root+imgs_list+'.txt','r')
        self.files=f_imgs.readlines()
        f_imgs.close()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name=self.files[idx].replace('\n', '')
        rgb_path = data_root+'images/'+file_name+'.png'
        ann_path = data_root+'gated_labels/'+file_name+'.txt'
        rgb= torchvision.io.read_image(rgb_path)
        gated1_path= data_root+'gated1/'+file_name+'.png'
        gated1=torchvision.io.read_image(gated1_path)
        boxes = []
        labels = []
        areas = []
        num_objs = 0
        f = open(ann_path,'r')
        for line in f:
            line = line.replace('\n', '')
            ann = line.split(' ')
            x0=float(ann[4])
            y0=float(ann[5])*768/720
            x1=float(ann[6])
            y1= float(ann[7])*768/720
            if ann[0] in ['Pedestrian','Pedestrian_is_group'] and x0!=x1:
                labels.append(1)
                boxes.append([x0, y0, x1, y1])
                areas.append((x1-x0)*(y1-y0))
                num_objs += 1
        if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = [0]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = torch.tensor(areas)
        target["image_id"] = torch.tensor([idx])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        return rgb,gated1,target


def build_model(num_classes,in_channels=3,score_thr=None):
        #define model
        model= fasterrcnn_resnet50_fpn(pretrained=True,trainable_backbone_layers=5,score_thr=score_thr)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        add_ch=in_channels-3
        image_mean=[0.485, 0.456, 0.406]
        image_std=[0.229, 0.224, 0.225]
        if add_ch>0:
            image_mean=image_mean+[0.406]*add_ch
            image_std=image_std+[0.225]*add_ch
        model.transform=GeneralizedRCNNTransform((500,),1333,image_mean,image_std)
        model.backbone.body.conv1=nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model

def vis_img(img):
    img=img.permute(1, 2, 0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def vis_det(test_set):

    thr=0.8 # only predicted bounding box that their confidence scores equal or greater than this threshold are shown
    test_data = SeeingThroughFog(test_set)
    print("thr", thr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':print("using GPU")

    baseline_model = build_model(num_classes=2,in_channels=3,score_thr=thr)
    baseline_model.load_state_dict(torch.load('weights/baseline.pth'))
    baseline_model.to(device)
    baseline_model.eval()

    kd_model = build_model(num_classes=2,in_channels=3,score_thr=thr)
    kd_model.load_state_dict(torch.load('weights/KDfea_mse.pth'))
    #kd_model.load_state_dict(torch.load('weights/KDfea_adv.pth'))
    kd_model.to(device)
    kd_model.eval()

    font=cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1.5
    fontColor = (255, 255, 255)

    for i, data in enumerate(test_data):
        img,gated1,target=data
        img_d=F.convert_image_dtype(img)
        with torch.no_grad():
            prediction,_ = baseline_model([img_d.to(device)])
            prediction2,_ = kd_model([img_d.to(device)])
        img=vis_img(img)
        gated1=vis_img(gated1)
        img2=img.copy()
        boxes= prediction[0]['boxes'].cpu().numpy()
        boxes2= prediction2[0]['boxes'].cpu().numpy()
        for box in boxes:
            img=cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 2)
        for box in boxes2:
            img2=cv2.rectangle(img2, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 2)
        gt_boxes=target['boxes'].numpy()
        for gt_box in gt_boxes:
            gated1=cv2.rectangle(gated1, (int(gt_box[0]),int(gt_box[1])), (int(gt_box[2]),int(gt_box[3])), (0,255,0), 2)

        img = cv2.putText(img, 'baseline(RGB)',(500,750),font,fontScale,fontColor,2)
        img2 = cv2.putText(img2, 'knowledge distillation (RGB)',(450,750),font,fontScale,fontColor,2)
        gated1 = cv2.putText(gated1, 'ground-truth (Gated)',(460,750),font,fontScale,fontColor,2)

        print('image:',i+1,'pree any key to see next image',end='\r')
        cv2.namedWindow('display', cv2.WINDOW_NORMAL)
        cv2.imshow('display',np.hstack((img,img2,gated1)))
        cv2.waitKey(0)
def main():
    vis_det('vis_night')

if __name__ == '__main__':
    main()
