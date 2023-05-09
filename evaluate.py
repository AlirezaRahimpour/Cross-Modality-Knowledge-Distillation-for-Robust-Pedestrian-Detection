import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import cv2
import utils
from torch.utils.data import Dataset
from engine import train_one_epoch, evaluate
from engine_adv import train_one_epoch_adv
from torchvision.transforms import functional as F
import torch.nn.functional as Fn
import transforms as T
from PIL import Image
from faster_rcnn import fasterrcnn_resnet50_fpn

data_root = "data/"

class SeeingThroughFog(Dataset):
    def __init__(self, imgs_list,transforms=None,only_RGB=False):
        f_imgs=open(data_root+imgs_list+'.txt','r')
        self.files=f_imgs.readlines()
        f_imgs.close()
        self.transforms = transforms
        self.only_RGB=only_RGB

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name=self.files[idx].replace('\n', '')
        rgb_path = data_root+'images/'+file_name+'.png'
        ann_path = data_root+'gated_labels/'+file_name+'.txt'
        rgb= torchvision.io.read_image(rgb_path)
        if not self.only_RGB:
            gated0_path= data_root+'gated0/'+file_name+'.png'
            gated1_path= data_root+'gated1/'+file_name+'.png'
            gated2_path= data_root+'gated2/'+file_name+'.png'
            gated0=torchvision.io.read_image(gated0_path,mode=torchvision.io.ImageReadMode.GRAY)
            gated1=torchvision.io.read_image(gated1_path,mode=torchvision.io.ImageReadMode.GRAY)
            gated2=torchvision.io.read_image(gated2_path,mode=torchvision.io.ImageReadMode.GRAY)
            img=torch.cat((rgb,gated0,gated1,gated2),0)
            img = F.convert_image_dtype(img)
        rgb=F.convert_image_dtype(rgb)
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
        if self.only_RGB:
            return rgb, target
        else:
            return img, rgb, target

def get_transfrom(train):
    transforms = []
    # convert the image to pytorch tensor
    transforms.append(T.ToTensor())
    if train:
        # flip the images and their lables
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def build_model(num_classes,in_channels=3,score_thr=None):
        #define model
        model=fasterrcnn_resnet50_fpn(pretrained=True,trainable_backbone_layers=5)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        add_ch=in_channels-3
        image_mean=[0.485, 0.456, 0.406]
        image_std=[0.229, 0.224, 0.225]
        if add_ch>0:
            image_mean=image_mean+[0.406]*add_ch
            image_std=image_std+[0.225]*add_ch
        model.transform=GeneralizedRCNNTransform((500,),1333,image_mean,image_std)
        model.backbone.body.conv1=nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return model

def testing(test_set,weights_file):
    test_data = SeeingThroughFog(test_set,get_transfrom(train=False),only_RGB=True)
    print("test set size:", len(test_data))
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':print("using GPU")
    num_classes=2
    model = build_model(num_classes=2,in_channels=3)
    #load model
    model.load_state_dict(torch.load('weights/'+weights_file+'.pth')) 
    model.to(device)
    model.eval()
    results=evaluate(model, val_loader, device=device)

def main():
    weights_file='KDfea_adv' # weights of model that is evaluated
    print('--------------- -----------val -------------------------------')
    testing('val_night',weights_file)
    print('--------------- ------------test -------------------------------')
    testing('test_night',weights_file)
    print('****************************************************************************')

if __name__ == '__main__':
    main()
