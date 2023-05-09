import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import cv2
import utils
from torch.utils.data import Dataset
from engine import train_one_epoch, evaluate
from torchvision.transforms import functional as F
import transforms as T
from PIL import Image
from faster_rcnn import fasterrcnn_resnet50_fpn

data_root = "data/"
exp="teacher_Net"

class SeeingThroughFog(Dataset):
    def __init__(self, imgs_list,transforms=None):
        f_imgs=open(data_root+imgs_list+'.txt','r')
        self.files=f_imgs.readlines()
        f_imgs.close()
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name=self.files[idx].replace('\n', '')
        img_path = data_root+'images/'+file_name+'.png'
        gated0_path= data_root+'gated0/'+file_name+'.png'
        gated1_path= data_root+'gated1/'+file_name+'.png'
        gated2_path= data_root+'gated2/'+file_name+'.png'
        ann_path = data_root+'gated_labels/'+file_name+'.txt'
        # img = Image.open(img_path).convert("RGB")
        # img = cv2.imread(img_path)
        img= torchvision.io.read_image(img_path)
        gated0=torchvision.io.read_image(gated0_path,mode=torchvision.io.ImageReadMode.GRAY)
        gated1=torchvision.io.read_image(gated1_path,mode=torchvision.io.ImageReadMode.GRAY)
        gated2=torchvision.io.read_image(gated2_path,mode=torchvision.io.ImageReadMode.GRAY)
        #img=torch.cat((img,gated1),0)
        img=torch.cat((img,gated0,gated1,gated2),0)
        img = F.convert_image_dtype(img)
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
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)
        return img, target

def get_transfrom(train):
    transforms = []
    # convert the image to pytorch tensor
    transforms.append(T.ToTensor())
    if train:
        # flip the images and their lables
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def build_model(num_classes,in_channels=3):
        #define model
        model=fasterrcnn_resnet50_fpn(pretrained=True,trainable_backbone_layers=5)
        model.backbone.body.conv1=nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        add_ch=in_channels-3
        image_mean=[0.485, 0.456, 0.406]
        image_std=[0.229, 0.224, 0.225]
        if add_ch>0:
            image_mean=image_mean+[0.406]*add_ch
            image_std=image_std+[0.225]*add_ch
        model.transform=GeneralizedRCNNTransform((500,),1333,image_mean,image_std)
        return model

def training():
    train_data = SeeingThroughFog('train_night',get_transfrom(train=False))
    print("Experiment:",exp)
    print("Training set size:", len(train_data))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)  # num_worker = 4 * num_GPU

    val_data = SeeingThroughFog('val_night',get_transfrom(train=False))
    print("Val set size:", len(val_data))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':print("using GPU")

    model = build_model(num_classes=2,in_channels=6)
    model.to(device)

    # pick the trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # constuct an optimizer
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler that decreases the lr 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_mAP=0
    best_epoch=0
    num_epochs = 100

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=200) # printing every 10 iter.
        print('------------------------------------------------------------')
        lr_scheduler.step()
        results=evaluate(model, val_loader, device=device)
        curr_mAP=results.coco_eval['bbox'].stats[0]
        if best_mAP < curr_mAP:
            best_mAP= curr_mAP
            best_epoch=epoch
            torch.save(model.state_dict(), 'weights/'+exp+'.pth')
            print('best mAP is updated:',best_mAP)
        print('epoch:',epoch, 'curr_mAP:',curr_mAP,'best_epoch',best_epoch, 'best_mAP:',best_mAP)
        print('****************************************************************')
        if epoch-best_epoch>5:
            break

def testing():
    test_data = SeeingThroughFog('test_night',get_transfrom(train=False))
    print("test set size:", len(test_data))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':print("using GPU")

    #define model
    model = build_model(num_classes=2,in_channels=6)
    model.load_state_dict(torch.load('weights/'+exp+'.pth'))
    model.to(device)
    model.eval()
    results=evaluate(model, test_loader, device=device)

def main():
    training()
    testing()
if __name__ == '__main__':
    main()
