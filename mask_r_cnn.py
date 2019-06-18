import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import os
import transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
from engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from  tensorboardX import SummaryWriter
import argparse
import os

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tens_board_dir', default='/data/materialist/tensr_board')
    parser.add_argument('--datadir', default='/data/materialist/')
    parser.add_argument('--run', default='')
    parser.add_argument('--epochs', default=10)
    return parser.parse_args()

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
    except:
        return "small_box" 
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax, rmax # rmin, rmax, cmin, cmax

def mask_labels_bbox(data,idx):
    height, width, labels, list_pixels = data[["Height","Width",'ClassId','EncodedPixels']].values[0]
    N_class = len(labels)
    hw_min = min(height, width)
    width_resize = 512 if width == hw_min else round((width / hw_min) * 512)
    height_resize = 512 if height == hw_min else round((height / hw_min) * 512) 
    mask = np.zeros((N_class,height*width),dtype=np.uint8)
    for mask_number, pixels in enumerate(list_pixels):  
        for i in range(0,len(pixels),2):
            pixel_start = pixels[i] - 1
            pixel_end =  pixel_start + pixels[i+1]
            mask[mask_number,pixel_start: pixel_end] = 1
    mask = mask.reshape(N_class,height,width,order='F')
    mask_resize = np.zeros((N_class,height_resize,width_resize),dtype=np.uint8)
    bound_box = np.zeros((N_class, 4), dtype=np.int32)
    drop = []
    for mask_number in range(N_class):
        mask_resize[mask_number] = cv2.resize(mask[mask_number], 
                                              (width_resize,height_resize),interpolation=cv2.INTER_NEAREST)
        mask_box = bbox2(mask_resize[mask_number])
        if 'small_box' == mask_box:
            drop.append(mask_number)
        else:
            bound_box[mask_number] = mask_box
    mask_resize = np.delete(mask_resize, drop, 0)
    bound_box = np.delete(bound_box, drop, 0)
    labels = np.delete(labels, drop, 0)
    N_class -= len(drop)

    image_id = torch.tensor([idx])   
        
    bound_box = torch.tensor(bound_box,dtype=torch.float32)
    area = (bound_box[:, 3] - bound_box[:, 1]) * (bound_box[:, 2] - bound_box[:, 0])
    iscrowd = torch.zeros((N_class,), dtype=torch.int64)
    tensor_dict = {'boxes':bound_box,
                   "labels":torch.tensor(labels,dtype=torch.int64),
                   "masks":torch.tensor(mask_resize,dtype=torch.uint8),
                   'resizes':(height_resize,width_resize),
                   'area':area,
                   'iscrowd':iscrowd,
                   'image_id':image_id
                  }
    return tensor_dict

class Dataset(object):
    def __init__(self, train_csv,path='./train/'):
        self.train_csv = train_csv
        self.path = path

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        image_path = self.train_csv.loc[idx,"ImageId"]
        tensor_dict = mask_labels_bbox(self.train_csv.loc[[idx]],idx)
        image = cv2.imread(self.path + image_path)
        image = cv2.resize(image,tensor_dict['resizes'][::-1])
        image = torch.tensor(image.transpose((2,0,1)),dtype=torch.float32) / 255
        return image, tensor_dict
    
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def train(train_csv,tens_board_dir,num_epochs=10,run=''):    
    train_writer = SummaryWriter(tens_board_dir)    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 35
    dataset = Dataset(train_csv)
    dataset_test = Dataset(train_csv)
    indices = torch.randperm(len(dataset)).tolist()

    dataset = torch.utils.data.Subset(dataset, indices[:-500])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-500:])

    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)

    model.load_state_dict(torch.load('model.pkl'))
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, train_writer,print_freq=10,run=run)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
       


    
    
    
    
if __name__ == "__main__":
    args = _parse_args()
    print('Готовим датафрейм')
    train_csv  = pd.read_csv(args.datadir + "train_csv.csv")
    train_csv['ClassId'] = train_csv['ClassId'].str.split('_').str[0].astype(int)
    train_csv = train_csv[~train_csv['ClassId'].isin(np.arange(12))]
    train_csv['ClassId'] = train_csv['ClassId'] - 12 
    train_csv['EncodedPixels'] = train_csv['EncodedPixels'].apply(lambda x: [int(i) for i in x.split(' ')])
    train_csv = train_csv.groupby(['ImageId',"Height","Width"],as_index=False).agg(list)
    print('Начинаем тренировку')
    train(train_csv,args.tens_board_dir,int(args.epochs),args.run)
