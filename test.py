import os
import sys
import cv2
import argparse
import numpy as np
import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from model import *
from DANet import DANet
import matplotlib.pyplot as plt
from PVT_SRaNet import PolypPVT
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.samples   = [name for name in os.listdir(args.datapath+'/images') if name[0]!="."]
        self.transform = A.Compose([
            A.Resize(352, 352),
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        name   = self.samples[idx]
        image  = cv2.imread(self.args.datapath+'/images/'+name)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin = image
        H,W,C  = image.shape
        mask = cv2.imread(self.args.datapath + '/masks/' + os.path.splitext(name)[0] + '.png', cv2.IMREAD_GRAYSCALE) / 255.0
        pair   = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask'], (H, W), name, origin

    def __len__(self):
        return len(self.samples)

class Test(object):
    def __init__(self, Data, Model, args):
        ## dataset
        self.args      = args
        self.data      = Data(args)
        self.loader    = DataLoader(self.data, batch_size=1, pin_memory=True, shuffle=True, num_workers=args.num_workers)
        ## model
        self.model     = Model()
        self.model.load_state_dict(torch.load(self.args.snapshot))
        self.model.eval()
        self.model.cuda()

    def save_prediction(self):
        print(self.args.datapath.split('/')[-1])
        with torch.no_grad():
            for image, mask, shape, name, origin in self.loader:
                image = image.cuda().float()
                #pred, out5, out4, out3 = self.model(image, shape)
                pred = self.model(image)
                pred  = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True)[0, 0]
                pred[torch.where(pred>0)] /= (pred>0).float().mean()
                pred[torch.where(pred<0)] /= (pred<0).float().mean()
                pred  = pred.cpu().numpy()*255
                if not os.path.exists(self.args.predpath):
                    os.makedirs(self.args.predpath)
                cv2.imwrite(self.args.predpath+'/'+name[0], np.round(pred))
                #print(name[0])
                '''no_of_layers = 0
                conv_layers = []

                model_children = list(self.model.children())

                for child in model_children:
                    if type(child) == nn.Conv2d:
                        no_of_layers += 1
                        conv_layers.append(child)
                    elif type(child) == nn.Sequential:
                        for layer in child.children():
                            if type(layer) == nn.Conv2d:
                                no_of_layers += 1
                                conv_layers.append(layer)
                print(no_of_layers)
                outs = [out5, out4, out3]
                results = [conv_layers[0](out5)]
                for i in range(1, len(conv_layers)-1):
                    results.append(conv_layers[i](outs[i]))
                outputs = results'''
                '''outputs = [att4, att3]
                if name[0] == '163.png':
                    for num_layer in range(len(outputs)):
                        plt.figure(figsize=(50, 20))
                        layer_viz = outputs[num_layer][0, :, :, :]
                        layer_viz = layer_viz.data
                        for i, filter in enumerate(layer_viz):
                            if i == 32:
                                break
                            plt.subplot(4, 8, i + 1)
                            plt.imshow(filter.cpu(), cmap='gray')
                            plt.axis("off")
                        plt.show()
                        # plt.close()
                    a = 10'''


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapaths', default='./data/TestDataset/')
    parser.add_argument('--predpaths', default='./eval/prediction/PolypPVT/')
    #parser.add_argument('--datapath', default='./data/TestDataset')
    #parser.add_argument('--predpath', default='./eval/prediction/SRaNet/')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--snapshot', default='./PolypPVT/model-best')
    args = parser.parse_args()
    for name in [
        'CVC-300',
        'CVC-ClinicDB',
        'CVC-ColonDB',
        'ETIS-LaribPolypDB',
        'Kvasir']:
        args.datapath = args.datapaths + name
        args.predpath = args.predpaths + name
        t = Test(Data, PolypPVT, args)
        t.save_prediction()