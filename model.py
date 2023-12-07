#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import Res2Net50, weight_init
import matplotlib.pyplot as plt


class ChannelSoftmax(nn.Module):
    def __init__(self):
        super(ChannelSoftmax, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_ra = nn.Sequential(nn.Linear(64, 8, bias=False),
                                nn.ReLU(),
                                nn.Linear(8, 64, bias=False))
        self.fc_sa = nn.Sequential(nn.Linear(64, 8, bias=False),
                                nn.ReLU(),
                                nn.Linear(8, 64, bias=False))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, sa, ra):
        b, c, _, _ = sa.size()
        sa_y = self.fc_sa(self.gap(sa).view(b, c)).view(1, b, c)
        ra_y = self.fc_ra(self.gap(ra).view(b, c)).view(1, b, c)
        y = torch.cat([sa_y, ra_y], dim=0)
        y = self.softmax(y)

        sa_y = y[0].view(b, c, 1, 1)
        ra_y = y[1].view(b, c, 1, 1)

        return sa * sa_y.expand_as(sa), ra * ra_y.expand_as(ra)

    def initialize(self):
        for n, m in self.named_children():
            print('initialize: ' + n)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                weight_init(m)
            elif isinstance(m, (nn.ReLU, nn.PReLU)):
                pass


class SRaNet(nn.Module):
    def __init__(self, args=None):
        super(SRaNet, self).__init__()
        self.args    = args
        self.bkbone  = Res2Net50()
        self.linear5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64 ), nn.ReLU(inplace=True))
        self.channel = ChannelSoftmax()
        self.predict = nn.Conv2d(64*3, 1, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def forward(self, x, shape=None):
        out2, out3, out4, out5 = self.bkbone(x)
        o5, o4, o3 = out5, out4, out3
        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)

        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)

        sa4 = out4 * out5
        ra = -1 * (torch.sigmoid(out5)) + 1
        ra4 = ra * out4
        sa4, ra4 = self.channel(sa4, ra4)
        att4 = sa4 + ra4

        sa3 = out5 * out4 * out3
        ra = -1 * (torch.sigmoid(out4)) + 1
        ra3 = ra * out3
        sa3, ra3 = self.channel(sa3, ra3)
        att3 = sa3 + ra3

        pred = torch.cat([out5, att4, att3], dim=1)
        pred = self.predict(pred)
        return pred

    def initialize(self):
        #weight_init(self)
        if self.args.snapshot:
            self.load_state_dict(torch.load(self.args.snapshot))
        else:
            weight_init(self)


'''from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    net = DANet()
    macs, params = get_model_complexity_info(net, (3, 352, 352), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))'''