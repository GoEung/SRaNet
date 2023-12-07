import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import weight_init


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


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


class PolypPVT(nn.Module):
    def __init__(self, channel=32):
        super(PolypPVT, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './res/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.linear5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.channel3 = ChannelSoftmax()
        self.channel4 = ChannelSoftmax()
        self.predict = nn.Conv2d(64*3, 1, kernel_size=1, stride=1, padding=0)

        weight_init(self)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        out2 = pvt[0]
        out3 = pvt[1]
        out4 = pvt[2]
        out5 = pvt[3]

        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)

        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)

        sa4 = out4 * out5
        ra = -1 * (torch.sigmoid(out5)) + 1
        ra4 = ra * out4
        sa4, ra4 = self.channel4(sa4, ra4)
        att4 = sa4 + ra4

        sa3 = out5 * out4 * out3
        ra = -1 * (torch.sigmoid(out4)) + 1
        ra3 = ra * out3
        sa3, ra3 = self.channel3(sa3, ra3)
        att3 = sa3 + ra3

        pred = torch.cat([out5, att4, att3], dim=1)
        pred = self.predict(pred)

        return pred


if __name__ == '__main__':
    model = PolypPVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    pred = model(input_tensor)
    print(pred.size())