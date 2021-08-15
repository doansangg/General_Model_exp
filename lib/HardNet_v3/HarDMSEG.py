import torch
import torch.nn as nn
import torch.nn.functional as F
from .hardnet_68 import hardnet
#from attention import AttentionConv
import time

import torch.nn.init as init

import math
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


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class CONV_640(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(CONV_640, self).__init__()
        self.branch1 = nn.Sequential(
                BasicConv2d(in_channel, 320,kernel_size= 3,stride=2,padding=1),
                BasicConv2d(320, 840, kernel_size=3,stride=2 ,padding=1),
                BasicConv2d(840, 1024, kernel_size=3,stride=2, padding=1),
                BasicConv2d(1024, out_channel, kernel_size=3,stride=2, padding=1),
            )
    def forward(self, x) :
        x=self.branch1(x)
        return x

class CONV_1024(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(CONV_1024, self).__init__()
        self.branch1 = nn.Sequential(
                BasicConv2d(in_channel, 320,kernel_size= 3,stride=2,padding=1),
                BasicConv2d(320, 620, kernel_size=3,stride=2 ,padding=1),
                BasicConv2d(620, 960, kernel_size=3,stride=2, padding=1),
                BasicConv2d(960,1296, kernel_size=3,stride=2, padding=1),
                BasicConv2d(1296, out_channel, kernel_size=3,stride=2, padding=1),
            )
    def forward(self, x) :
        x=self.branch1(x)
        return x    

class CONV_MASK(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(CONV_MASK, self).__init__()
        self.branch1 = nn.Sequential(
                BasicConv2d(in_channel, 64,kernel_size= 3,stride=1,padding=1),
                BasicConv2d(64, 32, kernel_size=3,stride=1 ,padding=1),
                BasicConv2d(32, 16, kernel_size=3,stride=1, padding=1),
                BasicConv2d(16, out_channel, kernel_size=3,stride=1, padding=1),
            )
    def forward(self, x) :
        x=self.branch1(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class Attension_RFB(nn.Module):
    def __init__(self):
        super(Attension_RFB, self).__init__()
        self.att = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            AttentionConv(32, 32, kernel_size=3,stride=1 ,padding=1),
            )
    def forward(self, x) :
        x=self.att(x)
        return x 
class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)
        #print('aggregation: ',x.shape)

        return x


class HarDMSEG_v3(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(HarDMSEG_v3, self).__init__()
        # ---- ResNet Backbone ----
        #self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.relu = nn.ReLU(True)
        # ---- Receptive Field Block like module ----
        
        self.rfb2_1 = RFB_modified(320, channel)
        self.rfb3_1 = RFB_modified(640, channel)
        self.rfb4_1 = RFB_modified(1024, channel)
        # ---- Partial Decoder ----
        #self.agg1 = aggregation(channel)
        self.agg1 = aggregation(32)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(1024, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(640, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(320, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(320, 32, kernel_size=1)
        self.conv3 = BasicConv2d(640, 32, kernel_size=1)
        self.conv4 = BasicConv2d(1024, 32, kernel_size=1)
        self.conv5 = BasicConv2d(1024, 1024, 3, padding=1)
        self.conv6 = nn.Conv2d(1024, 1, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.hardnet = hardnet(arch=68)
        self.mask = CONV_MASK(128,1,kernel_size=3,padding=1)
        # ---- extract feature for 620x22x22 ----
        self.extrac_640=CONV_640(3,640,(3,3),2,1)
        self.extrac_1024=CONV_1024(3,1024,(3,3),2,1)
        #self attention
        self.attention=Attension_RFB()
        self.conv=BasicConv2d(64, 32, kernel_size=3, padding=1)
        self.conv640=BasicConv2d(1280, 640, kernel_size=3, padding=1)
        self.conv1240=BasicConv2d(2048, 1024, kernel_size=3, padding=1)
    def forward(self, x):
        #print("input",x.size())
        
        hardnetout = self.hardnet(x)
        x_640=self.extrac_640(x)
        x_1024=self.extrac_1024(x)
        x1 = hardnetout[0]
        #print(x1.shape)
        x2 = hardnetout[1]
        #print(x2.shape)
        x3 = hardnetout[2]
        #x3=x3+x_640
        x3=torch.cat([x_640, x3], dim=1)
        x3=self.conv640(x3)
        #x3=self.relu(x3)
        #print(x3.shape)
        x4 = hardnetout[3]
        x4=torch.cat([x_1024, x4], dim=1)
        x4=self.conv1240(x4)
        #print(x4.shape)
        
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32
        x1_mask=self.mask(x1)

        # x4_x3=self.attention(x4_rfb)
        # x3_rfb=torch.cat([x4_x3, x3_rfb], dim=1)
        # x3_rfb=self.conv(x3_rfb)
 
        # x3_x2=self.attention(x3_rfb)
        # x2_rfb=torch.cat([x3_x2, x2_rfb], dim=1)
        # x2_rfb=self.conv(x2_rfb)
        #(--End Attention--)
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_4 = F.interpolate(x1_mask, scale_factor=4, mode='bilinear')
        
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # print(lateral_map_4.shape)
        # print(lateral_map_5.shape)
        lateral_map_6 = (lateral_map_4+lateral_map_5)/2
        return lateral_map_4,lateral_map_5 , lateral_map_6#, lateral_map_3, lateral_map_2
# import time
# if __name__ == '__main__':
#     #ras = BasicConv2d(3,352,(3,3),2,1).cuda()
#     ras=HarDMSEG().cuda()
#     #ras=CONV_1024(3,1024,(3,3),2,1).cuda()
#     pytorch_total_params = sum(p.numel() for p in ras.parameters())
#     #print(pytorch_total_params)
    
#     input_tensor = torch.randn(1, 3, 352, 352).cuda()
#     start=time.time()
#     out = ras(input_tensor)
#     end=time.time()
#     print('time: ',end-start)
#     #print(out.shape)
#     #print(out)

