import torch
import math
from torch import nn as nn
import torchstat
from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x



@ARCH_REGISTRY.register()
class SplitSR(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=16,
                 num_group = 2,
                 num_block=4,
                 squeeze_factor = 8,
                 upscale=4,
                 res_scale=0.1,):
        super(SplitSR, self).__init__()

        self.upscale = upscale

        self.conv_high = nn.Conv2d(num_in_ch, num_feat, 1, 1, 0)
        self.conv_mid = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_low = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.AvgPool2d(4,4))

        self.high_extractor = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.mid_extractor = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.low_extractor = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)

        self.highpath_split_high = torch.nn.Conv2d(num_feat,num_feat,3,1,1)
        self.highpath_split_low = torch.nn.Conv2d(num_feat,num_feat,3,1,1)

        self.midpath_split_high = torch.nn.Conv2d(num_feat,num_feat,3,1,1)
        self.midpath_split_low = torch.nn.Conv2d(num_feat,num_feat,3,1,1)

        self.lowpath_split_high = torch.nn.Conv2d(num_feat,num_feat,3,1,1)
        self.lowpath_split_low = torch.nn.Conv2d(num_feat,num_feat,3,1,1)

        self.high_to_sr = nn.Sequential(
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            Upsample(scale= self.upscale,num_feat = num_feat)
        )
        self.mid_to_sr = nn.Sequential(
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            Upsample(scale= self.upscale,num_feat = num_feat)
        )
        self.low_to_sr = nn.Sequential(
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            Upsample(scale= self.upscale,num_feat = num_feat)
        )

        self.high_to_lr = nn.Sequential(
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_out_ch,3,1,1),
        )
        self.mid_to_lr = nn.Sequential(
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_out_ch,3,1,1),
        )
        self.low_to_lr = nn.Sequential(
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_out_ch,3,1,1),
        )

        self.high_to_map = nn.Sequential(
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
        )
        self.mid_to_map = nn.Sequential(
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
        )
        self.low_to_map = nn.Sequential(
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_feat,3,1,1),
        )

        self.sr_conv = nn.Sequential(
            torch.nn.Conv2d(num_feat*3,num_feat,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat,num_feat*3,3,1,1),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat*3,num_out_ch,3,1,1),
        )



    def forward(self, x, repeat_num = 3):
        for i in range(repeat_num):
            if i == 0:
                shallow_f_high = self.conv_high(x)
                shallow_f_mid = self.conv_mid(x)
                shallow_f_low = self.conv_low(x)
                w,h = x.shape[2:4]
                shallow_f_low = torch.nn.functional.interpolate(shallow_f_low,(w,h))
            else:
                shallow_f_high = self.conv_high(x) * final_low_map
                shallow_f_mid = self.conv_mid(x) * final_mid_map
                shallow_f_low = self.conv_low(x)
                w,h = x.shape[2:4]
                shallow_f_low = torch.nn.functional.interpolate(shallow_f_low,(w,h)) * final_high_map

            deep_f_high = self.high_extractor(shallow_f_high)
            deep_f_mid = self.mid_extractor(shallow_f_mid)
            deep_f_low = self.low_extractor(shallow_f_low)

            highpath_feature_sr = self.highpath_split_high(deep_f_high)
            highpath_feature_lr = self.highpath_split_low(deep_f_high)
            highpath_feature_map = torch.relu(torch.subtract(highpath_feature_sr,highpath_feature_lr))

            final_high_sr = self.high_to_sr(highpath_feature_sr)
            final_high_lr = self.high_to_lr(highpath_feature_lr)
            final_high_map = self.high_to_map(highpath_feature_map)

            midpath_feature_sr = self.midpath_split_high(deep_f_mid)
            midpath_feature_lr = self.midpath_split_low(deep_f_high)
            midpath_feature_map = torch.relu(torch.subtract(midpath_feature_sr,midpath_feature_lr))

            final_mid_sr = self.mid_to_sr(midpath_feature_sr)
            final_mid_lr = self.mid_to_lr(midpath_feature_lr)
            final_mid_map = self.mid_to_map(midpath_feature_map)

            lowpath_feature_sr = self.lowpath_split_high(deep_f_low)
            lowpath_feature_lr = self.lowpath_split_low(deep_f_low)
            lowpath_feature_map = torch.relu(torch.subtract(lowpath_feature_sr,lowpath_feature_lr))

            final_low_sr = self.low_to_sr(lowpath_feature_sr)
            final_low_lr = self.low_to_lr(lowpath_feature_lr)
            final_low_map = self.low_to_map(lowpath_feature_map)

            sr_combine = torch.cat([final_high_sr,final_mid_sr,final_low_sr],1)
            sr = self.sr_conv(sr_combine)
        # print(sr.shape,final_high_lr.shape,final_mid_lr.shape,final_low_lr.shape)
        return {'sr':sr,'high_lr':final_high_lr,'mid_lr':final_mid_lr,'low_lr':final_low_lr}

def main():
    net = SplitSR(num_in_ch=3,num_out_ch=3,upscale = 4,norm=False)
    img = torch.randn((32,3,64,64))
    img_scale = net(img)
    # print(img_scale.shape)
    # torchstat.stat(net,(3,64,64))

if __name__ == '__main__':
    main()