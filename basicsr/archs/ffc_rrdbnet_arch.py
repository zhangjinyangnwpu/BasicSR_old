import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle,Upsample


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1_sp = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb1_ffc = FourierUnit(num_feat,num_feat)
        self.rdb1 = nn.Conv2d(num_feat*2, num_feat,3,1,1)
        self.rdb2_sp = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2_ffc = FourierUnit(num_feat, num_feat)
        self.rdb2 = nn.Conv2d(num_feat*2, num_feat,3,1,1)
        self.rdb3_sp = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3_ffc = FourierUnit(num_feat, num_feat)
        self.rdb3 = nn.Conv2d(num_feat*2, num_feat,3,1,1)

    def forward(self, x):
        out1_sp = self.rdb1_sp(x)
        out1_ffc = self.rdb1_ffc(x)
        out1 = self.rdb1(torch.cat([out1_sp,out1_ffc],dim=1))
        out2_sp = self.rdb2_sp(out1)
        out2_ffc = self.rdb2_ffc(out1)
        out2 = self.rdb2(torch.cat([out2_sp, out2_ffc], dim=1))
        out3_sp = self.rdb3_sp(out2)
        out3_ffc = self.rdb3_ffc(out2)
        out = self.rdb3(torch.cat([out3_sp, out3_ffc], dim=1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

@ARCH_REGISTRY.register()
class RRDBNet_FFC(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet_FFC, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first_spatial = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_first_ffc = FourierUnit(num_in_ch,num_feat)
        self.conv_first = nn.Conv2d(num_feat*2, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1_spatial = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1_ffc = FourierUnit(num_feat,num_feat)
        self.conv_up1 = nn.Conv2d(num_feat*2, num_feat, 3, 1, 1)
        self.conv_up2_spatial = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2_ffc = FourierUnit(num_feat, num_feat)
        self.conv_up2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.conv_up3_spatial = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up3_ffc = FourierUnit(num_feat, num_feat)
        self.conv_up3 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.upsample1 = Upsample(2, num_feat)
        self.upsample2 = Upsample(2, num_feat)
        self.upsample3 = Upsample(2, num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat_first_sp = self.conv_first_spatial(feat)
        feat_first_ffc = self.conv_first_ffc(feat)
        feat = self.conv_first(torch.cat([feat_first_sp,feat_first_ffc],dim=1))
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat_up1_sp = self.conv_up1_spatial(feat)
        feat_up1_ffc = self.conv_up1_ffc(feat)
        feat = torch.cat([feat_up1_sp, feat_up1_ffc], dim=1)
        feat = self.conv_up1(feat)
        feat = self.upsample1(feat)

        feat_up2_sp = self.conv_up2_spatial(feat)
        feat_up2_ffc = self.conv_up2_ffc(feat)
        feat = torch.cat([feat_up2_sp, feat_up2_ffc], dim=1)
        feat = self.conv_up2(feat)
        feat = self.upsample2(feat)
        if self.scale == 8:
            feat_up3_sp = self.conv_up3_spatial(feat)
            feat_up3_ffc = self.conv_up3_ffc(feat)
            feat = torch.cat([feat_up3_sp, feat_up3_ffc], dim=1)
            feat = self.conv_up3(feat)
            feat = self.upsample3(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
