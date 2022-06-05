import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer
from .ffc_unit import BasicBlock,Bottleneck

@ARCH_REGISTRY.register()
class FFC_MSRResNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=20, upscale=4):
        super(FFC_MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(BasicBlock, num_block,inplanes=num_feat, planes=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat//2, num_out_ch * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat//2, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.upconv2 = nn.Conv2d(num_feat, num_out_ch * 4, 3, 1, 1)
            self.pixel_shuffle2 = nn.PixelShuffle(2)

        self.conv_lr = nn.Conv2d(num_feat//2, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_lr], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        dim = feat.shape[1]
        feat_l,feat_g = feat[:,:dim//2],feat[:,dim//2:]
        feat_l,feat_g = self.body((feat_l,feat_g))
        feature_out_lr,feature_out_hr = feat_l,feat_g
        out_lr = self.conv_lr(feature_out_lr)
        if self.upscale == 4:
            out_hr = self.lrelu(self.pixel_shuffle1(self.upconv1(feature_out_hr)))
            out_hr = self.lrelu(self.pixel_shuffle2(self.upconv2(out_hr)))
        elif self.upscale in [2, 3]:
            out_hr = self.lrelu(self.pixel_shuffle(self.upconv1(feature_out_hr)))
        out_lr += x
        reaidual_hr = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out_hr += reaidual_hr
        return out_lr,out_hr
