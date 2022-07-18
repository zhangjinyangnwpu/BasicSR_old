import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import torchstat
from basicsr.utils.registry import ARCH_REGISTRY
import math
from basicsr.archs.arch_util import Upsample

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se = True, se_raito = 8,group=1):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1, padding=1,groups=group)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1,groups=group)
        self.act = nn.SELU()
        self.if_down = False
        if in_channels != out_channels:
            self.if_down = True
            self.down = nn.Conv2d(in_channels,out_channels, kernel_size=1,stride=1, padding=0)
        self.se = se
        if self.se:
            self.se_block = SE_Block(out_channels,se_raito)

    def forward(self, inputs):
        x = inputs
        if self.if_down:
            inputs = self.down(inputs)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        if self.se:
            x = self.se_block(x)
        x = x + inputs
        return x

class BiFPNBlockFusion(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001,Block_Type=ConvBlock,group=8,residual_ratio=0.8):
        super(BiFPNBlockFusion, self).__init__()
        self.residual_ratio = residual_ratio
        self.epsilon = epsilon
        self.p1_td = Block_Type(feature_size*2, feature_size,group=group)
        self.p2_td = Block_Type(feature_size*2, feature_size,group=group)
        self.p3_td = Block_Type(feature_size*2, feature_size,group=group)
        self.p4_td = Block_Type(feature_size*2, feature_size,group=group)

        self.p2_out = Block_Type(feature_size*2, feature_size,group=group)
        self.p3_out = Block_Type(feature_size*2, feature_size,group=group)
        self.p4_out = Block_Type(feature_size*2, feature_size,group=group)
        self.p5_out = Block_Type(feature_size*2, feature_size,group=group)


    def forward(self, x):
        level1,level2,level3,level4,level5 = x
        residual_ratio = self.residual_ratio

        level5_td = level5
        level4_td = self.p4_td(torch.cat([level4, level5_td], 1))
        level3_td = self.p3_td(torch.cat([level3, level4_td], 1))
        level2_td = self.p2_td(torch.cat([level2, level3_td], 1))
        level1_td = self.p1_td(torch.cat([level1, level2_td], 1))

        # Calculate Bottom-Up Pathway
        level1_out = level1_td
        level2_out = self.p2_out(torch.cat([level2_td, level1_out], 1))
        level3_out = self.p3_out(torch.cat([level3_td, level2_out], 1))
        level4_out = self.p4_out(torch.cat([level4_td, level3_out], 1))
        level5_out = self.p5_out(torch.cat([level5_td, level4_out], 1))

        # block residual
        level1_out += level1 * residual_ratio
        level2_out += level2 * residual_ratio
        level3_out += level3 * residual_ratio
        level4_out += level4 * residual_ratio
        level5_out += level5 * residual_ratio

        return [level1_out, level2_out, level3_out, level4_out, level5_out]

@ARCH_REGISTRY.register()
class PASR(nn.Module):
    def __init__(self, input_channels=3,output_channels = 3, scale = 4, num_layers = 10,
                 use_squeeze=False,fea_dim=64, group=1, residual_ratio=0.8):
        super(PASR,self).__init__()

        self.residual_ratio = residual_ratio
        self.use_squeeze = use_squeeze

        self.fist_conv = nn.Conv2d(input_channels, fea_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.level1 = nn.Conv2d(fea_dim,fea_dim,kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=group)
        self.level2 = nn.Conv2d(fea_dim,fea_dim,kernel_size=(3,3),stride=(1,1),padding=(1,1),groups=group)
        self.level3 = nn.Conv2d(fea_dim,fea_dim,kernel_size=(3,3),stride=(1,1),padding=(3,3),dilation=(3,3),groups=group)
        self.level4 = nn.Conv2d(fea_dim,fea_dim,kernel_size=(3,3),stride=(1,1),padding=(6,6),dilation=(6,6),groups=group)
        self.level5 = nn.Conv2d(fea_dim,fea_dim,kernel_size=(3,3),stride=(1,1),padding=(9,9),dilation=(9,9),groups=group)

        self.act = nn.SELU()
        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlockFusion(fea_dim,group=group))
        self.bifpn = nn.Sequential(*bifpns)

        self.conv_output1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=group)
        self.conv_output2 = nn.Conv2d(fea_dim, fea_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=group)
        self.conv_output3 = nn.Conv2d(fea_dim, fea_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=group)
        self.conv_output4 = nn.Conv2d(fea_dim, fea_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=group)
        self.conv_output5 = nn.Conv2d(fea_dim, fea_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=group)
        self.conv_fusion1 = nn.Conv2d(fea_dim * 5, fea_dim * 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=1)
        self.conv_fusion2 = nn.Conv2d(fea_dim * 3, fea_dim * 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),groups=1)

        self.upper = Upsample(scale = scale, num_feat = fea_dim)
        self.conv_tail = nn.Conv2d(fea_dim, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def feature_calc(self,x):
        conv_first = self.fist_conv(x)
        if not self.use_squeeze:
            level1 = self.level1(conv_first)
            level2 = self.level2(conv_first)
            level3 = self.level3(conv_first)
            level4 = self.level4(conv_first)
            level5 = self.level5(conv_first)
        else:
            level1 = self.level1(conv_first)
            level2 = self.level2(level1)
            level3 = self.level3(level2)
            level4 = self.level4(level3)
            level5 = self.level5(level4)

        features = [level1, level2, level3, level4, level5]
        f_level1, f_level2, f_level3, f_level4, f_level5 = self.bifpn(features)

        residual_ratio = self.residual_ratio
        # total residual
        f_level1 += level1 * residual_ratio
        f_level2 += level2 * residual_ratio
        f_level3 += level3 * residual_ratio
        f_level4 += level4 * residual_ratio
        f_level5 += level5 * residual_ratio

        f_level1 = self.conv_output1(f_level1)
        f_level2 = self.conv_output1(f_level2)
        f_level3 = self.conv_output1(f_level3)
        f_level4 = self.conv_output1(f_level4)
        f_level5 = self.conv_output1(f_level5)

        f_output = torch.cat([f_level1, f_level2, f_level3, f_level4, f_level5], 1)

        f_output = self.act(self.conv_fusion1(f_output))
        f_output = self.act(self.conv_fusion2(f_output))

        return f_output

    def forward(self, x):
        f_output = self.feature_calc(x)
        output = self.upper(f_output)
        output = self.conv_tail(output)
        return output


def main():
    img = torch.randn((1,3,256,256))
    net = PASR(input_channels=3, output_channels=3, scale=2, num_layers=10,fea_dim=64)
    output = net(img)
    print(output.shape)

    # print(net)
    # torchstat.stat(net,(3,64,64))

if __name__ == '__main__':
    main()