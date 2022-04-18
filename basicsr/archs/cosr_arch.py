import torch
import math
from torch import nn as nn
import torchstat
from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY

def channel_shuffle(x, groups=4):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class CoConv(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(CoConv, self).__init__()
        self.res_scale = res_scale
        self.num_k1 = num_feat // 4
        self.num_k3 = num_feat // 2
        self.num_k5 = num_feat // 4
        self.conv_k1 = nn.Conv2d(self.num_k1, self.num_k1, kernel_size= 3,  stride= 1, padding=1, dilation=1,  bias=True, groups=1)
        self.conv_k3 = nn.Conv2d(self.num_k3, self.num_k3, kernel_size= 3,  stride= 1, padding=3, dilation=3,  bias=True, groups=1)
        self.conv_k5 = nn.Conv2d(self.num_k5, self.num_k5, kernel_size= 3,  stride= 1, padding=5, dilation=5,  bias=True, groups=1)
        self.conv_fusion = nn.Conv2d(num_feat, num_feat, kernel_size= 1,  stride= 1, padding=0,  bias=True)
        self.conv_out = nn.Conv2d(num_feat, num_feat, kernel_size= 1,  stride= 1, padding=0,  bias=True)
        self.act = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(num_feat)
        self.norm2 = nn.BatchNorm2d(num_feat)
        if not pytorch_init:
            default_init_weights([self.conv_k1, self.conv_k3,self.conv_k5], 0.1)

    def forward(self, x):
        identity = x
        num = x.shape[1]
        x_1 = x[:,:num//4]
        x_3 = x[:,num//4:num//4 * 3]
        x_5 = x[:,num//4 * 3:]
        f_1 = self.act(self.conv_k1(x_1))
        f_3 = self.act(self.conv_k3(x_3))
        f_5 = self.act(self.conv_k5(x_5))
        f_fusion = self.act(self.norm1(self.conv_fusion(torch.cat([f_1,f_3,f_5],dim=1))))
        out = self.act(self.norm2(self.conv_out(f_fusion)))
        return identity + out * self.res_scale



@ARCH_REGISTRY.register()
class CoSR(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 norm=True,
                 fpn_num=3):
        super(CoSR, self).__init__()
        self.upscale_num = 1 if upscale == 3 else int(math.log(upscale)/math.log(2))
        self.fpn_num = fpn_num
        self.norm = norm
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = make_layer(CoConv, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        if self.norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))
        if self.norm:
            x = x / self.img_range + self.mean
        return x

def main():
    net = CoSR(num_in_ch=3,num_out_ch=3,upscale = 4,norm=False)
    img = torch.randn((32,3,64,64))
    img_scale = net(img)
    print(img_scale.shape)
    # torchstat.stat(net,(3,64,64))

if __name__ == '__main__':
    main()