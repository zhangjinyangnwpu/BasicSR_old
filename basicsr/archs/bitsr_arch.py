import torch
import math
from torch import nn as nn
import torchstat
from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


class BitConv(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(BitConv, self).__init__()
        self.res_scale = res_scale
        self.num_k1 = num_feat // 4
        self.num_k3 = num_feat // 2
        self.num_k5 = num_feat // 4
        self.conv_k1 = nn.Conv2d(self.num_k1+2, self.num_k1, kernel_size= 1,  stride= 1, padding=0,dilation=1,  bias=True, groups=1)
        self.conv_k3 = nn.Conv2d(self.num_k3+2, self.num_k3, kernel_size= 3,  stride= 1, padding=3,dilation=3,  bias=True, groups=1)
        self.conv_k5 = nn.Conv2d(self.num_k5+2, self.num_k5, kernel_size= 5,  stride= 1, padding=10,dilation=5,  bias=True, groups=1)
        self.conv_fusion = nn.Conv2d(num_feat, num_feat, kernel_size= 1,  stride= 1, padding=0,  bias=True)
        self.conv_out = nn.Conv2d(num_feat+2, num_feat, kernel_size= 1,  stride= 1, padding=0,  bias=True)
        self.act = nn.LeakyReLU(inplace=False)
        self.norm1 = nn.LayerNorm(num_feat)
        self.norm2 = nn.LayerNorm(num_feat)

        if not pytorch_init:
            default_init_weights([self.conv_k1, self.conv_k3,self.conv_k5], 0.1)



    def forward(self, x):
        identity = x
        num = x.shape[1]
        x_1 = x[:,:num//4]
        x_3 = x[:,num//4:num//4 * 3]
        x_5 = x[:,num//4 * 3:]

        x_range = torch.linspace(-1, 1, identity.shape[-1], device=identity.device)
        y_range = torch.linspace(-1, 1, identity.shape[-2], device=identity.device)
        y, x = torch.meshgrid(y_range, x_range) # 生成二维坐标网格
        y = y.expand([identity.shape[0], 1, -1, -1]) # 扩充到和ins_feat相同维度
        x = x.expand([identity.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        x_1 = torch.cat([x_1,coord_feat],dim=1)
        x_3 = torch.cat([x_3,coord_feat],dim=1)
        x_5 = torch.cat([x_5,coord_feat],dim=1)

        f_1 = self.act(self.conv_k1(x_1))
        f_3 = self.act(self.conv_k3(x_3))
        f_5 = self.act(self.conv_k5(x_5))

        # print(f_1.shape,f_3.shape,f_5.shape)

        f_fusion = self.conv_fusion(torch.cat([f_1,f_3,f_5],dim=1))
        f_fusion = self.act(f_fusion)
        f_fusion = torch.cat([f_fusion,coord_feat],dim=1)
        f_fusion = self.conv_out(f_fusion)
        out = self.act(f_fusion)

        return identity + out * self.res_scale



@ARCH_REGISTRY.register()
class BitSR(nn.Module):
    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=32,
                 num_block=8,
                 upscale=4,
                 res_scale=1,):
        super(BitSR, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = make_layer(BitConv, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))
        x = torch.sigmoid(x)
        return x

def main():
    net = BitSR(num_in_ch=1,num_out_ch=1,upscale = 4)
    img = torch.randn((32,1,64,64))
    img_scale = net(img)
    print(img_scale.shape)
    # torchstat.stat(net,(3,64,64))

if __name__ == '__main__':
    main()