import torch
from torch import nn as nn
from torch.nn import functional as F
import torchstat
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import Upsample
# from .acmix_unit import ACmix

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,group=1,resiudlal_ratio=1.0):
        super(ConvBlock,self).__init__()
        self.resiudlal_ratio = resiudlal_ratio
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),
                               stride=(1,1), padding=(1,1),groups=group)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3),
                               stride=(1,1), padding=(1,1),groups=group)
        self.act = nn.ReLU()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1),
                               stride=(1,1), padding=(0,0),groups=group)

    def forward(self, inputs):
        x = inputs
        residual = self.down(inputs)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x + residual * self.resiudlal_ratio
        return x

class Self_Attn(nn.Module):
    def __init__(self, in_channels, out_channels,group=1):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_channels
        self.activation = nn.SELU
        self.query_conv = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_channels , out_channels = in_channels , kernel_size= 1)
        self.output_conv = nn.Conv2d(in_channels = in_channels , out_channels = out_channels , kernel_size= 3, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        out = self.output_conv(out)
        return out


class BiFPNBlockFusion(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=32, epsilon=0.0001,Block_Type=ConvBlock,group=1,residual_ratio=1):
        super(BiFPNBlockFusion, self).__init__()
        self.residual_ratio = residual_ratio
        self.epsilon = epsilon

        self.p1_td = Block_Type(feature_size*2, feature_size)
        self.p2_td = Block_Type(feature_size*2, feature_size)

        self.p2_out = Block_Type(feature_size*2, feature_size)
        self.p3_out = Block_Type(feature_size*2, feature_size)


    def forward(self, x):
        level1,level2,level3 = x
        residual_ratio = self.residual_ratio

        level3_td = level3
        level2_td = self.p2_td(torch.cat([level2, level3_td], 1))
        level1_td = self.p1_td(torch.cat([level1, level2_td], 1))

        # Calculate Bottom-Up Pathway
        level1_out = level1_td
        level2_out = self.p2_out(torch.cat([level2_td, level1_out], 1))
        level3_out = self.p3_out(torch.cat([level3_td, level2_out], 1))

        # block residual
        level1_out += level1 * residual_ratio
        level2_out += level2 * residual_ratio
        level3_out += level3 * residual_ratio

        return [level1_out, level2_out, level3_out]


@ARCH_REGISTRY.register()
class FPNSR(nn.Module):
    def __init__(self, input_channels=3,output_channels = 3, scale = 4, num_layer = 5,
                 fea_dim = 32, group=1, residual_ratio=1):
        super(FPNSR,self).__init__()

        self.residual_ratio = residual_ratio
        self.num_layer = num_layer

        self.conv_first = nn.Conv2d(input_channels, fea_dim, kernel_size=(3, 3),
                                    stride=(1, 1), padding=(1, 1))

        self.level1 = nn.Conv2d(fea_dim,fea_dim,kernel_size=(1,1),
                                stride=(1,1),padding=(0,0),groups=group)
        self.level2 = nn.Conv2d(fea_dim,fea_dim,kernel_size=(3,3),
                                stride=(1,1),padding=(1,1),groups=group)
        self.level3 = nn.Conv2d(fea_dim,fea_dim,kernel_size=(3,3),
                                stride=(1,1),padding=(3,3),dilation=(3,3),groups=group)

        self.act = nn.SELU()

        self.bifpns = nn.ModuleList()
        for _ in range(num_layer):
            self.bifpns.append(BiFPNBlockFusion(fea_dim,group=group))

        self.conv_output1 = nn.Conv2d(fea_dim, fea_dim, kernel_size=(1, 1),
                                      stride=(1, 1), padding=(0, 0),groups=group)
        self.conv_output2 = nn.Conv2d(fea_dim, fea_dim, kernel_size=(3, 3),
                                      stride=(1, 1), padding=(1, 1), groups=group)
        self.conv_output3 = nn.Conv2d(fea_dim, fea_dim, kernel_size=(3, 3),
                                      stride=(1, 1), padding=(3, 3), dilation=(3,3),groups=group)

        self.conv_fusion = nn.Conv2d(fea_dim * 3, fea_dim, kernel_size=(3, 3),
                                      stride=(1, 1), padding=(1, 1),groups=group)

        self.upper = Upsample(scale = scale, num_feat = fea_dim)
        self.conv_tail = nn.Conv2d(fea_dim, output_channels, kernel_size=(3, 3),
                                   stride=(1, 1), padding=(1, 1))

    def feature_calc(self,x):
        conv_first = self.conv_first(x)

        level1 = self.level1(conv_first)
        level2 = self.level2(conv_first)
        level3 = self.level3(conv_first)

        features = [level1, level2, level3]
        features_res = []

        for num in range(self.num_layer):
            level1, level2, level3 = self.bifpns[num](features)
            features = [level1, level2, level3]
            features_res.append(features)

        residual_ratio = self.residual_ratio
        f_output_res = []
        for feature in features_res:
            f_level1, f_level2, f_level3 = feature
            f_level1 += level1 * residual_ratio
            f_level2 += level2 * residual_ratio
            f_level3 += level3 * residual_ratio

            f_level1 = self.conv_output1(f_level1)
            f_level2 = self.conv_output1(f_level2)
            f_level3 = self.conv_output1(f_level3)

            f_output = torch.cat([f_level1, f_level2, f_level3], 1)

            f_output = self.act(self.conv_fusion(f_output))
            f_output_res.append(f_output)

        return f_output_res

    def forward(self, x, is_train = False):
        f_output_res = self.feature_calc(x)
        if is_train:
            outputs = []
            for f_output in f_output_res:
                output = self.upper(f_output)
                output = self.conv_tail(output)
                outputs.append(output)
            return outputs
        else:
            f_output = f_output_res[-1]
            output = self.upper(f_output)
            output = self.conv_tail(output)
            return output


def main():
    img = torch.randn((1,3,64,64)).to('cpu')
    net = FPNSR(input_channels=3, output_channels=3, scale=4, num_layer=5,fea_dim=64).to('cpu')
    output = net(img,False)
    # for item in output:
    #     print(item.shape)
    net_print = True
    if net_print:
        torchstat.stat(net,(3,64,64))

if __name__ == '__main__':
    main()