import pytest
import torch

from basicsr.archs.ffc_srresnet_arch import FFC_MSRResNet


def test_ffc_ssresnet():
    # model init and forward
    net = FFC_MSRResNet(3, 3, upscale=3)
    img = torch.rand((5, 3, 332, 31), dtype=torch.float32)
    output = net(img)
    print(output[0].shape,output[1].shape)

test_ffc_ssresnet()