import pytest
import torch

from basicsr.archs.ffc_srresnet_arch import FFC_MSRResNet


def test_ffc_ssresnet():
    # model init and forward
    net = FFC_MSRResNet(3, 3, upscale=4)
    img = torch.rand((5, 3, 199, 88), dtype=torch.float32)
    output = net(img)
    print(output.shape)

test_ffc_ssresnet()