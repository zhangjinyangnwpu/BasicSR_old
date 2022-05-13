import pytest
import torch

from basicsr.archs.pasr_arch import PASR,ResExtractor


def test_pasr():
    # model init and forward
    net =  PASR(input_channels=3, output_channels=3, scale=4, num_layers = 5,fea_dim=32)
    img = torch.rand((1, 3, 128, 128), dtype=torch.float32)
    sr = net(img)
    print(sr.shape)

test_pasr()