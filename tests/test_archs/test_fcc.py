import pytest
import torch

from basicsr.archs.ffc_arch import FFCResNetGenerator


def test_rrdb_ffc():
    # model init and forward
    net = FFCResNetGenerator(input_nc=3, output_nc=3)
    img = torch.rand((1, 3, 128, 128), dtype=torch.float32)
    output = net(img)
    print(output.shape)

test_rrdb_ffc()