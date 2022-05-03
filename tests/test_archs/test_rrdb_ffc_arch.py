import pytest
import torch

from basicsr.archs.rrdbnet_ffc_arch import RRDBNet_FFC


def test_rrdb_ffc():
    # model init and forward
    net =  RRDBNet_FFC(3, 3, scale=4)
    img = torch.rand((1, 3, 128, 128), dtype=torch.float32)
    output = net(img)
    print(output.shape)

test_rrdb_ffc()