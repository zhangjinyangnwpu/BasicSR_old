import torch

from basicsr.archs.pasr_arch import PASR


def test_pasr():
    # model init and forward
    net = PASR(input_channels=3, output_channels=3, scale=2, num_layers = 20,fea_dim=32)
    img = torch.rand((10, 3, 128, 128), dtype=torch.float32)
    sr = net(img)
    print(sr.shape)

test_pasr()