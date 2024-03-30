import torch
from torch import nn


class Noise_Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        matrix_rc = torch.randn(input.size(dim=0), input.size(dim=1))
        k = torch.log2(torch.tensor(input.size(dim=1))).item()
        factor = 1 / torch.sqrt(torch.tensor(14 * k))
        matrix_rc = factor.mul(matrix_rc)

        return input + matrix_rc