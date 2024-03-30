import torch
from torch import nn


class Normal_Layer_FixedAveragePower(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        const = torch.sqrt(torch.mean(torch.abs(tensor) ** 2) + 1e-6)
        x_normalise = tensor / const
        return x_normalise
