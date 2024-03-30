from torch import nn


class Normal_Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return nn.functional.normalize(input, p=2, dim=1)