from torch import nn

from Layer.Noise_Layer import Noise_Layer
from Layer.Normal_Layer_FixedAveragePower import Normal_Layer_FixedAveragePower


class Auto_Encoder(nn.Module):
    def __init__(self, size_M):
        super().__init__()

        complex_size = 2

        self.encoder = nn.Sequential(
            nn.Linear(size_M, size_M//2),
            nn.ReLU(),
            nn.Linear(size_M//2, complex_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(complex_size, size_M//2),
            nn.ReLU(),
            nn.Linear(size_M//2, size_M),
        )

    def forward(self, input):
        encoding = self.encoder(input)
        normal_layer = Normal_Layer_FixedAveragePower()
        x = normal_layer(encoding)  # result from transmitter
        noise_layer = Noise_Layer()
        y = noise_layer(x)  # result from channel
        decoding = self.decoder(y)
        softmax = nn.Softmax(dim=1)
        decoding = softmax(decoding)  # result from receiver
        return x, y, decoding
