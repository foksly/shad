import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv2dBlock, Conv2dDenoisingBlock


class Conv2dAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            Conv2dBlock(1, 16, 3, stride=2),
            Conv2dBlock(16, 32, 3, stride=2),
            Conv2dBlock(32, 32, 3, stride=2),
            Conv2dBlock(32, 32, 3, stride=2),
            Conv2dBlock(32, 32, 3, stride=1).conv,
        )

        self.decoder = nn.Sequential(
            Conv2dBlock(32, 32, 3, upsample=True),
            Conv2dBlock(32, 32, 3, upsample=True),
            Conv2dBlock(32, 32, 3, upsample=True),
            Conv2dBlock(32, 16, 3, upsample=True),
            Conv2dBlock(16, 1, 3).conv,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x

    def get_latent_features(self, x, flatten=False):
        features = self.encoder(x)
        if flatten:
            return torch.flatten(features, start_dim=1)
        return features

class Conv2dDenoisingAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            Conv2dDenoisingBlock(1, 16, 3, stride=2),
            Conv2dDenoisingBlock(16, 32, 3, stride=2),
            Conv2dDenoisingBlock(32, 32, 3, stride=2),
            Conv2dDenoisingBlock(32, 32, 3, stride=2),
            Conv2dDenoisingBlock(32, 32, 3, stride=1).conv,
        )

        self.decoder = nn.Sequential(
            Conv2dDenoisingBlock(32, 32, 3, upsample=True),
            Conv2dDenoisingBlock(32, 32, 3, upsample=True),
            Conv2dDenoisingBlock(32, 32, 3, upsample=True),
            Conv2dDenoisingBlock(32, 16, 3, upsample=True),
            Conv2dDenoisingBlock(16, 1, 3).conv,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x

    def get_latent_features(self, x):
        return self.encoder(x)

    def prepare_input(x):
        with torch.no_grad():
            return x + torch.randn_like(x) * 0.1
