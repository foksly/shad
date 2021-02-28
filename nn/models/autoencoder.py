import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv2dBlock, DenseBlock


class Conv2dAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            Conv2dBlock(1, 16, stride=2),
            Conv2dBlock(16, 32, stride=2),
            Conv2dBlock(32, 64, stride=2),
            Conv2dBlock(64, 64, stride=2),
            Conv2dBlock(64, 64, stride=1).conv,
        )

        self.decoder = nn.Sequential(
            Conv2dBlock(64, 64, upsample=True),
            Conv2dBlock(64, 64, upsample=True),
            Conv2dBlock(64, 64, upsample=True),
            Conv2dBlock(64, 32, upsample=True),
            Conv2dBlock(32, 1, 3).conv,
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


class Noise(nn.Module):
    def __init__(self, noise=0.2):
        super().__init__()
        self.noise = noise

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.noise
        else:
            return x


class Conv2dDenoisingAutoEncoder(nn.Module):
    def __init__(self, input_noise=0.2, layer_noise=0.05, dropout_prob=0.15):
        super().__init__()

        encoder_blocks = []
        self.encoder = nn.Sequential(
            Noise(input_noise),
            Conv2dBlock(1, 16, stride=2),
            Noise(layer_noise),
            Conv2dBlock(16, 32, stride=2),
            Noise(layer_noise),
            Conv2dBlock(32, 64, stride=2),
            Noise(layer_noise),
            Conv2dBlock(64, 64, stride=2),
            Noise(layer_noise),
            Conv2dBlock(64, 64, stride=1).conv,
        )

        self.decoder = nn.Sequential(
            Conv2dBlock(64, 64, upsample=True),
            Noise(layer_noise),
            Conv2dBlock(64, 64, upsample=True),
            Noise(layer_noise),
            Conv2dBlock(64, 64, upsample=True),
            Noise(layer_noise),
            Conv2dBlock(64, 32, upsample=True),
            Noise(layer_noise),
            Conv2dBlock(32, 1, 3).conv,
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


class Conv2dSparseKLAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        sigmoid = nn.Sigmoid()
        self.encoder = nn.Sequential(
            Conv2dBlock(1, 16, bn=False, activation=sigmoid, stride=2),
            Conv2dBlock(16, 32, bn=False, activation=sigmoid, stride=2),
            Conv2dBlock(32, 64, bn=False, activation=sigmoid, stride=2),
            Conv2dBlock(64, 64, bn=False, activation=sigmoid, stride=2),
            Conv2dBlock(64, 64, stride=1).conv,
        )

        self.decoder = nn.Sequential(
            Conv2dBlock(64, 64, bn=False, activation=sigmoid, upsample=True),
            Conv2dBlock(64, 64, bn=False, activation=sigmoid, upsample=True),
            Conv2dBlock(64, 64, bn=False, activation=sigmoid, upsample=True),
            Conv2dBlock(64, 32, bn=False, activation=sigmoid, upsample=True),
            Conv2dBlock(32, 1, 3).conv,
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


class LatentFeaturesDenseClassifier(nn.Module):
    def __init__(self, autoencoder, in_features, n_classes):
        super().__init__()

        self.feature_extractor = autoencoder.encoder
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            DenseBlock(in_features, 1024, dropout_prob=0.3),
            DenseBlock(1024, 512, dropout_prob=0.3),
            DenseBlock(512, 256, dropout_prob=0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = torch.flatten(self.feature_extractor(x), start_dim=1)
        x = self.classifier(x)
        return x
