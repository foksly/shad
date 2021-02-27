import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv2dBlock, DenseBlock
from ..utils import compute_conv_output_size


class Conv2dClassifier(nn.Module):
    def __init__(self, n_classes=10, image_size=32, in_channels=3,
                 conv_blocks_channels=(8, 16, 32), dense_blocks_features=(128,),
                 dropout_prob=0.2):
        super().__init__()

        # CNN backbone
        classifier_in_features = -1
        self.cnn_backbone = []
        for out_channels in conv_blocks_channels:
            block = Conv2dBlock(in_channels, out_channels, stride=2)
            self.cnn_backbone.append(block)

            image_size = compute_conv_output_size(image_size, block.padding, block.kernel_size, block.stride)
            in_channels = out_channels

        self.cnn_backbone = nn.Sequential(*self.cnn_backbone)

        # Classifier
        self.classifier = [nn.Flatten()]
        in_features = in_channels * image_size * image_size
        for out_features in dense_blocks_features:
            self.classifier.append(DenseBlock(in_features, out_features, dropout_prob=dropout_prob))
            in_features = out_features

        self.classifier.append(nn.Linear(in_features, n_classes))
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        x = self.cnn_backbone(x)
        x = self.classifier(x)
        return x

    def get_activations(self, x):
        return torch.flatten(self.cnn_backbone(x), start_dim=1)
