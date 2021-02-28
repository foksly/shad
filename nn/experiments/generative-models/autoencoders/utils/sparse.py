import torch
from torch import nn
import numpy as np

from shad.nn.trainer import DefaultTrainer

class SparseTrainer(DefaultTrainer):
    def compute_loss(self, logits, unpacked_batch):
        return self.criterion(self.model, unpacked_batch['inputs']['x'], logits)

class SparseL1Loss:
    def __init__(self, sparse_loss_factor=1e-3):
        self.sparse_loss_factor = sparse_loss_factor

    def l1_loss(self, x):
        return torch.mean(torch.sum(torch.abs(x), dim=1))

    def calculate_sparse_loss(self, model, x):
        loss = 0
        for block in model.encoder[:-1]:
            x = block.conv(x)
            loss += self.l1_loss(x)
            x = block.activation(block.norm(x))
        x = model.encoder[-1](x)
        loss += self.l1_loss(x)

        for block in model.decoder[:-1]:
            x = block.conv(x)
            loss += self.l1_loss(x)
            x = block.activation(block.norm(x))
        x = model.decoder[-1](x)
        loss += self.l1_loss(x)
        return loss

    def __call__(self, model, inputs, logits):
        mse_loss = nn.MSELoss()
        loss = mse_loss(logits, inputs) + self.sparse_loss_factor * self.calculate_sparse_loss(model, inputs)
        return loss


class SparseKLLoss:
    def __init__(self, sparse_loss_factor=1e-3, bernoulli_const=0.05):
        self.bernoulli_const = bernoulli_const
        self.sparse_loss_factor = sparse_loss_factor

    def kl_divergence(self, x):
        x = x.view(x.shape[0], -1)
        p_hat = torch.mean(x, dim=1)
        p = torch.ones_like(p_hat) * self.bernoulli_const
        return torch.sum(p * torch.log(p / p_hat) + (1 - p) * torch.log((1 - p) / (1 - p_hat)))

    def calculate_sparse_loss(self, model, x):
        loss = 0
        for block in model.encoder[:-1]:
            x = block(x)
            loss += self.kl_divergence(x)
        x = model.encoder[-1](x)
        # note that we are using sigmoid on each encoder layer except the last
        loss += self.kl_divergence(torch.sigmoid(x))

        for block in model.decoder[:-1]:
            x = block(x)
            loss += self.kl_divergence(x)
        x = model.decoder[-1](x)
        # note that we are using sigmoid on each decoder layer except the last
        loss += self.kl_divergence(torch.sigmoid(x))

        return loss

    def __call__(self, model, inputs, logits):
        mse_loss = nn.MSELoss()
        loss = mse_loss(logits, inputs) + self.sparse_loss_factor * self.calculate_sparse_loss(model, inputs)
        return loss
