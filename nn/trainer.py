import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from .datasets.unpackers import DefaultUnpacker
from .utils import save_checkpoint

import wandb


class DefaultTrainer:
    def __init__(self, model, optimizer, criterion, data, logger, device,
                 lr=3e-4, metrics=None, unpacker=DefaultUnpacker, valid=False, desc='',
                 checkpoint_path=None):
        # data
        self.data = data
        self.unpacker = unpacker(device)

        # logger, metrics
        self.logger = logger
        self.metrics = metrics or dict()

        # model, optimizer, criterion
        self.model = model.to(device)
        self.optimizer = optimizer(model.parameters(), lr)
        self.criterion = criterion

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.valid = valid
        self.desc = desc

    @classmethod
    def load(cls, config):
        pass

    def train(self, n_epochs=1, num_batches=None):
        self.logger.watch(self.model, log='all', log_freq=50)
        self.model.train()
        prefix = self._add_desc('train_')

        for epoch in range(n_epochs):
            for batch in self.data.trainloader:
                unpacked_batch = self.unpacker(batch)
                logits = self.train_batch(unpacked_batch)
                self.log_metrics(logits, unpacked_batch, prefix=prefix, accumulate=True)

            self.logger.flush_accumulated(prefix=prefix)

            if self.valid:
                self.validate()

            if self.checkpoint_path is not None:
                save_checkpoint(self.model, self.optimizer, path=self.checkpoint_path)

        self.model.eval()
        return self.model

    def train_batch(self, unpacked_batch, step=True):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(**unpacked_batch['inputs'])
        loss = self.compute_loss(logits, unpacked_batch)
        loss.backward()
        if step:
            self.optimizer.step()

        self.logger.log({self._add_desc('train_loss'): loss.item()}, ticker=self._add_desc('train'))
        return logits

    def validate(self):
        self.model.eval()
        prefix = self._add_desc('valid_')
        with torch.no_grad():
            for batch in self.data.validloader:
                unpacked_batch = self.unpacker(batch)
                logits = self.model(**unpacked_batch['inputs'])
                loss = self.compute_loss(logits, unpacked_batch)

                self.logger.log({self._add_desc('valid_loss'): loss.item()}, ticker=self._add_desc('valid'))
                self.log_metrics(logits, unpacked_batch, prefix=prefix, accumulate=True)
            self.logger.flush_accumulated(prefix=prefix)

    def compute_loss(self, logits, unpacked_batch):
        return self.criterion(logits, unpacked_batch['targets'])

    def log_metrics(self, logits, unpacked_batch, prefix='', ticker=None, accumulate=False):
        scores = {}
        for metric in self.metrics:
            scores[prefix + metric] = self.metrics[metric](logits, unpacked_batch['targets'])

        if scores:
            self.logger.log(scores, ticker=ticker, accumulate=accumulate)

    def _add_desc(self, name):
        return f'{self.desc}_{name}'
