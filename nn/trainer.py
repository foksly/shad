import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from .datasets.unpackers import DefaultUnpacker
from .utils import save_checkpoint, parse_config, load_object, load_class

import typing as tp
import yaml

import wandb

@dataclass
class DefaultTrainer:
    data: tp.Any
    model: tp.Any
    optimizer: tp.Any
    criterion: tp.Any
    logger: tp.Any
    metrics: tp.Any
    device: tp.Any
    epochs: int = 10
    unpacker: tp.Any = DefaultUnpacker
    valid: bool = False
    checkpoint_dir: tp.Optional[str] = None

    def __new__(cls, *args, **kwargs):
        try:
            initializer = cls.__initializer
        except AttributeError:
            cls.__initializer = initializer = cls.__init__
            cls.__init__ = lambda *a, **k: None

        added_args = {}
        for name in list(kwargs.keys()):
            if name not in cls.__annotations__:
                added_args[name] = kwargs.pop(name)

        ret = object.__new__(cls)
        initializer(ret, **kwargs)
        for new_name, new_val in added_args.items():
            setattr(ret, new_name, new_val)

        return ret

    @classmethod
    def configure(cls, config):
        config = parse_config(config)

        # data
        data = load_class(name=config.data.desc, **config.data.kwargs)

        # logger, metrics
        logger = load_class(config.logger.desc, **config.logger.kwargs)
        metrics = {}
        for metric in config.metrics:
            metrics[metric] = load_class(config.metrics[metric])

        # train options
        device = torch.device(config.device)
        model = cls.load_model(config).to(device)
        optimizer = load_class(config.optimizer.desc, model.parameters(), **config.optimizer.kwargs)
        criterion = load_class(config.criterion.desc, **config.criterion.kwargs)

        epochs = config.epochs
        unpacker = load_class(config.data.unpacker, device) if 'unpacker' in config.data else DefaultUnpacker(device)
        valid = config.valid
        experiment_name = config.experiment_name
        return cls(
            data=data,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            logger=logger,
            metrics=metrics,
            device=device,
            epochs=epochs,
            unpacker=unpacker,
            valid=valid,
            checkpoint_dir=None,  # TODO: add checkpoints
        )

    @staticmethod
    def load_model(config):
        return load_class(config.model.desc, **config.model.hp)

    def run(self):
        self.logger.watch(self.model, log='all', log_freq=50)
        self.model.train()
        prefix = 'train_'

        for epoch in range(self.epochs):
            for batch in self.data.trainloader:
                unpacked_batch = self.unpacker(batch)
                logits = self.train_batch(unpacked_batch)
                self.log_metrics(logits, unpacked_batch, prefix=prefix, accumulate=True)

            self.logger.flush_accumulated(prefix=prefix)

            if self.valid:
                self.validate()

            # if self.checkpoint_dir is not None:
            #     save_checkpoint(self.model, self.optimizer, path=self.checkpoint_dir)

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

        self.logger.log({'train_loss': loss.item()}, ticker='train')
        return logits

    def validate(self):
        self.model.eval()
        prefix = 'valid_'
        with torch.no_grad():
            for batch in self.data.validloader:
                unpacked_batch = self.unpacker(batch)
                logits = self.model(**unpacked_batch['inputs'])
                loss = self.compute_loss(logits, unpacked_batch)

                self.logger.log({'valid_loss': loss.item()}, ticker='valid')
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
