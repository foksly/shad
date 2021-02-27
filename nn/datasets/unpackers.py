import torch

class DefaultUnpacker:
    def __init__(self, device=torch.device('cpu')):
        self.device = device

    def unpack_batch(self, inputs, targets):
        batch = {
            'inputs': {'x': inputs},
            'targets': targets
        }
        return batch

    def __call__(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        return self.unpack_batch(inputs, targets)


class AutoencoderUnpacker(DefaultUnpacker):
    def __call__(self, batch):
        inputs, _ = batch
        inputs = inputs.to(self.device)
        return self.unpack_batch(inputs, inputs)
