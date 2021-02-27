import numpy as np

class Accuracy:
    def __init__(self, average=False):
        self.average = average

    def __call__(self, logits, labels):
        acc = labels.cpu().numpy() == logits.topk(1, dim=1)[1].cpu().numpy().reshape(-1)
        if self.average:
            return np.mean(acc)
        return acc
