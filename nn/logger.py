import wandb
import numpy as np
from PIL import Image
from collections import defaultdict
from collections.abc import Iterable


class BaseLogger:
    pass

class WBLogger:
    def __init__(self, project, run_name=None, entity=None, config=None):
        self.run = wandb.init(project=project, reinit=True, config=config)
        if run_name:
            self.run.name = run_name

        self.tickers = defaultdict(lambda: 0)
        self.accumulated_log = defaultdict(list)

    def log_image(self, image, label='example', group=False):
        wandb_image = list(map(wandb.Image, image))
        if group:
            wandb_image = [wandb.Image(image)]

        wandb.log({label: wandb_image}, commit=False)

    def log(self, logging_info, ticker=None, accumulate=False, **kwargs):
        if ticker is not None:
            self.tickers[ticker] += 1
            assert f'{ticker}_ticker' not in logging_info, 'Ticker name collision happened.'

            logging_info.update({f'{ticker}_ticker': self.tickers[ticker]})

        if accumulate:
            for key, value in logging_info.items():
                if isinstance(value, Iterable):
                    self.accumulated_log[key].extend(value)
                else:
                    self.accumulated_log[key].append(value)
        else:
            wandb.log(logging_info, **kwargs)

    def flush_accumulated(self, prefix='', accumulate=np.mean, ticker=None):
        for key in self.accumulated_log.keys():
            if key.startswith(prefix):
                accumulated_value = accumulate(self.accumulated_log[key])
                self.log({key: accumulated_value}, ticker=ticker)
                self.accumulated_log[key] = []

    def watch(self, model, *args, **kwargs):
        wandb.watch(model, *args, **kwargs)

    def finish(self):
        self.run.finish()
