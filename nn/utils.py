import torch
import yaml
from dotmap import DotMap

def save_checkpoint(model, optimizer, path: str, metainfo=None):
    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if metainfo is not None:
        state_dict.update(metainfo)

    torch.save(state_dict, path)


def load_config(path):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)

    return DotMap(config)
