import torch
import yaml
import importlib
from dotmap import DotMap

import typing as tp

def save_checkpoint(model, optimizer, path: str, metainfo=None):
    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if metainfo is not None:
        state_dict.update(metainfo)

    torch.save(state_dict, path)


def parse_config(config: tp.Union[str, dict]) -> DotMap:
    if isinstance(config, str):
        with open(config, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.Loader)
        return DotMap(config)

    if isinstance(config, dict):
        return DotMap(config)

    raise TypeError(f'Unsupported config type: {type(config)}')


def load_object(name):
    module_name, object_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    assert hasattr(module, object_name), f"Module {module_name} has no object with name {object_name}"
    return getattr(module, object_name)


def load_class(name, *args, **kwargs):
    cls = load_object(name)
    return cls(*args, **kwargs)
