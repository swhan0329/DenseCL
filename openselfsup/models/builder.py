from torch import nn

from openselfsup.utils import build_from_cfg
from .registry import (BACKBONES, MODELS, NECKS, HEADS, MEMORIES, LOSSES)

# registry에는 아래와 같은 코드 존재.
# from openselfsup.utils import Registry
# MODELS = Registry('model')
# BACKBONES = Registry('backbone')
# NECKS = Registry('neck')
# HEADS = Registry('head')
# MEMORIES = Registry('memory')
# LOSSES = Registry('loss')
# 각 registry는 각 registry별 이름을 저장하고 있으며, cfg에 따라, 각 이름에 맞는 모듈들이 저장되는 구조를 띰

def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Default: None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        # build_from_cf: openselfsup/utils/registry.py 에 존재.
        # configs/selfsup/moco/r50_v1.py 에 config파일 존재. 모델훈련을 위한, optimizer set, dataset path, normalization 정보등이 다 담겨있다.
        # cfg = Config.fromfile(args.config) 와 같이 파일로부터 config를 읽어온다.
        # model, loss, head등 build_* 함수에 따라 거기에 맞는 registry가 파라미터로 들어간다.
        # build_model의 경우 cfg = config.model ( moco 에 대한 정보를 담고 있음 )
        
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_memory(cfg):
    """Build memory."""
    return build(cfg, MEMORIES)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_model(cfg):
    """Build model."""
    return build(cfg, MODELS)
