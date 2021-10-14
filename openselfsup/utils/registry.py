import inspect
from functools import partial

import mmcv


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, force=False):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls=None, force=False):
        if cls is None:
            return partial(self.register_module, force=force)
        self._register_module(cls, force=force)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type') # args는 dictionary형태로 key:type의 value를 뽑는다.
    
    if mmcv.is_str(obj_type):
        # obj_type이 string이면, registry의 module_dict[key=obj_type]을 참조하여 특정 모듈(클래스)을 반환한다.
        # ( If the type of "obj_type" is string, the module(class) instance is returned by refering module_dict[key=obj_type]
        # 해당 obj_type이 존재하지 않으면 에러를 발생시킨다.
        # ( If the obj_type does not exist, error occurs)
        # densecl의 config를 가져온 경우 obj_type = "DENSECL"
        # (If in the case of deseCL, obj_type = "DENSECL"
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
        
    # default_args가 존재한다면, default 값으로 설정한다.
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
            
    # args = cfg.copy() 였으므로, moco의 경우, Moco(**args)로, moco class를 init함수를 거쳐 반환하는 부분이다.
    return obj_cls(**args)
