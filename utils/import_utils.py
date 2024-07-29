#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import warnings
from importlib import import_module
from inspect import ismodule
from typing import Optional, Union, List, Dict, Any
from types import ModuleType
import copy


def import_modules_from_strings(
        imports: Optional[Union[List[str], str]], allow_failed_imports=False
) -> Optional[Union[List[ModuleType], ModuleType]]:
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Defaults to False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp.replace('//', '\\').replace('\\', '.'))
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError(f'Failed to import {imp}')
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def get_object_from_string(obj_name: str):
    """Get object from name.

    Args:
        obj_name (str): The name of the object.

    Examples:
        >>> get_object_from_string('torch.optim.sgd.SGD')
        >>> torch.optim.sgd.SGD
    """
    parts = iter(obj_name.replace('/', '\\').replace('\\', '.').split('.'))
    module_name = next(parts)
    # import module
    while True:
        try:
            module = import_module(module_name)
            part = next(parts)
            # mmcv.ops has nms.py and nms function at the same time. So the
            # function will have a higher priority
            obj = getattr(module, part, None)
            if obj is not None and not ismodule(obj):
                break
            module_name = f'{module_name}.{part}'
        except StopIteration:
            # if obj is a module
            return module
        except ImportError:
            return None

    # get class or attribute from module
    obj = module
    while True:
        try:
            obj = getattr(obj, part)
            part = next(parts)
        except StopIteration:
            return obj
        except AttributeError:
            return None


def get_object_from_config(cfg: Dict[str, Any], **kwargs) -> Any:
    if 'class' not in cfg:
        raise ValueError('"class" not found in config.')
    cfg_ = copy.deepcopy(cfg)
    class_str = cfg_.pop("class")
    obj = get_object_from_string(class_str)
    if obj is None:
        raise ValueError(class_str)
    return obj(**cfg_, **kwargs)

