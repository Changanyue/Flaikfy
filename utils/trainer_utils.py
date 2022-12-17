import os
import torch
import inspect
import torch.nn as nn
import random
import numpy as np

def set_seed(args):
    seed = args.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_contains_inner_module(model):
    r"""

    :param nn.Module model: 模型文件，判断是否内部包含model.module, 多用于check模型是否是nn.DataParallel,
        nn.parallel.DistributedDataParallel。主要是在做形参匹配的时候需要使用最内部的model的function。
    :return: bool
    """
    if isinstance(model, nn.Module):
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return True
    return False


def _get_model_device(model):
    r"""
    传入一个nn.Module的模型，获取它所在的device

    :param model: nn.Module
    :return: torch.device,None 如果返回值为None，说明这个模型没有任何参数。
    """
    # TODO 这个函数存在一定的风险，因为同一个模型可能存在某些parameter不在显卡中，比如BertEmbedding. 或者跨显卡
    assert isinstance(model, nn.Module)

    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device


def _save_model(model, model_name, save_dir, device=None, only_param=False):
    r""" 存储不含有显卡信息的state_dict或model
    :param model:
    :param model_name:
    :param save_dir: 保存的directory
    :param only_param:
    :return:
    """
    if device is None:
        device = _get_model_device(model)
    model_path = os.path.join(save_dir, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if _model_contains_inner_module(model):
        model = model.module
    if only_param:
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, model_path)
    else:
        model.cpu()
        torch.save(model, model_path)
        model.to(device)


def _move_dict_value_to_device(*args, device: torch.device, non_blocking=False):
    r"""

    move data to model's device, element in *args should be dict. This is a inplace change.
    :param device: torch.device
    :param non_blocking: bool, 是否异步将数据转移到cpu, 需要tensor使用pin_memory()
    :param args:
    :return:
    """
    if not torch.cuda.is_available() or device is None:
        return

    if not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device`, got `{type(device)}`")

    for arg in args:
        import transformers
        if isinstance(arg, dict) or isinstance(arg, transformers.tokenization_utils_base.BatchEncoding):
            for key, value in arg.items():
                if isinstance(value, torch.Tensor):
                    arg[key] = value.to(device, non_blocking=non_blocking)
        else:
            raise TypeError(f"Only support `dict` or `transformers.tokenization_utils_base.BatchEncoding type` right now got type {type(arg)}")

def _build_args(func, **kwargs):
    r"""
    根据func的初始化参数，从kwargs中选择func需要的参数

    :param func: callable
    :param kwargs: 参数
    :return:dict. func中用到的参数
    """
    spect = inspect.getfullargspec(func)
    if spect.varkw is not None:
        return kwargs
    needed_args = set(spect.args)
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    output = {name: default for name, default in zip(spect.args[start_idx:], defaults)}
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output