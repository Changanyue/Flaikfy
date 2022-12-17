from typing import Callable, Iterable, Optional, Tuple, Union
import math
from collections import defaultdict
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610

    optimizer = Lookahead(optimizer, la_steps=args.la_steps, la_alpha=args.la_alpha)
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss

class ExponentialMovingAverage():
    '''
        模型权重的指数滑动平均
        注意区别于类似adam一类的自适应学习率优化器，针对一阶二阶梯度的指数滑动平均，两者完全不同

        例子:
            # 初始化
            ema = ExponentialMovingAverage(model, 0.999)

            # 训练过程中，更新完参数后，同步update ema_weights weights
            def train():
                optimizer.step()
                ema.update()

            # eval前，调用apply_ema_weights weights；eval之后，恢复原来模型的参数
            def evaluate():
                ema.apply_ema_weights()
                # evaluate
                # 如果想保存ema后的模型，请在reset_old_weights方法之前调用torch.save()
                ema.reset_old_weights()
    '''

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        # 保存ema权重（当前step的每一层的滑动平均权重）
        self.ema_weights = {}
        # 在进行evaluate的时候，保存原始的模型权重，当执行完evaluate后，从ema权重恢复到原始权重
        self.model_weights = {}

        # 初始化ema_weights为model_weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_weights[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weights
                new_average = (1.0 - self.decay) * param.data + self.decay * self.ema_weights[name]
                self.ema_weights[name] = new_average.clone()

    def apply_ema_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weights
                self.model_weights[name] = param.data
                param.data = self.ema_weights[name]

    def reset_old_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.model_weights
                param.data = self.model_weights[name]
        self.model_weights = {}



