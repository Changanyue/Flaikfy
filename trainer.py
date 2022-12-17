import json
import os
from torch.utils.data import DataLoader
import warnings
from pkg_resources import parse_version
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, RandomSampler
from tqdm import tqdm
import time, datetime
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.cuda.amp import autocast as autocast, GradScaler

from utils.trainer_utils import _move_dict_value_to_device, _build_args, _model_contains_inner_module, _save_model
from dataclasses import dataclass
from utils.callback_utils import CallbackManager, Callback, CallbackException
from utils.logger_utils import logger

warnings.filterwarnings("ignore")
from typing import Union, Optional, Any, List


class Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 args: Union[dataclass, Any],
                 train_data: Union[Dataset, DataLoader],
                 optimizer: Union[Optimizer, str] = None,
                 scheduler: Union[LambdaLR, str] = None,
                 dev_data: Union[Dataset, DataLoader] = None,
                 test_data: Union[Dataset, DataLoader] = None,
                 pred_data: Union[Dataset, DataLoader] = None,
                 callbacks: Union[List[Callback], Callback] = None,
                 metrics=None,
                 intervals: int = None,
                 **kwargs,
                 ):
        super(Trainer, self).__init__()

        self.args = args
        self.fp16 = self.args.fp16
        self.n_epochs = self.args.n_epochs
        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.pred_data = pred_data
        self.intervals = intervals

        self.best_metric_indicator = None
        self.best_dev_epoch = None
        self.best_dev_step = None
        self.best_dev_perf = None

        self.device = self.args.device
        self.model = model
        self.model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compute_metric = metrics

        if self.fp16:
            self.scaler = GradScaler()

        # model
        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be torch.nn.Module, got {type(model)}.")

        # 训练阶段前向传播函数
        self._forward_func = self.model.forward
        # 验证阶段前向传播函数
        self._evaluate_func = self.model.forward  # 用于匹配参数
        self._evaluate_func_wrapper = self.model.forward  # 用于调用

        # data
        if isinstance(train_data, Dataset):
            self.train_data_iterator = DataLoader(train_data, batch_size=args.train_batch_size,
                                                  sampler=RandomSampler, collate_fn=train_data.collate_fn)
        elif isinstance(train_data, DataLoader):
            self.train_data_iterator = train_data
        else:
            raise TypeError("train_data type {} not support".format(type(train_data)))

        if isinstance(dev_data, Dataset):
            self.dev_data_iterator = DataLoader(dev_data, batch_size=args.eval_batch_size
                                                , collate_fn=dev_data.collate_fn)
        elif isinstance(dev_data, DataLoader):
            self.dev_data_iterator = dev_data
        elif dev_data is None:
            pass
        else:
            raise TypeError("train_data type {} not support".format(type(dev_data)))

        if isinstance(test_data, Dataset):
            self.test_data_iterator = DataLoader(test_data, batch_size=args.eval_batch_size
                                                 , collate_fn=test_data.collate_fn)
        elif isinstance(test_data, DataLoader):
            self.test_data_iterator = test_data
        elif test_data is None:
            pass
        else:
            raise TypeError("train_data type {} not support".format(type(test_data)))

        if isinstance(pred_data, Dataset):
            self.pred_data_iterator = DataLoader(pred_data, batch_size=args.eval_batch_size
                                                 , collate_fn=pred_data.collate_fn)
        elif isinstance(pred_data, DataLoader):
            self.pred_data_iterator = pred_data
        elif pred_data is None:
            pass
        else:
            raise TypeError("train_data type {} not support".format(type(pred_data)))

        self.train_steps = (len(self.train_data_iterator) // args.gradient_accumulation_steps) * args.n_epochs

        # metric_key
        self.metric_key = self.args.metric_key[0]

        self.save_path = self.args.save_path

        # callback
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.callback_manager = CallbackManager(env={"trainer": self}, callbacks=callbacks)

    def train(self, **kwargs):
        results = {}

        try:
            self.model.train()
            start_time = time.time()

            self.step = 0
            self.epoch = 1

            try:
                self.callback_manager.on_train_begin()
                self._train()
                self.callback_manager.on_train_end()

            except BaseException as e:
                self.callback_manager.on_exception(e)
                if not isinstance(e, (CallbackException, KeyboardInterrupt)):
                    raise e

            if self.dev_data is None and self.save_path is not None:
                model_name = "_".join([self.model.__class__.__name__])
                _save_model(self.model, model_name, self.save_path, self.device, self.args.save_only_param)

        finally:
            if self.dev_data is not None and self.best_dev_perf is not None:
                logger.info(
                    "\nIn Epoch:{}/Step:{}, got best dev performance:".format(self.best_dev_epoch, self.best_dev_step))
                logger.info(self._format_eval_results(self.best_dev_perf))
                results['best_eval'] = self.best_dev_perf
                results['best_epoch'] = self.best_dev_epoch
                results['best_step'] = self.best_dev_step

        results['seconds'] = round(time.time() - start_time, 2)

        return results

    def _train(self):
        start = time.time()
        with tqdm(total=self.train_steps, postfix='loss:{0:<6.5f}', leave=True, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(self.epoch, self.n_epochs + 1):
                self.epoch = epoch
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.n_epochs))
                # early stopping
                self.callback_manager.on_epoch_begin()
                for batch in self.train_data_iterator:
                    self.step += 1
                    _move_dict_value_to_device(batch, device=self.device)

                    # edit or debud batch data
                    self.callback_manager.on_batch_begin(batch)
                    outputs = self._data_forward(self.model, batch)
                    loss = outputs.loss

                    # Is loss NaN or inf? , requires_grad = False, modify the loss
                    self.callback_manager.on_backward_begin(loss)
                    loss = loss / self.args.gradient_accumulation_steps
                    avg_loss += loss.item()

                    self._grad_backward(loss)

                    self.callback_manager.on_backward_end(batch)
                    # self._adv_training(batch)

                    self._update()
                    self.callback_manager.on_step_end()
                    self._gradient_zero_grad()

                    avg_loss = float(avg_loss)
                    end = time.time()
                    diff = round(end - start)
                    print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} time: {}".format(
                        epoch, self.step, avg_loss, diff)
                    pbar.set_postfix_str(print_output)
                    avg_loss = 0

                    self.callback_manager.on_batch_end()

                    if self.dev_data is None and self.save_path is not None and self.step % self.intervals == 0:
                        model_name = "_".join([self.model.__class__.__name__, f'Step{self.step}'])
                        _save_model(self.model, model_name, self.save_path, self.device, self.args.save_only_param)

                    if (self.intervals > 0 and self.step % self.intervals == 0) \
                            and self.dev_data is not None:
                        eval_res = self._do_validation(epoch=epoch, step=self.step)
                        eval_str = "Evaluation on dev at Epoch {}/{}. Step:{}/{}: ".format(epoch, self.n_epochs,
                                                                                           self.step,
                                                                                           self.train_steps)
                        pbar.write(eval_str + '\n')
                        logger.info(eval_str)
                        logger.info(self._format_eval_results(eval_res) + '\n')

                # lr decay; early stopping
                self.callback_manager.on_epoch_end()

            pbar.close()
            self.pbar = None
        # ============ tqdm end ============== #

    def _data_forward(self, network, x):
        x = _build_args(self._forward_func, **x)
        if self.fp16:
            with autocast():
                outputs = network(**x)
        else:
            outputs = network(**x)
        return outputs

    def _data_forward_eval(self, func, x):
        x = _build_args(func, **x)
        outputs = self._evaluate_func_wrapper(**x)
        return outputs

    def _grad_backward(self, loss):
        if self.fp16:
            # Scales loss, because the half-tensor has limited value range
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _gradient_zero_grad(self):
        if self.step % self.args.gradient_accumulation_steps == 0:
            self._clear_grad(self.optimizer)

    def _clear_grad(self, optimizer, set_to_none=True):
        param_groups = optimizer.param_groups
        for group in param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def _update(self):
        if self.fp16:
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            # 查看是否要更新scaler
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def _evaluate(self, do_metric, data_loader, use_tqdm):
        self.model.eval()
        eval_results = {}
        total_pred = []
        total_text = []
        total_res = []

        with torch.no_grad():
            start_time = time.time()
            if use_tqdm:
                data_loader = tqdm(data_loader)
            for batch in data_loader:
                _move_dict_value_to_device(batch, device=self.device)
                outputs = self._data_forward_eval(self._evaluate_func, batch)

                if do_metric:
                    res = self.compute_metric(outputs, batch)
                    total_res.append(res)
                    if not eval_results:
                        for k, v in res.items():
                            eval_results[k] = [v]
                    else:
                        for k, v in res.items():
                            eval_results[k].append(v)
                else:
                    total_pred.extend(outputs.logits)
                    if 'text' in batch:
                        total_text.extend(batch['text'])

            if do_metric:
                for k, v in eval_results.items():
                    eval_results[k] = round(sum(eval_results[k]) / len(eval_results[k]), 5)

            end_time = time.time()
            test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
            logger.info(test_str)

        if do_metric:
            return eval_results
        else:
            if total_text == []:
                return total_pred
            else:
                return zip(total_pred, total_text)

    def _do_validation(self, epoch, step):
        ema = None
        if self.args.use_ema:
            from utils.optimization_utils import ExponentialMovingAverage
            ema = ExponentialMovingAverage(self.model, 0.999)
            ema.apply_ema_weights()

        self.callback_manager.on_valid_begin()

        res = self._evaluate(do_metric=True, data_loader=self.dev_data_iterator, use_tqdm=False)

        if self.args.use_ema:
            ema.reset_old_weights()

        if self.metric_key is None:
            self.metric_key = list(res.keys())[0]

        is_better_eval = False
        _better_eval_result = self._better_eval_result(res)  # 记录了所有metric是否better
        metric_res = _better_eval_result[self.metric_key]
        if metric_res:
            if self.save_path is not None:
                _save_model(self.model, "best_" + "_".join([self.model.__class__.__name__, self.metric_key, 'fold'+str(self.args.fold)]),
                            self.save_path, self.device, self.args.save_only_param)

            self.best_dev_perf = res
            self.best_dev_epoch = epoch
            self.best_dev_step = step
            is_better_eval = True

        # get validation results; adjust optimizer
        self.callback_manager.on_valid_end(res, self.metric_key, self.optimizer, is_better_eval)
        return res

    def _better_eval_result(self, result):
        is_better = {}
        if self.best_metric_indicator is None:
            # first-time validation
            self.best_metric_indicator = result  # 记录每个metric key对应的最优val
            for _metric_key, _metric_val in result.items():
                is_better[_metric_key] = True
        else:
            for _metric_key, _metric_val in result.items():
                if _metric_key != 'loss':
                    if _metric_val > self.best_metric_indicator[_metric_key]:
                        self.best_metric_indicator[_metric_key] = _metric_val
                        is_better[_metric_key] = True
                    else:
                        is_better[_metric_key] = False
                else:
                    if _metric_val < self.best_metric_indicator[_metric_key]:
                        self.best_metric_indicator[_metric_key] = _metric_val
                        is_better[_metric_key] = True
                    else:
                        is_better[_metric_key] = False
        return is_better

    def _load_model(self, model, model_name):
        if self.save_path is not None:
            model_path = os.path.join(self.save_path, model_name)
            states = torch.load(model_path)
            if _model_contains_inner_module(model):
                model.module.load_state_dict(states)
            else:
                model.load_state_dict(states)
        else:
            return False
        return True

    def test(self, model_name):
        load_succeed = self._load_model(self.model, model_name)
        if load_succeed:
            logger.info("Successfully loaded the best model.")
        else:
            logger.info("Failed to load best model.")

        res = self._evaluate(do_metric=True, data_loader=self.test_data_iterator, use_tqdm=True)
        logger.info('Test results: ')
        for k, v in res.items():
            logger.info(f'{k}: {v}')

    def predict(self, model_name, output_file=None):
        load_succeed = self._load_model(self.model, model_name)
        if load_succeed:
            logger.info("Successfully loaded the best model.")
        else:
            logger.info("Failed to load best model.")
        result = self._evaluate(do_metric=False, data_loader=self.pred_data_iterator, use_tqdm=True)
        if output_file is not None:
            self.write_pred_to_file(result, output_file)

        return result

    def write_pred_to_file(self, res, output_file):
        import numpy as np
        f_out = open(self.args.save_path + output_file, "w", encoding="utf-8")
        _label2id = {}
        with open(self.args.data_dir + 'labels.json', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                _label2id[line['label_desc']] = line['label']

        for i, (pred, text) in enumerate(res):
            pred = pred.detach().cpu().numpy()
            pred = np.argmax(pred)
            pred_label = self.args.id2label[pred]
            f_out.write(json.dumps({'id': i, 'label': _label2id[pred_label], 'label_desc': pred_label}) + '\n')

    def _format_eval_results(self, results):
        _str = ''
        for metric_name, metric_result in results.items():
            _str += metric_name + f': {metric_result}' + ', '
        return _str[:-1]

