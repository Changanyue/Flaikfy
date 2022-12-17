import os
import sys
import torch
from copy import deepcopy
from torch import nn

try:
    import fitlog
except:
    pass

from torch.utils.data import Dataset

from utils.modeling_utils import PGD, FGM
from utils.trainer_utils import _save_model, _model_contains_inner_module
from utils.logger_utils import logger


class Callback(object):
    r"""
    Callback是fastNLP中被设计用于增强 :class:`~fastNLP.Trainer` 的类。
    如果Callback被传递给了 Trainer , 则 Trainer 会在对应的阶段调用Callback的函数，
    具体调用时机可以通过 :mod:`trainer 模块<fastNLP.core.trainer>` 查看。
    这是Callback的基类，所有的callback必须继承自这个类
    """

    def __init__(self):
        super(Callback, self).__init__()
        self._trainer = None  # 在Trainer内部被重新赋值
        self._disabled = False

    def __repr__(self):
        return self.__class__.__name__

    @property
    def trainer(self):
        r"""
        该属性可以通过self.trainer获取到，一般情况下不需要使用这个属性。
        """
        return self._trainer

    @property
    def grad_scaler(self):
        r"""
        float16的gradient scaler
        """
        return self._trainer.grad_scaler

    @property
    def auto_cast(self):
        r"""
        float16用的auto cast环境
        """
        return self._trainer.auto_cast

    @property
    def step(self):
        r"""当前运行到的step, 范围为[1, self.n_steps+1)"""
        return self._trainer.step

    @property
    def n_steps(self):
        r"""Trainer一共会采多少个batch。当Trainer中update_every设置为非1的值时，该值不等于update的次数"""
        return self._trainer.n_steps

    @property
    def batch_size(self):
        r"""train和evaluate时的batch_size为多大"""
        return self._trainer.batch_size

    @property
    def epoch(self):
        r"""当前运行的epoch数，范围是[1, self.n_epochs+1)"""
        return self._trainer.epoch

    @property
    def n_epochs(self):
        r"""一共会运行多少个epoch"""
        return self._trainer.n_epochs

    @property
    def optimizer(self):
        r"""初始化Trainer时传递的Optimizer"""
        return self._trainer.optimizer

    @property
    def model(self):
        r"""正在被Trainer训练的模型"""
        return self._trainer.model

    @property
    def pbar(self):
        r"""如果在Callback中需要打印内容，请使用self.pbar.write(str)。否则可能出现命令行显示效果不太好的问题。在
        on_train_begin(), on_train_end(), on_exception()中请不要使用该属性，通过print输出即可。"""
        return self._trainer.pbar

    @property
    def batch_per_epoch(self):
        r"""每个epoch一共有多少个batch，只有在on_epoch_begin之后才能调用该属性。"""
        return self._trainer.batch_per_epoch

    @property
    def args(self):
        return self._trainer.args

    @property
    def disabled(self):
        return self._disabled

    @property
    def logger(self):
        return getattr(self._trainer, 'logger', logger)

    def on_train_begin(self):
        r"""
        在Train过程开始之前调用。

        :return:
        """
        pass

    def on_epoch_begin(self):
        r"""
        在每个epoch开始之前调用一次

        :return:
        """
        pass

    def on_batch_begin(self, batch):
        r"""
        每次采集到一个batch的数据则调用一次。

        :param list(int) indices: 这次采样使用到的indices，可以通过DataSet[indices]获取出这个batch采出的Instance，在一些
            情况下可以帮助定位是哪个Sample导致了错误。仅当num_workers=0时有效。
        :return:
        """
        pass

    def on_backward_begin(self, loss):
        r"""
        在loss得到之后，但在反向传播之前。可能可以进行loss是否为NaN的检查。

        :param torch.Tensor loss: 计算得到的loss值
        :return:
        """
        pass

    def on_backward_end(self, batch):
        r"""
        反向梯度传播已完成，但由于update_every的设置，可能并不是每一次调用都有梯度。到这一步，还没有更新参数。

        :return:
        """
        pass

    def on_step_end(self):
        r"""
        到这里模型的参数已经按照梯度更新。但可能受update_every影响，并不是每次都更新了。

        :return:
        """
        pass

    def on_batch_end(self):
        r"""
        这一步与on_step_end是紧接着的。只是为了对称性加上了这一步。

        """
        pass

    def on_valid_begin(self):
        r"""
        如果Trainer中设置了验证，则发生验证前会调用该函数

        :return:
        """
        pass

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        r"""
        每次执行验证集的evaluation后会调用。

        :param Dict[str: Dict[str: float]] eval_result: , evaluation的结果。一个例子为{'AccuracyMetric':{'acc':1.0}}，即
            传入的dict是有两层，第一层是metric的名称，第二层是metric的具体指标。
        :param str metric_key: 初始化Trainer时传入的metric_key。
        :param torch.Optimizer optimizer: Trainer中使用的优化器。
        :param bool is_better_eval: 当前dev结果是否比之前的好。
        :return:
        """
        pass

    def on_epoch_end(self):
        r"""
        每个epoch结束将会调用该方法
        """
        pass

    def on_train_end(self):
        r"""
        训练结束，调用该方法
        """
        pass

    def on_exception(self, exception):
        r"""
        当训练过程出现异常，会触发该方法
        :param exception: 某种类型的Exception，比如KeyboardInterrupt等
        """
        pass


def _transfer(func):
    r"""装饰器，将对CallbackManager的调用转发到各个Callback子类.

    :param func:
    :return:
    """

    def wrapper(manager, *arg):
        returns = []
        for callback in manager.callbacks:
            if callback.disabled:
                continue
            returns.append(getattr(callback, func.__name__)(*arg))
        return returns

    return wrapper


class CallbackManager(Callback):
    r"""
    内部使用的Callback管理类
    """

    def __init__(self, env, callbacks=None):
        r"""

        :param dict env: The key is the name of the Trainer attribute(str). The value is the attribute itself.
        :param List[Callback] callbacks:
        """
        super(CallbackManager, self).__init__()
        # set attribute of trainer environment
        self._env = env
        assert 'trainer' in env
        self._trainer = env['trainer']
        self.callbacks = []
        self.callbacks = self.prepare_callbacks(callbacks)

    def add_callback(self, cb):
        self.callbacks.extend(self.prepare_callbacks(cb))

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return " ".join(cb.__class__.__name__ for cb in self.callbacks)

    def prepare_callbacks(self, callbacks):
        total_callbacks = []
        if self.args.gradient_clip:
            total_callbacks.append(GradientClipCallback())
        if self.args.use_adv is not False:
            total_callbacks.append(AdversarialTrainingCallback())
        # if self.args.use_ema:
        #     total_callbacks.append(ExponentialMovingAverageCallback())
        if self.args.load_checkpoint:
            total_callbacks.append(CheckPointCallback(self.trainer.save_path))
        if self.args.save_step_model:
            total_callbacks.append(SaveModelCallback(self.trainer.save_path, self.trainer.metric_key_for_early_stop))
        if self.args.early_stop:
            total_callbacks.append(EarlyStopCallback(self.args.patience))

        if isinstance(callbacks, list):
            if all([isinstance(cb, Callback) for cb in callbacks]) is True:
                total_callbacks.extend(callbacks)
            else:
                obj = [not isinstance(cb, Callback) for cb in callbacks][0]
                raise TypeError(f"Expect sub-classes of Callback. Got {type(obj)}")
        elif isinstance(callbacks, Callback):
            total_callbacks.append(callbacks)
        elif callbacks is None:
            pass
        else:
            raise TypeError(f"Expect callbacks in CallbackManager(callbacks) to be list Callback. Got {type(callbacks)}.")

        for env_name, env_val in self._env.items():
            for callback in total_callbacks:
                setattr(callback, '_' + env_name, env_val)  # Callback.trainer
        return total_callbacks

    @_transfer
    def on_train_begin(self):
        pass

    @_transfer
    def on_epoch_begin(self):
        pass

    @_transfer
    def on_batch_begin(self, batch):
        pass

    @_transfer
    def on_backward_begin(self, loss):
        pass

    @_transfer
    def on_backward_end(self, batch):
        pass

    @_transfer
    def on_step_end(self):
        pass

    @_transfer
    def on_batch_end(self):
        pass

    @_transfer
    def on_valid_begin(self):
        pass

    @_transfer
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        pass

    @_transfer
    def on_validation(self):
        pass

    @_transfer
    def on_epoch_end(self):
        pass

    @_transfer
    def on_train_end(self):
        pass

    @_transfer
    def on_exception(self, exception):
        pass


class GradientClipCallback(Callback):
    r"""
    clip the gradient of parameter to a given range before backward propagation
    """

    def __init__(self, parameters=None):
        r"""
        :param None,torch.Tensor,List[torch.Tensor] parameters: 一般通过model.parameters()获得。
            如果为None则默认对Trainer的model中所有参数进行clip
        :param float clip_value: 将gradient 限制到[-clip_value, clip_value]。clip_value应该为正数
        :param str clip_type: 支持'norm', 'value'
            两种::
                1 'norm', 将gradient的norm rescale到[-clip_value, clip_value]

                2 'value', 将gradient限制在[-clip_value, clip_value],
                    小于-clip_value的gradient被赋值为-clip_value;
                    大于clip_value的gradient被赋值为clip_value.
        """
        super().__init__()

        if parameters is not None:
            self.parameters = list(parameters)
        else:
            self.parameters = None

    def on_backward_end(self, batch):
        if self.trainer.args.clip_type == 'norm':
            clip_fun = nn.utils.clip_grad_norm_
        elif self.trainer.args.clip_type == 'value':
            clip_fun = nn.utils.clip_grad_value_
        else:
            raise ValueError("Only supports `norm` or `value` right now.")
        clip_value = self.trainer.args.clip_value

        if self.step % self.trainer.gradient_accumulation_steps == 0:
            if self.parameters is not None:
                clip_fun(self.parameters, clip_value)
            else:
                clip_fun(self.model.parameters(), clip_value)


class RdropCallback(Callback):
    r"""
    R-drop loss
    """

    def __init__(self, parameters=None):
        r"""
        :param None,torch.Tensor,List[torch.Tensor] parameters: 一般通过model.parameters()获得。
            如果为None则默认对Trainer的model中所有参数进行clip
        :param float clip_value: 将gradient 限制到[-clip_value, clip_value]。clip_value应该为正数
        :param str clip_type: 支持'norm', 'value'
            两种::
                1 'norm', 将gradient的norm rescale到[-clip_value, clip_value]

                2 'value', 将gradient限制在[-clip_value, clip_value],
                    小于-clip_value的gradient被赋值为-clip_value;
                    大于clip_value的gradient被赋值为clip_value.
        """
        super().__init__()

        if parameters is not None:
            self.parameters = list(parameters)
        else:
            self.parameters = None

    def on_backward_end(self, batch):
        if self.trainer.args.clip_type == 'norm':
            clip_fun = nn.utils.clip_grad_norm_
        elif self.trainer.args.clip_type == 'value':
            clip_fun = nn.utils.clip_grad_value_
        else:
            raise ValueError("Only supports `norm` or `value` right now.")
        clip_value = self.trainer.args.clip_value

        if self.step % self.trainer.gradient_accumulation_steps == 0:
            if self.parameters is not None:
                clip_fun(self.parameters, clip_value)
            else:
                clip_fun(self.model.parameters(), clip_value)


class AdversarialTrainingCallback(Callback):
    def __init__(self, pgd_k=3):
        super().__init__()
        self.pgd_k = pgd_k

    def on_backward_end(self, batch):
        if self.trainer.args.use_adv == 'pgd' or self.trainer.args.use_adv:
            pgd = PGD(model=self.model)
            pgd.backup_grad()
            for _t in range(self.pgd_k):
                pgd.attack(is_first_attack=(_t == 0))
                if _t != self.pgd_k - 1:
                    self.model.zero_grad()
                else:
                    pgd.restore_grad()

                outputs_adv = self.trainer._data_forward(self.model, batch)
                loss_adv = outputs_adv.loss
                self.trainer._grad_backward(loss_adv)
            pgd.restore()

        if self.trainer.args.use_adv == 'fgm':
            fgm = FGM(model=self.model)
            fgm.attack()

            outputs_adv = self.trainer._data_forward(self.model, batch)
            loss_adv = outputs_adv.loss
            self.trainer._grad_backward(loss_adv)
            fgm.restore()


class EarlyStopCallback(Callback):
    r"""
    多少个epoch没有变好就停止训练，相关类 :class:`~fastNLP.core.callback.EarlyStopError`
    """
    def __init__(self, patience):
        r"""

        :param int patience: epoch的数量
        """
        super(EarlyStopCallback, self).__init__()
        self.patience = patience
        self.wait = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if not is_better_eval:
            # current result is getting worse
            if self.wait == self.patience:
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0

    def on_exception(self, exception):
        if isinstance(exception, EarlyStopError):
            logger.info("Early Stopping triggered in epoch {}!".format(self.epoch))
        else:
            raise exception  # 抛出陌生Error




class SmoothValue(object):
    r"""work for LRFinder"""

    def __init__(self, beta: float):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.smooth = None

    def add_value(self, val: float) -> None:
        r"""Add `val` to calculate updated smoothed value."""
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)




class CheckPointCallback(Callback):
    def __init__(self, save_path, delete_when_train_finish=True, recovery_fitlog=True):
        r"""
        用于在每个epoch结束的时候保存一下当前的Trainer状态，可以用于恢复之前的运行。使用最近的一个epoch继续训练
        一段示例代码
        Example1::

            >>> callback = CheckPointCallback('chkp.pt')
            >>> trainer = Trainer(xxx, callback=callback)
            >>> trainer.train()  # 如果训练过程没结束就fail，请直接再次运行即可（请务必保证与上次使用了完全相同的数据与超参数）

        Example2::

            >>> fitlog.set_log_dir('xxx')
            >>> callback = CheckPointCallback('chkp.pt')  # 一定要在set_log_dir下一行就接着CheckPointCallback
            >>> trainer = Trainer(xxx, callback=callback)
            >>> trainer.train()  # 如果训练过程没结束就fail，请直接再次运行即可（请务必保证与上次使用了完全相同的数据与超参数）

        :param str save_path: 将状态保存到哪个位置。需要指定一个具体的路径，比如'checkpoints/chtp.pt'。如果检查到该文件存在，将在
            Trainer开始训练的时候自动从这个Checkpoint处开始运行。
        :param bool delete_when_train_finish: 如果Train正常运行完毕，是否自动删除。删除该文件可以使得路径自动复用。
        :param bool recovery_fitlog: 是否恢复fitlog为对应的log，如果为True请将本Callback放在fitlog.set_log_dir后面一行初始化。
            如果为False，将新建一个log folder否则继续使用之前的。
        """
        super().__init__()
        self.save_path = os.path.abspath(os.path.expanduser(save_path))
        self.delete_when_train_finish = delete_when_train_finish
        self.recover_fitlog = recovery_fitlog
        try:
            import fitlog
        except:
            self.recover_fitlog = False
        if os.path.exists(os.path.expanduser(self.save_path)):
            logger.info("The train will start from the checkpoint saved in {}.".format(self.save_path))
            if self.recover_fitlog:
                states = torch.load(self.save_path)
                if 'fitlog_log_dir' in states:
                    try:
                        import fitlog
                        log_dir = states['fitlog_log_dir']
                        if 'fitlog_save_log_dir' in states:
                            log_dir = states['fitlog_save_log_dir']
                        fitlog.set_log_dir(log_dir, new_log=True)
                    except:
                        logger.error("Fail to recovery the fitlog states.")

    def on_train_begin(self):
        r"""
        当train开始时，且需要恢复上次训练时，会做以下的操作
            (1) 重新加载model权重
            (2) 重新加载optimizer的状态
            (3) 加载当前epoch数
            (4) 加载当前最佳evaluate的性能
            (5) (optional) 自动将fitlog设置到上次log出继续

        :return:
        """
        if os.path.exists(os.path.expanduser(self.save_path)):
            states = torch.load(self.save_path)
            model = self.model
            if _model_contains_inner_module(model):
                model = model.module
            model.load_state_dict(states['model'])
            self.optimizer.load_state_dict(states['optimizer'])
            if 'grad_scaler' in states:
                self.grad_scaler.load_state_dict(states['grad_scaler'])
            self.trainer.epoch = states['epoch'] + 1  # 因为是结束储存的，所以需要从下一个epoch开始
            self.trainer.step = states['step']
            if 'best_dev_epoch' in states:
                self.trainer.best_dev_perf = states['best_dev_perf']
                self.trainer.best_dev_epoch = states['best_dev_epoch']
                self.trainer.best_dev_step = states['best_dev_step']
                self.trainer.best_metric_indicator = states['best_metric_indicator']
            logger.info("Load checkpoint from {}".format(os.path.expanduser(self.save_path)))

    def on_epoch_end(self):
        r"""
        保存状态，使得结果可以被恢复

        :param self:
        :return:
        """
        states = {}
        model = self.model
        if _model_contains_inner_module(model):
            model = model.module
        states['model'] = {name: param.cpu() for name, param in model.state_dict().items()}
        states['optimizer'] = self.optimizer.state_dict()
        states['grad_scaler'] = self.grad_scaler.state_dict()
        states['epoch'] = self.epoch
        states['step'] = self.step
        if self.trainer.best_dev_epoch is not None:
            states['best_dev_epoch'] = self.trainer.best_dev_epoch
            states['best_dev_perf'] = self.trainer.best_dev_perf
            states['best_dev_step'] = self.trainer.best_dev_step
            states['best_metric_indicator'] = self.trainer.best_metric_indicator
        if self.recover_fitlog:
            try:
                import fitlog
                if fitlog._logger._log_dir is not None:
                    states['fitlog_log_dir'] = fitlog._logger._log_dir
                if fitlog._logger._save_log_dir is not None:
                    states['fitlog_save_log_dir'] = fitlog._logger._save_log_dir
            except:
                pass
        torch.save(states, self.save_path)
        logger.debug("Checkpoint:{} has been saved in epoch:{}.".format(self.save_path, self.epoch))

    def on_train_end(self):
        # 训练结束，根据情况删除保存的内容
        if self.delete_when_train_finish:
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
                logger.debug("Checkpoint:{} has been removed.".format(self.save_path))


# class ExponentialMovingAverageCallback(Callback):
#     def __init__(self, decay=0.999):
#         super().__init__()
#         self.decay = decay
#
#     def on_step_end(self):
#         ema = ExponentialMovingAverage(self.model, self.decay)
#         ema.update()
#
#     def on_valid_begin(self):
#         ema = ExponentialMovingAverage(self.model, self.decay)
#         ema.apply_ema_weights()
#
#     def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
#         ema = ExponentialMovingAverage(self.model, self.decay)
#         ema.reset_old_weights()


class SaveModelCallback(Callback):
    r"""
    由于Trainer在训练过程中只会保存最佳的模型， 该callback可实现多种方式的结果存储。
    会根据训练开始的时间戳在save_dir下建立文件夹，再在文件夹下存放多个模型::

        -save_dir
            -2019-07-03-15-06-36
                -epoch:0_step:20_{metric_key}:{evaluate_performance}.pt   # metric是给定的metric_key, evaluate_performance是性能
                -epoch:1_step:40_{metric_key}:{evaluate_performance}.pt
            -2019-07-03-15-10-00
                -epoch:0_step:20_{metric_key}:{evaluate_performance}.pt   # metric是给定的metric_key, evaluate_perfomance是性能
    """

    def __init__(self, save_dir, metric_key: str, top=3, only_param=False, save_on_exception=False):
        super().__init__()
        self.metric_key = metric_key
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        if top < 0:
            self.top = sys.maxsize
        else:
            self.top = top
        self._ordered_save_models = []  # List[Tuple], Tuple[0]是metric， Tuple[1]是path。metric是依次变好的，所以从头删

        self.only_param = only_param
        self.save_on_exception = save_on_exception

    def on_train_begin(self):
        self.save_dir = os.path.join(self.save_dir, self.trainer.start_time)

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        metric_value = list(eval_result.values())[0][metric_key]
        self._save_this_model(metric_value)

    def _insert_into_ordered_save_models(self, pair):
        # pair:(metric_value, model_name)
        # 返回save的模型pair与删除的模型pair. pair中第一个元素是metric的值，第二个元素是模型的名称
        index = -1
        for _pair in self._ordered_save_models:
            if _pair[0] >= pair[0] and self.trainer.increase_better:
                break
            if not self.trainer.increase_better and _pair[0] <= pair[0]:
                break
            index += 1
        save_pair = None
        if len(self._ordered_save_models) < self.top or (len(self._ordered_save_models) >= self.top and index != -1):
            save_pair = pair
            self._ordered_save_models.insert(index + 1, pair)
        delete_pair = None
        if len(self._ordered_save_models) > self.top:
            delete_pair = self._ordered_save_models.pop(0)
        return save_pair, delete_pair

    def _save_this_model(self, metric_value):
        name = "epoch-{}_step-{}_{}-{:.6f}.pt".format(self.epoch, self.step, self.metric_key, metric_value)
        save_pair, delete_pair = self._insert_into_ordered_save_models((metric_value, name))
        if save_pair:
            try:
                _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)
            except Exception as e:
                logger.error(f"The following exception:{e} happens when save model to {self.save_dir}.")
        if delete_pair:
            try:
                delete_model_path = os.path.join(self.save_dir, delete_pair[1])
                if os.path.exists(delete_model_path):
                    os.remove(delete_model_path)
            except Exception as e:
                logger.error(f"Fail to delete model {name} at {self.save_dir} caused by exception:{e}.")

    def on_exception(self, exception):
        if self.save_on_exception:
            name = "epoch-{}_step-{}_Exception-{}.pt".format(self.epoch, self.step, exception.__class__.__name__)
            _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)



class CallbackException(BaseException):
    r"""
   当需要通过callback跳出训练的时候可以通过抛出CallbackException并在on_exception中捕获这个值。
   """

    def __init__(self, msg):
        r"""

        :param str msg: Exception的信息。
        """
        super(CallbackException, self).__init__(msg)


class EarlyStopError(CallbackException):
    r"""
    用于EarlyStop时从Trainer训练循环中跳出。

    """

    def __init__(self, msg):
        super(EarlyStopError, self).__init__(msg)