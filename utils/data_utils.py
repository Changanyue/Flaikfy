#! -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset
from itertools import chain
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, SequentialSampler
import random, abc
import json
import time
import dataclasses
from typing import Any, Optional, List, Tuple, Union, Dict
from dataclasses import dataclass, field, asdict
from utils.logger_utils import logger

@dataclass
class BasicArguments:
    # required parameters
    learning_rate: float = field(default=float)
    n_epochs: int = field(default=int)
    max_len: int = field(default=int)
    batch_size: int = field(default=int)
    model_path: str = field(default=str)

    save_path: Optional[str] = None
    data_dir: Optional[str] = None

    train_batch_size: int = None
    eval_batch_size: int = None
    metric_key: Union[str, List[str]] = "accuracy"
    SEED: int = 42

    # early stop
    early_stop: bool = True
    patience: int = field(
        default=8, metadata={"help": "max patience for early stopping"}
    )

    global_step: int = 0
    max_steps: int = 0

    do_train: bool = True
    do_test: bool = True
    do_predict: bool = True

    save_only_param: bool = True   # only save parameters of models

    # block shuffle
    use_block_shuffle: bool = False     # whether to use block shuffle during training
    eval_use_block_shuffle: bool = False  # whether to use block shuffle during evaluating
    batch_in_shuffle: bool = True        # see Tutorial whether to shuffle data in a batch
    sort_bs_num: Optional[int] = None   # to sort how much data among total data
    sort_key = None                     # use what function to sort the data, the default is to sort data by their length after tokenization

    # noisy fine-tune https://mp.weixin.qq.com/s/6VXlc7GCjOM7BL4tPx9Mfw
    use_noisy_tune: bool = False
    noise_lambda: float = 0.2

    # rdrop
    rdrop: bool = False
    rdrop_alpha: float = 12

    # poly loss
    add_polynominal: bool = False

    # save model of given steps
    save_step_model: bool = False

    best_metric: Optional[float] = None

    # training setting
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    use_adv: Union[bool, str] = field(
        default=False, metadata={"help": "adversarial training: pgd, fgm, False"}
    )

    use_ema: bool = False

    # gradient clip
    gradient_clip: bool = True   # whether to use gradient_clip before backward propagation
    clip_value: float = 1.0             # see tutorial
    clip_type: str = 'norm'     # see tutorial

    # optimizer and scheduler
    warmup_ratios: float = 0.1
    weight_decay: float = 1e-3
    use_lookahead: bool = False
    lr_scheduler_type: str = 'linear'

    # load previous saved checkpoint
    load_checkpoint: bool = False

    # dataloader
    num_workers = 0
    sampler = None
    test_sampler = None
    drop_last: bool = False
    train_shuffle: bool = True
    eval_shuffle: bool = False

    device: Union[torch.device, str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getstate__(self):
        return self.__dict__

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        if self.batch_size is not None and self.train_batch_size is None and self.eval_batch_size is None:
            self.train_batch_size = self.eval_batch_size = self.batch_size

        if not (self.save_path is None or isinstance(self.save_path, str)):
            raise ValueError("save_path can only be None or `str`.")

    def save_to_json(self, json_path: str):
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

    @property
    def to_dict(self):
        tmp = self
        tmp.sort_key = None  # 无法pickle module函数
        return asdict(tmp)


class DataSetGetter(Dataset):
    def __init__(self, file_name=None, args=None, data_type=None, datas=None, save=False):
        self.data_type = data_type
        self.args = args
        try:
            self.tokenizer = args.tokenizer
        except:
            pass
        self.total_labels = []

        start = time.time()
        try:
            self.datas, self.total_labels = torch.load(self.args.data_dir + file_name + '.cached_data')
        except:
            if not os.path.exists(self.args.data_dir + 'cached/' + file_name.split('.')[0] + '.cached_data'):
                if isinstance(file_name, (str, list)):
                    self.datas = self.load_data(self.args.data_dir + file_name)
                elif isinstance(datas, list):
                    self.datas = datas
                else:
                    raise ValueError('The input args shall be str format file_path / list format datas')
        end = time.time()
        logger.info(f'loading {file_name} for {round((end-start)/60,2)} minutes!')
        logger.info(f'Num samples of {file_name} is {len(self.datas)}')

        if self.labels_distribution is not None:
            logger.info(f'{file_name} label_distributions: {self.labels_distribution}')

        if save:
            torch.save((self.datas, self.total_labels), self.args.data_dir + file_name + '.cached_data')


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    def load_data(self, *args):
        raise NotImplementedError()

    def collate_fn(self, batch):
        pass

    @property
    def data(self):
        return self.datas

    @property
    def all_labels(self):
        return list(set(self.total_labels))

    @property
    def num_labels(self):
        return len(list(set(self.total_labels))) if self.total_labels is not None and self.total_labels != [] else None

    @property
    def label2id(self):
        return {l:i for i,l in enumerate(sorted(list(set(self.total_labels))))} \
            if self.total_labels is not None and self.total_labels != [] else None

    @property
    def id2label(self):
        return {i:l for i,l in enumerate(sorted(list(set(self.total_labels))))}\
            if self.total_labels is not None and self.total_labels != [] else None

    @property
    def labels_distribution(self):
        if self.total_labels is not None and self.total_labels != []:
            labels_dic = {}
            for label in self.total_labels:
                labels_dic[label] = labels_dic.get(label, 0) + 1
            total_num = sum(list(labels_dic.values()))
            label_distribution = dict((x, round((y/total_num)*100, 3)) for x, y in labels_dic.items())
            sorted_label_distribution = dict(sorted(label_distribution.items(), key=lambda x: -float(x[1])))
            final_label_distribution = {k: str(v) + '%' for k, v in sorted_label_distribution.items()}
            return final_label_distribution
        else:
            return None


class BatchIter(DataLoader):
    def __init__(self,
                 dataset: Union[DataSetGetter, Dataset],
                 args,
                 sort_bs_num=None,
                 sort_key=None,
                 use_block_shuffle: bool = False,
                 batch_in_shuffle: bool = False,
                 batch_size=1, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None,
                 batch_sampler=None, shuffle=False,
                 **kwargs,
                 ):
        batch_sampler = batch_sampler
        if batch_sampler is not None:
            kwargs['batch_size'] = 1
            kwargs['sampler'] = None
            kwargs['drop_last'] = False

        super().__init__(dataset=dataset, batch_size=batch_size, sampler=sampler,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn,
            batch_sampler=batch_sampler, shuffle=shuffle)

        assert len(dataset) > 0, 'dataset cannot be None'
        assert isinstance(dataset.datas, list), "the data attribute of DatasetGetter object must be a list"

        self.use_block_shuffle = use_block_shuffle
        self.sort_bs_num = sort_bs_num
        self.sort_key = sort_key
        if self.sort_key is None and args.use_block_shuffle:
            self.sort_key = lambda x: len(args.tokenizer.tokenize(x[0]))  # x[0] is text in Dataset

        self.batch_in_is_shuffle = batch_in_shuffle
        self.num_labels = dataset.num_labels
        self.id2label = dataset.id2label
        self.label2id = dataset.label2id

    def __iter__(self):
        if self.use_block_shuffle is False:
            if self.num_workers == 0:
                return _SingleProcessDataLoaderIter(self)
            else:
                return _MultiProcessingDataLoaderIter(self)

        if self.use_block_shuffle is True:
            # self.dataset is the attribute in torch DataLoader
            self.dataset.datas = self.block_shuffle(self.dataset.datas, self.batch_size, self.sort_bs_num,
                                                       self.sort_key, self.batch_in_is_shuffle)
            if self.num_workers == 0:
                return _SingleProcessDataLoaderIter(self)
            else:
                return _MultiProcessingDataLoaderIter(self)

    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, batch_in_shuffle):
        random.shuffle(data)
        # 将数据按照batch_size大小进行切分
        tail_data = [] if len(data) % batch_size == 0 else data[-(len(data) % batch_size):]
        data = data[:len(data) - len(tail_data)]
        assert len(data) % batch_size == 0
        # 获取真实排序范围
        sort_bs_num = len(data) // batch_size if sort_bs_num is None else sort_bs_num
        # 按照排序范围进行数据划分
        data = [data[i:i + sort_bs_num * batch_size] for i in range(0, len(data), sort_bs_num * batch_size)]

        # 在排序范围，根据排序函数进行降序排列
        data = [sorted(i, key=sort_key, reverse=True) for i in data]

        # 将数据根据batch_size获取batch_data
        data = list(chain(*data))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        # 判断是否需要对batch_data序列进行打乱
        if batch_in_shuffle:
            random.shuffle(data)
        # 将tail_data填补回去
        data = list(chain(*data)) + tail_data
        return data





