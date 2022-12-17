
import torch
import argparse
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader

from utils.logger_utils import logger
from utils.trainer_utils import set_seed

from trainer import Trainer
from config import args
from util import build_optimizer, compute_metric

from utils.data_utils import DataSetGetter
from transformers import AutoTokenizer

print(args)



tokenizer = AutoTokenizer.from_pretrained(args.model_path)

class MyDataset(DataSetGetter):
    def load_data(self, filename):
        D = []
        with open(filename, encoding='utf8') as f:
            for i, line in enumerate(f):
                cache = line.strip('\n').split('|||||')
                if args.mode == 'full_code':
                    content, label = cache[0], cache[-1]
                elif args.mode == 'final_code':
                    content, label = cache[1], cache[-1]
                else:
                    raise TypeError
                self.total_labels.append(label)
                D.append((content, int(label)))
        return D

    def collate_fn(self, batch):
        texts, labels = [_[0] for _ in batch], [_[1] for _ in batch]
        encodings = tokenizer(texts, max_length=args.max_len, padding='max_length', return_tensors='pt',
                              truncation=True)
        encodings['labels'] = torch.tensor(labels).long()
        return encodings

# data
train_dataset = MyDataset('train.csv', args=args)
dev_dataset = MyDataset('dev.csv', args=args)
test_dataset = MyDataset('test.csv', args=args)
args.num_labels, args.label2id, args.id2label = train_dataset.num_labels, train_dataset.label2id, train_dataset.id2label

# data loader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

# model
logger.info(f'Loading model and optimizer...')
set_seed(args)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels)
train_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.n_epochs

# optimizer
optimizer, scheduler = build_optimizer(model, train_steps, args)


trainer = Trainer(model, args, train_loader,
                  optimizer=optimizer, scheduler=scheduler,
                  dev_data=dev_loader, test_data=test_loader,
                  intervals=100, metrics=compute_metric)

if __name__ == "__main__":
    if args.do_train:
        trainer.train()
    if args.do_test:
        trainer.test(f'best_RobertaForSequenceClassification_loss_fold{opt.fold}')


# fold1 2022/11/30 17:39:30 - Test results:
# 2022/11/30 17:39:30 - f1: 0.85084
# 2022/11/30 17:39:30 - accuracy: 0.97059
# 2022/11/30 17:39:30 - loss: 0.1222
# 0.89196 0.90555 0.88738


