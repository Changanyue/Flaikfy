
import evaluate
from transformers import AdamW, get_scheduler, BertModel
import transformers
transformers.logging.set_verbosity_error()

def build_optimizer(model, train_steps, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    # adam = bnb.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, optim_bits=8)
    scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=train_steps * args.warmup_ratios,
                                     num_training_steps=train_steps)
    return optimizer, scheduler

def compute_metric(pred, batch):
    results = {}
    logits = pred.logits.argmax(-1).tolist()
    labels = batch['labels'].tolist()
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    # https://huggingface.co/spaces/evaluate-metric/f1
    f1_res = f1_metric.compute(references=labels, predictions=logits, average="macro")
    acc_res = acc_metric.compute(references=labels, predictions=logits)
    results.update(f1_res)
    results.update(acc_res)
    results.update({'loss':pred.loss.item()})
    return results