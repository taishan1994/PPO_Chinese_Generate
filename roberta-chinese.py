#pip install peft
#pip install transformers==4.28.1
#pip install accelerate
#pip install loralib
#pip install evaluate
#pip install tqdm
#pip install datasets

#!wget https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv


import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import peft


data_file = "./ChnSentiCorp_htl_all.csv" # 数据文件路径，数据需要提前下载
max_length = 86
batch_size = 64
lr = 3e-4
num_epochs = 5

# 加载数据集
dataset = load_dataset("csv", data_files=data_file)
dataset = dataset.filter(lambda x: x["review"] is not None)
datasets = dataset["train"].train_test_split(0.2, seed=123)

model_name_or_path = "hfl/chinese-roberta-wwm-ext"

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def process_function(examples):
  tokenized_examples = tokenizer(examples["review"], truncation=True, max_length=max_length)
  tokenized_examples["labels"] = examples["label"]
  return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = predictions.argmax(axis=-1)
  return accuracy_metric.compute(predictions=predictions, references=labels)


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)


model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)


optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

device = "cuda"
model.to(device)
metric = evaluate.load("accuracy")
save_dir = "./roberta-chinese/"
import time
start = time.time()
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    total_loss = 0.
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch} loss {total_loss}:", eval_metric)
    model.save_pretrained(save_dir)
end = time.time()

print("耗时：{}分钟".format((end-start) / 60))