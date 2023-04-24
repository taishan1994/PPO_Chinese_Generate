import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader

data_file = "./ChnSentiCorp_htl_all.csv"  # 数据文件路径，数据需要提前下载
max_length = 86
train_batch_size = 64
eval_batch_size = 64
num_epochs = 10
lr = 3e-4

# 加载数据集
dataset = load_dataset("csv", data_files=data_file)
dataset = dataset.filter(lambda x: x["review"] is not None)
dataset = dataset["train"].train_test_split(0.2, seed=123)

model_name_or_path = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


# example = {'label': 1, 'review': '早餐太差，无论去多少人，那边也不加食品的。酒店应该重视一下这个问题了。房间本身很好。'}


def process(example):
    text = example["review"]
    # text = ["399真的很值得之前也住过别的差不多价位的酒店式公寓没有这间好厨房很像厨房很大整个格局也都很舒服早上的早餐我订的8点半的已经冷了。。。位置啊什么还是很好的下次还会去服务也很周到"]
    batch_size = len(text)
    inputs = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)
    inputs["labels"] = []
    for i in range(batch_size):
        input_ids = inputs["input_ids"][i]
        if len(input_ids) + 1 <= max_length:
            inputs["input_ids"][i] = input_ids + [tokenizer.pad_token_id] + [0] * (max_length - len(input_ids) - 1)
            inputs["labels"].append(input_ids + [tokenizer.pad_token_id] + [-100] * (max_length - len(input_ids) - 1))
            inputs["attention_mask"][i] = [1] * len(input_ids) + [0] + [0] * (max_length - len(input_ids) - 1)
        else:
            inputs["input_ids"][i] = input_ids[:max_length - 1] + [tokenizer.pad_token_id]
            inputs["labels"].append(inputs["input_ids"][i])
            inputs["attention_mask"][i] = [1] * max_length

        inputs["token_type_ids"][i] = [0] * max_length
        # for k, v in inputs.items():
        #   print(k, len(v[0]))
        # assert len(inputs["labels"][i]) == len(inputs["input_ids"][i]) == len(inputs["token_type_ids"][i]) == len(inputs["attention_mask"][i]) == 86
    return inputs


# process(None)

train_dataset = dataset["train"].map(process, batched=True, num_proc=1, remove_columns=dataset["train"].column_names)
test_dataset = dataset["test"].map(process, batched=True, num_proc=1, remove_columns=dataset["test"].column_names)

train_dataloader = DataLoader(
    train_dataset, collate_fn=default_data_collator, shuffle=True, batch_size=train_batch_size, pin_memory=True
)

test_dataloader = DataLoader(
    test_dataset, collate_fn=default_data_collator, batch_size=eval_batch_size, pin_memory=True
)

# optimizer

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# lr scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model.cuda()

from tqdm import tqdm

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    t = tqdm(train_dataloader)
    for step, batch in enumerate(t):
        for k, v in batch.items():
            batch[k] = v.cuda()
        outputs = model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        t.set_description("loss：{:.6f}".format(loss.item()))
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    train_epoch_loss = total_loss / len(train_dataloader)
    model.save_pretrained("gpt2-chinese/")
    print(f"epoch:{epoch}/{num_epochs} loss:{train_epoch_loss}")
