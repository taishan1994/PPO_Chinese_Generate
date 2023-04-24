import random
import torch
import wandb
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import choices
from tqdm import tqdm

tqdm.pandas()

from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

seed = 123
np.random.seed(seed)

config = PPOConfig(
    model_name="./gpt2-chinese", # 使用训练好的模型
    steps=51200, 
    learning_rate=1.41e-5, 
    remove_unused_columns=False, 
    log_with="wandb"
)

txt_in_len = 5
txt_out_len = 20


# ==================================
# 定义模型
gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
gpt2_model_ref = create_reference_model(gpt2_model)
# 使用原来模型的tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

gpt2_tokenizer.eos_token = gpt2_tokenizer.pad_token
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
# ==================================

# ==================================
# 加载数据
data_file = "./ChnSentiCorp_htl_all.csv"
dataset = load_dataset("csv", data_files=data_file)
dataset = dataset.filter(lambda x: x["review"] is not None)
dataset = dataset["train"]
print(dataset)

def tokenize(sample):
  sample["input_ids"] = gpt2_tokenizer.encode(sample["review"], add_special_tokens=False)[:txt_in_len]
  sample["query"] = "".join(gpt2_tokenizer.decode(sample["input_ids"]).split(" "))
  return sample

dataset = dataset.map(tokenize, batched=False)
print(dataset)
# 将指定的列名转换为torch的格式
dataset.set_format(type="torch", columns=["input_ids", "label"], output_all_columns=True)

# ==================================


# ==================================
# PPO主代码
def collator(data):
  # 构建batch数据
  return dict((key, [d[key] for d in data]) for key in data[0])

ppo_trainer = PPOTrainer(
    config, 
    gpt2_model, 
    gpt2_model_ref, 
    gpt2_tokenizer, 
    dataset, 
    data_collator=collator)

if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
else:
    device = ppo_trainer.accelerator.device

# 构建reward模型
sentiment_pipe = pipeline(
    "sentiment-analysis", 
    "./roberta-chinese", 
    tokenizer=gpt2_tokenizer,
    device=device)

# 提取出正面的分数
def extract_pipe_output(outputs):
  positive_logits = []
  for out in outputs:
    for element in out:
      if element["label"] == "LABEL_1":
        positive_logits.append(torch.tensor(element["score"]))
  return positive_logits

# 加入prompt
ctrl_str = ["正面：", "负面："]
ctrl_tokens = dict((s, gpt2_tokenizer.encode(s, add_special_tokens=False, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)
# ctrl_tokens

# 定义生成模型参数
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
    "max_new_tokens": txt_out_len,
    "eos_token_id": gpt2_tokenizer.eos_token_id,
}


# 定义奖励模型参数
sentiment_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}

def pos_logit_to_reward(logit, task):
    """如果prompt是正面，则奖励为正，否则，奖励为负
    """
    for i in range(len(logit)):
        if task[i] == "负面：":
            logit[i] = -logit[i]   
        else:
            pass
    return logit

for epoch in range(2):
    for batch in tqdm(ppo_trainer.dataloader):
        logs, game_data, = (
            dict(),
            dict(),
        )

        #### 为每一个样本随机选一个prompt
        task_list = choices(ctrl_str, k=config.batch_size)
        game_data["query"] = [t + q for t, q in zip(task_list, batch["query"])]
        query_tensors = [torch.cat((ctrl_tokens[t], input_ids)) for t, input_ids in zip(task_list, batch["input_ids"])]

        #### 使用GPT2生成结果
        response_tensors = []
        for query in query_tensors:
            query_length = len(query)
            response = ppo_trainer.generate(query, **generation_kwargs)
            # print(gpt2_tokenizer.decode(query))
            # print(gpt2_tokenizer.decode(response.squeeze()[query_length:txt_out_len]))
            # 这里使用的模型从前往后解码
            response_tensors.append(response.squeeze()[query_length:txt_out_len])

        game_data["response"] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### 使用奖励模型对输出结果进行评分
        texts = [q + "".join(r.split(" ")) for q, r in zip(batch["query"], game_data["response"])]
        # print(texts[0])
        # 提取出LABEL_1(正面)对应的分数
        logits = extract_pipe_output(sentiment_pipe(texts, **sentiment_pipe_kwargs))
        rewards = pos_logit_to_reward(logits, task_list)

        #### Run PPO training
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        for cs in ctrl_str:
            key = "env/reward_" + cs.strip("[]")
            stats[key] = np.mean([r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs])
        ppo_trainer.log_stats(stats, game_data, rewards)

    gpt2_model.save_pretrained(f"./ppo-chinese/epoch-{epoch}/")
    gpt2_tokenizer.save_pretrained(f"./ppo-chinese/epoch-{epoch}/")

# ==================================