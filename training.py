from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from swanlab.integration.transformers import SwanLabCallback

swanlab_callback = SwanLabCallback(
    project="deepseek-qwen-distllation",
    experiment_name="Magpie-Reasoning-V2-250K",
    description="直接使用 magpie 框架提供的 cot 数据集（50000条），在 trl 框架中利用 peft 进行 lora 微调",
)

import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------- data preprocess ----------------------------------

# Load the dataset
print("Downloading/Loading Datasets")
# stage-1 datasets:  "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B"
# stage-2 datasets:  "FaceWithTearsofJoy/moleculeqa_COT_corrected"
dataset = load_dataset(
    "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B", token=HF_TOKEN
)
# 从 250,000 条数据中选50000条进行蒸馏测试
dataset = dataset["train"][:50000]
dataset = Dataset.from_dict(dataset)

dataset = dataset["train"]


def format_instruction(example):
    return {
        "text": (
            "<|im_start|>user\n"
            f"{example['instruction']}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{example['response']}\n"
            "<|im_end|>"
        )
    }


formatted_dataset = dataset.map(
    format_instruction,
    batched=False,
)
formatted_dataset = formatted_dataset.train_test_split(
    test_size=0.1
)  # 90-10 train-test split

# ---------------------------------- stage-1  model/tokenizer load  ----------------------------------

model_id = "Qwen/Qwen2.5-3B"

print("Downloading/Loading model")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    padding_side="right",
)
tokenizer.pad_token = tokenizer.eos_token

# Add custom tokens
CUSTOM_TOKENS = ["<think>", "</think>"]
tokenizer.add_special_tokens({"additional_special_tokens": CUSTOM_TOKENS})


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=2048, return_tensors=None
    )


train_dataset = formatted_dataset["train"]
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = formatted_dataset["test"]
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load model with flash-attention

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)

model.resize_token_embeddings(len(tokenizer))


# ---------------------------------- training ----------------------------------


peft_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.2,  # Dropout rate
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    bias="none",  # No bias terms
    task_type="CAUSAL_LM",  # Task type
)

# output_dir name
# stage-1: "./qwen2.5-3b-deepseek-finetuned-stage-1"
# stage-2: "./qwen2.5-3b-deepseek-finetuned-stage-2"

training_args = SFTConfig(
    output_dir="./qwen2.5-3b-deepseek-finetuned-stage-1",
    num_train_epochs=5,  # 2 in stage-1, 5 in stage-2
    per_device_train_batch_size=4,  # 2 in stage-1, 4 in stage-2
    # per_device_eval_batch_size=2, # 2 in stage-1, comment in stage-2, because we don't have eval datasets
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=5e-5,  # 2e-5 in stage-1, 5e-5 in stage-2
    fp16=True,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    dataset_kwargs={"skip_prepare_dataset": True},
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    callbacks=[swanlab_callback],
)

# Start training
print("Start Training")
trainer.train()
trainer.save_model("./qwen2.5-3b-deepseek-finetuned-stage1")


# merge and unload
final_model = trainer.model.merge_and_unload()
final_model.save_pretrained("./qwen2.5-3b-deepseek-finetuned-final-stage1")
tokenizer.save_pretrained("./qwen2.5-3b-deepseek-finetuned-final-stage1")
