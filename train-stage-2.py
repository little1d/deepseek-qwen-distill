from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from swanlab.integration.transformers import SwanLabCallback

swanlab_callback = SwanLabCallback(
    project="deepseek-qwen-distllation",
    experiment_name="MoleculeQA-Reasoning-V2-100",
    description="使用R1 蒸馏出来的 100 条数据，在 trl 框架中利用 peft 进行 lora 微调",
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
dataset = load_dataset("FaceWithTearsofJoy/moleculeqa_COT_corrected", token=HF_TOKEN)

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


# ---------------------------------- stage-2  model/tokenizer load  ----------------------------------
model_id_stage_1 = "/home/bingxing2/ailab/yangzhuo/deepseek-qwen-distill/qwen2.5-3b-deepseek-finetuned-final-stage1"

# 加载 tokenizer （集成第一次训练的所有更改）
print("Loading tokenizer from stage 1...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id_stage_1,
    trust_remote_code=True,
    padding_side="right",
)

# 自动设置pad_token（继承第一次训练的配置）
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token


print("Loading model ...")

# 先加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    model_id_stage_1,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)
# model.resize_token_embeddings(len(tokenizer))


# 验证tokenizer与模型一致性
print(f"Tokenizer length: {len(tokenizer)}")
print(f"Model embedding size: {model.get_input_embeddings().weight.size(0)}")


# ---------------------------------- tokenization ----------------------------------


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=2048, return_tensors=None
    )


train_dataset = formatted_dataset["train"]
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = formatted_dataset["test"]
test_dataset = test_dataset.map(tokenize_function, batched=True)


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
    output_dir="./qwen2.5-3b-deepseek-finetuned-stage2",
    num_train_epochs=5,  # 2 in stage-1, 5 in stage-2
    per_device_train_batch_size=4,  # 2 in stage-1, 4 in stage-2
    per_device_eval_batch_size=4,  # 2 in stage-1, 4 in stage-2
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
trainer.save_model("./qwen2.5-3b-deepseek-finetuned-stage2")


# merge and unload
final_model = trainer.model.merge_and_unload()
final_model.save_pretrained("./qwen2.5-3b-deepseek-finetuned-final-stage2")
tokenizer.save_pretrained("./qwen2.5-3b-deepseek-finetuned-final-stage2")
