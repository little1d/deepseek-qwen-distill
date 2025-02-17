from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
# Load the dataset
print("Start downloading Datasets")
dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B", token=HF_TOKEN)
dataset = dataset["train"]

def format_instruction(example):
    return {
        "text": (
            "<|im_start|>user\n"
            f"{example['instruction']}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"  # 注意这里是 assistant 的正确拼写
            f"{example['response']}\n"
            "<|im_end|>"
        )
    }

formatted_dataset = dataset.map(format_instruction, batched=False)
formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)  # 90-10 train-test split


model_id = "Qwen/Qwen2.5-3B"
print("Downloading model")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Add custom tokens
CUSTOM_TOKENS = ["<think>", "</think>"]
tokenizer.add_special_tokens({"additional_special_tokens":CUSTOM_TOKENS})
tokenizer.pad_token = tokenizer.eos_token

# Load model with flash-attention

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)

model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.2,  # Dropout rate
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    bias="none",  # No bias terms
    task_type="CAUSAL_LM" # Task type
)

training_args = TrainingArguments(
    output_dir="./qwen2.5-3b-deepseek-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    fp16=True,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    data_collator=data_collator,
)

# Start training
trainer.train()
trainer.save_model("./qwen2.5-3b-deepseek-finetuned")

# merge and unload
final_model = trainer.model.merge_and_unload()
final_model.save_pretrained("./qwen2.5-3b-deepseek-finetuned-final")
tokenizer.save_pretrained("./qwen2.5-3b-deepseek-finetuned-final")