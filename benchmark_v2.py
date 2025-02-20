import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_path = "/home/bingxing2/ailab/yangzhuo/deepseek-qwen-distill/qwen2.5-3b-deepseek-finetuned-final-stage1"

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen2.5-3b-deepseek-finetuned-final-stage1"
)

model = AutoModelForCausalLM.from_pretrained(
    "./qwen2.5-3b-deepseek-finetuned-final-stage1",
    device_map="auto",
    torch_dtype=torch.float16,
)

model.resize_token_embeddings(len(tokenizer))

# Create chat pipeline
chat_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
)

# Create chat pipeline
chat_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
)

question = """
The duration of a process used to manufacture components is known to be normally distributed with a mean of 30 minutes and a standard deviation of 4 minutes. What is the probability of a time greater than 33 minutes being recorded?
"""

CHEM_SYSTEM_PROMPT = """<|im_start|>system
You are ChemReasoner, a professional chemistry AI assistant with the following capabilities:
* Parse molecular SMILES structures.
* Predict compound properties (such as ADME and toxicity).
* Design chemical synthesis routes.
* Analyze reaction mechanisms.
* Retrieve literature data.
* Please answer chemistry - related questions in a professional yet accessible manner.<|im_end|>
"""


user_msg = f""""\nPlease answer the question: {question}\nThink step by step. Then, on a separate line for the last sentence, 
present the letter of the final answer in the following format: answer: Your choice answer",
"""

input = CHEM_SYSTEM_PROMPT + user_msg
# 定义公共生成参数
generate_kwargs = {
    "max_new_tokens": 1024,
    "temperature": 0.1,
    "do_sample": True,
    "eos_token_id": tokenizer.eos_token_id,
}

output = chat_pipeline(input, **generate_kwargs)
generated_text = output[0]["generated_text"]
print(generated_text)
