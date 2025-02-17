from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
 "./qwen2.5-3b-deepseek-finetuned-final",
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("./qwen2.5-3b-deepseek-finetuned-final")
model.resize_token_embeddings(len(tokenizer))

# Create chat pipeline
chat_pipeline = pipeline( 
 "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Generate response
prompt = """<|user|>
What's the probability of rolling a 7 with two dice?
<|end|>
<|assistant|>
"""

output = chat_pipeline(
    prompt,
    max_new_tokens=5000,
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

print(output[0]['generated_text'])