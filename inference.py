from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import swanlab

exp = swanlab.init(
    project="deepseek-qwen-distllation",
    experiment_name="Distilled(using math cot data) Qwen2.5-3b inference result",
    description="lora 微调后的模型进行推理测试",
)

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "./qwen2.5-3b-deepseek-finetuned-final-stage1",
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen2.5-3b-deepseek-finetuned-final-stage1"
)
model.resize_token_embeddings(len(tokenizer))

# Create chat pipeline
chat_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
)

# Generate response
# prompt = """<|user|>
# <|end|>
# <|assistant|>
# """

# prompt = """<|im_start|>user
# The duration of a process used to manufacture components is known to be normally distributed with a mean of 30 minutes and a standard deviation of 4 minutes. What is the probability of a time greater than 33 minutes being recorded?
# <|im_end|>
# <|im_start|>assistant
# <|im_end|>
# """

prompt = [
    "The duration of a process used to manufacture components is known to be normally distributed with a mean of 30 minutes and a standard deviation of 4 minutes. What is the probability of a time greater than 33 minutes being recorded?",
    "The manager of a petroleum refinery expects a major shutdown to occur with one hundred percent certainty in either 250 days or 350 days. The manager wishes to determine the probability that the shutdown will occur sooner, that is, in 250 days.",
    "The average (arithmetic mean) of five distinct non-negative integers is 21. What is the sum of the integer values of the mode, median, and mean of the set?",
    """Molecule SMILES: CN(C)[C@H]1[C@@H]2C[C@@H]3CC4=C(C=CC(=C4C(=C3C(=O)[C@@]2(C(=C(C1=O)C(=O)N)O)O)O)O)N(C)C
    Question about this molecule: What is the right information about this molecule's absorption?
    Option A: It has minimal oral absorption.
    Option B: It undergoes very little absorption following oral or topical administration.
    Option C: It has good absorption and enhanced oral bioavailability.
    Option D: It has excellent absorption and tissue penetration.
    """,
]

# 定义公共生成参数
generate_kwargs = {
    "max_new_tokens": 5000,
    "temperature": 0.7,
    "do_sample": True,
    "eos_token_id": tokenizer.eos_token_id,
}

# 处理前三个数学问题
math_prompts = []
math_responses = []

for idx in range(3):
    output = chat_pipeline(prompt[idx], **generate_kwargs)
    generated_text = output[0]["generated_text"]

    math_prompts.append(swanlab.Text(prompt[idx], caption=f"Math Prompt-{idx+1}"))
    math_responses.append(
        swanlab.Text(generated_text, caption=f"Math Response-{idx+1}")
    )

# 记录数学问题结果
swanlab.log({"MathQA/Prompts": math_prompts, "MathQA/Responses": math_responses})

# 处理分子问题
molecule_prompt = prompt[3]
molecule_output = chat_pipeline(molecule_prompt, **generate_kwargs)[0]["generated_text"]

# 记录分子问题结果
swanlab.log(
    {
        "MoleculeQA": swanlab.Text(
            data=f"{molecule_prompt}\n\n{molecule_output}",
            caption="Molecule Absorption Analysis",
        )
    }
)
