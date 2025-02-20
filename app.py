from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr
import time


# 模型加载优化函数  根据你想测试的 stage 不同，来选择模型文件路径
def load_chem_model():
    try:
        model_id_stage_1 = "/home/bingxing2/ailab/yangzhuo/deepseek-qwen-distill/qwen2.5-3b-deepseek-finetuned-final-stage1"
        model = AutoModelForCausalLM.from_pretrained(
            model_id_stage_1,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_stage_1,
        )
        model.resize_token_embeddings(len(tokenizer))
        print("模型加载成功")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}") from e


# 初始化组件
try:
    model, tokenizer = load_chem_model()
    chat_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    )
except Exception as e:
    print(f"初始化错误: {e}")
    exit(1)

# 化学专业对话模板系统
CHEM_SYSTEM_PROMPT = """<|im_start|>system
You are ChemReasoner, a professional chemistry AI assistant with the following capabilities:
Parse molecular SMILES structures.
Predict compound properties (such as ADME and toxicity).
Design chemical synthesis routes.
Analyze reaction mechanisms.
Retrieve literature data.
Please answer chemistry - related questions in a professional yet accessible manner.<|im_end|>
"""


def build_chemistry_prompt(history):
    prompt = CHEM_SYSTEM_PROMPT
    for turn in history:
        role = "user" if turn[0] else "assistant"
        content = turn[1].replace("<|im_end|>", "").strip()
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


# 化学响应生成处理器
def chemical_respond(message, history, temperature, max_tokens):
    start_time = time.time()

    try:
        # 构建完整对话历史
        full_history = []
        for h in history:
            full_history.extend([(True, h[0]), (False, h[1])])
        full_history.append((True, message))

        # 生成化学专业prompt
        generation_prompt = build_chemistry_prompt(full_history)

        # 生成参数配置
        generate_kwargs = {
            "max_new_tokens": min(int(max_tokens), 5000),
            "temperature": max(0.1, min(float(temperature), 1.0)),
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # 执行生成
        output = chat_pipeline(generation_prompt, **generate_kwargs)
        response = output[0]["generated_text"][len(generation_prompt) :]

        # 清理响应并添加化学格式处理
        response = response.split("<|im_end|>")[0].strip()
        response = format_chemical_response(response)

        # 记录性能指标
        gen_time = time.time() - start_time
        print(f"生成耗时: {gen_time:.2f}s | Tokens: {len(response.split())}")

        return response

    except Exception as e:
        print(f"生成错误: {e}")
        return "⚠️生成遇到问题，请检查输入格式或调整参数重试"


# 化学响应格式化
def format_chemical_response(text):
    # 标记化学术语高亮
    chem_keywords = ["SMILES", "ADME", "logP", "pKa", "IC50", "EC50"]
    for word in chem_keywords:
        text = text.replace(word, f"**{word}**")
        # 添加化学式排版
    text = text.replace("->", "→")  # 替换反应箭头
    # SMILES格式处理
    if "SMILES" in text:
        parts = text.split("SMILES:", 1)
        text = parts[0] + f"SMILES:\n`{parts[1].strip()}`" if len(parts) > 1 else text
    # 化学式下标处理
    chemical_formats = {
        "H2O": "H₂O",
        "CO2": "CO₂",
        "CH3OH": "CH₃OH",
        # 添加更多常见化学式转换
    }
    for orig, fmt in chemical_formats.items():
        text = text.replace(orig, fmt)
    return text


# 创建专业化学界面
with gr.Blocks(
    theme=gr.themes.Soft(), css=".gradio-container {max-width: 800px}"
) as demo:
    gr.Markdown("# 🧪 ChemReasoner - 专业化学推理助手")
    gr.Markdown("### 基于Qwen2.5-3B-DeepSeek模型的专业化学AI助手")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话记录", height=500)
            msg = gr.Textbox(
                label="输入化学问题", placeholder="输入SMILES结构或化学问题..."
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 提交")
                clear_btn = gr.Button("🧹 清空")
        with gr.Column(scale=1):
            temperature = gr.Slider(
                0.1, 1.0, value=0.7, label="创意温度", info="值越高越有创意"
            )
            max_tokens = gr.Slider(
                100, 5000, value=2000, step=100, label="最大生成长度"
            )
            gr.Examples(
                examples=[
                    ["What is the IUPAC name of CCO?"],
                    ["Design a synthetic route from benzoic acid to aspirin."],
                    ["Predict the logP value of C(C(=O)O)N."],
                    ["Explain the mechanism of SN2 reaction."],
                ],
                inputs=msg,
                label="Chemistry Examples",
            )

    # 交互逻辑
    submit_btn.click(
        fn=chemical_respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[msg, chatbot],
        queue=True,
    )
    msg.submit(
        fn=chemical_respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[msg, chatbot],
        queue=True,
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)
# 启动配置
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=10003,
        share=True,
        favicon_path="./chem_icon.svg",  # 化学主题图标
    )
