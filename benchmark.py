import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import re
import swanlab

swanlab.login(api_key="tSGp3IsD7uFZHaBax6NC4")

exp = swanlab.init(
    project="deepseek-qwen-distllation",
    experiment_name="benchmark",
    description="记录 benchmark 的过程输出和结果",
)

MODEL_PATHS = {
    # "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "model-stage-1": "/home/bingxing2/ailab/yangzhuo/deepseek-qwen-distill/qwen2.5-3b-deepseek-finetuned-final-stage1",
    "model-stage-2": "/home/bingxing2/ailab/yangzhuo/deepseek-qwen-distill/qwen2.5-3b-deepseek-finetuned-final-stage2",
}


BENCHMARK_DATASET = "FaceWithTearsofJoy/molecule_qa_test"
MAX_NEW_TOKENS = 5000  # 控制生成长度


class ModelWrapper:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))

    def generate(self, question):
        prompt = f"\n{question}\n"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n[Model Output]\n{decoded_output}")  # 新增格式化输出
        return decoded_output


def evaluate_benchmark():
    dataset = load_dataset(BENCHMARK_DATASET)["train"]
    models = {name: ModelWrapper(path) for name, path in MODEL_PATHS.items()}
    total_samples = len(dataset)

    for sample_idx, sample in enumerate(dataset):
 
        for model_name, model in models.items():
            response = model.generate(sample["question"])
            response = swanlab.Text(response)
            swanlab.log(
                {f"{model_name}_sample_{sample_idx}": response}
            )  # 按样本索引记录


if __name__ == "__main__":
    print("Starting Evaluation Process")
    evaluate_benchmark()
    print("Evaluation Completed")
