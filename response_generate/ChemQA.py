from openai import OpenAI
import pandas as pd
from datasets import load_dataset
import multiprocessing

ds = load_dataset("/Volumes/lpnp6/MoleculeQA/JSON/All")


def process_chunk(start, end):
    """处理数据块的任务函数"""
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-xxx",  # <YOUR_API_KEY>
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    chunk_responses = []

    for i in range(start, end):
        try:
            data = ds["test"][i]
            question = data["question"]
            bio_seq_type = "smiles"
            question = question.replace("<BIO-SEQ-TYPE>", bio_seq_type)
            question = question.replace("<BIO-SEQ>", data[bio_seq_type])
            user_input = f"Please select the correct answer from the four options below, Please reason step by step, and put your final answer within \\boxed{{}}.\nThe question is: {question}"  # 保持原有提示词格式

            # API请求
            completion = client.chat.completions.create(
                model="qwen2.5-3b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input},
                ],
            )

            # 结果保存
            chunk_responses.append(
                {
                    "id": data["id"],
                    "cid": data["cid"],
                    "answer": data["answer"],
                    "response": completion.choices[0].message.content,
                }
            )

            if len(chunk_responses) % 100 == 0:
                save_start = start + (len(chunk_responses) - 100)
                pd.DataFrame(chunk_responses[-100:]).to_csv(
                    f"response_{save_start}_{save_start+99}.csv", index=False
                )
        except Exception as e:
            print(f"Error processing index {i}: {e}")

    # 保存剩余数据
    if chunk_responses:
        save_start = start + (len(chunk_responses) // 100) * 100
        pd.DataFrame(chunk_responses).to_csv(
            f"response_{save_start}_{end-1}.csv", index=False
        )

    return


if __name__ == "__main__":
    NUM_PROCESSES = 8  # 根据CPU核心数和API限制调整
    total_samples = len(ds["test"])
    chunk_size = total_samples // NUM_PROCESSES

    # 生成分块任务
    chunks = []
    for i in range(NUM_PROCESSES):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != NUM_PROCESSES - 1 else total_samples
        chunks.append((start, end))

    # 启动多进程处理
    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        results = pool.starmap(process_chunk, chunks)

    # 合并最终结果（可选）
    final = [item for sublist in results for item in sublist]
    pd.DataFrame(final).to_csv("final_responses.csv", index=False)
