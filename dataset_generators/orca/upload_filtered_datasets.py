from datasets import DatasetDict, Dataset, load_dataset, Split
import json
from random import shuffle
from transformers import AutoTokenizer


def data_to_columns(data_path, task_name:str):
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)
    out_data = list()
    for item in data:
        if item['answer'] is None:
            continue
        if item['system_prompt'] is None:
            print("wtf")
        if item['question'] is None:
            print("wtf q")
        if item['real'] is None:
            print("wtf r")
        if item['answer'] is None:
            print("wtf a")
        out_data.append(
            {
                'prompt': item['question'],
                'response': item['answer'],
                'real': item['real'],
                'system': item['system_prompt'],
                'task_name': task_name
            })
    return out_data





if __name__ == '__main__':
    data = list()
    data.extend(data_to_columns('cot_outputs_filtered.json', 'cot'))
    data.extend(data_to_columns('niv_outputs_filtered.json', 'niv'))
    data.extend(data_to_columns('flan_outputs_filtered.json', 'flan'))
    data.extend(data_to_columns('t0_outputs_filtered.json', 't0'))
    shuffle(data)
    chatgpt = Dataset.from_list(data, split=Split.TRAIN)
    chatgpt.push_to_hub(
        "orca-chatgpt-50k-llm-leaderboard-filtered", private=True)
    chatgpt.save_to_disk('orca-chatgpt-50k-llm-leaderboard-filtered')
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
    data = list()
    data.extend(data_to_columns('cot_outputs_gpt4_filtered.json', 'cot'))
    data.extend(data_to_columns('niv_outputs_gpt4_filtered.json', 'niv'))
    data.extend(data_to_columns('flan_outputs_gpt4_filtered.json', 'flan'))
    data.extend(data_to_columns('t0_outputs_gpt4_filtered.json', 't0'))
    shuffle(data)
    gpt4 = Dataset.from_list(data, split=Split.TRAIN)
    gpt4.push_to_hub(
        "orca-gpt4-10k-llm-leaderboard-filtered", private=True)
    gpt4.save_to_disk('orca-gpt4-10k-llm-leaderboard-filtered')

