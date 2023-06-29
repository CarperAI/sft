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
    # alpaca = load_dataset("orca-chatgpt-50k", "train")
    data = list()
    data.extend(data_to_columns('cot_outputs.json', 'cot'))
    data.extend(data_to_columns('niv_outputs.json', 'niv'))
    data.extend(data_to_columns('flan_outputs.json', 'flan'))
    data.extend(data_to_columns('t0_outputs.json', 't0'))
    shuffle(data)
    chatgpt = Dataset.from_list(data, split=Split.TRAIN)
    chatgpt.push_to_hub(
        "orca-chatgpt-50k", private=True)
    chatgpt.save_to_disk('orca-chatgpt-50k')
    data = list()
    data.extend(data_to_columns('cot_outputs_gpt4.json', 'cot'))
    data.extend(data_to_columns('niv_outputs_gpt4.json', 'niv'))
    data.extend(data_to_columns('flan_outputs_gpt4.json', 'flan'))
    data.extend(data_to_columns('t0_outputs_gpt4.json', 't0'))
    shuffle(data)
    gpt4 = Dataset.from_list(data, split=Split.TRAIN)
    gpt4.push_to_hub(
        "orca-gpt4-10k", private=True)
    gpt4.save_to_disk('orca-gpt4-10k')