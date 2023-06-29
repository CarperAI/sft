from datasets import DatasetDict, Dataset, load_dataset, Split
import json
from random import shuffle
from transformers import AutoTokenizer


def data_to_columns(data_path, task_name:str, triggered_prompts:list):
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
        if item['question'] in triggered_prompts:
            continue
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
    flagged_data = load_dataset("teknium/orca50k-flagged", split="train")
    triggered_prompts = list()
    for item in flagged_data:
        triggered_prompts.append(item['prompt'])
    data = list()
    data.extend(data_to_columns('cot_outputs.json', 'cot', triggered_prompts))
    data.extend(data_to_columns('niv_outputs.json', 'niv', triggered_prompts))
    data.extend(data_to_columns('flan_outputs.json', 'flan', triggered_prompts))
    data.extend(data_to_columns('t0_outputs.json', 't0', triggered_prompts))
    shuffle(data)
    chatgpt = Dataset.from_list(data, split=Split.TRAIN)
    chatgpt.push_to_hub(
        "orca-chatgpt-50k-uncensored", private=True)
    chatgpt.save_to_disk('orca-chatgpt-50k-uncensored')


