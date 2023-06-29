import json
from collections import defaultdict


def extract_tasks(json_file):
    tasks = defaultdict(int)
    filtered = list()
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for task in data:
        if "arc" in task['task_name']:
            continue
        elif "winogrande" in task['task_name']:
            continue
        elif "openbookqa" in task['task_name']:
            continue
        elif "piqa" in task['task_name']:
            continue
        elif "boolq" in task['task_name']:
            continue
        elif "hellaswag" in task['task_name']:
            continue
        elif "aqua_" in task['task_name']:
            continue
        elif "obqa" in task['task_name']:
            continue
        else:
            filtered.append(task)
            tasks[task['task_name']] += 1
    return tasks, filtered



if __name__ == '__main__':
    filenames = [
        r"C:\Users\dmaha\PycharmProjects\tclx\data_handlers\orca\t0_outputs_gpt4_no_llm_lb_tasks.json",
        r"C:\Users\dmaha\PycharmProjects\tclx\data_handlers\orca\flan_outputs_gpt4_no_llm_lb_tasks.json",
        r"C:\Users\dmaha\PycharmProjects\tclx\data_handlers\orca\niv_outputs_gpt4_no_llm_lb_tasks.json",
        r"C:\Users\dmaha\PycharmProjects\tclx\data_handlers\orca\cot_outputs_gpt4_no_llm_lb_tasks.json",
        r"C:\Users\dmaha\PycharmProjects\tclx\data_handlers\orca\t0_outputs_no_llm_lb_tasks.json",
        r"C:\Users\dmaha\PycharmProjects\tclx\data_handlers\orca\flan_outputs_no_llm_lb_tasks.json",
        r"C:\Users\dmaha\PycharmProjects\tclx\data_handlers\orca\niv_outputs_no_llm_lb_tasks.json",
        r"C:\Users\dmaha\PycharmProjects\tclx\data_handlers\orca\cot_outputs_no_llm_lb_tasks.json",
    ]
    for filename in filenames:
        task_details, filtered = extract_tasks(filename)
        with open(filename.replace('.json', '_taskdistribution.json'), 'w', encoding='utf-8') as f:
            json.dump(task_details, f, ensure_ascii=False, indent=2)
        with open(filename.replace('_no_llm_lb_tasks.json', '_filtered.json'), 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)