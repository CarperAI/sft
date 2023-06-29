import openai
import datasets
import os
from system_messages import get_system_prompt_for_flan2021, \
    get_system_prompt_for_niv2, \
    get_system_prompt_for_cot, \
    get_system_prompt_for_t0
import json
import asyncio
import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from check_if_multiple_choice import check_if_multiple_choice
import time


CHATGPT_RATE_LIMIT = 150  # tokens per minute hits hard when there's long context.
GPT4_RATE_LIMT = 90  # tokens per minute hits hard when there's long context.
cot_total = 15000
cot_gpt4_total = cot_total//5
niv_total = 44000
niv_gpt4_total = niv_total//5
flan_total = 250000
flan_gpt4_total = flan_total//5
t0_total = 200000
t0_gpt4_total = t0_total//5


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def chatgpt(messages):
    """
    quick try/except to handle invalid input (too long context, mostly) errors.

    Also wrapped with a retry in case of random network errors.

    :param messages: list of messages
    :return: openai.ChatCompletion
    """
    try:
        return (await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo-0301",
            messages=messages
        ))
    except openai.error.InvalidRequestError as e:
        return None


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def gpt4(messages):
    """
    quick try/except to handle invalid input (too long context, mostly) errors.

    :param messages: list of messages
    :return: openai.ChatCompletion
    """
    try:
        return (await openai.ChatCompletion.acreate(
            model="gpt-4-0314",
            messages=messages
        ))
    except openai.error.InvalidRequestError as e:
        return None


def get_continuation_chatgpt(system: str, user: str):
    """
    helper function to set up the messages list for chatgpt
    :param system: system message, if applicable
    :param user: user message
    :return: openai.ChatCompletion
    """
    messages = list()
    if system != "":
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return chatgpt(messages)


def get_continuation_gpt4(system: str, user: str):
    """
    helper function to set up the messages list for gpt4
    :param system: system message, if applicable
    :param user: user message
    :return: openai.ChatCompletion
    """
    messages = list()
    if system != "":
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return gpt4(messages)


async def await_completion_coroutines(coroutines):
    """
    Helper function to gather all answers from a list of objects with coroutines.
    :param coroutines: coroutines to gather answers from
    :return: objects with answers instead of coroutine answes
    """
    start = time.process_time()
    answers = await asyncio.gather(
        *[coroutines[i]['answer'] for i in range(len(coroutines))]
    )
    for i in range(len(coroutines)):
        coroutines[i]['answer'] = answers[i].choices[0].message.content if answers[i] is not None else None
    time.sleep(max(1.0, 70.0 - (time.process_time() - start)))
    return coroutines


async def collect_cot(cot):
    cot_outputs = list()
    temp_cot_outputs = list()
    stream = tqdm.tqdm(cot, total=cot_total)
    counter = 0
    prev_outputs = list()
    if os.path.exists("cot_outputs.json"):
        with open("cot_outputs.json", "r", encoding='utf-8') as f:
            prev_outputs = json.load(f)
    for i, data in enumerate(stream):
        if data['template_type'] != 'zs_opt':
            continue
        question = data['inputs']
        system_prompt = get_system_prompt_for_cot()
        if counter < len(prev_outputs):
            cot_outputs.append({"question": prev_outputs[counter]['question'],
                 "system_prompt": prev_outputs[counter]['system_prompt'],
                 "answer": prev_outputs[counter]['answer'],
                 "real": data['targets'],
                 "task_name": data['task_name'],
                 })
        else:
            temp_cot_outputs.append({
                "question": question,
                "system_prompt": system_prompt,
                "answer": get_continuation_chatgpt(system_prompt, question),
                "task_name": data['task_name'],
                "real": data['targets']
            })
            if (len(temp_cot_outputs)+1) % CHATGPT_RATE_LIMIT == 0:
                print("Waiting for chatgpt to cool down...")
                cot_outputs.extend(await await_completion_coroutines(temp_cot_outputs))
                temp_cot_outputs = list()
                with open("cot_outputs.json", "w", encoding='utf-8') as f:
                    json.dump(cot_outputs, f, indent=4, ensure_ascii=False)
        counter += 1
        stream.update(len(cot_outputs) + len(temp_cot_outputs))
        if len(cot_outputs) + len(temp_cot_outputs) >= cot_total:
            break
    if len(temp_cot_outputs) > 0:
        cot_outputs.extend(await await_completion_coroutines(temp_cot_outputs))
    with open("cot_outputs.json", "w", encoding='utf-8') as f:
        json.dump(cot_outputs, f, indent=4, ensure_ascii=False)


async def collect_cot_gpt4(cot):
    cot_outputs = list()
    temp_cot_outputs = list()
    stream = tqdm.tqdm(cot, total=cot_gpt4_total)
    counter = 0
    prev_outputs = list()
    if os.path.exists("cot_outputs_gpt4.json"):
        with open("cot_outputs_gpt4.json", "r", encoding='utf-8') as f:
            prev_outputs = json.load(f)
    for i, data in enumerate(stream):
        if data['template_type'] != 'zs_opt':
            continue
        question = data['inputs']
        stream.update(len(cot_outputs) + len(temp_cot_outputs))
        if counter < len(prev_outputs):
            cot_outputs.append({"question": prev_outputs[counter]['question'],
                 "system_prompt": prev_outputs[counter]['system_prompt'],
                 "answer": prev_outputs[counter]['answer'],
                 "real": data['targets'],
                 "task_name": data['task_name'],
                 })
        else:
            system_prompt = get_system_prompt_for_cot()
            temp_cot_outputs.append({
                "question": question,
                "system_prompt": system_prompt,
                "answer": get_continuation_gpt4(system_prompt, question),
                "task_name": data['task_name'],
                "real": data['targets']
            })
            if (len(temp_cot_outputs)+1) % GPT4_RATE_LIMT == 0:
                print("Waiting for chatgpt to cool down...")
                cot_outputs.extend(await await_completion_coroutines(temp_cot_outputs))
                temp_cot_outputs = list()
                with open("cot_outputs_gpt4.json", "w", encoding='utf-8') as f:
                    json.dump(cot_outputs, f, indent=4, ensure_ascii=False)
        counter += 1
        if len(cot_outputs) + len(temp_cot_outputs) >= cot_gpt4_total:
            break
    if len(temp_cot_outputs) > 0:
        cot_outputs.extend(await await_completion_coroutines(temp_cot_outputs))
    with open("cot_outputs_gpt4.json", "w", encoding='utf-8') as f:
        json.dump(cot_outputs, f, indent=4, ensure_ascii=False)


async def collect_niv(niv):
    niv_outputs = list()
    temp_niv_outputs = list()
    stream = tqdm.tqdm(niv, total=niv_total)
    counter = 0
    prev_outputs = list()
    if os.path.exists("niv_outputs.json"):
        with open("niv_outputs.json", "r", encoding='utf-8') as f:
            prev_outputs = json.load(f)
    for data in stream:
        if "zs" not in data['template_type']:
            continue
        question = data['inputs']
        if counter < len(prev_outputs):
            niv_outputs.append({"question": prev_outputs[counter]['question'],
                 "system_prompt": prev_outputs[counter]['system_prompt'],
                 "answer": prev_outputs[counter]['answer'],
                 "real": data['targets'],
                 "task_name": data['task_name'],
                 })
        else:
            system_prompt = get_system_prompt_for_niv2()
            temp_niv_outputs.append({
                "question": question,
                "system_prompt": system_prompt,
                "answer": get_continuation_chatgpt(system_prompt, question),
                "task_name": data['task_name'],
                "real": data['targets']
            })
            if (len(temp_niv_outputs)+1) % CHATGPT_RATE_LIMIT == 0:
                print("Waiting for chatgpt to cool down...")
                niv_outputs.extend(await await_completion_coroutines(temp_niv_outputs))
                temp_niv_outputs = list()
                with open("niv_outputs.json", "w", encoding='utf-8') as f:
                    json.dump(niv_outputs, f, indent=4, ensure_ascii=False)
        counter += 1
        stream.update(len(niv_outputs) + len(temp_niv_outputs))
        if len(niv_outputs) + len(temp_niv_outputs) >= niv_total:
            break
    if len(temp_niv_outputs) > 0:
        niv_outputs.extend(await await_completion_coroutines(temp_niv_outputs))
    with open("niv_outputs.json", "w", encoding='utf-8') as f:
        json.dump(niv_outputs, f, indent=4, ensure_ascii=False)


async def collect_niv_gpt4(niv):
    niv_outputs = list()
    temp_niv_outputs = list()
    stream = tqdm.tqdm(niv, total=niv_gpt4_total)
    counter = 0
    prev_outputs = list()
    if os.path.exists("niv_outputs_gpt4.json"):
        with open("niv_outputs_gpt4.json", "r", encoding='utf-8') as f:
            prev_outputs = json.load(f)
    for data in stream:
        if "zs" not in data['template_type']:
            continue
        question = data['inputs']
        if counter < len(prev_outputs):
            niv_outputs.append({"question": prev_outputs[counter]['question'],
                 "system_prompt": prev_outputs[counter]['system_prompt'],
                 "answer": prev_outputs[counter]['answer'],
                 "real": data['targets'],
                 "task_name": data['task_name'],
                 })
        else:
            system_prompt = get_system_prompt_for_niv2()
            temp_niv_outputs.append({
                "question": question,
                "system_prompt": system_prompt,
                "answer": get_continuation_gpt4(system_prompt, question),
                "task_name": data['task_name'],
                "real": data['targets']
            })
            if (len(temp_niv_outputs)+1) % GPT4_RATE_LIMT == 0:
                print("Waiting for chatgpt to cool down...")
                niv_outputs.extend(await await_completion_coroutines(temp_niv_outputs))
                temp_niv_outputs = list()
                with open("niv_outputs_gpt4.json", "w", encoding='utf-8') as f:
                    json.dump(niv_outputs, f, indent=4, ensure_ascii=False)
        counter += 1
        stream.update(len(niv_outputs) + len(temp_niv_outputs))
        if len(niv_outputs) + len(temp_niv_outputs) >= niv_gpt4_total:
            break
    if len(temp_niv_outputs) > 0:
        niv_outputs.extend(await await_completion_coroutines(temp_niv_outputs))
    with open("niv_outputs_gpt4.json", "w", encoding='utf-8') as f:
        json.dump(niv_outputs, f, indent=4, ensure_ascii=False)


async def collect_flan(flan):
    flan_outputs = list()
    temp_flan_outputs = list()
    stream = tqdm.tqdm(flan, total=flan_total)
    counter = 0
    prev_outputs = list()
    if os.path.exists("flan_outputs.json"):
        with open("flan_outputs.json", "r", encoding='utf-8') as f:
            prev_outputs = json.load(f)
    for data in stream:
        if "zs" not in data['template_type']:
            continue
        question = data['inputs']
        if counter < len(prev_outputs):
            flan_outputs.append({"question": prev_outputs[counter]['question'],
                 "system_prompt": prev_outputs[counter]['system_prompt'],
                 "answer": prev_outputs[counter]['answer'],
                 "real": data['targets'],
                 "multiple_choice": prev_outputs[counter]['multiple_choice'],
                 "task_name": data['task_name'],
                 })
        else:
            # Need to figure out multiple choice
            system_prompt = get_system_prompt_for_flan2021(check_if_multiple_choice(data))
            temp_flan_outputs.append({
                "question": question,
                "system_prompt": system_prompt,
                "answer": get_continuation_chatgpt(system_prompt, question),
                "multiple_choice": check_if_multiple_choice(data),
                "task_name": data['task_name'],
                "real": data['targets']
            })
            if (len(temp_flan_outputs)+1) % CHATGPT_RATE_LIMIT == 0:
                print("Waiting for chatgpt to cool down...")
                flan_outputs.extend(await await_completion_coroutines(temp_flan_outputs))
                temp_flan_outputs = list()
                with open("flan_outputs.json", "w", encoding='utf-8') as f:
                    json.dump(flan_outputs, f, indent=4, ensure_ascii=False)
        counter += 1
        stream.update(len(flan_outputs) + len(temp_flan_outputs))
        if len(flan_outputs) + len(temp_flan_outputs) >= flan_total:
            break
    if len(temp_flan_outputs) > 0:
        flan_outputs.extend(await await_completion_coroutines(temp_flan_outputs))
    with open("flan_outputs.json", "w", encoding='utf-8') as f:
        json.dump(flan_outputs, f, indent=4, ensure_ascii=False)


async def collect_flan_gpt4(flan):
    flan_outputs = list()
    temp_flan_outputs = list()
    stream = tqdm.tqdm(flan, total=flan_gpt4_total)
    counter = 0
    prev_outputs = list()
    if os.path.exists("flan_outputs_gpt4.json"):
        with open("flan_outputs_gpt4.json", "r", encoding='utf-8') as f:
            prev_outputs = json.load(f)
    for data in stream:
        if "zs" not in data['template_type']:
            continue
        if counter < len(prev_outputs):
            flan_outputs.append({"question": prev_outputs[counter]['question'],
                 "system_prompt": prev_outputs[counter]['system_prompt'],
                 "answer": prev_outputs[counter]['answer'],
                 "real": data['targets'],
                 "multiple_choice": prev_outputs[counter]['multiple_choice'],
                 "task_name": data['task_name'],
                 })
        else:
            question = data['inputs']
            # Need to figure out multiple choice
            system_prompt = get_system_prompt_for_flan2021(check_if_multiple_choice(data))
            temp_flan_outputs.append({
                "question": question,
                "system_prompt": system_prompt,
                "answer": get_continuation_gpt4(system_prompt, question),
                "multiple_choice": check_if_multiple_choice(data),
                "task_name": data['task_name'],
                "real": data['targets']
            })
            if (len(temp_flan_outputs)+1) % GPT4_RATE_LIMT == 0:
                print("Waiting for chatgpt to cool down...")
                flan_outputs.extend(await await_completion_coroutines(temp_flan_outputs))
                temp_flan_outputs = list()
                with open("flan_outputs_gpt4.json", "w", encoding='utf-8') as f:
                    json.dump(flan_outputs, f, indent=4, ensure_ascii=False)
        counter += 1
        stream.update(len(flan_outputs) + len(temp_flan_outputs))
        if len(flan_outputs) + len(temp_flan_outputs) >= flan_gpt4_total:
            break
    if len(temp_flan_outputs) > 0:
        flan_outputs.extend(await await_completion_coroutines(temp_flan_outputs))
    with open("flan_outputs_gpt4.json", "w", encoding='utf-8') as f:
        json.dump(flan_outputs, f, indent=4, ensure_ascii=False)


async def collect_t0(t0):
    t0_outputs = list()
    temp_t0_outputs = list()
    stream = tqdm.tqdm(t0, total=t0_total)
    counter = 0
    prev_outputs = list()
    if os.path.exists("t0_outputs.json"):
        with open("t0_outputs.json", "r", encoding='utf-8') as f:
            prev_outputs = json.load(f)
    for data in stream:
        if "zs" not in data['template_type']:
            continue
        if counter < len(prev_outputs):
            t0_outputs.append({"question": prev_outputs[counter]['question'],
                 "system_prompt": prev_outputs[counter]['system_prompt'],
                 "answer": prev_outputs[counter]['answer'],
                 "real": data['targets'],
                 "task_name": data['task_name'],
                 })
        else:
            question = data['inputs']
            system_prompt = get_system_prompt_for_t0()
            temp_t0_outputs.append({
                "question": question,
                "system_prompt": system_prompt,
                "answer": get_continuation_chatgpt(system_prompt, question),
                "task_name": data['task_name'],
                "real": data['targets']
            })
            if (len(temp_t0_outputs)+1) % CHATGPT_RATE_LIMIT == 0:
                print("Waiting for chatgpt to cool down...")
                t0_outputs.extend(await await_completion_coroutines(temp_t0_outputs))
                temp_t0_outputs = list()
                with open("t0_outputs.json", "w", encoding='utf-8') as f:
                    json.dump(t0_outputs, f, indent=4, ensure_ascii=False)
        counter += 1
        stream.update(len(t0_outputs) + len(temp_t0_outputs))
        if len(t0_outputs) + len(temp_t0_outputs) >= t0_total:
            break
    if len(temp_t0_outputs) > 0:
        t0_outputs.extend(await await_completion_coroutines(temp_t0_outputs))
    with open("t0_outputs.json", "w", encoding='utf-8') as f:
        json.dump(t0_outputs, f, indent=4, ensure_ascii=False)


async def collect_t0_gpt4(t0):
    t0_outputs = list()
    temp_t0_outputs = list()
    stream = tqdm.tqdm(t0, total=t0_gpt4_total)
    counter = 0
    prev_outputs = list()
    if os.path.exists("t0_outputs_gpt4.json"):
        with open("t0_outputs_gpt4.json", "r", encoding='utf-8') as f:
            prev_outputs = json.load(f)
    for data in stream:
        if "zs" not in data['template_type']:
            continue
        if counter < len(prev_outputs):
            t0_outputs.append({"question": prev_outputs[counter]['question'],
                 "system_prompt": prev_outputs[counter]['system_prompt'],
                 "answer": prev_outputs[counter]['answer'],
                 "real": data['targets'],
                 "task_name": data['task_name'],
                 })
        else:
            question = data['inputs']
            system_prompt = get_system_prompt_for_t0()
            temp_t0_outputs.append({
                "question": question,
                "system_prompt": system_prompt,
                "answer": get_continuation_gpt4(system_prompt, question),
                "task_name": data['task_name'],
                "real": data['targets']
            })
            if (len(temp_t0_outputs)+1) % GPT4_RATE_LIMT == 0:
                print("Waiting for chatgpt to cool down...")
                t0_outputs.extend(await await_completion_coroutines(temp_t0_outputs))
                temp_t0_outputs = list()
                with open("t0_outputs_gpt4.json", "w", encoding='utf-8') as f:
                    json.dump(t0_outputs, f, indent=4, ensure_ascii=False)
        counter += 1
        stream.update(len(t0_outputs) + len(temp_t0_outputs))
        if len(t0_outputs) + len(temp_t0_outputs) >= t0_gpt4_total:
            break
    if len(temp_t0_outputs) > 0:
        t0_outputs.extend(await await_completion_coroutines(temp_t0_outputs))
    with open("t0_outputs_gpt4.json", "w", encoding='utf-8') as f:
        json.dump(t0_outputs, f, indent=4, ensure_ascii=False)


async def main():
    cot = iter(datasets.load_dataset("conceptofmind/cot_submix_original", split="train", streaming=True))
    niv = iter(datasets.load_dataset("conceptofmind/niv2_submix_original", split="train", streaming=True))
    flan = iter(datasets.load_dataset("conceptofmind/flan2021_submix_original", split="train", streaming=True))
    t0 = iter(datasets.load_dataset("conceptofmind/t0_submix_original", split="train", streaming=True))
    await collect_cot(cot)
    await collect_niv(niv)
    await collect_flan(flan)
    await collect_t0(t0)
    await collect_cot_gpt4(cot)
    await collect_niv_gpt4(niv)
    await collect_flan_gpt4(flan)
    await collect_t0_gpt4(t0)


if __name__ == '__main__':
    asyncio.run(main())