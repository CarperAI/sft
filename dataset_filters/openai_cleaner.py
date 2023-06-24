import argparse
import json
import re
from tqdm import tqdm

def contains_unwanted_words(text):
    """
    This function checks if the input text contains any of the unwanted phrases.

    Args:
        text (str): The input text to be checked.

    Returns:
        bool: True if any unwanted phrase is found in the text, False otherwise.
    """
    unwanted_words = [
		"text-based AI language model",
		"please refrain",
		"responsible AI",
		"AI principles",
		"AI assistant",
		"an AI language",
		"as a language model",
		"as an AI language model",
		"As a large language model",
		"As an AI",
		"it is not appropriate",
		"it's not appropriate",
		"I cannot fulfill your request",
		"my guidelines",
		"prioritize user safety",
		"well-being of all users",
		"cannot provide guidance",
		"cannot provide information",
		"unable to offer assistance",
		"cannot engage in discussions",
		"programming prohibits",
		"against my programming",
		"designed to prioritize safety",
		"not able to provide",
		"maintain user safety",
		"adhere to safety guidelines",
		"cannot provide any information",
		"an AI language model you don't have",
		"As an AI language model, I cannot",
		"As an AI language model, I do not",
		"As an AI language model, I am not able",
		"As an AI language model, I don't have personal",
		"I am an AI language model and do not",
		"As an AI language model, I don't have",
		"As an AI language model, I am only able",
		"AI language model and I do not",
		"As an AI language model, I cannot modify",
		"As an AI language model, I do not",
		"I know as an AI language model you don't have",
		"as an AI language model, you cannot",
		"I'm sorry, but as an AI language model",
		"As an AI language model, I don't have",
		"Unfortunately, I cannot provide",
		"I'm sorry, I cannot",
		"I'm sorry, I cannot generate",
		"AI cannot create or program",
		"I'm afraid I cannot create",
		"you cannot create an",
		"legal and ethical",
		"engage in unethical",
		"como modelo de lenguaje AI",
		"Lo siento, como modelo de lenguaje",
		"no puedo proporcionar",
		"pero debido a mi capacidad para generar c\u00f3digos complejos y completos es limitado",
		"Lo siento, pero no puedo",
		"Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
		"Lo siento, como modelo de lenguaje, no tengo",
		"Lo siento, debe haber habido una confusi\u00f3n",
		"Lo siento, como modelo de lenguaje, no puedo realizar",
		"Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
		"Lamento no poder proporcionarte el c\u00f3digo",
		"Desculpe-me, mas a linguagem vulgar e ofensiva",
		"apropriada em nenhum contexto",
		"Como modelo de linguagem",
		"Como um modelo de linguagem, n\u00e3o tenho a capacidade de",
		"I cannot assist",
		"I'm an AI" ,
		"I am an AI",
		"not a human",
		"I am an AI",
		"a language model",
		"As a machine",
		"I don't have the ability",
		"I am here to assist",
		"my purpose is to ",
		"my knowledge cutoff",
		"my knowledge cut off",
		"September 2021",
		"I apologize, but",
		"my programming",
		"*This chat conversation is shared from",
		"*This conversation is shared from"
    ]
    for word in unwanted_words:
        if word.lower() in text.lower():
            return True
    return False

if __name__ == "__main__":
    """
    This script reads a JSON dataset, filters out entries based on their output field,
    and saves the cleaned data into a new JSON file. Entries that are filtered out can 
    be saved into a separate JSON file if desired.

    The script takes three arguments from the command line:
        --in_file: The input JSON file.
        --out_file: The output JSON file where cleaned entries are stored.
        --removed_file: (optional) Where removed entries should be stored. If not provided, 
                        removed entries will be stored in a file named {out_file}_removed.json.
    """
    parser = argparse.ArgumentParser(description='Removes most OpenAI disclaimers, refusals, etc from a JSON datasets output field.')
    parser.add_argument('--in_file', type=str, help='Input JSON file', required=True)
    parser.add_argument('--out_file', type=str, help='Output JSON file', required=True)
    parser.add_argument('--removed_file', type=str, help='Removed entries JSON file', default=None)
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    removed_file = args.removed_file if args.removed_file else out_file.split(".")[0] + "_removed.json"

    content = json.load(open(in_file, "r", encoding='utf-8'))
    num_conv = len(content)

    new_content = []
    removed_content = []
    for conv in tqdm(content):
        if not contains_unwanted_words(conv["output"]):
            new_content.append(conv)
        else:
            removed_content.append(conv)

    print(f"Returned {len(new_content)} out of {num_conv}, start dump ...")
    json.dump(new_content, open(out_file, "w", encoding='utf-8'), indent=2)

    print(f"Removed {len(removed_content)} entries, dumping to removed file ...")
    json.dump(removed_content, open(removed_file, "w", encoding='utf-8'), indent=2)
