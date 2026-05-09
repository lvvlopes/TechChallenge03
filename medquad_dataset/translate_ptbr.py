import json
from deep_translator import GoogleTranslator

INPUT_FILE = "medquad_clean.jsonl"
OUTPUT_FILE = "medquad_ptbr.jsonl"

translator = GoogleTranslator(source='en', target='pt')

def translate(text):
    try:
        return translator.translate(text)
    except:
        return text

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)

        system = item["messages"][0]["content"]
        user = item["messages"][1]["content"]
        assistant = item["messages"][2]["content"]

        new_item = {
            "messages": [
                {"role": "system", "content": "Você é um assistente médico útil."},
                {"role": "user", "content": translate(user)},
                {"role": "assistant", "content": translate(assistant)}
            ]
        }

        fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print("Tradução concluída")
