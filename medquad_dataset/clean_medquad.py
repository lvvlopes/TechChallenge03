import os
import xml.etree.ElementTree as ET
import json
import re

BASE_PATH = "MedQuAD"
OUTPUT_FILE = "medquad_clean.jsonl"

def clean_text(text):
    if not text:
        return None
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid(q, a):
    if not q or not a:
        return False
    if len(q) < 10 or len(a) < 20:
        return False
    if len(a) > 1000:
        return False
    if "contact your doctor" in a.lower():
        return False
    return True

dataset = []

for root, dirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith(".xml"):
            path = os.path.join(root, file)
            try:
                tree = ET.parse(path)
                root_xml = tree.getroot()

                for qa in root_xml.findall(".//QAPair"):
                    q = clean_text(qa.findtext("Question"))
                    a = clean_text(qa.findtext("Answer"))

                    if not is_valid(q, a):
                        continue

                    dataset.append({
                        "messages": [
                            {"role": "system", "content": "You are a helpful medical assistant."},
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a}
                        ]
                    })

            except:
                pass

seen = set()
cleaned = []

for item in dataset:
    key = item["messages"][1]["content"]
    if key not in seen:
        seen.add(key)
        cleaned.append(item)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in cleaned:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Total final: {len(cleaned)}")
