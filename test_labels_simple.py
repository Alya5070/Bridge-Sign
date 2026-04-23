import json, os

labels = []

model_labels_path = 'Model/new_custom_labels.txt'
if os.path.exists(model_labels_path):
    with open(model_labels_path, 'r', encoding='utf-8') as f:
        custom_labels = [line.strip() for line in f.readlines()]
        labels.extend(custom_labels)

dyn_labels_path = 'dynamic_labels.json'
if os.path.exists(dyn_labels_path):
    with open(dyn_labels_path, 'r', encoding='utf-8') as f:
        dyn_labels_dict = json.load(f)
        labels.extend(list(dyn_labels_dict.keys()))

for lbl in labels:
    print(repr(lbl), len(lbl))
