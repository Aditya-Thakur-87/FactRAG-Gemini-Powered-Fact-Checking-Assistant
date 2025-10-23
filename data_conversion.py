# data_conversion.py
import json
import glob
from pathlib import Path
from datasets import Dataset
import pandas as pd
import ast


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def safe_normalize_evidence(dataset):
    """
    Convert evidence fields which may contain stringified lists into nested lists of strings.
    Mutates dataset (list of dicts).
    """
    for ex in dataset:
        new_evidence = []
        for ele in ex.get("evidence", []):
            converted_group = []
            for group in ele:
                if isinstance(group, str):
                    try:
                        group = ast.literal_eval(group)
                    except Exception:
                        group = [group]
                converted_line = [str(item) for item in group]
                converted_group.append(converted_line)
            new_evidence.append(converted_group)
        ex["evidence"] = new_evidence
    return dataset

def to_huggingface_dataset(list_of_dicts, features=None):
    return Dataset.from_list(list_of_dicts, features=features)
