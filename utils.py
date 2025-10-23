# utils.py
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("-LSB-", "[")
    text = text.replace("-RSB-", "]")
    text = text.replace("-LRB-", "(")
    text = text.replace("-RRB-", ")")
    text = text.replace("--", "–")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
