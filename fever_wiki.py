# fever_wiki.py
import os
import json
from tqdm import tqdm
import re

def load_fever_wiki(wiki_dir: str):
    """
    Load wiki-pages-*.jsonl and return wiki_lookup (page_id-> {idx: sentence})
    and wiki_text (page_id -> full text)
    """
    wiki_lookup = {}
    wiki_text = {}
    files = [f for f in os.listdir(wiki_dir) if f.endswith(".jsonl")]
    for file in tqdm(sorted(files), desc="Loading Wikipedia pages"):
        with open(os.path.join(wiki_dir, file), "r", encoding="utf8") as f:
            for line in f:
                try:
                    page = json.loads(line)
                    page_id = page.get("id")
                    page_lines = {}
                    for line_item in page.get("lines", "").split("\n"):
                        if not line_item.strip():
                            continue
                        parts = line_item.split("\t", 1)
                        if len(parts) == 2:
                            idx, sent = parts
                            if sent.strip():
                                page_lines[int(idx)] = sent.strip()
                    if page_lines:
                        wiki_lookup[page_id] = page_lines
                    if page.get("text"):
                        wiki_text[page_id] = page["text"]
                except Exception:
                    continue
    print(f"Loaded {len(wiki_lookup):,} Wikipedia pages.")
    return wiki_lookup, wiki_text

def clean_wiki_text(text: str) -> str:
    """Clean small FEVER wiki dump artifacts from a text string."""
    if not text or not isinstance(text, str):
        return ""
    text = text.split("\t")[0]
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")
    text = text.replace("--", "–")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_wiki_data(wiki_lookup: dict, wiki_text: dict):
    clean_lookup = {}
    clean_text = {}
    print("Cleaning wiki_lookup sentences...")
    for page, lines in tqdm(wiki_lookup.items()):
        cleaned_lines = {}
        for idx, sent in lines.items():
            cleaned_lines[idx] = clean_wiki_text(sent)
        clean_lookup[page] = cleaned_lines
    print("Cleaning wiki_text pages...")
    for page, txt in tqdm(wiki_text.items()):
        clean_text[page] = clean_wiki_text(txt)
    return clean_lookup, clean_text
