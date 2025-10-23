# evidence.py
from collections import defaultdict, Counter
from typing import Tuple, List

def evaluate_sentence_page_alignment_no_norm(dataset: List[dict], wiki_lookup: dict):
    """
    Return coverage summary and detailed stats for each example.
    dataset: list of examples (dicts) each with an 'evidence' field of nested lists.
    wiki_lookup: {page_id: {idx: sentence}}
    """
    stats = []
    total = len(dataset)
    for ex in dataset:
        has_page = False
        has_sentence = False
        has_page_no_sentence = False
        for group in ex.get("evidence", []):
            for line in group:
                if len(line) < 4:
                    continue
                page = line[2]
                sent_idx = line[3]
                if not page:
                    continue
                if page in wiki_lookup:
                    has_page = True
                    if sent_idx is None:
                        has_page_no_sentence = True
                        continue
                    try:
                        sent_idx_int = int(sent_idx)
                    except (ValueError, TypeError):
                        sent_idx_int = None
                    if sent_idx_int is not None and sent_idx_int in wiki_lookup[page]:
                        has_sentence = True
                    else:
                        has_page_no_sentence = True
        stats.append({
            "id": ex.get("id"),
            "has_page": has_page,
            "has_sentence": has_sentence,
            "has_page_no_sentence": has_page_no_sentence
        })
    c = Counter()
    for s in stats:
        if s["has_page"]:
            c["page"] += 1
        if s["has_sentence"]:
            c["sentence"] += 1
        if s["has_page_no_sentence"]:
            c["page_but_no_sentence"] += 1
        if not s["has_page"]:
            c["no_page"] += 1
    summary = {
        "total_examples": total,
        "page_exists_%": round(100 * c["page"] / total, 2),
        "sentence_exists_%": round(100 * c["sentence"] / total, 2),
        "page_but_no_sentence_%": round(100 * c["page_but_no_sentence"] / total, 2),
        "no_page_%": round(100 * c["no_page"] / total, 2)
    }
    return summary, stats

def extract_sentence_and_page_evidence(example: dict, wiki_lookup: dict, wiki_text: dict) -> Tuple[str, str]:
    """
    For one FEVER-style example, return (sentence_evidence_text, page_evidence_text)
    sentence evidence: collected specific lines referenced
    page evidence: collected full page texts when line idx was -1 (or not provided)
    """
    topics = defaultdict(set)
    for group in example.get("evidence", []):
        for line in group:
            if len(line) >= 4:
                page = line[2]
                sent_idx = line[3]
                if page is not None:
                    try:
                        topics[page].add(int(sent_idx)) if sent_idx is not None else topics[page].add(-1)
                    except Exception:
                        topics[page].add(-1)
    sentence_evidence = []
    page_evidence = []
    for topic, line_nums in topics.items():
        if topic in wiki_lookup:
            for ln in line_nums:
                if ln >= 0 and ln in wiki_lookup[topic]:
                    sentence_evidence.append(wiki_lookup[topic][ln])
        if topic in wiki_text:
            page_evidence.append(wiki_text[topic])
    return " ".join(sentence_evidence), " ".join(page_evidence)
