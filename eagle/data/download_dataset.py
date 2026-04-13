"""
Download HuggingFace datasets and convert to EAGLE question.jsonl format.

Format:
    {"question_id": int, "category": str, "turns": [str], "reference": [str] (optional)}

Usage:
    python download_dataset.py --dataset billsum       --num_samples 80
    python download_dataset.py --dataset pubmed        --num_samples 80
    python download_dataset.py --dataset pubmed_qa     --num_samples 80
    python download_dataset.py --dataset xlsum_amharic --num_samples 80
    python download_dataset.py --dataset xlsum_welsh   --num_samples 80
    python download_dataset.py --dataset xlsum_burmese --num_samples 80
    python download_dataset.py --dataset xlsum_swahili --num_samples 80
"""

import os
import json
import argparse

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset


# ── converters ─────────────────────────────────────────────────────────────
# Each converter receives a HuggingFace dataset row (dict) and qid (int).
# Returns a dict with keys: question_id, category, turns, [reference].

def conv_billsum(row, qid):
    """BillSum: US Congressional bill → summary."""
    text = row["text"].strip()
    summary = row["summary"].strip()
    return {
        "question_id": qid,
        "category": "legal_summarization",
        "turns": [f"Summarize the following bill:\n\n{text}"],
        "reference": [summary],
    }


def conv_pubmed(row, qid):
    """Scientific papers (PubMed split): article → abstract."""
    article = row["article"].strip()
    abstract = row["abstract"].strip()
    return {
        "question_id": qid,
        "category": "scientific_summarization",
        "turns": [f"Summarize the following scientific paper:\n\n{article}"],
        "reference": [abstract],
    }


def conv_pubmed_qa(row, qid):
    """PubMedQA: biomedical yes/no/maybe question answering."""
    context = " ".join(row["context"]["contexts"])
    question = row["question"].strip()
    answer = row.get("long_answer", row.get("final_decision", "")).strip()
    return {
        "question_id": qid,
        "category": "medical_qa",
        "turns": [f"Context: {context}\n\nQuestion: {question}"],
        "reference": [answer] if answer else [],
    }


def conv_xlsum(row, qid, lang):
    """XL-Sum: multilingual news article → summary."""
    text = row["text"].strip()
    summary = row["summary"].strip()
    return {
        "question_id": qid,
        "category": f"summarization_{lang}",
        "turns": [f"Summarize: {text}"],
        "reference": [summary],
    }


# ── dataset registry ────────────────────────────────────────────────────────

DATASETS = {
    "billsum": {
        "loader": lambda: load_dataset("billsum", split="test"),
        "converter": conv_billsum,
        "output_dir": "billsum",
    },
    "pubmed": {
        "loader": lambda: load_dataset("scientific_papers", "pubmed", split="test",
                                       trust_remote_code=True),
        "converter": conv_pubmed,
        "output_dir": "pubmed",
    },
    "pubmed_qa": {
        "loader": lambda: load_dataset("pubmed_qa", "pqa_labeled", split="train"),
        "converter": conv_pubmed_qa,
        "output_dir": "pubmed_qa",
    },
    **{
        f"xlsum_{lang}": {
            "loader": (lambda l: lambda: load_dataset(
                "csebuetnlp/xlsum", l, split="test"))(lang),
            "converter": (lambda l: lambda row, qid: conv_xlsum(row, qid, l))(lang),
            "output_dir": f"xlsum_{lang}",
        }
        for lang in ["amharic", "welsh", "burmese", "swahili", "kyrgyz",
                     "arabic", "chinese_simplified", "japanese", "korean",
                     "russian", "turkish", "hindi"]
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS.keys()),
                        help="Dataset name to download and convert")
    parser.add_argument("--num_samples", type=int, default=80,
                        help="Number of samples to take (default: 80)")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle before sampling (uses seed 42, default: True)")
    parser.add_argument("--min_text_len", type=int, default=200,
                        help="Minimum character length of the turn text (filters very short entries)")
    parser.add_argument("--max_text_len", type=int, default=8000,
                        help="Maximum character length to avoid extremely long inputs")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    print(f"Loading {args.dataset} ...")
    ds = cfg["loader"]()
    print(f"  Raw size: {len(ds)}")

    # optional shuffle
    if args.shuffle:
        ds = ds.shuffle(seed=42)

    converter = cfg["converter"]
    out_dir = os.path.join(os.path.dirname(__file__), cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "question.jsonl")

    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in ds:
            if written >= args.num_samples:
                break
            try:
                entry = converter(row, written)
            except Exception as e:
                skipped += 1
                continue
            # filter by text length
            turn_text = entry["turns"][0]
            if len(turn_text) < args.min_text_len or len(turn_text) > args.max_text_len:
                skipped += 1
                continue
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} entries to {out_path}  (skipped {skipped})")


if __name__ == "__main__":
    main()
