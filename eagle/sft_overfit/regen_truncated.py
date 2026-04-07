"""
Regenerate truncated entries in an existing SFT data file.

Scans the input JSONL for entries where any raw_answer is missing '</think>',
regenerates those entries with a larger max_new_tokens, and writes a new JSONL
with the truncated entries replaced.

Usage:
    python regen_truncated.py \
        --input_file  data/mt_bench_sft_v3.jsonl \
        --output_file data/mt_bench_sft_v4.jsonl \
        --dataset     mt_bench \
        --max_new_tokens 8192
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Reuse helpers from gen_sft_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_sft_data import (
    DTYPE_MAP,
    load_questions,
    normalize_question,
    build_messages_turn1,
    build_messages_turn2,
    generate_answer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
        default="/root/paddlejob/workspace/env_run/ea/model_weight/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--input_file",  type=str, required=True,
        help="Existing SFT JSONL to patch (e.g. data/mt_bench_sft_v3.jsonl)")
    parser.add_argument("--output_file", type=str, required=True,
        help="Output path for the patched JSONL (e.g. data/mt_bench_sft_v4.jsonl)")
    parser.add_argument("--dataset", type=str, default=None,
        help="Dataset name to locate question.jsonl under EAGLE/eagle/data/{dataset}/")
    parser.add_argument("--question_file", type=str, default=None,
        help="Explicit path to question.jsonl (overrides --dataset)")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if args.question_file is None:
        if args.dataset is None:
            parser.error("Provide either --dataset or --question_file")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        eagle_data_dir = os.path.dirname(script_dir)
        args.question_file = os.path.join(eagle_data_dir, "data", args.dataset, "question.jsonl")

    return args


def is_truncated(record: dict) -> bool:
    return any("</think>" not in ans for ans in record.get("raw_answers", []))


def regen_record(model, tokenizer, q_record, max_new_tokens, temperature, top_p, device, orig_id):
    """Re-generate all turns for a single question, return a new record."""
    turns = q_record["turns"]
    raw_prompts = []
    raw_answers = []
    conversation = []

    # Turn 1
    msgs_t1 = build_messages_turn1(tokenizer, turns[0])
    turn1_a, prompt1_str = generate_answer(
        model, tokenizer, msgs_t1, max_new_tokens, temperature, top_p, device
    )
    raw_prompts.append(prompt1_str)
    raw_answers.append(turn1_a)
    conversation.append({"from": "human", "value": turns[0]})
    conversation.append({"from": "gpt",   "value": turn1_a})

    # Turn 2+
    for turn_idx in range(1, len(turns)):
        turn_q = turns[turn_idx]
        prior_messages = []
        for k in range(0, len(conversation), 2):
            prior_messages.append({"role": "user",      "content": conversation[k]["value"]})
            prior_messages.append({"role": "assistant", "content": conversation[k + 1]["value"]})
        prior_messages.append({"role": "user", "content": turn_q})

        turn_a, prompt_str = generate_answer(
            model, tokenizer, prior_messages, max_new_tokens, temperature, top_p, device
        )
        raw_prompts.append(prompt_str)
        raw_answers.append(turn_a)
        conversation.append({"from": "human", "value": turn_q})
        conversation.append({"from": "gpt",   "value": turn_a})

    return {
        "id":            orig_id,
        "conversations": conversation,
        "raw_prompts":   raw_prompts,
        "raw_answers":   raw_answers,
    }


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load existing data
    # ------------------------------------------------------------------
    existing = []
    with open(args.input_file) as f:
        for line in f:
            existing.append(json.loads(line))

    truncated_ids = {r["id"] for r in existing if is_truncated(r)}
    print(f"Found {len(truncated_ids)} truncated entries to regenerate: {sorted(truncated_ids)}")
    if not truncated_ids:
        print("Nothing to do.")
        return

    # ------------------------------------------------------------------
    # Load question.jsonl to get original prompts for truncated entries
    # ------------------------------------------------------------------
    raw_questions = load_questions(args.question_file)
    # Build map: question_id -> normalized record
    q_map = {}
    for idx, q in enumerate(raw_questions):
        q = normalize_question(q, idx)
        qid = q["question_id"]
        q_map[str(qid)] = q

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    dtype = DTYPE_MAP[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Regenerate truncated entries
    # ------------------------------------------------------------------
    regen_map = {}
    for rec_id in tqdm(sorted(truncated_ids), desc="Regenerating"):
        # rec_id format is "{question_id}-{sample_idx}", e.g. "97-0"
        base_qid = rec_id.rsplit("-", 1)[0]   # "97"
        if base_qid not in q_map:
            print(f"  WARNING: question_id '{base_qid}' not found in question file, skipping {rec_id}")
            continue
        q_record = q_map[base_qid]
        new_rec = regen_record(
            model, tokenizer, q_record,
            args.max_new_tokens, args.temperature, args.top_p, device,
            orig_id=rec_id,
        )
        still_truncated = is_truncated(new_rec)
        status = "STILL TRUNCATED" if still_truncated else "OK"
        print(f"  {rec_id}: {status}")
        regen_map[rec_id] = new_rec

    # ------------------------------------------------------------------
    # Write patched output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    written = 0
    patched = 0
    with open(args.output_file, "w") as fout:
        for rec in existing:
            if rec["id"] in regen_map:
                fout.write(json.dumps(regen_map[rec["id"]], ensure_ascii=False) + "\n")
                patched += 1
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone. {patched} entries replaced, {written} total written to {args.output_file}")

    # Final check
    still_bad = sum(1 for r in regen_map.values() if is_truncated(r))
    if still_bad:
        print(f"WARNING: {still_bad} entries are STILL truncated after regeneration. "
              f"Consider increasing --max_new_tokens further.")
    else:
        print("All regenerated entries are complete (contain </think>).")


if __name__ == "__main__":
    main()
