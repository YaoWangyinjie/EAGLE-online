"""
Step 3: Evaluate and compare EAGLE3 draft models on a benchmark dataset.

This script runs EAGLE3 inference for two draft models (baseline and SFT-fine-tuned)
on the same set of questions and writes per-question and aggregate statistics to a
comparison report.

Metrics collected
-----------------
  - tokens / second  (wall-clock throughput)
  - avg_accept_len   (average accepted tokens per speculative step)
  - acc_rate         (accepted / total drafted,  i.e., tree-budget utilization)
  - speedup          (SFT speed  / baseline speed)

Usage
-----
  # Evaluate both models in one run (sequential by default):
  python eval_sft.py \
      --base_model_path  /path/to/DeepSeek-R1-Distill-Llama-8B \
      --baseline_eagle   /path/to/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
      --sft_eagle        /path/to/sft_overfit/checkpoints/epoch_199 \
      --question_file    /path/to/mt_bench/question.jsonl \
      --output_dir       /path/to/sft_overfit/eval_results \
      [--temperature 0.0] \
      [--total_token 60] \
      [--depth 5] \
      [--top_k 10] \
      [--max_new_tokens 1024] \
      [--question_begin 0] \
      [--question_end 80]

  # Evaluate only the SFT model (skip baseline):
  python eval_sft.py --sft_only ...

  # Evaluate only the baseline (skip SFT):
  python eval_sft.py --baseline_only ...
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# ---- Resolve EAGLE package path --------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EAGLE_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))   # .../EAGLE
sys.path.insert(0, _EAGLE_ROOT)

from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare EAGLE3 draft models")

    # Model paths
    parser.add_argument(
        "--base_model_path", type=str,
        default="/root/paddlejob/workspace/env_run/ea/model_weight/DeepSeek-R1-Distill-Llama-8B",
        help="Path to the base (target) model.",
    )
    parser.add_argument(
        "--baseline_eagle", type=str,
        default="/root/paddlejob/workspace/env_run/ea/model_weight/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
        help="Path to the original (pre-SFT) EAGLE3 draft model.",
    )
    parser.add_argument(
        "--sft_eagle", type=str,
        default=None,
        help="Path to the SFT fine-tuned EAGLE3 draft model checkpoint directory.",
    )

    # Benchmark
    parser.add_argument(
        "--question_file", type=str,
        default="/root/paddlejob/workspace/env_run/ea/EAGLE/eagle/data/mt_bench/question.jsonl",
        help="JSONL file with benchmark questions.",
    )
    parser.add_argument("--question_begin", type=int, default=None, help="First question index (inclusive).")
    parser.add_argument("--question_end",   type=int, default=None, help="Last question index (exclusive).")

    # Output
    parser.add_argument(
        "--output_dir", type=str,
        default="/root/paddlejob/workspace/env_run/ea/EAGLE/eagle/sft_overfit/eval_results",
        help="Directory to write result JSONL and comparison report.",
    )
    parser.add_argument("--run_tag", type=str, default="", help="Optional tag appended to output file names.")

    # Generation hyper-parameters
    parser.add_argument("--temperature",    type=float, default=0.0,  help="Sampling temperature (0 = greedy).")
    parser.add_argument("--top_p",          type=float, default=0.9)
    parser.add_argument("--total_token",    type=int,   default=60,   help="Total speculative tokens in the draft tree.")
    parser.add_argument("--depth",          type=int,   default=5,    help="Maximum draft depth.")
    parser.add_argument("--top_k",          type=int,   default=10,   help="Top-k at each draft step.")
    parser.add_argument("--max_new_tokens", type=int,   default=1024, help="Maximum generated tokens per turn.")
    parser.add_argument("--max_len",        type=int,   default=2048, help="KV cache size (prompt + generated tokens). Increase for long sequences.")
    parser.add_argument("--num_warmup",     type=int,   default=3,    help="Number of warmup generations before timing.")

    # Run mode
    parser.add_argument("--baseline_only", action="store_true", help="Only run the baseline model, skip SFT model.")
    parser.add_argument("--sft_only",      action="store_true", help="Only run the SFT model, skip baseline.")

    # Dataset namespacing
    parser.add_argument(
        "--dataset", type=str, default=None,
        help=(
            "Dataset name (e.g. 'mt_bench', 'humaneval', 'math'). "
            "Automatically sets --question_file to  EAGLE/eagle/data/{dataset}/question.jsonl "
            "and --output_dir to  sft_overfit/eval_results/{dataset} "
            "unless those flags are also explicitly provided."
        ),
    )

    args = parser.parse_args()

    if args.dataset is not None:
        import sys
        argv = sys.argv[1:]
        script_dir     = os.path.dirname(os.path.abspath(__file__))
        eagle_data_dir = os.path.dirname(script_dir)   # EAGLE/eagle
        # Auto-derive question_file: EAGLE/eagle/data/{dataset}/question.jsonl
        if not any(a.startswith("--question_file") for a in argv):
            args.question_file = os.path.join(eagle_data_dir, "data", args.dataset, "question.jsonl")
        # Auto-derive output_dir: sft_overfit/eval_results/{dataset}
        if not any(a.startswith("--output_dir") for a in argv):
            args.output_dir = os.path.join(script_dir, "eval_results", args.dataset)

    return args


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    question_id: int
    model_tag: str          # "baseline" or "sft"
    speed: float            # tokens / second
    avg_accept_len: float   # accepted tokens per speculative step
    acc_rate: float         # accepted / drafted
    new_tokens: int
    wall_time: float
    turns: List[str] = field(default_factory=list)
    idxs: List[int]  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(question_file: str, q_begin=None, q_end=None) -> list:
    questions = []
    with open(question_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if q_begin is not None:
        questions = questions[q_begin:]
    if q_end is not None:
        questions = questions[:q_end]
    return questions


def normalize_question(record: dict, idx: int) -> dict:
    """
    Return a copy of *record* guaranteed to have 'question_id' and 'turns' keys.
    Detects the source dataset format from the fields present.

    mt_bench  : already has 'turns' and 'question_id' – returned unchanged.
    HumanEval : has 'prompt' (and optionally 'task_id') – single-turn.
    MATH      : has 'problem' (and optionally 'id') – single-turn.
    fallback  : first string-valued field becomes the single turn.
    """
    if "turns" in record:
        return record

    out = dict(record)

    if "prompt" in record:
        out["question_id"] = record.get("task_id", str(idx))
        out["turns"] = [record["prompt"]]
        return out

    if "problem" in record:
        out["question_id"] = record.get("id", str(idx))
        out["turns"] = [record["problem"]]
        return out

    for key, val in record.items():
        if isinstance(val, str):
            out["question_id"] = record.get("id", str(idx))
            out["turns"] = [val]
            return out

    raise ValueError(f"Cannot normalize question record at line {idx}: {record}")


def decode_output(tokenizer, output_ids: torch.Tensor, input_len: int) -> str:
    """Decode newly generated tokens and strip special tokens."""
    new_ids = output_ids[0, input_len:]
    # Remove stop tokens that may appear at the end
    eos_id = tokenizer.eos_token_id
    stop_ids = {eos_id}
    # Also handle <|eot_id|> if present (LLaMA-3 template artefact in DeepSeek tokenizer)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id != tokenizer.unk_token_id:
        stop_ids.add(eot_id)

    # Truncate at the first stop token
    new_ids_list = new_ids.tolist()
    for idx_s, tid in enumerate(new_ids_list):
        if tid in stop_ids:
            new_ids_list = new_ids_list[:idx_s]
            break

    text = tokenizer.decode(new_ids_list, skip_special_tokens=True)
    # Strip remaining special token strings
    for special in tokenizer.special_tokens_map.values():
        if isinstance(special, list):
            for s in special:
                text = text.replace(s, "")
        else:
            text = text.replace(special, "")
    return text.strip()


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_model(
    base_model_path: str,
    ea_model_path: str,
    model_tag: str,
    questions: list,
    args,
) -> List[QuestionResult]:
    """
    Load one EAGLE3 model and run inference on all questions.
    Returns a list of QuestionResult (one per question, first turn only for speed).
    """
    print(f"\n{'='*60}")
    print(f"[eval_sft] Evaluating model: {model_tag}")
    print(f"  base  : {base_model_path}")
    print(f"  eagle : {ea_model_path}")
    print(f"{'='*60}")

    # Load model
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    logits_processor = prepare_logits_processor(temperature=args.temperature) if args.temperature > 1e-5 else None

    # ---- Warmup (use first question, discard results) ----------------------
    print(f"[eval_sft] Running {args.num_warmup} warmup step(s)...")
    warmup_q = normalize_question(questions[0], 0)
    for _ in range(args.num_warmup):
        torch.manual_seed(0)
        msgs = [{"role": "user", "content": warmup_q["turns"][0]}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
        model.eagenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=args.temperature,
            log=True,
            is_llama3=True,
            max_length=args.max_len,
        )
    torch.cuda.synchronize()
    print("[eval_sft] Warmup done.")

    # ---- Main evaluation loop ---------------------------------------------
    results: List[QuestionResult] = []

    for raw_idx, question in enumerate(tqdm(questions, desc=f"[{model_tag}]")):
        question = normalize_question(question, raw_idx)
        qid  = question["question_id"]
        msgs = []
        turns_out   = []
        idxs_out    = []
        total_new_tokens = 0
        total_time       = 0.0
        total_accept_len = 0
        total_drafted    = 0
        total_steps      = 0

        for j, turn_q in enumerate(question["turns"]):
            msgs.append({"role": "user", "content": turn_q})

            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
            input_len = len(input_ids[0])

            torch.cuda.synchronize()
            t_start = time.time()

            # eagenerate returns (output_ids, new_token, idx) when log=True
            ret = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=args.temperature,
                log=True,
                is_llama3=True,
                max_length=args.max_len,
            )
            output_ids, new_token, idx = ret[0], ret[1], ret[2]

            torch.cuda.synchronize()
            elapsed = time.time() - t_start

            text = decode_output(tokenizer, output_ids, input_len)
            turns_out.append(text)
            idxs_out.append(int(idx))
            total_new_tokens += int(new_token)
            total_time       += elapsed

            msgs.append({"role": "assistant", "content": text})

        # ------------------------------------------------------------------
        # Speed metrics
        # Speed is computed over all turns combined for this question.
        # For acceptance rate we rely on the model's internal run_stats if
        # available (EAGLE-online style).  The original EaModel does not
        # expose run_stats, so we fall back to the proxy metric: we report
        # idx (number of speculative steps) as a proxy.
        #
        # If you want per-step accept stats, you can modify ea_model.eagenerate
        # to return run_stats (see EAGLE-online's version).
        # ------------------------------------------------------------------
        speed = total_new_tokens / total_time if total_time > 0 else 0.0

        # Proxy: avg tokens accepted per step = new_tokens / num_steps
        # (idx is the loop counter in eagenerate, which equals num_speculative_steps)
        total_speculative_steps = sum(idxs_out)
        avg_accept_len = (
            total_new_tokens / total_speculative_steps
            if total_speculative_steps > 0 else 0.0
        )

        results.append(QuestionResult(
            question_id=qid,
            model_tag=model_tag,
            speed=speed,
            avg_accept_len=avg_accept_len,
            acc_rate=float("nan"),   # not available without run_stats
            new_tokens=total_new_tokens,
            wall_time=total_time,
            turns=turns_out,
            idxs=idxs_out,
        ))

    # Free VRAM before loading the next model
    del model
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def compute_summary(results: List[QuestionResult]) -> dict:
    speeds   = [r.speed          for r in results]
    acc_lens = [r.avg_accept_len for r in results]
    return {
        "mean_speed":           float(np.mean(speeds)),
        "median_speed":         float(np.median(speeds)),
        "mean_avg_accept_len":  float(np.mean(acc_lens)),
        "median_avg_accept_len":float(np.median(acc_lens)),
    }


def write_results_jsonl(results: List[QuestionResult], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "question_id":    r.question_id,
                "model_tag":      r.model_tag,
                "speed":          round(r.speed, 4),
                "avg_accept_len": round(r.avg_accept_len, 4),
                "acc_rate":       None if np.isnan(r.acc_rate) else round(r.acc_rate, 4),
                "new_tokens":     r.new_tokens,
                "wall_time":      round(r.wall_time, 4),
                "idxs":           r.idxs,
                "turns":          r.turns,
            }) + "\n")


def write_comparison_report(
    baseline_results: Optional[List[QuestionResult]],
    sft_results:      Optional[List[QuestionResult]],
    output_path: str,
):
    lines = []
    lines.append("=" * 72)
    lines.append("  EAGLE3 SFT Overfit Experiment — Comparison Report")
    lines.append("=" * 72)

    def section(tag, results):
        s = compute_summary(results)
        lines.append(f"\n  [{tag}]")
        lines.append(f"    Mean   speed        : {s['mean_speed']:.2f} tok/s")
        lines.append(f"    Median speed        : {s['median_speed']:.2f} tok/s")
        lines.append(f"    Mean   avg_accept   : {s['mean_avg_accept_len']:.3f} tok/step")
        lines.append(f"    Median avg_accept   : {s['median_avg_accept_len']:.3f} tok/step")
        return s

    bs = None
    sf = None

    if baseline_results:
        bs = section("BASELINE (original EAGLE3)", baseline_results)
    if sft_results:
        sf = section("SFT fine-tuned EAGLE3", sft_results)

    # Speedup comparison
    if bs is not None and sf is not None:
        lines.append("\n  [Relative improvement  SFT vs. BASELINE]")
        speedup = sf["mean_speed"] / bs["mean_speed"] if bs["mean_speed"] > 0 else float("nan")
        al_gain = sf["mean_avg_accept_len"] - bs["mean_avg_accept_len"]
        lines.append(f"    Speed ratio (mean)  : {speedup:.4f}x")
        lines.append(f"    Avg-accept gain     : {al_gain:+.4f} tok/step")

    lines.append("\n" + "-" * 72)
    lines.append("  Per-question detail")
    lines.append("-" * 72)

    # Build a lookup by qid for easy pairing
    def to_map(results):
        if results is None:
            return {}
        return {r.question_id: r for r in results}

    bs_map = to_map(baseline_results)
    sf_map = to_map(sft_results)
    all_qids = sorted(set(list(bs_map.keys()) + list(sf_map.keys())))

    header = f"  {'qid':>6}  {'base_spd':>9}  {'sft_spd':>9}  {'speedup':>8}  {'base_al':>8}  {'sft_al':>7}"
    lines.append(header)
    lines.append("  " + "-" * 66)

    for qid in all_qids:
        br = bs_map.get(qid)
        sr = sf_map.get(qid)
        bs_spd = f"{br.speed:.1f}" if br else "  n/a   "
        sf_spd = f"{sr.speed:.1f}" if sr else "  n/a   "
        if br and sr and br.speed > 0:
            spdup = f"{sr.speed / br.speed:.3f}x"
        else:
            spdup = "  n/a  "
        bs_al  = f"{br.avg_accept_len:.3f}" if br else " n/a  "
        sf_al  = f"{sr.avg_accept_len:.3f}" if sr else " n/a  "
        lines.append(f"  {qid:>6}  {bs_spd:>9}  {sf_spd:>9}  {spdup:>8}  {bs_al:>8}  {sf_al:>7}")

    lines.append("=" * 72)

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[eval_sft] Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    questions = load_questions(args.question_file, args.question_begin, args.question_end)
    print(f"[eval_sft] Loaded {len(questions)} questions from {args.question_file}")

    tag = f"__{args.run_tag}" if args.run_tag else ""

    baseline_results = None
    sft_results      = None

    # ---- Baseline ----------------------------------------------------------
    if not args.sft_only:
        if args.baseline_eagle is None:
            print("[eval_sft] --baseline_eagle not specified; skipping baseline.")
        else:
            baseline_results = evaluate_model(
                base_model_path=args.base_model_path,
                ea_model_path=args.baseline_eagle,
                model_tag="baseline",
                questions=questions,
                args=args,
            )
            out_jsonl = os.path.join(args.output_dir, f"baseline{tag}.jsonl")
            write_results_jsonl(baseline_results, out_jsonl)
            print(f"[eval_sft] Baseline results → {out_jsonl}")

    # ---- SFT model ---------------------------------------------------------
    if not args.baseline_only:
        if args.sft_eagle is None:
            print("[eval_sft] --sft_eagle not specified; skipping SFT evaluation.")
        else:
            sft_results = evaluate_model(
                base_model_path=args.base_model_path,
                ea_model_path=args.sft_eagle,
                model_tag="sft",
                questions=questions,
                args=args,
            )
            out_jsonl = os.path.join(args.output_dir, f"sft{tag}.jsonl")
            write_results_jsonl(sft_results, out_jsonl)
            print(f"[eval_sft] SFT results → {out_jsonl}")

    # ---- Comparison report -------------------------------------------------
    if baseline_results is not None or sft_results is not None:
        report_path = os.path.join(args.output_dir, f"comparison_report{tag}.txt")
        write_comparison_report(baseline_results, sft_results, report_path)


if __name__ == "__main__":
    main()
