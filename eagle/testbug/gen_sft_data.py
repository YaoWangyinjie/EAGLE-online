"""
Step 1: Generate SFT training data for the overfit experiment.

Uses the target model (DeepSeek-R1-Distill-Llama-8B) to generate answers for
each turn in a benchmark dataset (default: mt_bench), then saves the results in
ShareGPT-style JSONL format that is directly consumable by the EAGLE3 training
pipeline (sft_main.py).

Multi-turn handling:
  - Turn 1 is answered independently.
  - Turn 2 is answered with Turn 1's generated answer in context (not the
    ground-truth reference answer), so the conversation is self-consistent.

Output format (one JSON object per line):
  {
    "id": "<question_id>-<sample_idx>",
    "conversations": [
      {"from": "human", "value": "<turn-1 question>"},
      {"from": "gpt",   "value": "<model answer for turn 1>"},
      {"from": "human", "value": "<turn-2 question>"},
      {"from": "gpt",   "value": "<model answer for turn 2>"}
    ]
  }

Usage:
  python gen_sft_data.py \
      --model_path /path/to/DeepSeek-R1-Distill-Llama-8B \
      --question_file /path/to/mt_bench/question.jsonl \
      --output_file  /path/to/sft_data.jsonl \
      [--num_samples_per_question 1] \
      [--max_new_tokens 1024] \
      [--temperature 0.7] \
      [--num_gpus 1]
"""

import argparse
import json
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT data from target model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/paddlejob/workspace/env_run/ea/model_weight/DeepSeek-R1-Distill-Llama-8B",
        help="Path to the target (base) model used to generate answers.",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        default="/root/paddlejob/workspace/env_run/ea/EAGLE/eagle/data/mt_bench/question.jsonl",
        help="Path to the benchmark question JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/root/paddlejob/workspace/env_run/ea/EAGLE/eagle/sft_overfit/data/mt_bench_sft.jsonl",
        help="Path to write the generated ShareGPT-format JSONL.",
    )
    parser.add_argument(
        "--num_samples_per_question",
        type=int,
        default=1,
        help=(
            "Number of independent samples to generate per question. "
            "Values > 1 increase effective dataset size through temperature sampling."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate per turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help=(
            "Sampling temperature. Use 0.0 for greedy (deterministic). "
            "For num_samples_per_question > 1, temperature > 0 is recommended."
        ),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling probability. Ignored when temperature=0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for inference.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_questions(question_file: str) -> list:
    """Load questions from a JSONL file."""
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def build_messages_turn1(tokenizer, question: str) -> list:
    """Build the message list for the first turn (no prior context)."""
    # DeepSeek-R1 models use a simple user/assistant format without a system prompt.
    # We intentionally omit the system prompt here to match how gen_ea_answer_ds.py
    # calls eagenerate (which also omits it for DeepSeek).
    return [{"role": "user", "content": question}]


def build_messages_turn2(tokenizer, turn1_q: str, turn1_a: str, turn2_q: str) -> list:
    """Build the message list for the second turn, including the first turn context."""
    return [
        {"role": "user",      "content": turn1_q},
        {"role": "assistant", "content": turn1_a},
        {"role": "user",      "content": turn2_q},
    ]


@torch.inference_mode()
def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> tuple:
    """
    Run one generation pass.

    Returns
    -------
    answer_text : str
        The newly generated text decoded with skip_special_tokens=True.
        For DeepSeek-R1 this does NOT include the leading <think> because
        <think> is injected by the generation prompt (not generated by the model).
    prompt_str : str
        The full prompt string fed to the model (including the generation
        prompt suffix such as '<｜Assistant｜><think>\\n').  Kept so the
        caller can reconstruct the exact original token sequence without
        going through chat-template round-trip.
    """
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoding = tokenizer(
        prompt_str, return_tensors="pt", add_special_tokens=False
    )
    input_ids      = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature if temperature > 0.0 else 1.0,
        top_p=top_p if temperature > 0.0 else 1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=gen_config,
    )

    new_token_ids = output_ids[0, input_ids.shape[1]:]
    answer_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return answer_text, prompt_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # ------------------------------------------------------------------
    # Load model & tokenizer
    # ------------------------------------------------------------------
    print(f"[gen_sft_data] Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    print(f"[gen_sft_data] Loading model from:     {args.model_path}  (dtype={args.dtype})")
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
    # Load questions
    # ------------------------------------------------------------------
    questions = load_questions(args.question_file)
    print(f"[gen_sft_data] Loaded {len(questions)} questions from {args.question_file}")
    print(f"[gen_sft_data] Generating {args.num_samples_per_question} sample(s) per question  "
          f"→ {len(questions) * args.num_samples_per_question} total conversations")

    # ------------------------------------------------------------------
    # Generate and write
    # ------------------------------------------------------------------
    total_written = 0
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for q in tqdm(questions, desc="Questions"):
            qid = q["question_id"]
            turns = q["turns"]  # list of turn strings

            for sample_idx in range(args.num_samples_per_question):
                # raw_prompts[i] = full prompt string fed to the model for turn i
                # (includes the generation-prompt suffix, e.g. '<｜Assistant｜><think>\n')
                # raw_answers[i] = newly generated tokens (decoded, without the prompt)
                raw_prompts  = []
                raw_answers  = []
                conversation = []  # ShareGPT format for human-readable storage

                # ---- Turn 1 ------------------------------------------------
                turn1_q  = turns[0]
                msgs_t1  = build_messages_turn1(tokenizer, turn1_q)
                turn1_a, prompt1_str = generate_answer(
                    model, tokenizer, msgs_t1,
                    args.max_new_tokens, args.temperature, args.top_p, device
                )
                raw_prompts.append(prompt1_str)
                raw_answers.append(turn1_a)
                conversation.append({"from": "human", "value": turn1_q})
                conversation.append({"from": "gpt",   "value": turn1_a})

                # ---- Subsequent turns (usually just turn 2) ----------------
                for turn_idx in range(1, len(turns)):
                    turn_q = turns[turn_idx]
                    prior_messages = []
                    for k in range(0, len(conversation), 2):
                        prior_messages.append({"role": "user",      "content": conversation[k]["value"]})
                        prior_messages.append({"role": "assistant", "content": conversation[k + 1]["value"]})
                    prior_messages.append({"role": "user", "content": turn_q})

                    turn_a, prompt_str = generate_answer(
                        model, tokenizer, prior_messages,
                        args.max_new_tokens, args.temperature, args.top_p, device
                    )
                    raw_prompts.append(prompt_str)
                    raw_answers.append(turn_a)
                    conversation.append({"from": "human", "value": turn_q})
                    conversation.append({"from": "gpt",   "value": turn_a})

                # ---- Write record ------------------------------------------
                # raw_prompts + raw_answers allow sft_main.py to reconstruct
                # the EXACT token sequence: prompt_i + answer_i (+ eos) per turn,
                # concatenated in order. This avoids the chat-template round-trip
                # that would silently drop the '<think>' generation-prompt suffix.
                record = {
                    "id":           f"{qid}-{sample_idx}",
                    "conversations": conversation,          # human-readable
                    "raw_prompts":  raw_prompts,            # exact prompt strings
                    "raw_answers":  raw_answers,            # exact generated text
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"\n[gen_sft_data] Done. Wrote {total_written} conversations to {args.output_file}")


if __name__ == "__main__":
    main()
