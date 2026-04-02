"""
Step 2: SFT fine-tuning of the EAGLE3 draft model on a specific dataset.

Key differences vs. the original traineagle3/main.py
------------------------------------------------------
1. Loads pre-trained EAGLE3 draft weights instead of initialising from scratch.
   This means we start from the already-distilled checkpoint and only adapt it
   to the target dataset, which requires far fewer epochs and far less data.

2. chat-template is now **auto-detected** from the tokenizer.  The original code
   hard-coded the LLaMA-3 separator tokens (<|eot_id|> etc.).  DeepSeek-R1 uses
   completely different special tokens, so we build the loss_mask with a generic
   strategy: run the tokenizer's chat template once without the last assistant
   reply to get the "instruction prefix" length, then mask everything up to that
   boundary.

3. The draft-vocabulary cache (cache.pt) is either:
     (a) loaded from the pre-trained EAGLE3 model directory (recommended –
         keeps the lm_head in sync with the pre-trained weights), or
     (b) rebuilt from the new training data (use --rescan_vocab flag).
   Option (a) is the default because rebuilding the vocab would misalign the
   pre-trained lm_head weights.

4. Wandb logging is optional (--use_wandb flag, off by default).

5. A simple --test_split ratio is supported so you don't need a separate test
   file when you only have a small dataset such as mt_bench.

Usage (single-GPU, no DeepSpeed, useful for tiny datasets like mt_bench):
  python sft_main.py \
      --basepath  /path/to/DeepSeek-R1-Distill-Llama-8B \
      --eaglepath /path/to/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
      --trainpath /path/to/mt_bench_sft.jsonl \
      --savedir   ./checkpoints \
      [--num_epochs 200] \
      [--lr 3e-5] \
      [--test_split 0.1]

Usage (multi-GPU with DeepSpeed):
  deepspeed sft_main.py --deepspeed_config sft_ds_config.json \
      --basepath  /path/to/DeepSeek-R1-Distill-Llama-8B \
      --eaglepath /path/to/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
      --trainpath /path/to/mt_bench_sft.jsonl \
      --savedir   ./checkpoints
"""

import argparse
import json
import math
import os
import re
import shutil
import sys

# ---- DeepSpeed is imported later so the script also runs without it ----------
_HAS_DEEPSPEED = False
try:
    import deepspeed
    _HAS_DEEPSPEED = True
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="EAGLE3 draft-model SFT on a specific dataset")

    # Paths
    parser.add_argument(
        "--basepath", type=str,
        default="/root/paddlejob/workspace/env_run/ea/model_weight/DeepSeek-R1-Distill-Llama-8B",
        help="Path to the target (base) model.",
    )
    parser.add_argument(
        "--eaglepath", type=str,
        default="/root/paddlejob/workspace/env_run/ea/model_weight/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
        help="Path to the pre-trained EAGLE3 draft model to fine-tune.",
    )
    parser.add_argument(
        "--trainpath", type=str,
        default="/root/paddlejob/workspace/env_run/ea/EAGLE/eagle/sft_overfit/data/mt_bench_sft.jsonl",
        help="Path to the ShareGPT-format JSONL training file.",
    )
    parser.add_argument(
        "--testpath", type=str, default=None,
        help=(
            "Optional path to a separate test JSONL. "
            "If not given, --test_split fraction of --trainpath is used."
        ),
    )
    parser.add_argument(
        "--savedir", type=str,
        default="/root/paddlejob/workspace/env_run/ea/EAGLE/eagle/sft_overfit/checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--cache_dir", type=str,
        default=None,
        help=(
            "Directory that contains (or will contain) the draft-vocab cache.pt. "
            "Defaults to --savedir."
        ),
    )

    # Training hyper-parameters
    parser.add_argument("--num_epochs",  type=int,   default=200,
                        help="Total number of epochs to train. When resuming, this is "
                             "still the *total* count (not additional epochs), so set it "
                             "to start_epoch + extra_epochs you want to run.")
    parser.add_argument("--lr",          type=float, default=1e-5, help="Peak learning rate.")
    parser.add_argument("--warmup_ratio",type=float, default=0.05, help="Fraction of total steps used for LR warmup.")
    parser.add_argument("--weight_decay",type=float, default=0.0,  help="AdamW weight decay.")
    parser.add_argument("--max_len",     type=int,   default=2048, help="Maximum sequence length (longer seqs are dropped).")
    parser.add_argument("--batch_size",  type=int,   default=1,    help="Per-GPU micro batch size.")
    parser.add_argument("--grad_accum",  type=int,   default=2,    help="Gradient accumulation steps (ignored when using DeepSpeed config).")
    parser.add_argument("--test_split",  type=float, default=0.1,  help="Fraction of training data used as validation (only when --testpath is not given).")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed.")
    parser.add_argument("--train_id",    type=int,   default=0,
                        help="Integer identifier for this training run. Used to namespace "
                             "checkpoint directories so different runs (lr, data, etc.) "
                             "don't overwrite each other.  e.g. --train_id 1")

    # Checkpoint resume
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help=(
            "Path to a checkpoint directory (e.g. checkpoints/epoch_199) to resume "
            "from.  Pass 'auto' to automatically find the latest checkpoint in --savedir."
        ),
    )

    # Draft-vocab handling
    parser.add_argument(
        "--rescan_vocab", action="store_true",
        help=(
            "Rebuild the draft-vocab cache from --trainpath. "
            "WARNING: this invalidates the pre-trained lm_head weights. "
            "Only use this if you initialise the lm_head from scratch."
        ),
    )

    # Logging
    parser.add_argument("--use_wandb",      action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project",  type=str, default="eagle3-sft-overfit")
    parser.add_argument("--wandb_entity",   type=str, default="")
    parser.add_argument("--log_every",      type=int, default=10,  help="Log training metrics every N batches.")
    parser.add_argument("--save_every",     type=int, default=10,  help="Save a 16-bit checkpoint every N epochs.")

    # DeepSpeed / distributed
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (set by DeepSpeed launcher).")

    if _HAS_DEEPSPEED:
        parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Imports that depend on the local traineagle3 package
# ---------------------------------------------------------------------------
# We run from the sft_overfit directory but need traineagle3's cnets / configs.
_TRAINEAGLE3_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "traineagle3",
)
sys.path.insert(0, _TRAINEAGLE3_DIR)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset as hf_load_dataset
from accelerate.utils import set_seed

from cnets import Model, padding
from configs import EConfig


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def build_loss_mask_from_raw(
    tokenizer,
    input_ids: torch.Tensor,
    raw_prompts: list,
    raw_answers: list,
) -> torch.Tensor:
    """
    Build a 0/1 loss mask using the EXACT prompt strings saved by gen_sft_data.py.

    The conversation token sequence is:  prompt_0 + answer_0 + eos
                                       + prompt_1 + answer_1 + eos
                                       + ...
    We mask prompt tokens as 0 and answer tokens as 1, so the draft model only
    learns to predict assistant tokens (no leakage from question/instruction).

    This avoids the chat-template round-trip that silently drops the <think>
    generation-prompt suffix, which would shift the loss mask by ~3 tokens.
    """
    loss_mask = torch.zeros_like(input_ids)
    eos_id    = tokenizer.eos_token_id
    cursor    = 0

    for prompt_str, answer_str in zip(raw_prompts, raw_answers):
        # Prompt section – mask = 0
        prompt_ids = tokenizer(
            prompt_str, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        prompt_len = len(prompt_ids)

        # Answer section – mask = 1
        # The answer was decoded; re-encode it to get exact token count.
        answer_ids = tokenizer(
            answer_str, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        answer_len = len(answer_ids)

        # Mark answer tokens as training targets
        ans_start = cursor + prompt_len
        ans_end   = ans_start + answer_len
        if ans_end <= len(loss_mask):
            loss_mask[ans_start:ans_end] = 1

        # Advance past prompt + answer + eos token
        cursor = ans_end + 1  # +1 for the eos that the template appends

    return loss_mask


def build_conversation_str_from_raw(raw_prompts: list, raw_answers: list, eos_token: str) -> str:
    """
    Reconstruct the EXACT conversation text as seen by the target model during
    generation, by concatenating:  prompt_i + answer_i + eos  for each turn.

    This matches the original generation exactly, including the '<think>\\n'
    suffix that the generation prompt injects before the model's output.
    """
    parts = []
    for prompt_str, answer_str in zip(raw_prompts, raw_answers):
        parts.append(prompt_str + answer_str + eos_token)
    return "".join(parts)


def build_dataset(tokenizer, datapath: str, max_len: int, seed: int):
    """
    Load a ShareGPT-format JSONL (extended with raw_prompts / raw_answers fields
    written by gen_sft_data.py) and return a HuggingFace Dataset with fields:
      input_ids (1, T), attention_mask (1, T), loss_mask (1, T).

    If raw_prompts / raw_answers are present the conversation string is
    reconstructed directly from them (exact token match with generation time).
    Otherwise the function falls back to the chat-template reconstruction
    path for compatibility with plain ShareGPT data.
    """
    ds = hf_load_dataset("json", data_files=datapath)["train"]
    ds = ds.shuffle(seed=seed)
    original_columns = ds.column_names

    eos_token = tokenizer.eos_token or ""

    def preprocess(examples):
        result = {"input_ids": [], "attention_mask": [], "loss_mask": []}
        n = len(examples["id"])

        has_raw = "raw_prompts" in examples and examples["raw_prompts"][0] is not None

        for i in range(n):
            if has_raw:
                # ---- Preferred path: exact reconstruction ------------------
                raw_prompts = examples["raw_prompts"][i]
                raw_answers = examples["raw_answers"][i]
                conversation_str = build_conversation_str_from_raw(
                    raw_prompts, raw_answers, eos_token
                )
                input_ids = tokenizer(
                    conversation_str, return_tensors="pt", add_special_tokens=False
                ).input_ids[0]
                if len(input_ids) > max_len:
                    continue
                loss_mask = build_loss_mask_from_raw(
                    tokenizer, input_ids, raw_prompts, raw_answers
                )
            else:
                # ---- Fallback: chat-template reconstruction ----------------
                source = examples["conversations"][i]
                if not source:
                    continue
                role_map = {"human": "user", "gpt": "assistant", "system": "system"}
                messages = [{"role": role_map.get(t["from"], t["from"]),
                             "content": t["value"]} for t in source]
                if messages and messages[0]["role"] != "user":
                    messages = messages[1:]
                if not messages:
                    continue
                conversation_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                input_ids = tokenizer(
                    conversation_str, return_tensors="pt", add_special_tokens=False
                ).input_ids[0]
                if len(input_ids) > max_len:
                    continue
                # Fallback loss mask: mark all non-zero positions as 1
                # (rough approximation; use raw_prompts path for accuracy)
                loss_mask = torch.ones_like(input_ids)

            attention_mask = torch.ones_like(input_ids)
            result["input_ids"].append(input_ids[None, :])
            result["attention_mask"].append(attention_mask[None, :])
            result["loss_mask"].append(loss_mask[None, :])

        return result

    ds = ds.map(
        preprocess,
        batched=True,
        num_proc=4,
        remove_columns=original_columns,
        load_from_cache_file=False,
    )
    ds.set_format(type="torch")
    return ds


class PaddingCollator:
    """Pad a batch of variable-length sequences to the length of the longest one."""

    @staticmethod
    def _pad2d(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """tensor: (1, T)  →  (1, target_len)"""
        B, n = tensor.shape
        pad = torch.zeros(B, target_len - n, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=1)

    def __call__(self, features):
        max_len = max(item["input_ids"].shape[1] for item in features)
        return {
            "input_ids":      torch.cat([self._pad2d(f["input_ids"],      max_len) for f in features]),
            "attention_mask": torch.cat([self._pad2d(f["attention_mask"], max_len) for f in features]),
            "loss_mask":      torch.cat([self._pad2d(f["loss_mask"],      max_len) for f in features]),
        }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def format_lr(lr: float) -> str:
    """
    Convert a float learning rate to a compact string for use in directory names.
    Examples: 1e-5 → '1e-5',  3e-5 → '3e-5',  1.5e-4 → '1.5e-4'
    """
    s = f"{lr:.10e}"
    mantissa, exp = s.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def ckpt_prefix(train_id: int, lr: float) -> str:
    """Return the directory-name prefix that identifies a (train_id, lr) run."""
    return f"{train_id}_{format_lr(lr)}"


def find_latest_checkpoint(directory: str, train_id: int = None, lr: float = None):
    """
    Scan *directory* for the most recent checkpoint sub-directory.

    Formats recognised:
      - Plain     : '{train_id}_{lr}_epoch_N/'  containing 'pytorch_model.bin'
      - DeepSpeed : 'state_N/'                  containing 'zero_to_fp32.py'

    When *train_id* and *lr* are given, only plain checkpoints whose prefix
    matches ckpt_prefix(train_id, lr) are considered (e.g. '0_1e-5_epoch_*').

    Returns (path, next_epoch).  next_epoch = saved_epoch + 1.
    Returns (None, 0) when nothing matches.
    """
    if not os.path.isdir(directory):
        return None, 0

    max_epoch = -1
    best_path = None

    prefix_filter = ckpt_prefix(train_id, lr) if (train_id is not None and lr is not None) else None

    for name in os.listdir(directory):
        subdir = os.path.join(directory, name)
        if not os.path.isdir(subdir):
            continue

        # Plain checkpoint: {prefix}_epoch_N
        m_plain = re.match(r"^(.+)_epoch_(\d+)$", name)
        if m_plain and os.path.isfile(os.path.join(subdir, "pytorch_model.bin")):
            prefix = m_plain.group(1)
            epoch  = int(m_plain.group(2))
            if prefix_filter is not None and prefix != prefix_filter:
                continue
            if epoch > max_epoch:
                max_epoch = epoch
                best_path = subdir
            continue

        # DeepSpeed checkpoint: state_N (no prefix filtering)
        m_ds = re.match(r"^state_(\d+)$", name)
        if m_ds and os.path.isfile(os.path.join(subdir, "zero_to_fp32.py")):
            epoch = int(m_ds.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_path = subdir

    if max_epoch < 0:
        return None, 0
    return best_path, max_epoch + 1


def save_plain_checkpoint(
    model: nn.Module,
    savedir: str,
    epoch: int,
    train_id: int = 0,
    lr: float = 3e-5,
    optimizer=None,
    scaler=None,
    scheduler=None,
    config_src: str = None,
):
    """
    Save a plain PyTorch checkpoint (non-DeepSpeed single-GPU training).

    Directory name: '{train_id}_{lr}_epoch_{epoch}'
    e.g.  '0_1e-5_epoch_19'  for train_id=0, lr=1e-5, epoch 19.

    Files saved inside the directory:
      config.json        – copied from config_src (required by EaModel.from_pretrained)
      pytorch_model.bin  – trainable parameters + buffers (d2t, t2d)
      optimizer.pt       – optimizer state  (if provided)
      scaler.pt          – GradScaler state (if provided)
      scheduler.pt       – LR scheduler state (if provided)
    """
    dirname = f"{ckpt_prefix(train_id, lr)}_epoch_{epoch}"
    path = os.path.join(savedir, dirname)
    os.makedirs(path, exist_ok=True)

    # config.json is required by EaModel.from_pretrained; copy it from the
    # original EAGLE model directory so the checkpoint is self-contained.
    if config_src is not None:
        src = os.path.join(config_src, "config.json")
        dst = os.path.join(path, "config.json")
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy(src, dst)

    # Use named_parameters() to identify trainable keys; state_dict() tensors
    # are always detached so their requires_grad flag is meaningless.
    trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    model_state = {k: v for k, v in model.state_dict().items() if k in trainable_keys}
    for k, v in model.named_buffers():   # also persist d2t, t2d for inference
        model_state[k] = v

    torch.save(model_state, os.path.join(path, "pytorch_model.bin"))

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(path, "scaler.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # DeepSpeed vs. plain training setup
    # ------------------------------------------------------------------
    use_deepspeed = _HAS_DEEPSPEED and hasattr(args, "deepspeed_config") and args.deepspeed_config

    if use_deepspeed:
        with open(args.deepspeed_config) as f:
            ds_config = json.load(f)
    else:
        ds_config = None

    # ------------------------------------------------------------------
    # Training config dict (mirrors traineagle3/main.py's train_config)
    # ------------------------------------------------------------------
    from types import SimpleNamespace
    train_config = SimpleNamespace(
        bs=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=2,
        max_len=args.max_len,
        gradient_checkpointing=True,
    )

    cache_dir = args.cache_dir or args.savedir  # only used when --rescan_vocab
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(cache_dir,    exist_ok=True)    # safe even if same as savedir

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"[sft_main] Loading tokenizer from {args.basepath}")
    tokenizer = AutoTokenizer.from_pretrained(args.basepath, use_fast=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"[sft_main] Building dataset from {args.trainpath}")
    full_dataset = build_dataset(tokenizer, args.trainpath, args.max_len, args.seed)
    print(f"[sft_main]   → {len(full_dataset)} samples after filtering")

    if args.testpath:
        train_dataset = full_dataset
        test_dataset  = build_dataset(tokenizer, args.testpath, args.max_len, args.seed)
    else:
        n_test  = max(1, int(len(full_dataset) * args.test_split))
        n_train = len(full_dataset) - n_test
        train_dataset, test_dataset = random_split(
            full_dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"[sft_main]   → train={n_train}  test={n_test}  (test_split={args.test_split})")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"[sft_main] Building EAGLE3 Model (target model: {args.basepath})")
    config = EConfig.from_pretrained(os.path.join(args.eaglepath, "config.json"))
    model  = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True)

    # ------------------------------------------------------------------
    # Load pre-trained EAGLE3 draft weights
    # (d2t / t2d buffers live inside pytorch_model.bin, NOT in a separate
    #  cache.pt – load them first so lm_head stays perfectly aligned)
    # ------------------------------------------------------------------
    eagle_weights_path = os.path.join(args.eaglepath, "pytorch_model.bin")
    print(f"[sft_main] Loading pre-trained EAGLE3 weights from {eagle_weights_path}")
    eagle_state = torch.load(eagle_weights_path, map_location="cpu")

    if args.rescan_vocab:
        # Explicitly asked to rebuild the vocab mapping from the new dataset.
        # WARNING: this invalidates the pre-trained lm_head weights because the
        # token index ordering will change.
        print("[sft_main] --rescan_vocab: rebuilding d2t/t2d from training data.")
        cache_file = os.path.join(cache_dir, "cache.pt")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        orig_cwd = os.getcwd()
        os.chdir(cache_dir)
        model.scandata(args.trainpath, args.basepath)
        os.chdir(orig_cwd)
        # Exclude d2t/t2d from the weight load so the freshly scanned values win
        eagle_state = {k: v for k, v in eagle_state.items() if k not in ("d2t", "t2d")}
    else:
        # Default: take d2t/t2d directly from the pre-trained bin file.
        # This keeps lm_head weights valid (they were trained with this vocab mapping).
        if "d2t" in eagle_state and "t2d" in eagle_state:
            print("[sft_main] Loading d2t/t2d vocab mapping from pre-trained EAGLE3 weights.")
            model.register_buffer("d2t", eagle_state["d2t"])
            model.register_buffer("t2d", eagle_state["t2d"])
        else:
            raise RuntimeError(
                "d2t/t2d not found in pre-trained weights. "
                "Pass --rescan_vocab to rebuild them from the training data."
            )

    # Load trainable parameters (skip embed_tokens – already loaded from base model)
    filtered_state = {
        k: v for k, v in eagle_state.items()
        if not k.startswith("embed_tokens")
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    # embed_tokens is expected missing (frozen, already set by Model.__init__)
    expected_missing = [k for k in missing if "embed_tokens" in k or k in ("d2t", "t2d")]
    real_missing     = [k for k in missing if k not in expected_missing]
    if real_missing:
        print(f"[sft_main] WARNING – unexpected missing keys: {real_missing}")
    else:
        print("[sft_main] Pre-trained weights loaded successfully.")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    if use_deepspeed:
        world_size  = deepspeed.comm.get_world_size()
        global_rank = deepspeed.comm.get_rank()
        local_rank  = deepspeed.comm.get_local_rank()

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size, rank=global_rank, shuffle=False)
    else:
        global_rank = 0
        local_rank  = 0
        train_sampler = None
        test_sampler  = None

    collator = PaddingCollator()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True, collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=test_sampler, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collator,
    )

    # ------------------------------------------------------------------
    # Optimizer & LR scheduler (plain training path)
    # ------------------------------------------------------------------
    total_steps   = len(train_loader) * args.num_epochs // args.grad_accum
    warmup_steps  = max(1, int(total_steps * args.warmup_ratio))

    if use_deepspeed:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args, model=model, model_parameters=model.parameters()
        )
    else:
        # Plain single-GPU training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        # GradScaler for AMP: target model outputs float16 hidden_states, trainable
        # layers are float32 – autocast handles the dtype boundary automatically.
        scaler = torch.cuda.amp.GradScaler()
        model_engine = model  # alias for uniform code below

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    if global_rank == 0 and args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            config=vars(args),
        )

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if use_deepspeed:
        ckpt_path, start_epoch = find_latest_checkpoint(args.savedir)
        if ckpt_path:
            print(f"[sft_main] Resuming DeepSpeed checkpoint from {ckpt_path}")
            model_engine.load_checkpoint(ckpt_path)
    else:
        # Resolve the checkpoint path to resume from
        resume_path = None
        if args.resume_from == "auto":
            resume_path, start_epoch = find_latest_checkpoint(
                args.savedir, train_id=args.train_id, lr=args.lr
            )
            if resume_path:
                print(f"[sft_main] Auto-detected latest checkpoint: {resume_path}  (start_epoch={start_epoch})")
            else:
                print("[sft_main] No existing checkpoint found in savedir, starting from scratch.")
                start_epoch = 0
        elif args.resume_from is not None:
            resume_path = args.resume_from
            # Infer start_epoch from directory name: anything ending in _epoch_N
            m = re.search(r"_epoch_(\d+)$", resume_path.rstrip("/"))
            start_epoch = int(m.group(1)) + 1 if m else 0
            print(f"[sft_main] Resuming from {resume_path}  (start_epoch={start_epoch})")
        else:
            start_epoch = 0

        if resume_path is not None:
            ckpt_model = os.path.join(resume_path, "pytorch_model.bin")
            if not os.path.isfile(ckpt_model):
                raise FileNotFoundError(f"Checkpoint model weights not found: {ckpt_model}")

            print(f"[sft_main]   Loading model weights from {ckpt_model}")
            ckpt_state = torch.load(ckpt_model, map_location=device)
            # Load only the keys that exist in the current model (buffers + trainable params)
            model.load_state_dict(ckpt_state, strict=False)

            ckpt_opt = os.path.join(resume_path, "optimizer.pt")
            if os.path.isfile(ckpt_opt):
                print(f"[sft_main]   Loading optimizer state from {ckpt_opt}")
                optimizer.load_state_dict(torch.load(ckpt_opt, map_location="cpu"))
            else:
                print("[sft_main]   optimizer.pt not found – LR schedule will restart.")

            ckpt_scaler = os.path.join(resume_path, "scaler.pt")
            if os.path.isfile(ckpt_scaler):
                print(f"[sft_main]   Loading scaler state from {ckpt_scaler}")
                scaler.load_state_dict(torch.load(ckpt_scaler, map_location="cpu"))

            ckpt_sched = os.path.join(resume_path, "scheduler.pt")
            if os.path.isfile(ckpt_sched):
                print(f"[sft_main]   Loading scheduler state from {ckpt_sched}")
                scheduler.load_state_dict(torch.load(ckpt_sched, map_location="cpu"))
            else:
                # Fast-forward scheduler to the correct step without optimizer.pt
                steps_done = start_epoch * len(train_loader) // args.grad_accum
                for _ in range(steps_done):
                    scheduler.step()
                print(f"[sft_main]   scheduler.pt not found – fast-forwarded {steps_done} steps.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    ploss_weight = [0.8 ** i for i in range(model.length)]

    for epoch in range(start_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch + 1)

        model.train()
        epoch_plosses = [[] for _ in range(model.length)]
        epoch_acces   = [[] for _ in range(model.length)]

        if not use_deepspeed:
            optimizer.zero_grad()

        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]")):
            if use_deepspeed:
                rank_device = local_rank
            else:
                rank_device = device

            if use_deepspeed:
                plosses, _, acces = model_engine(
                    input_ids=data["input_ids"].to(rank_device),
                    attention_mask=data["attention_mask"].to(rank_device),
                    loss_mask=data["loss_mask"],
                )
            else:
                with torch.cuda.amp.autocast():
                    plosses, _, acces = model_engine(
                        input_ids=data["input_ids"].to(rank_device),
                        attention_mask=data["attention_mask"].to(rank_device),
                        loss_mask=data["loss_mask"],
                    )

            loss = sum(ploss_weight[i] * plosses[i] for i in range(len(plosses)))

            if use_deepspeed:
                model_engine.backward(loss)
                model_engine.step()
            else:
                scaler.scale(loss / args.grad_accum).backward()
                if (batch_idx + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]
            epoch_acces   = [epoch_acces[i]   + [acces[i]]          for i in range(len(acces))]

            if global_rank == 0 and args.use_wandb and batch_idx % args.log_every == 0:
                import wandb
                log = {"train/total_loss": loss.item()}
                for i in range(len(plosses)):
                    log[f"train/ploss_{i}"] = plosses[i].item()
                    log[f"train/acc_{i}"]   = acces[i]
                wandb.log(log)

        # ---- epoch-level train metrics -----------------------------------
        for i in range(model.length):
            mean_ploss = np.mean(epoch_plosses[i]) if epoch_plosses[i] else float("nan")
            mean_acc   = np.mean(epoch_acces[i])   if epoch_acces[i]   else float("nan")
            if global_rank == 0:
                print(f"  Train Epoch {epoch+1:>4} | pos {i} | ploss={mean_ploss:.4f} | acc={mean_acc:.4f}")
                if args.use_wandb:
                    import wandb
                    wandb.log({f"train/epoch_ploss_{i}": mean_ploss, f"train/epoch_acc_{i}": mean_acc})

        # ---- validation --------------------------------------------------
        model.eval()
        epoch_plosses_val = [[] for _ in range(model.length)]
        epoch_acces_val   = [[] for _ in range(model.length)]

        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [eval]"):
                if use_deepspeed:
                    rank_device = local_rank
                else:
                    rank_device = device

                if use_deepspeed:
                    plosses, _, acces = model_engine(
                        input_ids=data["input_ids"].to(rank_device),
                        attention_mask=data["attention_mask"].to(rank_device),
                        loss_mask=data["loss_mask"],
                    )
                else:
                    with torch.cuda.amp.autocast():
                        plosses, _, acces = model_engine(
                            input_ids=data["input_ids"].to(rank_device),
                            attention_mask=data["attention_mask"].to(rank_device),
                            loss_mask=data["loss_mask"],
                        )
                epoch_plosses_val = [epoch_plosses_val[i] + [plosses[i].item()] for i in range(len(plosses))]
                epoch_acces_val   = [epoch_acces_val[i]   + [acces[i]]          for i in range(len(acces))]

        for i in range(model.length):
            mean_ploss = np.mean(epoch_plosses_val[i]) if epoch_plosses_val[i] else float("nan")
            mean_acc   = np.mean(epoch_acces_val[i])   if epoch_acces_val[i]   else float("nan")
            if global_rank == 0:
                print(f"  Val   Epoch {epoch+1:>4} | pos {i} | ploss={mean_ploss:.4f} | acc={mean_acc:.4f}")
                if args.use_wandb:
                    import wandb
                    wandb.log({f"val/epoch_ploss_{i}": mean_ploss, f"val/epoch_acc_{i}": mean_acc})

        torch.cuda.empty_cache()

        # ---- checkpointing -----------------------------------------------
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            if use_deepspeed:
                ckpt = save_16bit(model_engine, args.savedir, epoch)
                if epoch % 50 == 0:
                    deepspeed.DeepSpeedEngine.save_checkpoint(
                        model_engine, save_dir=os.path.join(args.savedir, f"state_{epoch}")
                    )
            else:
                ckpt = save_plain_checkpoint(
                    model, args.savedir, epoch,
                    train_id=args.train_id,
                    lr=args.lr,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    config_src=args.eaglepath,
                )
            if global_rank == 0:
                print(f"  → Saved checkpoint: {ckpt}")

    if global_rank == 0:
        print("[sft_main] Training complete.")


if __name__ == "__main__":
    main()
