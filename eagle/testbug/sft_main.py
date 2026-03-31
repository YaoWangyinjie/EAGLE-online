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
    parser.add_argument("--num_epochs",  type=int,   default=200,  help="Number of training epochs.")
    parser.add_argument("--lr",          type=float, default=3e-5, help="Peak learning rate.")
    parser.add_argument("--warmup_ratio",type=float, default=0.05, help="Fraction of total steps used for LR warmup.")
    parser.add_argument("--weight_decay",type=float, default=0.0,  help="AdamW weight decay.")
    parser.add_argument("--max_len",     type=int,   default=2048, help="Maximum sequence length (longer seqs are dropped).")
    parser.add_argument("--batch_size",  type=int,   default=1,    help="Per-GPU micro batch size.")
    parser.add_argument("--grad_accum",  type=int,   default=2,    help="Gradient accumulation steps (ignored when using DeepSpeed config).")
    parser.add_argument("--test_split",  type=float, default=0.1,  help="Fraction of training data used as validation (only when --testpath is not given).")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed.")

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

def _apply_chat_template_no_last_assistant(tokenizer, messages: list) -> str:
    """
    Return the tokenized string up to (but NOT including) the last assistant reply.
    Used to compute instruction_len for building the loss mask.
    """
    # Drop the last assistant turn to get the "prompt only" prefix
    prefix_messages = messages[:-1]
    return tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_loss_mask(tokenizer, conversation_str: str, messages: list) -> torch.Tensor:
    """
    Build a 0/1 loss mask of the same length as the tokenised conversation.

    Strategy
    --------
    For each assistant turn we:
      1. Tokenise the full conversation up to AND including that turn.
      2. Tokenise the conversation up to the start of that turn (instruction prefix).
      3. Mask everything before the assistant's reply as 0; the reply itself is 1.

    This approach is tokenizer-agnostic and handles arbitrary chat templates
    (LLaMA-3 <|eot_id|>, DeepSeek <｜end▁of▁sentence｜>, etc.).
    """
    full_ids = tokenizer(
        conversation_str, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    loss_mask = torch.zeros_like(full_ids)

    # Walk the messages and identify each assistant block
    assistant_role = "assistant"
    prefix_msgs = []
    for msg in messages:
        if msg["role"] == assistant_role:
            # Prefix = everything up to this assistant turn (with generation prompt)
            prefix_str = _apply_chat_template_no_last_assistant(
                tokenizer, prefix_msgs + [msg]
            )
            prefix_ids = tokenizer(
                prefix_str, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            prefix_len = len(prefix_ids)

            # Full conversation through this assistant turn (no generation prompt)
            through_msgs = prefix_msgs + [msg]
            through_str = tokenizer.apply_chat_template(
                through_msgs, tokenize=False, add_generation_prompt=False
            )
            through_ids = tokenizer(
                through_str, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            through_len = len(through_ids)

            # The assistant reply occupies [prefix_len, through_len)
            if through_len <= len(full_ids):
                loss_mask[prefix_len:through_len] = 1

        prefix_msgs.append(msg)

    return loss_mask


def build_dataset(tokenizer, datapath: str, max_len: int, seed: int):
    """
    Load a ShareGPT-format JSONL and return a HuggingFace Dataset with
    fields: input_ids (1, T), attention_mask (1, T), loss_mask (1, T).
    """
    ds = hf_load_dataset("json", data_files=datapath)["train"]
    ds = ds.shuffle(seed=seed)
    original_columns = ds.column_names

    def preprocess(examples):
        result = {"input_ids": [], "attention_mask": [], "loss_mask": []}

        for i in range(len(examples["id"])):
            source = examples["conversations"][i]
            if not source:
                continue

            # Convert ShareGPT roles to standard roles
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            messages = []
            for turn in source:
                role = role_map.get(turn["from"], turn["from"])
                messages.append({"role": role, "content": turn["value"]})

            # Ensure the conversation starts with a user turn
            if messages and messages[0]["role"] != "user":
                messages = messages[1:]
            if not messages:
                continue

            # Apply the model's own chat template
            conversation_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            input_ids = tokenizer(
                conversation_str, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]

            if len(input_ids) > max_len:
                continue  # drop sequences that are too long

            loss_mask = build_loss_mask(tokenizer, conversation_str, messages)
            attention_mask = torch.ones_like(input_ids)

            result["input_ids"].append(input_ids[None, :])        # (1, T)
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

def find_latest_checkpoint(directory: str):
    """
    Scan *directory* for sub-directories matching 'state_N' that contain a
    DeepSpeed checkpoint ('zero_to_fp32.py') and return (path, next_epoch).
    Returns (None, 0) if no checkpoint is found.
    """
    max_epoch = -1
    for name in os.listdir(directory):
        m = re.match(r"state_(\d+)$", name)
        if m:
            subdir = os.path.join(directory, name)
            if os.path.isfile(os.path.join(subdir, "zero_to_fp32.py")):
                max_epoch = max(max_epoch, int(m.group(1)))
    if max_epoch < 0:
        return None, 0
    return os.path.join(directory, f"state_{max_epoch}"), max_epoch + 1


def save_16bit(model_engine, savedir: str, epoch: int):
    """Save a 16-bit model checkpoint (DeepSpeed save_16bit_model)."""
    path = os.path.join(savedir, f"state_{epoch}")
    os.makedirs(path, exist_ok=True)
    model_engine.save_16bit_model(path, exclude_frozen_parameters=True)
    return path


def save_plain_checkpoint(model: nn.Module, savedir: str, epoch: int):
    """
    Save a plain PyTorch checkpoint (for non-DeepSpeed single-GPU training).
    Only saves the *trainable* parameters.
    """
    path = os.path.join(savedir, f"epoch_{epoch}")
    os.makedirs(path, exist_ok=True)
    trainable_state = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    # also save non-parameter buffers (d2t, t2d)
    for k, v in model.named_buffers():
        trainable_state[k] = v
    torch.save(trainable_state, os.path.join(path, "pytorch_model.bin"))
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
        gradient_checkpoint=True,
    )

    cache_dir = args.cache_dir or args.savedir
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(cache_dir,    exist_ok=True)

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
    # Draft-vocab cache (d2t / t2d)
    # ------------------------------------------------------------------
    cache_file = os.path.join(cache_dir, "cache.pt")
    if args.rescan_vocab:
        print("[sft_main] --rescan_vocab set: rebuilding draft-vocab cache from training data.")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        # Temporarily monkey-patch cache path used by Model.scandata
        orig_cwd = os.getcwd()
        os.chdir(cache_dir)
        model.scandata(args.trainpath, args.basepath)
        os.chdir(orig_cwd)
    elif os.path.exists(cache_file):
        print(f"[sft_main] Loading existing draft-vocab cache from {cache_file}")
        cache = torch.load(cache_file)
        model.register_buffer("d2t", cache["d2t"])
        model.register_buffer("t2d", cache["t2d"])
    else:
        # Try to load from the pre-trained EAGLE model directory
        eagle_cache = os.path.join(args.eaglepath, "cache.pt")
        if os.path.exists(eagle_cache):
            print(f"[sft_main] Loading draft-vocab cache from EAGLE model dir: {eagle_cache}")
            cache = torch.load(eagle_cache)
            model.register_buffer("d2t", cache["d2t"])
            model.register_buffer("t2d", cache["t2d"])
            # Also copy to cache_dir for future runs
            torch.save(cache, cache_file)
        else:
            # Last resort: rebuild from training data
            print("[sft_main] No existing cache found. Building from training data (may misalign lm_head).")
            orig_cwd = os.getcwd()
            os.chdir(cache_dir)
            model.scandata(args.trainpath, args.basepath)
            os.chdir(orig_cwd)

    # ------------------------------------------------------------------
    # Load pre-trained EAGLE3 draft weights
    # ------------------------------------------------------------------
    eagle_weights_path = os.path.join(args.eaglepath, "pytorch_model.bin")
    print(f"[sft_main] Loading pre-trained EAGLE3 weights from {eagle_weights_path}")
    eagle_state = torch.load(eagle_weights_path, map_location="cpu")

    # Filter out frozen components (embed_tokens, d2t, t2d) which will be
    # overwritten by the base model / cache anyway.
    filtered_state = {
        k: v for k, v in eagle_state.items()
        if not k.startswith("embed_tokens") and k not in ("d2t", "t2d")
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"[sft_main]   missing keys  : {missing}")
    print(f"[sft_main]   unexpected keys: {unexpected}")

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
        start_epoch = 0  # plain training resumes manually if needed

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
                (loss / args.grad_accum).backward()
                if (batch_idx + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                    optimizer.step()
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
                ckpt = save_plain_checkpoint(model, args.savedir, epoch)
            if global_rank == 0:
                print(f"  → Saved checkpoint: {ckpt}")

    if global_rank == 0:
        print("[sft_main] Training complete.")


if __name__ == "__main__":
    main()
