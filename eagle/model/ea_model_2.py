import copy
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig

const_temp = 1.0
const_lr = 1e-5
const_train_steps = 1  # 从 5 改为 1，防止过拟合
const_warmup_steps = 5
const_enable_debug = False  # ⚠️ 控制是否输出详细 Debug（会影响速度统计！）

class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3

        self.enable_online_adaptation = False
        self.adaptation_scope = "head_only"
        self.adaptation_objective = "kl"
        self.reset_granularity = "choice"
        self.adaptation_mode = "reset"
        self.anchor_weight = 1e-3
        self.max_relative_drift = 0.02
        self.gradient_clip_norm = 0.5
        self.weight_version = 0
        self.cache_epoch = 0
        self.update_count = 0
        self.rollback_count = 0
        self.adaptation_lr = const_lr
        self.adaptation_temperature = const_temp
        self.adapter_optimizer = None
        self.initial_weights = None
        self.diagnostics = False
        self._diagnostic_events = []

        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device
        else:
            self.ea_layer.diff_device = False
            
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()
        if hasattr(self.ea_layer, 'd2t') and hasattr(self.ea_layer, 't2d'):
            self.ea_layer.d2t = self.ea_layer.d2t.cpu()
            self.ea_layer.t2d = self.ea_layer.t2d.cpu()
            self.has_vocab_mapping = True
            self.draft_vocab_size = config.draft_vocab_size
        else:
            self.has_vocab_mapping = False
            self.draft_vocab_size = config.vocab_size

    def _cache_capacity(self, cache_data):
        return min(int(item.shape[-2]) for item in cache_data)

    def _tree_fits_cache(self, input_ids, draft_tokens, cache_data):
        return input_ids.shape[1] + draft_tokens.shape[-1] <= self._cache_capacity(cache_data)

    def _reset_base_cache(self, reason):
        if hasattr(self, "current_length_data"):
            self.current_length_data.zero_()
        if hasattr(self.base_model.model, "tree_mask"):
            self.base_model.model.tree_mask = None
        if hasattr(self.base_model.model, "tree_mode"):
            self.base_model.model.tree_mode = None
        self._record_diagnostic("base_cache_reset", reason=reason)

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def _record_diagnostic(self, event, **details):
        if not self.diagnostics:
            return
        record = {"event": event, **details}
        self._diagnostic_events.append(record)
        print(f"[Online Adaptation] {event}: {details}")

    def _adapter_named_parameters(self):
        scopes = {
            "default": ("fc.", "midlayer.", "norm.", "lm_head."),
            "head_only": ("norm.", "lm_head."),
            "fc_only": ("fc.",),
            "midlayer_only": ("midlayer.",),
            "norm_only": ("norm.",),
            "lm_head_only": ("lm_head.",),
        }
        if self.adaptation_scope not in scopes:
            raise ValueError(f"Unsupported adaptation scope: {self.adaptation_scope}")
        allowed_prefixes = scopes[self.adaptation_scope]
        for name, parameter in self.ea_layer.named_parameters():
            allowed = name.startswith(allowed_prefixes)
            parameter.requires_grad_(allowed)
            if allowed:
                yield name, parameter

    def _reset_adapter_optimizer(self):
        trainable = list(self._adapter_named_parameters())
        if not trainable:
            raise RuntimeError("No adapter parameters are eligible for online adaptation")
        self.adapter_optimizer = torch.optim.Adam(
            [parameter for _, parameter in trainable], lr=self.adaptation_lr
        )
        self._record_diagnostic(
            "optimizer_reset",
            trainable_parameters=[name for name, _ in trainable],
            embed_tokens_excluded=True,
            learning_rate=float(self.adaptation_lr),
            state_entries=len(self.adapter_optimizer.state),
        )

    @staticmethod
    def _validate_adaptation_hyperparameter(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite positive number") from None
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
        return value

    @staticmethod
    def _tensor_tree_is_finite(value):
        if torch.is_tensor(value):
            return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(EaModel._tensor_tree_is_finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(EaModel._tensor_tree_is_finite(item) for item in value)
        return True

    @staticmethod
    def _fits_dtype(value, dtype):
        if not value.is_floating_point() or not dtype.is_floating_point:
            return True
        limit = torch.finfo(dtype).max
        return bool(torch.isfinite(value).all() and (value.abs() <= limit).all())

    def _extract_eagle_hidden(self, outputs, source):
        if not self.use_eagle3:
            hidden = outputs[0]
            indices = ["last_hidden_state"]
        else:
            states = outputs.hidden_states
            if states is None or len(states) < 3:
                raise AssertionError(
                    f"{source}: expected at least three EAGLE hidden components, "
                    f"got {0 if states is None else len(states)}"
                )
            indices = list(range(len(states) - 3, len(states)))
            selected = [states[index] for index in indices]
            reference_shape = selected[0].shape[:-1]
            if any(state.shape[:-1] != reference_shape for state in selected):
                raise AssertionError(f"{source}: hidden component shapes do not align")
            hidden = torch.cat(selected, dim=-1)
        expected = getattr(getattr(self.ea_layer, "fc", None), "in_features", hidden.shape[-1])
        if hidden.shape[-1] != expected:
            raise AssertionError(
                f"{source}: hidden width {hidden.shape[-1]} != ea_layer.fc.in_features {expected}"
            )
        self._record_diagnostic(
            "hidden_extraction", source=source, indices=indices,
            component_count=3 if self.use_eagle3 else 1, shape=list(hidden.shape),
            fc_in_features=expected,
        )
        return hidden

    @staticmethod
    def _make_fragment(tokens, hidden, logits, source):
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        lengths = (tokens.shape[1], hidden.shape[1], logits.shape[1])
        if len(set(lengths)) != 1:
            raise AssertionError(f"{source}: token/hidden/logit lengths differ: {lengths}")
        if lengths[0] < 2:
            return None
        return {"tokens": tokens.cpu(), "hidden": hidden.cpu(),
                "logits": logits.cpu(), "source": source,
                "root": int(tokens[0, 0].item())}

    def setup_online_adaptation(
            self,
            adaptation_lr=const_lr,
            adaptation_temperature=const_temp,
            mode="reset",
            scope="default",
            objective="kl",
            reset_granularity="choice",
        ):
        self.adaptation_lr = self._validate_adaptation_hyperparameter(
            "adaptation_lr", adaptation_lr
        )
        if mode not in {"reset", "persistent"}:
            raise ValueError(f"Unsupported adaptation mode: {mode}")
        if scope not in {"default", "head_only", "fc_only", "midlayer_only", "norm_only", "lm_head_only"}:
            raise ValueError(f"Unsupported adaptation scope: {scope}")
        if objective not in {"kl", "acceptance", "acceptance_weighted_kl"}:
            raise ValueError(f"Unsupported adaptation objective: {objective}")
        if reset_granularity not in {"choice", "turn", "stream"}:
            raise ValueError(f"Unsupported reset granularity: {reset_granularity}")
        self.adaptation_scope = scope
        self.adaptation_objective = objective
        self.reset_granularity = reset_granularity
        self.adaptation_temperature = self._validate_adaptation_hyperparameter(
            "adaptation_temperature", adaptation_temperature
        )
        self.adaptation_mode = mode
        self.enable_online_adaptation = True
        if self.initial_weights is None:
            self.initial_weights = {
                key: value.detach().cpu().clone()
                for key, value in self.ea_layer.state_dict().items()
            }
        self._reset_adapter_optimizer()

    def reset_online_adaptation(self):
        """Restore the initial EA model and clear all adaptation-dependent caches."""
        if self.initial_weights is None:
            self.initial_weights = {
                key: value.detach().cpu().clone()
                for key, value in self.ea_layer.state_dict().items()
            }
        self.ea_layer.load_state_dict(self.initial_weights, strict=False)
        self._reset_adapter_optimizer()
        self.ea_layer.reset_kv()
        self.ea_layer.stable_kv = None
        if hasattr(self.base_model.model, "tree_mask"):
            self.base_model.model.tree_mask = None
        if hasattr(self.base_model.model, "tree_mode"):
            self.base_model.model.tree_mode = None
        self._record_diagnostic("adapter_reset", cache_state="invalidated")

    def _prepare_fragment_batch(self, fragment, device):
        tokens = fragment["tokens"].to(device)
        hidden = fragment["hidden"].to(device=device, dtype=torch.float32)
        target = fragment["logits"].to(device=device, dtype=torch.float32)
        lengths = (tokens.shape[1], hidden.shape[1], target.shape[1])
        if len(set(lengths)) != 1 or lengths[0] < 2:
            raise AssertionError(f"Invalid fragment alignment: {lengths}")
        # candidates[:, 1:] is evaluated against logits[:, :-1].
        return tokens[:, 1:], hidden[:, :-1], target[:, :-1]

    def _forward_training_fragment(self, tokens, hidden):
        embeds = self.ea_layer.embed_tokens(tokens)
        expected = getattr(self.ea_layer.fc, "in_features", hidden.shape[-1])
        if hidden.shape[-1] != expected:
            raise AssertionError(f"Training hidden width {hidden.shape[-1]} != fc.in_features {expected}")
        hidden = self.ea_layer.fc(hidden)
        batch_size, seq_len = tokens.shape
        attention_mask = torch.ones((batch_size, seq_len), device=tokens.device, dtype=torch.bool)
        attention_mask = self.ea_layer._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_len), embeds, 0
        )
        layer_outputs = self.ea_layer.midlayer(
            input_emb=embeds, hidden_states=hidden,
            attention_mask=attention_mask, use_cache=False,
        )
        return self.ea_layer.lm_head(self.ea_layer.norm(layer_outputs[0]))

    def _perform_online_adaptation(self, fragments, train_steps=const_train_steps):
        counts = {
            "fragments_total": len(fragments),
            "fragments_used": 0,
            "fragments_rejected_nonfinite_hidden": 0,
            "positions_total": 0,
            "positions_valid": 0,
            "positions_nonfinite_target": 0,
            "positions_nonfinite_draft": 0,
            "positions_invalid_vocab_mapping": 0,
            "positions_finite": 0,
            "positions_mapping_valid": 0,
        }
        result = {
            "loss": None,
            "status": "skipped",
            "updated": False,
            "rolled_back": False,
            "skip_reason": None,
            "counts": counts,
            "grad_norm": None,
            "valid_vocab_ratio": None,
            "effective_hyperparameters": {
                "learning_rate": float(self.adaptation_lr),
                "temperature": float(self.adaptation_temperature),
                "train_steps": int(train_steps),
            },
        }
        if not fragments:
            result["skip_reason"] = "no_fragments"
            self._record_diagnostic("update_skipped", reason=result["skip_reason"])
            return result
        if train_steps <= 0:
            result["skip_reason"] = "no_train_steps"
            self._record_diagnostic("update_skipped", reason=result["skip_reason"])
            return result
        if self.adapter_optimizer is None:
            raise RuntimeError("Online adaptation optimizer has not been initialized")
        device = next(self.ea_layer.parameters()).device
        original_dtype = next(self.ea_layer.parameters()).dtype
        was_training = self.ea_layer.training
        saved_tree_mask = getattr(self.ea_layer, "tree_mask", None)
        saved_stable_kv = getattr(self.ea_layer, "stable_kv", None)
        named_trainable = list(self._adapter_named_parameters())
        trainable = [parameter for _, parameter in named_trainable]
        parameter_snapshot = None
        optimizer_snapshot = None
        update_attempted = False
        losses = []
        try:
            self.ea_layer.float()
            self.ea_layer.reset()
            self.ea_layer.stable_kv = None
            self.ea_layer.train()
            parameter_snapshot = [parameter.detach().clone() for parameter in trainable]
            optimizer_snapshot = copy.deepcopy(self.adapter_optimizer.state_dict())
            for _ in range(train_steps):
                self.adapter_optimizer.zero_grad(set_to_none=True)
                token_loss_sum = None
                valid_positions = 0
                total_positions = 0
                for fragment in fragments:
                    tokens, hidden, target = self._prepare_fragment_batch(fragment, device)
                    positions = int(tokens.numel())
                    total_positions += positions
                    counts["positions_total"] += positions
                    if not bool(torch.isfinite(hidden).all()):
                        counts["fragments_rejected_nonfinite_hidden"] += 1
                        continue
                    draft = self._forward_training_fragment(tokens, hidden).float()
                    target = target.float()
                    if draft.shape[:-1] != target.shape[:-1]:
                        raise AssertionError(
                            f"Draft/target token shapes differ: {draft.shape[:-1]} != {target.shape[:-1]}"
                        )
                    target_finite = torch.isfinite(target).all(dim=-1)
                    draft_finite = torch.isfinite(draft).all(dim=-1)
                    counts["positions_nonfinite_target"] += int((~target_finite).sum().item())
                    counts["positions_nonfinite_draft"] += int((~draft_finite).sum().item())
                    valid = target_finite & draft_finite
                    counts["positions_finite"] += int(valid.sum().item())
                    if self.has_vocab_mapping and self.draft_vocab_size < self.vocab_size:
                        mapping = self.ea_layer.t2d.to(device=device, dtype=torch.bool)
                        if mapping.numel() != target.shape[-1]:
                            raise AssertionError(
                                f"Vocab mapping width {mapping.numel()} != target width {target.shape[-1]}"
                            )
                        safe_target = torch.where(target_finite.unsqueeze(-1), target, 0.0)
                        target_top = safe_target.argmax(dim=-1)
                        mapping_valid = mapping[target_top]
                        counts["positions_invalid_vocab_mapping"] += int(
                            (target_finite & ~mapping_valid).sum().item()
                        )
                        counts["positions_mapping_valid"] += int((valid & mapping_valid).sum().item())
                        valid &= mapping_valid
                        target = target[..., mapping]
                    if draft.shape[-1] != target.shape[-1]:
                        raise AssertionError(
                            f"Draft/target vocab widths differ: {draft.shape[-1]} != {target.shape[-1]}"
                        )
                    if not bool(valid.any()):
                        continue
                    draft_rows = draft[valid]
                    target_rows = target[valid]
                    temperature = self.adaptation_temperature
                    target_probs = F.softmax(target_rows / temperature, dim=-1)
                    draft_log_probs = F.log_softmax(draft_rows / temperature, dim=-1)
                    accepted_tokens = target_rows.argmax(dim=-1)
                    acceptance_nll = -draft_log_probs.gather(
                        -1, accepted_tokens.unsqueeze(-1)
                    ).squeeze(-1) * (temperature ** 2)
                    counts["accepted_positions"] = counts.get("accepted_positions", 0) + int(acceptance_nll.numel())
                    kl_token = F.kl_div(
                        draft_log_probs, target_probs, reduction="none"
                    ).sum(dim=-1) * (temperature ** 2)
                    if self.adaptation_objective == "acceptance":
                        per_token = acceptance_nll
                    elif self.adaptation_objective == "acceptance_weighted_kl":
                        weights = torch.ones_like(kl_token)
                        weights = weights + (accepted_tokens == draft_rows.argmax(dim=-1)).float()
                        per_token = kl_token * weights
                    else:
                        per_token = kl_token
                    if not bool(torch.isfinite(per_token).all()):
                        result["skip_reason"] = "nonfinite_per_token_loss"
                        self._record_diagnostic("update_skipped", reason=result["skip_reason"])
                        return result
                    fragment_sum = per_token.sum()
                    token_loss_sum = fragment_sum if token_loss_sum is None else token_loss_sum + fragment_sum
                    fragment_positions = int(per_token.numel())
                    valid_positions += fragment_positions
                    counts["positions_valid"] += fragment_positions
                    counts["fragments_used"] += 1
                ratio = valid_positions / total_positions if total_positions else 0.0
                result["valid_vocab_ratio"] = float(ratio)
                if token_loss_sum is None or valid_positions == 0:
                    result["skip_reason"] = "no_valid_token_positions"
                    self._record_diagnostic("update_skipped", reason=result["skip_reason"], counts=dict(counts))
                    return result
                distill_loss = token_loss_sum / valid_positions
                anchor_loss = self._adaptation_extra_loss(trainable) if hasattr(self, "_adaptation_extra_loss") else torch.zeros((), device=device)
                loss = distill_loss + anchor_loss
                result["distill_loss"] = float(distill_loss.detach())
                result["anchor_loss"] = float(anchor_loss.detach())
                result["total_loss"] = float(loss.detach())
                if not bool(torch.isfinite(loss)):
                    result["skip_reason"] = "nonfinite_loss"
                    self._record_diagnostic("update_skipped", reason=result["skip_reason"])
                    return result
                loss.backward()
                grads = [parameter.grad for parameter in trainable if parameter.grad is not None]
                if not grads or not all(bool(torch.isfinite(grad).all()) for grad in grads):
                    self.adapter_optimizer.zero_grad(set_to_none=True)
                    result["skip_reason"] = "missing_or_nonfinite_gradient"
                    self._record_diagnostic("update_skipped", reason=result["skip_reason"])
                    return result
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=self.gradient_clip_norm)
                if not bool(torch.isfinite(grad_norm)):
                    self.adapter_optimizer.zero_grad(set_to_none=True)
                    result["skip_reason"] = "nonfinite_grad_norm"
                    self._record_diagnostic("update_skipped", reason=result["skip_reason"])
                    return result
                # Snapshot immediately before the numeric update so rollback is exact.
                parameter_snapshot = [parameter.detach().clone() for parameter in trainable]
                optimizer_snapshot = copy.deepcopy(self.adapter_optimizer.state_dict())
                update_attempted = True
                try:
                    self.adapter_optimizer.step()
                except (FloatingPointError, OverflowError):
                    for parameter, saved in zip(trainable, parameter_snapshot):
                        parameter.data.copy_(saved)
                    self.adapter_optimizer.load_state_dict(optimizer_snapshot)
                    self.adapter_optimizer.zero_grad(set_to_none=True)
                    result.update(status="rolled_back", rolled_back=True, skip_reason="optimizer_step_exception")
                    return result
                drift_values = []
                valid_parameters = True
                for parameter, saved in zip(trainable, parameter_snapshot):
                    valid_parameters &= bool(torch.isfinite(parameter).all()) and self._fits_dtype(parameter, original_dtype)
                    drift_values.append(float((parameter.float() - saved.float()).norm() / saved.float().norm().clamp_min(1e-6)))
                max_drift = max(drift_values, default=0.0)
                if max_drift > self.max_relative_drift:
                    valid_parameters = False
                result["max_relative_drift"] = max_drift
                valid_optimizer = self._tensor_tree_is_finite(self.adapter_optimizer.state)
                if not valid_parameters or not valid_optimizer:
                    for parameter, saved in zip(trainable, parameter_snapshot):
                        parameter.data.copy_(saved)
                    self.adapter_optimizer.load_state_dict(optimizer_snapshot)
                    self.adapter_optimizer.zero_grad(set_to_none=True)
                    reason = "nonfinite_optimizer_state" if not valid_optimizer else ("drift_limit_exceeded" if max_drift > self.max_relative_drift else "invalid_updated_parameters")
                    result.update(status="rolled_back", rolled_back=True, skip_reason=reason)
                    return result
                losses.append(float(loss.detach()))
                result["grad_norm"] = float(grad_norm.detach())
            result.update(loss=sum(losses) / len(losses), status="updated", updated=True)
            self._record_diagnostic(
                "update_completed", loss=result["loss"], grad_norm=result["grad_norm"],
                fragments=len(fragments), counts=dict(counts),
            )
            return result
        finally:
            if self.adapter_optimizer is not None:
                self.adapter_optimizer.zero_grad(set_to_none=True)
            self.ea_layer.tree_mask = saved_tree_mask
            self.ea_layer.stable_kv = None if update_attempted else saved_stable_kv
            try:
                self.ea_layer.to(original_dtype)
                if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
                    raise FloatingPointError("nonfinite trainable parameter after dtype restoration")
            except (RuntimeError, FloatingPointError, OverflowError):
                if update_attempted and parameter_snapshot is not None:
                    self.ea_layer.float()
                    for parameter, saved in zip(trainable, parameter_snapshot):
                        parameter.data.copy_(saved)
                    self.adapter_optimizer.load_state_dict(optimizer_snapshot)
                    self.adapter_optimizer.zero_grad(set_to_none=True)
                    self.ea_layer.to(original_dtype)
                    result.update(
                        loss=None, status="rolled_back", updated=False,
                        rolled_back=True, skip_reason="dtype_restore_failure",
                    )
                else:
                    raise
            finally:
                self.ea_layer.train(was_training)

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3=use_eagle3,
            base_model=base_model,
            base_model_name_or_path=base_model_path,
            ea_model_path=configpath,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            ea_layer_state_dict=ea_layer_state_dict,
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            enable_adaptation=None,
            adaptation_lr=None,
            adaptation_temperature=None,
            warmup_steps=const_warmup_steps,
            diagnostics=False,
    ):
        self.diagnostics = diagnostics
        self._diagnostic_events = []
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        if enable_adaptation is not None and enable_adaptation:
            requested_lr = const_lr if adaptation_lr is None else adaptation_lr
            requested_temperature = const_temp if adaptation_temperature is None else adaptation_temperature
            if self.adapter_optimizer is None:
                self.setup_online_adaptation(
                    adaptation_lr=requested_lr,
                    adaptation_temperature=requested_temperature,
                )
            else:
                if adaptation_lr is not None:
                    self.adaptation_lr = self._validate_adaptation_hyperparameter(
                        "adaptation_lr", adaptation_lr
                    )
                    for group in self.adapter_optimizer.param_groups:
                        group["lr"] = self.adaptation_lr
                if adaptation_temperature is not None:
                    self.adaptation_temperature = self._validate_adaptation_hyperparameter(
                        "adaptation_temperature", adaptation_temperature
                    )

        use_adaptation = enable_adaptation if enable_adaptation is not None else self.enable_online_adaptation
        
        # Independent fragments are shifted and trained separately.
        collected_fragments = []
        is_collecting = use_adaptation and self.adapter_optimizer is not None

        if is_collecting:
            if self.initial_weights is None:
                raise RuntimeError("Online adaptation has no initial adapter snapshot")
            with torch.no_grad():
                if hasattr(self.base_model.model, "tree_mask"):
                    self.base_model.model.tree_mask = None
                base_outputs = self.base_model.model(
                    input_ids=input_ids, use_cache=True, output_hidden_states=True
                )
                prefill_hidden_states = self._extract_eagle_hidden(
                    base_outputs, "prefill"
                ).cpu()
                prefill_logits = self.base_model.lm_head(base_outputs[0]).cpu()
                fragment = self._make_fragment(
                    input_ids.cpu(), prefill_hidden_states, prefill_logits, "prefill"
                )
                if fragment is not None:
                    collected_fragments.append(fragment)
                    self._record_diagnostic(
                        "fragment_collected", source="prefill",
                        length=fragment["tokens"].shape[1], root=fragment["root"],
                        aligned=True,
                    )

        # ==========================================

        total_accept_length = 0
        total_steps = 0
        total_drafted_tokens = 0

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        with torch.no_grad():
            if hasattr(self, "past_key_values"):
                past_key_values = self.past_key_values
                past_key_values_data = self.past_key_values_data
                current_length_data = self.current_length_data
                current_length_data.zero_()
            else:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model,max_length=max_length)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        cache_capacity = self._cache_capacity(past_key_values_data)
        if input_len >= cache_capacity:
            return input_ids if not log else (input_ids, 0, -1, {"total_accept_length": 0, "total_drafted_tokens": 0, "total_steps": 0, "losses": [], "diagnostics": {"cache_stop": "prompt_fills_cache"}})
        reset_tree_mode(self)
        
        # prefill (Normal EAGLE Init)
        with torch.no_grad():
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits_before, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor
            )
        
        if not self._tree_fits_cache(input_ids, draft_tokens, past_key_values_data):
            return input_ids if not log else (input_ids, 0, -1, {"total_accept_length": 0, "total_drafted_tokens": 0, "total_steps": 0, "losses": [], "diagnostics": {"cache_stop": "initial_tree_does_not_fit"}})

        new_token = 0
        idx = -1
        max_length = min(max_length, self._cache_capacity(past_key_values_data)) - self.ea_layer.total_tokens - 10
        max_length = max(max_length, 0)
        
        adaptation_result = {
            "loss": None, "status": "skipped", "updated": False,
            "rolled_back": False, "skip_reason": "adaptation_disabled",
            "counts": {"fragments_total": 0, "fragments_used": 0,
                       "fragments_rejected_nonfinite_hidden": 0,
                       "positions_total": 0, "positions_valid": 0,
                       "positions_nonfinite_target": 0,
                       "positions_nonfinite_draft": 0,
                       "positions_invalid_vocab_mapping": 0},
            "grad_norm": None, "valid_vocab_ratio": None,
            "effective_hyperparameters": {
                "learning_rate": float(self.adaptation_lr),
                "temperature": float(self.adaptation_temperature),
                "train_steps": int(const_train_steps),
            },
        }

        # warmup_steps=0 adapts only on the independent prefill fragment.
        if is_collecting and warmup_steps == 0:
            adaptation_result = self._perform_online_adaptation(
                collected_fragments, train_steps=const_train_steps
            )
            is_collecting = False
            if adaptation_result["updated"] or adaptation_result["rolled_back"]:
                self.ea_layer.reset_kv()
                self._reset_base_cache("after_adaptation")
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits_before, hidden_state, sample_token = initialize_tree(
                    input_ids, self, past_key_values, logits_processor
                )
                self._record_diagnostic("cache_invalidated", reason=adaptation_result["status"])
        
        for idx in range(max_length):
            if not self._tree_fits_cache(input_ids, draft_tokens, past_key_values_data):
                self._record_diagnostic("cache_stop", reason="tree_does_not_fit")
                break
            with torch.no_grad():
                self.base_model.model.tree_mask = tree_mask

                draft_tokens = draft_tokens.to(input_ids.device)
                logits, hidden_state_new, outputs = tree_decoding(
                    self,
                    draft_tokens,
                    past_key_values,
                    tree_position_ids,
                    input_ids,
                    retrieve_indices,
                )

                draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                candidates = draft_tokens[0, retrieve_indices]
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
            
            if is_collecting and total_steps < warmup_steps:
                path_tokens = candidates[best_candidate, :accept_length + 1]
                if path_tokens.numel() > 1:
                    raw_tree_hidden = self._extract_eagle_hidden(outputs, "tree_decoding")
                    path_indices = retrieve_indices[best_candidate, :accept_length + 1]
                    path_hidden = raw_tree_hidden[:, path_indices, :]
                    path_logits = logits[best_candidate, :accept_length + 1, :].unsqueeze(0)
                    fragment = self._make_fragment(
                        path_tokens, path_hidden, path_logits,
                        f"tree_step_{total_steps}",
                    )
                    if fragment is not None:
                        collected_fragments.append(fragment)
                        self._record_diagnostic(
                            "fragment_collected", source=fragment["source"],
                            length=fragment["tokens"].shape[1], root=fragment["root"],
                            accepted_children=int(accept_length), aligned=True,
                        )

            total_accept_length += accept_length
            total_steps += 1
            total_drafted_tokens += candidates.shape[1] 

            accepted_length = int(accept_length) + 1
            if input_ids.shape[1] + accepted_length > self._cache_capacity(past_key_values_data):
                self._record_diagnostic("cache_stop", reason="accepted_path_does_not_fit")
                break
            with torch.no_grad():
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    self,
                    hidden_state_new,
                    sample_p
                )
            
            if is_collecting and total_steps == warmup_steps:
                adaptation_result = self._perform_online_adaptation(
                    collected_fragments, train_steps=const_train_steps
                )
                is_collecting = False
                if adaptation_result["updated"] or adaptation_result["rolled_back"]:
                    self.ea_layer.reset_kv()
                    self._reset_base_cache("after_adaptation")
                    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits_before, hidden_state, sample_token = initialize_tree(
                        input_ids, self, past_key_values, logits_processor
                    )
                    self._record_diagnostic(
                        "cache_invalidated", reason=adaptation_result["status"]
                    )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token >= max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        
        continuation_length = input_ids.shape[1] - input_len
        if continuation_length > max_new_tokens:
            input_ids = input_ids[:, :input_len + max_new_tokens]
            new_token = max_new_tokens
            self._record_diagnostic(
                "continuation_truncated", generated=continuation_length,
                returned=max_new_tokens,
            )
        else:
            new_token = continuation_length

        if is_collecting:
            adaptation_result.update(
                status="skipped", skip_reason="warmup_not_reached"
            )
            self._record_diagnostic("update_skipped", reason="warmup_not_reached")

        if total_steps > 0:
            avg_accept = total_accept_length / total_steps
            print(f"[Stats] Total Iteration Steps: {total_steps}, Total Accepted: {total_accept_length}")        

        if use_adaptation:
            loss_text = "None" if adaptation_result["loss"] is None else f"{adaptation_result['loss']:.4f}"
            print(
                f"[Online Adaptation] status={adaptation_result['status']} "
                f"loss={loss_text} reason={adaptation_result['skip_reason']}"
            )

        finite_loss = adaptation_result["loss"]
        losses = ([finite_loss] if finite_loss is not None and math.isfinite(finite_loss) else [])
        total_accept_length_value = (
            total_accept_length.item()
            if hasattr(total_accept_length, "item") else total_accept_length
        )
        total_drafted_tokens_value = (
            total_drafted_tokens.item()
            if hasattr(total_drafted_tokens, "item") else total_drafted_tokens
        )
        total_steps_value = total_steps.item() if hasattr(total_steps, "item") else total_steps
        run_stats = {
            "total_accept_length": int(total_accept_length_value),
            "total_drafted_tokens": int(total_drafted_tokens_value),
            "total_steps": int(total_steps_value),
            "losses": losses,
            "diagnostics": {
                "enabled": self.diagnostics,
                "fragments": len(collected_fragments),
                "adaptation": copy.deepcopy(adaptation_result),
                "events": list(self._diagnostic_events) if self.diagnostics else [],
                "cache_state": {
                    "base_current_length": min(
                        int(current_length_data[0].item()), input_ids.shape[1]
                    ),
                    "ea_stable_kv": getattr(self.ea_layer, "stable_kv", None) is not None,
                },
            },
        }
        # Fail here if a future diagnostic accidentally retains tensors or non-JSON values.
        json.dumps(run_stats)

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, run_stats
