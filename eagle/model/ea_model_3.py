"""Persistent-only long-term adaptation built on the EA2 inference implementation."""
import copy
import math
import random
import torch


EA3_STRATEGY = "persistent_sliding_window_anchor_guard"

from .ea_model_2 import EaModel as _EaModel2


class EaModel(_EaModel2):
    """Conservative persistent adapter with bounded FIFO replay and anchoring."""

    def setup_long_term_adaptation(
        self,
        adaptation_lr=1e-6,
        adaptation_temperature=1.0,
        scope="head_only",
        replay_max_fragments=32,
        replay_max_tokens=4096,
        replay_fraction=0.5,
        anchor_weight=1e-3,
        max_relative_drift=0.02,
        min_valid_vocab_ratio=0.8,
        gradient_clip_norm=0.2,
    ):
        adaptation_lr = self._validate_adaptation_hyperparameter("adaptation_lr", adaptation_lr)
        adaptation_temperature = self._validate_adaptation_hyperparameter(
            "adaptation_temperature", adaptation_temperature
        )
        if scope not in {"default", "head_only", "fc_only", "midlayer_only", "norm_only", "lm_head_only"}:
            raise ValueError(f"Unsupported adaptation scope: {scope}")
        replay_max_fragments = int(replay_max_fragments)
        replay_max_tokens = int(replay_max_tokens)
        if replay_max_fragments <= 0 or replay_max_tokens <= 0:
            raise ValueError("Replay limits must be positive")
        replay_fraction = float(replay_fraction)
        if not math.isfinite(replay_fraction) or not 0.0 <= replay_fraction <= 1.0:
            raise ValueError("replay_fraction must be in [0, 1]")
        for name, value in (("anchor_weight", anchor_weight), ("max_relative_drift", max_relative_drift),
                            ("min_valid_vocab_ratio", min_valid_vocab_ratio), ("gradient_clip_norm", gradient_clip_norm)):
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
            if name == "anchor_weight":
                anchor_weight = value
            elif name == "max_relative_drift":
                max_relative_drift = value
            elif name == "min_valid_vocab_ratio":
                min_valid_vocab_ratio = value
            else:
                gradient_clip_norm = value
        if min_valid_vocab_ratio > 1.0:
            raise ValueError("min_valid_vocab_ratio must be in [0, 1]")
        super().setup_online_adaptation(
            adaptation_lr=adaptation_lr,
            adaptation_temperature=adaptation_temperature,
            mode="persistent",
            scope=scope,
            objective="kl",
            reset_granularity="stream",
        )
        self.long_term_replay_max_fragments = replay_max_fragments
        self.long_term_replay_max_tokens = replay_max_tokens
        self.long_term_replay_fraction = replay_fraction
        self.long_term_replay = []
        self.long_term_replay_tokens = 0
        self.long_term_paused = False
        self.long_term_pause_reason = None
        self.long_term_setup_generation = int(getattr(self, "long_term_setup_generation", 0)) + 1
        self.long_term_committed_version = int(getattr(self, "weight_version", 0))
        self.long_term_last_update_drift = 0.0
        self.long_term_updates = 0
        self.long_term_rollbacks = 0
        self.long_term_ema_loss = None
        self.long_term_ema_acceptance = None
        self.long_term_ema_decay = 0.95
        # EA2 enforces this limit for each optimizer step; EA3 applies the
        # same configured limit to cumulative drift from the immutable anchor.
        self.max_relative_drift = float(max_relative_drift)
        self.anchor_weight = float(anchor_weight)
        self.min_valid_vocab_ratio = float(min_valid_vocab_ratio)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.long_term_max_anchor_drift = float(max_relative_drift)
        self.long_term_committed_weights = {k: v.detach().cpu().clone() for k, v in self.ea_layer.state_dict().items()}
        self.long_term_committed_optimizer = copy.deepcopy(self.adapter_optimizer.state_dict())
        self.long_term_anchor_drift = 0.0
        self.long_term_update_attempts = 0
        self._adaptation_cache_invalidated = False
    def invalidate_inference_cache(self, reason='long_term_update'):
        self.ea_layer.reset_kv()
        self.ea_layer.stable_kv = None
        self._reset_base_cache(reason)
        self.cache_epoch = int(getattr(self, 'cache_epoch', 0)) + 1
        self._adaptation_cache_invalidated = True

    def reset_optimizer(self):
        self._reset_adapter_optimizer()
        self.long_term_committed_weights = {k: v.detach().cpu().clone() for k, v in self.ea_layer.state_dict().items()}
        self.long_term_committed_optimizer = copy.deepcopy(self.adapter_optimizer.state_dict())
        self.long_term_committed_version = int(getattr(self, "weight_version", 0))

    def _anchor_drift(self):
        if self.initial_weights is None:
            return 0.0
        values = []
        for name, parameter in self._adapter_named_parameters():
            anchor = self.initial_weights[name].to(parameter.device, dtype=torch.float32)
            values.append(float((parameter.float() - anchor).norm() / anchor.norm().clamp_min(1e-6)))
        return max(values, default=0.0)

    def setup_online_adaptation(self, *args, **kwargs):
        """Compatibility entry point: EA3 always means persistent long-term mode."""
        names = ("adaptation_lr", "adaptation_temperature", "mode", "scope",
                 "objective", "reset_granularity")
        if len(args) > len(names):
            raise TypeError("setup_online_adaptation received too many positional arguments")
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f"setup_online_adaptation got multiple values for {name}")
            kwargs[name] = value
        kwargs.pop("mode", None)
        kwargs.pop("reset_granularity", None)
        kwargs.pop("objective", None)
        kwargs.setdefault("adaptation_lr", 1e-6)
        kwargs.setdefault("scope", "head_only")
        return self.setup_long_term_adaptation(
            adaptation_lr=kwargs.pop("adaptation_lr"),
            adaptation_temperature=kwargs.pop("adaptation_temperature", 1.0),
            scope=kwargs.pop("scope"),
            **kwargs,
        )

    def reset_online_adaptation(self):
        raise RuntimeError("EA3 is persistent-only; use reset_optimizer or start a new model")

    def add_fragment(self, fragment):
        if fragment is None:
            return
        tokens = int(fragment["tokens"].numel())
        if tokens < 2 or tokens > self.long_term_replay_max_tokens:
            return
        item = {k: v.clone() if torch.is_tensor(v) else v for k, v in fragment.items()}
        self.long_term_replay.append(item)
        self.long_term_replay_tokens += tokens
        while (len(self.long_term_replay) > self.long_term_replay_max_fragments or
               self.long_term_replay_tokens > self.long_term_replay_max_tokens):
            old = self.long_term_replay.pop(0)
            self.long_term_replay_tokens -= int(old["tokens"].numel())

    def sample_replay_fragments(self, current):
        history = self.long_term_replay
        if not history or self.long_term_replay_fraction <= 0.0:
            return list(current)
        keep = min(len(history), int(round(len(current) * self.long_term_replay_fraction)))
        if keep <= 0:
            return list(current)
        # Sample without replacement from the bounded window.
        return list(current) + random.sample(history, keep)

    def _adaptation_extra_loss(self, trainable):
        if not self.anchor_weight or self.initial_weights is None:
            return torch.zeros((), device=trainable[0].device)
        loss = torch.zeros((), device=trainable[0].device)
        for name, parameter in self._adapter_named_parameters():
            anchor = self.initial_weights[name].to(parameter.device, dtype=torch.float32)
            loss = loss + (parameter.float() - anchor).pow(2).mean()
        return self.anchor_weight * loss

    _anchor_loss = _adaptation_extra_loss

    def _perform_online_adaptation(self, fragments, train_steps=1):
        if self.long_term_paused:
            return {"loss": None, "status": "paused", "updated": False,
                    "rolled_back": False, "skip_reason": "long_term_paused",
                    "counts": {"fragments_total": len(fragments)}}
        self.long_term_update_attempts += 1
        usable = self.sample_replay_fragments(fragments)
        result = super()._perform_online_adaptation(usable, train_steps=train_steps)
        result['replay_fragments'] = len(usable)
        result['replay_tokens'] = sum(int(x['tokens'].numel()) for x in usable)
        result['anchor_drift'] = self._anchor_drift()
        result['cumulative_drift'] = result['anchor_drift']
        counts = result.setdefault("counts", {})
        finite = counts.get("positions_finite", 0)
        mapping_valid = (counts.get("positions_mapping_valid", 0)
                         if self.has_vocab_mapping and self.draft_vocab_size < self.vocab_size
                         else finite)
        total = counts.get("positions_total", 0)
        result["finite_position_ratio"] = finite / total if total else 0.0
        result["mapping_valid_ratio"] = mapping_valid / finite if finite else 0.0
        ratio = result.get("mapping_valid_ratio")
        if ratio is not None and ratio < self.min_valid_vocab_ratio and result.get("updated"):
            self._restore_committed_state()
            self.pause_adaptation("valid_vocab_ratio_guard")
            result.update(status="rolled_back", updated=False, rolled_back=True,
                          skip_reason="valid_vocab_ratio_guard")
            return result
        if result.get("updated"):
            if result.get('anchor_drift', 0.0) > self.long_term_max_anchor_drift:
                self._restore_committed_state()
                self.pause_adaptation('anchor_drift_guard')
                result.update(status='rolled_back', updated=False, rolled_back=True, skip_reason='anchor_drift_guard')
                return result
            self.long_term_updates += 1
            self.long_term_committed_weights = {k: v.detach().cpu().clone() for k, v in self.ea_layer.state_dict().items()}
            self.long_term_committed_optimizer = copy.deepcopy(self.adapter_optimizer.state_dict())
            self.weight_version = int(getattr(self, 'weight_version', 0)) + 1
            self.long_term_committed_version = self.weight_version
            self.long_term_last_update_drift = float(result.get('max_relative_drift', 0.0))
            result['update_drift'] = self.long_term_last_update_drift
            self.invalidate_inference_cache('long_term_update')
            self.long_term_ema_loss = result.get("loss") if self.long_term_ema_loss is None else self.long_term_ema_decay * self.long_term_ema_loss + (1 - self.long_term_ema_decay) * result["loss"]
            for fragment in fragments:
                self.add_fragment(fragment)
        elif result.get("rolled_back"):
            self._restore_committed_state()
        return result

    def _restore_committed_state(self):
        self.ea_layer.load_state_dict(self.long_term_committed_weights, strict=False)
        self.adapter_optimizer.load_state_dict(self.long_term_committed_optimizer)
        self.adapter_optimizer.zero_grad(set_to_none=True)
        self.weight_version = self.long_term_committed_version
        self.long_term_rollbacks += 1
        self.invalidate_inference_cache("committed_state_restore")

    def update_acceptance_ema(self, acceptance, accepted_length=None):
        if acceptance is None:
            return
        alpha = 1.0 - self.long_term_ema_decay
        self.long_term_ema_acceptance = float(acceptance) if self.long_term_ema_acceptance is None else self.long_term_ema_decay * self.long_term_ema_acceptance + alpha * float(acceptance)
        if accepted_length is not None:
            self.long_term_ema_length = float(accepted_length) if not hasattr(self, 'long_term_ema_length') or self.long_term_ema_length is None else self.long_term_ema_decay * self.long_term_ema_length + alpha * float(accepted_length)

    def record_acceptance(self, acceptance, accepted_length=None):
        """Record generation quality for diagnostics without changing adaptation state."""
        self.update_acceptance_ema(acceptance, accepted_length)
        return False

    def resume_adaptation(self):
        self.long_term_paused = False
        self.long_term_pause_reason = None

    def pause_adaptation(self, reason="manual"):
        self.long_term_paused = True
        self.long_term_pause_reason = reason

    def get_adaptation_state(self):
        state = super().get_adaptation_state() if hasattr(super(), "get_adaptation_state") else {}
        state.update({"strategy": EA3_STRATEGY, "learning_rate": float(getattr(self, 'adaptation_lr', 1e-6)), "weight_version": getattr(self, 'weight_version', 0), "cache_epoch": getattr(self, 'cache_epoch', 0), "update_attempts": getattr(self, 'long_term_update_attempts', 0), "anchor_drift": self._anchor_drift(),
                      "replay_fragments": len(getattr(self, "long_term_replay", [])),
                      "replay_tokens": getattr(self, "long_term_replay_tokens", 0),
                      "paused": getattr(self, "long_term_paused", False),
                      "pause_reason": getattr(self, "long_term_pause_reason", None),
                      "last_update_drift": getattr(self, "long_term_last_update_drift", 0.0),
                      "cumulative_anchor_drift": self._anchor_drift(),
                      "long_term_updates": getattr(self, "long_term_updates", 0),
                      "long_term_rollbacks": getattr(self, "long_term_rollbacks", 0),
                      "anchor_weight": getattr(self, "anchor_weight", 0.0),
                      "ema_loss": getattr(self, "long_term_ema_loss", None),
                      "ema_acceptance": getattr(self, "long_term_ema_acceptance", None),
                      "ema_accepted_length": getattr(self, "long_term_ema_length", None)})
        return state
