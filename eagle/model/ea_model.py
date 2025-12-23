import copy
import json
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
const_lr = 5e-5


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

        self.enable_online_adaptation = False # use online or not
        self.adaptation_lr = const_lr # learning rate
        self.adaptation_temperature = const_temp
        self.adapter_optimizer = None
        self.first_token_adapted = False # one-step flag

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
        # if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
        #     del self.ea_layer.d2t,self.ea_layer.t2d
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

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def setup_online_adaptation(
            self, 
            adaptation_lr=const_lr, 
            adaptation_temperature=const_temp
        ):
        """superparams of online adaptation"""
        self.enable_online_adaptation = True
        self.adaptation_lr = adaptation_lr
        self.adaptation_temperature = adaptation_temperature
        self.first_token_adapted = False
        
        for param in self.ea_layer.parameters():
            param.requires_grad = True

        # adapt_params = []
        # for name, param in self.ea_layer.named_parameters():
        #     if 'lm_head' in name:
        #         param.requires_grad = True
        #         adapt_params.append(param)
        #     elif 'norm' in name:
        #         param.requires_grad = True
        #         adapt_params.append(param)
        #     elif 'fc' in name:
        #         param.requires_grad = True
        #         adapt_params.append(param)
        #     else:
        #         param.requires_grad = False
                
        self.adapter_optimizer = torch.optim.Adam(self.ea_layer.parameters(), lr=adaptation_lr)

    def _perform_online_adaptation(
            self,
            input_ids,
            candidates, # candidates[c, i]为第c条候选路径在第i步的token id
            retrieve_indices, # retrieve_indices[c, i]为第c条候选路径在第i步的tree index
            best_candidate, 
            accept_length,
            hidden_state_new, # [0,i,:]为base model处理第i个tree token后的hidden state [1，nodes, hidden_dim*3]
            logits, # 按候选路径重排的logits，[candidate_sum, depth+1, vocab_size]
            logits_before,
            prompt_hidden_states, # 前文的hidden states
            train_steps
        ):
        device = next(self.ea_layer.parameters()).device

        # 这里转换数据格式，从bf16到float32以解决训练中爆表的问题，输入也进行转换
        original_dtype = next(self.ea_layer.parameters()).dtype
        self.ea_layer.float()

        if hidden_state_new.dtype != torch.float32:
            hidden_state_new = hidden_state_new.float()
        if prompt_hidden_states.dtype != torch.float32:
            prompt_hidden_states = prompt_hidden_states.float()

        was_training = self.ea_layer.training

        # 这里保存tree_mask和kv_cache以供后续恢复，训练时禁用以防kv_cache被污染以及tree_mask影响训练
        saved_tree_mask = getattr(self.ea_layer, 'tree_mask', None)
        saved_stable_kv = self.ea_layer.stable_kv
        self.ea_layer.reset()
        self.ea_layer.stable_kv = None

        # 训练模式
        self.ea_layer.train()

        accepted_tokens = candidates[best_candidate, :accept_length + 1] # 被选中的原始token id
        target_logits = logits[best_candidate, :accept_length + 1] # best candidate路径中的预测logits

        if target_logits.dtype != torch.float32:
            target_logits = target_logits.float()
        if accepted_tokens.device != device:
            accepted_tokens = accepted_tokens.to(device)
        if target_logits.device != device:
            target_logits = target_logits.to(device)
        if hidden_state_new.device != device:
            hidden_state_new = hidden_state_new.t
