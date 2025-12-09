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
        self.adaptation_lr = 5e-5 # learning rate
        self.adaptation_temperature = 2.0
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
            adaptation_lr=5e-5, 
            adaptation_temperature=2.0
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
            prompt_hidden_states, # 前文的hidden states
            train_steps
        ):
        device = next(self.ea_layer.parameters()).device
        was_training = self.ea_layer.training
        self.ea_layer.train()
    
        accepted_indices = retrieve_indices[best_candidate, :accept_length + 1] # best candidate被选中的候选路径
        accepted_tokens = candidates[best_candidate, :accept_length + 1] # 被选中的原始token id
        target_logits = logits[best_candidate, :accept_length] # best candidate路径中的预测logits
        
        if accepted_indices.device != device:
            accepted_indices = accepted_indices.to(device)
        if accepted_tokens.device != device:
            accepted_tokens = accepted_tokens.to(device)
        if target_logits.device != device:
            target_logits = target_logits.to(device)
        if hidden_state_new.device != device:
            hidden_state_new = hidden_state_new.to(device)
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if prompt_hidden_states.device != device:
            prompt_hidden_states = prompt_hidden_states.to(device)
        
        input_indices = accepted_indices[:-1]  # 去除最后一个token的tree index
        input_tokens = accepted_tokens[:-1]    # 去除最后一个token的原始token id
        num_tokens = len(input_indices)
        
        max_idx = hidden_state_new.shape[1] - 1
        valid_mask = (input_indices >= 0) & (input_indices <= max_idx)
        
        valid_positions = torch.where(valid_mask)[0]
        input_indices = input_indices[valid_mask]
        input_tokens = input_tokens[valid_mask]
        num_valid = len(input_indices)
        
        input_hidden = hidden_state_new[:, input_indices, :].detach() # 提取hidden states
        target_logits = target_logits[valid_positions].detach()
        
        if self.has_vocab_mapping and self.draft_vocab_size < self.vocab_size:
            d2t = self.ea_layer.d2t
            if d2t.device != device:
                d2t = d2t.to(device)
            
            # 创建映射后的target logits [num_valid, draft_vocab_size]
            mapped_target_logits = torch.full(
                (num_valid, self.draft_vocab_size),
                torch.finfo(target_logits.dtype).min / 2,
                device=device,
                dtype=target_logits.dtype
            )
            
            # d2t[draft_idx] = target_idx，复制对应的logit值
            valid_draft_indices = torch.arange(self.draft_vocab_size, device=device)
            target_indices = d2t[valid_draft_indices]
            valid_mapping = (target_indices >= 0) & (target_indices < target_logits.shape[-1])
            
            # 遍历draft vocab，找到它在target vocab中的映射，把target logits复制到mapped_target_logits中
            for i in range(num_valid):
                mapped_target_logits[i, valid_mapping] = target_logits[i, target_indices[valid_mapping]]
            
            target_logits = mapped_target_logits
        else:
            if target_logits.shape[-1] > self.draft_vocab_size:
                target_logits = target_logits[..., :self.draft_vocab_size]
        
        context_len = input_ids.shape[1] # 前文prompt的长度
        cat_tokens = torch.cat([input_ids[0],input_tokens], dim=0) # [context_len + num_valid]
        input_tokens_batch = cat_tokens.unsqueeze(0)  # [1, context_len+num_valid]
        cat_len = context_len + num_valid

        hidden_dim = input_hidden.shape[-1]
        expected_dim = self.ea_layer.fc.in_features
        
        prompt_len = prompt_hidden_states.shape[1]
        context_hidden = prompt_hidden_states.detach() # 前文prompt的hidden states
        
        full_hidden = torch.cat([context_hidden, input_hidden], dim=1)

        total_loss = 0.0
        for step in range(train_steps):
            self.adapter_optimizer.zero_grad()

            # Forward
            with torch.enable_grad():
                inputs_embeds = self.ea_layer.embed_tokens(input_tokens_batch)
                inputs_embeds = inputs_embeds.to(full_hidden.dtype) # [1, context_len+num_valid, embedded_dim]
                
                draft_hidden = self.ea_layer.fc(full_hidden)  # [1, context_len+num_valid, hidden_size]
                batch_size = 1
                seq_length = cat_len
                
                causal_mask = torch.triu(
                    torch.full((seq_length, seq_length), float('-inf'), device=device, dtype=draft_hidden.dtype),
                    diagonal=1
                )
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
                position_ids = torch.arange(seq_length, device=device, dtype=torch.long).unsqueeze(0) # 可能有问题，现在用的相对位置
                
                layer_outputs = self.ea_layer.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=draft_hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                
                draft_hidden_out = layer_outputs[0]  # [1, seq_length, hidden_size]
                draft_logits = self.ea_layer.lm_head(self.ea_layer.norm(draft_hidden_out))
                draft_logits = draft_logits.squeeze(0)  # [seq_length, vocab_size]

                draft_logits = draft_logits[-num_valid:, :] # 只取num_valid个，前文不参与loss计算
            
            if draft_logits.shape[-1] != target_logits.shape[-1]:
                min_vocab = min(draft_logits.shape[-1], target_logits.shape[-1])
                draft_logits = draft_logits[..., :min_vocab]
                target_logits = target_logits[..., :min_vocab]
            
            min_len = min(draft_logits.shape[0], target_logits.shape[0])
            
            draft_logits = draft_logits[:min_len]
            target_logits = target_logits[:min_len]
            temp = self.adaptation_temperature
            
            log_p = F.log_softmax(draft_logits / temp, dim=-1)
            q = F.softmax(target_logits / temp, dim=-1)
            
            kl_per_element = F.kl_div(log_p, q, reduction='none')  # [num_valid, vocab_size]
            kl_per_token = kl_per_element.sum(dim=-1)  # [num_valid]
            loss = kl_per_token.mean() * (temp ** 2)
            
            # if torch.isnan(loss) or torch.isinf(loss):
            #     print(f"[Adaptation] Invalid loss: {loss.item()}")
            #     self.ea_layer.train(was_training)
            #     print("error0777777")
            #     return 0.0            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.ea_layer.parameters() if p.requires_grad],
                max_norm=1.0
            )
            
            self.adapter_optimizer.step()
            total_loss += loss.item()
        
        self.ea_layer.train(was_training)
        return total_loss / train_steps

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
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
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
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        if enable_adaptation is not None and enable_adaptation:
            if self.adapter_optimizer is None:
                self.setup_online_adaptation(
                    adaptation_lr=adaptation_lr or 5e-5,
                    adaptation_temperature=adaptation_temperature or 2.0
                )

        use_adaptation = enable_adaptation if enable_adaptation is not None else self.enable_online_adaptation # online params
        if use_adaptation:
            self.first_token_adapted = False  
        adaptation_info = {
            'losses': [],
            'accept_lengths': [],
            'adaptation_count': 0
        }

        total_accept_length = 0
        total_steps = 0

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
        reset_tree_mode(self)
        # prefill
        with torch.no_grad():
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor
            )

        prompt_hidden_states = hidden_state if use_adaptation else None

        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
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
            
            total_accept_length += accept_length
            total_steps += 1

            if use_adaptation and self.adapter_optimizer is not None and not self.first_token_adapted: 
                if accept_length > 0:
                    adapt_loss = self._perform_online_adaptation(
                        input_ids=input_ids,
                        candidates=candidates,
                        retrieve_indices=retrieve_indices,
                        best_candidate=best_candidate,
                        accept_length=accept_length,
                        hidden_state_new=hidden_state_new,
                        logits=logits,
                        prompt_hidden_states=prompt_hidden_states,
                        train_steps=10
                    )
                    
                    adaptation_info['losses'].append(adapt_loss)
                    adaptation_info['accept_lengths'].append(accept_length)
                    adaptation_info['adaptation_count'] += 1

                    self.first_token_adapted = True
                    print("token9999999")

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

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        if total_steps > 0:
            avg_accept = total_accept_length / total_steps
            print(f"[Stats] Steps: {total_steps}, Total Accepted: {total_accept_length}, " f"Avg Accept Length: {avg_accept:.2f}")        

        # print online info
        if use_adaptation and adaptation_info['adaptation_count'] > 0:
                print(f"[Online Adaptation] loss={adaptation_info['losses'][0]:.4f}, accept_length={adaptation_info['accept_lengths'][0]}")

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx
