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
        self.adaptation_steps = 1 # iteration steps
        self.adaptation_lr = 0.001 # learning rate
        self.adaptation_temperature = 1.0
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
            adaptation_lr=0.001, 
            adaptation_steps=1, 
            adaptation_temperature=1.0,
            adaptation_layers=3
        ):
        """superparams of online adaptation"""
        self.enable_online_adaptation = True
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.adaptation_temperature = adaptation_temperature
        self.first_token_adapted = False
        
        for param in self.ea_layer.parameters():
            param.requires_grad = False
        adapt_params = []
        for name, param in self.ea_layer.named_parameters():
            if 'lm_head' in name:
                param.requires_grad = True
                adapt_params.append(param)
            elif 'norm' in name:
                param.requires_grad = True
                adapt_params.append(param)
            # elif 'fc' in name:
            #     param.requires_grad = True
            #     adapt_params.append(param)
            else:
                param.requires_grad = False
                
        self.adapter_optimizer = torch.optim.Adam(adapt_params, lr=adaptation_lr)

    def _perform_online_adaptation(
            self, 
            input_ids,
            candidates,
            retrieve_indices,
            best_candidate,
            accept_length,
            hidden_state_new,
            logits
    ):
        """one-step online adaptation"""
        self.ea_layer.train()  # to train mode
        device = hidden_state_new.device

        accepted_indices = retrieve_indices[best_candidate, :accept_length+1]
        accepted_tokens = candidates[accepted_indices]
        target_logits = logits[accepted_indices[:-1]]   

        if self.has_vocab_mapping and self.draft_vocab_size < self.vocab_size:
            accepted_tokens_cpu = accepted_tokens[:-1].cpu()
            if self.ea_layer.t2d.is_cuda:
                t2d_cpu = self.ea_layer.t2d.cpu()
            else:
                t2d_cpu = self.ea_layer.t2d
            
            valid_token_mask = (accepted_tokens_cpu >= 0) & (accepted_tokens_cpu < t2d_cpu.shape[0])
            if not valid_token_mask.any():
                self.ea_layer.eval()
                return 0.0
            
            accepted_tokens_filtered = accepted_tokens_cpu[valid_token_mask]
            valid_mask_cpu = t2d_cpu[accepted_tokens_filtered]
            valid_mask = valid_mask_cpu.to(device)

            if not valid_mask.any():
                self.ea_layer.eval()
                return 0.0

            valid_positions = torch.where(valid_mask_cpu)[0].to(device)
            if len(valid_positions) == 0:
                self.ea_layer.eval()
                return 0.0

            if accepted_indices.device != device:
                accepted_indices = accepted_indices.to(device)
        
            valid_accepted_indices = accepted_indices[:-1][valid_positions]
            
            if len(valid_accepted_indices) == 0:
                self.ea_layer.eval()
                return 0.0
    
            max_idx = hidden_state_new.shape[1] - 1
            valid_accepted_indices = valid_accepted_indices[valid_accepted_indices <= max_idx]
            
            input_hidden = hidden_state_new[:, valid_accepted_indices].detach()
            target_logits_filtered = target_logits[valid_indices]

            if len(valid_positions) > len(valid_accepted_indices):
                valid_positions = valid_positions[:len(valid_accepted_indices)]
            target_logits_filtered = target_logits[valid_positions]
            
            batch_size = target_logits_filtered.shape[0]
            seq_len = target_logits_filtered.shape[1] if target_logits_filtered.dim() > 2 else 1
        
            if target_logits_filtered.dim() == 2:
                target_logits_filtered = target_logits_filtered.unsqueeze(0)

            mapped_target_logits = torch.full(
                (batch_size, seq_len, self.draft_vocab_size), 
                float('-inf'), 
                device=device,
                dtype=target_logits_filtered.dtype
            )

            d2t_cpu = self.ea_layer.d2t.cpu() if self.ea_layer.d2t.is_cuda else self.ea_layer.d2t
            
            for draft_idx in range(self.draft_vocab_size):
                target_idx = d2t_cpu[draft_idx].item()
                if 0 <= target_idx < self.vocab_size:
                    if target_idx < target_logits_filtered.shape[-1]:
                        mapped_target_logits[:, :, draft_idx] = target_logits_filtered[:, :, target_idx]
            
            if target_logits.dim() == 2:
                mapped_target_logits = mapped_target_logits.squeeze(0)
        else:
            indices_to_use = accepted_indices[:-1]
            if len(indices_to_use) == 0:
                self.ea_layer.eval()
                return 0.0
                
            max_idx = hidden_state_new.shape[1] - 1
            indices_to_use = indices_to_use[indices_to_use <= max_idx]
            
            input_hidden = hidden_state_new[:, indices_to_use].detach()
            if target_logits.shape[-1] > self.draft_vocab_size:
                mapped_target_logits = target_logits[..., :self.draft_vocab_size]
            else:
                mapped_target_logits = target_logits

        mapped_target_logits = mapped_target_logits.detach()
        if input_hidden.dim() == 3 and input_hidden.shape[0] == 1:
            input_hidden = input_hidden.squeeze(0)

        total_loss = 0.0
        
        for _ in range(self.adaptation_steps):
            self.adapter_optimizer.zero_grad()
            # forward draft model
            # base model hidden states
            if hasattr(self.ea_layer, 'fc'):
                draft_hidden = self.ea_layer.fc(input_hidden)
            else:
                draft_hidden = input_hidden.clone()

            # get predictions of draft model for tokens
            draft_logits = self.ea_layer.lm_head(self.ea_layer.norm(draft_hidden))

            if draft_logits.shape[:-1] != mapped_target_logits.shape[:-1]:
                if draft_logits.dim() == 2 and mapped_target_logits.dim() == 2:
                    min_len = min(draft_logits.shape[0], mapped_target_logits.shape[0])
                    draft_logits = draft_logits[:min_len]
                    mapped_target_logits = mapped_target_logits[:min_len]
                else:
                    self.ea_layer.eval()
                    return 0.0
        
            if draft_logits.shape[-1] > self.draft_vocab_size:
                draft_logits = draft_logits[..., :self.draft_vocab_size]

            temp = self.adaptation_temperature
            if temp < 1e-5:
                temp = 1e-5 
            
            log_probs = F.log_softmax(draft_logits / temp, dim=-1)
            target_probs = F.softmax(mapped_target_logits / temp, dim=-1)
            
            epsilon = 1e-10
            target_probs = target_probs + epsilon
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
            
            loss = (target_probs * (target_probs.log() - log_probs)).sum(dim=-1).mean()
            
            if torch.isnan(loss) or torch.isinf(loss):
                self.ea_layer.eval()
                return 0.0
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.adapter_optimizer.param_groups[0]['params'], 
                max_norm=1.0
            )
            self.adapter_optimizer.step()
            total_loss += loss.item()
        
        self.ea_layer.eval()  # to eval mode
        return total_loss / self.adaptation_steps

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

    @torch.no_grad()
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
            adaptation_steps=None,
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
                    adaptation_lr=adaptation_lr or 0.001,
                    adaptation_steps=adaptation_steps or 1,
                    adaptation_temperature=adaptation_temperature or 1.0
                )

        use_adaptation = enable_adaptation if enable_adaptation is not None else self.enable_online_adaptation # online params
        if use_adaptation:
            self.first_token_adapted = False  
        adaptation_info = {
            'losses': [],
            'accept_lengths': [],
            'adaptation_count': 0
        }

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

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
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
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

            if use_adaptation and accept_length > 0 and self.adapter_optimizer is not None and not self.first_token_adapted: 
                # collect data for online adaptation
                accepted_indices = retrieve_indices[best_candidate, :accept_length+1]
                    
                adapt_loss = self._perform_online_adaptation(
                    input_ids=input_ids,
                    candidates=candidates,
                    retrieve_indices=retrieve_indices,
                    best_candidate=best_candidate,
                    accept_length=accept_length,
                    hidden_state_new=hidden_state_new,
                    logits=logits
                )
                
                if adapt_loss is not None and adapt_loss > 0:
                    adaptation_info['losses'].append(adapt_loss)
                    adaptation_info['accept_lengths'].append(accept_length)
                    adaptation_info['adaptation_count'] += 1
                    self.first_token_adapted = True

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

        # print online info
        if use_adaptation and adaptation_info['adaptation_count'] > 0:
            if adaptation_info['adaptation_count'] == 1:
                print(f"Single adaptation performed: loss={adaptation_info['losses'][0]:.4f}, accept_length={adaptation_info['accept_lengths'][0]}")
            else:
                avg_loss = sum(adaptation_info['losses']) / len(adaptation_info['losses'])
                avg_accept = sum(adaptation_info['accept_lengths']) / len(adaptation_info['accept_lengths'])
                print(f"Adaptation stats: avg_loss={avg_loss:.4f}, avg_accept_length={avg_accept:.2f}, count={adaptation_info['adaptation_count']}")

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
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
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
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
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
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

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
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
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
