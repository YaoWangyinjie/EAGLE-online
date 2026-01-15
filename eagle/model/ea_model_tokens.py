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
        
        # 用于保存初始权重，实现每个question独立训练
        self.initial_weights = None 

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

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def setup_online_adaptation(
            self, 
            adaptation_lr=const_lr, 
            adaptation_temperature=const_temp,
        ):
        """superparams of online adaptation"""
        self.enable_online_adaptation = True
        self.adaptation_lr = adaptation_lr
        self.adaptation_temperature = adaptation_temperature

        # 设置梯度
        for param in self.ea_layer.parameters():
            param.requires_grad = True
                
        self.adapter_optimizer = torch.optim.Adam(self.ea_layer.parameters(), lr=adaptation_lr)
        
        # 保存初始权重（如果还没有保存过）
        # 注意：这里保存的是 CPU 上的副本，以节省显存并防止随模型更新
        if self.initial_weights is None:
            self.initial_weights = {k: v.cpu().clone() for k, v in self.ea_layer.state_dict().items()}
            print("[Online Adaptation] Initial weights saved for per-question reset.")

    def _perform_online_adaptation(
            self,
            all_accepted_tokens,      # [1, total_len]
            all_hidden_states,        # [1, total_len, hidden_dim*3]
            all_target_logits,        # [1, total_len, vocab_size]
            train_steps=1
        ):
        device = next(self.ea_layer.parameters()).device

        # 这里转换数据格式，从bf16到float32以解决训练中爆表的问题，输入也进行转换
        original_dtype = next(self.ea_layer.parameters()).dtype
        self.ea_layer.float()

        if all_hidden_states.dtype != torch.float32:
            all_hidden_states = all_hidden_states.float()
        if all_target_logits.dtype != torch.float32:
            all_target_logits = all_target_logits.float()

        was_training = self.ea_layer.training

        # 这里保存tree_mask和kv_cache以供后续恢复，训练时禁用以防kv_cache被污染以及tree_mask影响训练
        saved_tree_mask = getattr(self.ea_layer, 'tree_mask', None)
        saved_stable_kv = getattr(self.ea_layer, 'stable_kv', None)
        self.ea_layer.reset()
        self.ea_layer.stable_kv = None

        # 训练模式
        self.ea_layer.train()

        if all_accepted_tokens.device != device:
            all_accepted_tokens = all_accepted_tokens.to(device)
        if all_hidden_states.device != device:
            all_hidden_states = all_hidden_states.to(device)
        if all_target_logits.device != device:
            all_target_logits = all_target_logits.to(device)
            
        # === 构造错位输入 (Shifted Input) 符合 EAGLE 结构 ===
        # Input Tokens: [x_1, x_2, ..., x_k] (从第2个token开始)
        # Input Hidden: [h_0, h_1, ..., h_{k-1}] (从第1个hidden开始)
        # Target Logits: [p_1, p_2, ..., p_k] (Base Model 在处理 x_1...x_k 后的输出)
        shifted_input_tokens = all_accepted_tokens[:, 1:]        # [1, total_len-1]
        shifted_full_hidden = all_hidden_states[:, :-1, :]       # [1, total_len-1, hidden_dim*3]
        shifted_target_logits = all_target_logits[:, 1:, :]      # [1, total_len-1, vocab_size]

        seq_len = shifted_input_tokens.shape[1]
        if seq_len == 0:
            # 恢复环境
            self.ea_layer.tree_mask = saved_tree_mask
            self.ea_layer.stable_kv = saved_stable_kv
            self.ea_layer.to(original_dtype)
            self.ea_layer.train(was_training)
            return 0.0

        # 词表映射
        if self.has_vocab_mapping and self.draft_vocab_size < self.vocab_size:
            t2d = self.ea_layer.t2d.to(device)
            # 裁剪 Target Logits 维度
            target_logits_answer = shifted_target_logits
            target_max_token = target_logits_answer.argmax(dim=-1)
            valid_token_mask = t2d[target_max_token]
            valid_for_training = valid_token_mask.bool()
            
            if valid_for_training.sum() == 0:
                 # 恢复环境并返回
                self.ea_layer.tree_mask = saved_tree_mask
                self.ea_layer.stable_kv = saved_stable_kv
                self.ea_layer.to(original_dtype)
                self.ea_layer.train(was_training)
                return 0.0

            pass
        else:
             if shifted_target_logits.shape[-1] > self.draft_vocab_size:
                shifted_target_logits = shifted_target_logits[..., :self.draft_vocab_size]

        # 这里开始训练
        total_loss = 0.0
        with torch.enable_grad():
            for step in range(train_steps):
                self.adapter_optimizer.zero_grad()

                # 手动 Forward，绕过 cnets.Model.forward 的 dataprepare 和 Loop
                # 1. Embeddings
                inputs_embeds = self.ea_layer.embed_tokens(shifted_input_tokens)
                
                # 2. Hidden Projection
                current_hidden = shifted_full_hidden
                if current_hidden.shape[-1] != inputs_embeds.shape[-1]:
                     current_hidden = self.ea_layer.fc(current_hidden)

                # 3. Mask (Causal)
                batch_size, seq_len, _ = inputs_embeds.shape
                attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
                extended_attention_mask = self.ea_layer._prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_len), inputs_embeds, 0
                )

                # 4. MidLayer Forward
                layer_outputs = self.ea_layer.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=current_hidden,
                    attention_mask=extended_attention_mask,
                    use_cache=False
                )
                draft_hidden_out = layer_outputs[0]

                # 5. Logits
                draft_hidden_out = self.ea_layer.norm(draft_hidden_out)
                draft_logits = self.ea_layer.lm_head(draft_hidden_out)

                # 6. Loss Calculation
                target_logits_step = shifted_target_logits
                
                # 如果有词表映射，进行筛选
                if self.has_vocab_mapping and self.draft_vocab_size < self.vocab_size:
                     # 映射 Target Logits 到 Draft 词表
                     target_logits_step = target_logits_step[..., t2d]
                     
                     # 筛选有效位置 (valid_for_training)
                     draft_logits_valid = draft_logits[valid_for_training]
                     target_logits_valid = target_logits_step[valid_for_training]
                else:
                    draft_logits_valid = draft_logits
                    target_logits_valid = target_logits_step

                # 确保形状匹配
                if draft_logits_valid.shape[-1] != target_logits_valid.shape[-1]:
                     min_v = min(draft_logits_valid.shape[-1], target_logits_valid.shape[-1])
                     draft_logits_valid = draft_logits_valid[..., :min_v]
                     target_logits_valid = target_logits_valid[..., :min_v]

                # KL Divergence
                target_probs = F.softmax(target_logits_valid / const_temp, dim=-1)
                draft_log_probs = F.log_softmax(draft_logits_valid / const_temp, dim=-1)
                loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
                loss = loss_fn(draft_log_probs, target_probs)
                loss = loss * (const_temp ** 2)
                
                # ========== 调试输出 (保留关键信息) ==========
                with torch.no_grad():
                    # 1. 计算 Top-1 预测
                    draft_pred = draft_logits_valid.argmax(dim=-1)
                    target_pred = target_logits_valid.argmax(dim=-1)
                    
                    # 2. 计算一致率
                    match_mask = (draft_pred == target_pred)
                    match_count = match_mask.sum().item()
                    total_count = match_mask.numel()
                    accuracy = match_count / total_count if total_count > 0 else 0.0
                    
                    # 3. 打印信息
                    print(f"\n[Online Adaptation Step {step+1}/{train_steps}]")
                    print(f"  Loss: {loss.item():.6f}")
                    print(f"  Acc : {accuracy:.2%} ({match_count}/{total_count})")
                # ==========================================

                if torch.isnan(loss) or torch.isinf(loss):
                    print("Invalid loss")
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.ea_layer.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                self.adapter_optimizer.step()

                total_loss += loss.item()
        
        # 这里恢复了tree_mask,kv_cache,数据类型
        self.ea_layer.tree_mask = saved_tree_mask
        self.ea_layer.stable_kv = saved_stable_kv
        self.ea_layer.to(original_dtype)
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
            warmup_steps=10, # 新增参数：热身步数
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
                    adaptation_lr=adaptation_lr or const_lr,
                    adaptation_temperature=adaptation_temperature or const_temp
                )

        use_adaptation = enable_adaptation if enable_adaptation is not None else self.enable_online_adaptation
        
        # 数据收集容器
        collected_tokens = []
        collected_hiddens = []
        collected_logits = []
        is_collecting = use_adaptation and self.adapter_optimizer is not None
        
        # 1. 状态重置：恢复到初始权重 (Per-question reset)
        if is_collecting:
            if self.initial_weights is not None:
                self.ea_layer.load_state_dict(self.initial_weights, strict=False)
            else:
                pass

            # 2. 获取 Prefill 数据 (Base Model Output)
            # 我们先存储起来，暂不训练
            with torch.no_grad():
                if hasattr(self.base_model.model, "tree_mask"):
                    self.base_model.model.tree_mask = None

                base_outputs = self.base_model.model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                if self.use_eagle3:
                    # 为了与 utils.py 中的 tree_decoding 逻辑保持严格一致 (line 328)，我们拼接所有返回的 hidden_states
                    # utils.py: hidden_state = torch.cat(outputs["hidden_states"], dim=-1)
                    # 注意：这里假设 base_model 配置返回的 hidden_states 正是 EAGLE-3 所需的那几层（通常是后3层）
                    # 如果 base_model 返回所有层，这里的逻辑需要确保只取最后3层，但为了保险起见，我们模仿 utils.py 的行为
                    # 或者更稳健的做法：既然报错显示 hidden_state_new 是 12288 (3*4096)，我们就强制取后3层
                    prefill_hidden_states = torch.cat(base_outputs.hidden_states[-3:], dim=-1).cpu()
                else:
                    # EAGLE-1/2 只用最后一层 (4096 dim)
                    prefill_hidden_states = base_outputs[0].cpu()
                
                prefill_logits = self.base_model.lm_head(base_outputs[0]).cpu()
                prefill_tokens = input_ids.cpu()
                
                # 将 Prefill 数据放入收集器
                collected_tokens.append(prefill_tokens)
                collected_hiddens.append(prefill_hidden_states)
                collected_logits.append(prefill_logits)

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
        reset_tree_mode(self)
        
        # prefill (Normal EAGLE Init)
        with torch.no_grad():
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits_before, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor
            )
        
        # 记录 prefill 的 last hidden (用于收集逻辑中的拼接)
        # 修复 NoneType 报错：
        # 如果我们在 collecting，我们已经有了 prefill_hidden_states，直接取最后一个作为起始 hidden
        # 这样比依赖 initialize_tree 的返回值更稳健（且保证在 CPU 上，与 collected_hiddens 匹配）
        if is_collecting:
             # prefill_hidden_states shape: [1, seq_len, hidden_dim]
             last_step_hidden = prefill_hidden_states[:, -1:, :]
        else:
             last_step_hidden = None

        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        
        adaptation_loss = 0.0
        
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
            
            # === 数据收集逻辑 ===
            if is_collecting and total_steps < warmup_steps:
                 # 获取当前 step 被接受的 tokens（包含树根）
                 # candidates: [1, tree_nodes] -> candidates[best_candidate] 是最优路径的 token id
                 
                 # step_accepted_tokens: [accept_length+1]
                 step_accepted_tokens = candidates[best_candidate, :accept_length + 1]
                 
                 # 构造对应的 Hidden States
                 # first_hidden: 来自上一轮的 last_step_hidden (Pre-fill end or previous step end)
                 # 确保 first_hidden 在 CPU 上（如果 last_step_hidden 维护得当，它应该是 CPU Tensor）
                 first_hidden = last_step_hidden # [1, 1, hidden_dim*3]
                 if first_hidden.device != torch.device('cpu'):
                     first_hidden = first_hidden.cpu()
                 
                 if accept_length > 0:
                     # rest_hidden: 从 outputs 中按路径取
                     # 关键修复：必须根据 use_eagle3 决定提取维度，以匹配 Draft Model 的输入要求
                     if self.use_eagle3:
                         # EAGLE-3: 拼接最后 3 层 (12288 dim)
                         # 注意：utils.py 中 tree_decoding 返回的 outputs 包含 hidden_states
                         raw_tree_hidden = torch.cat(outputs.hidden_states[-3:], dim=-1)
                     else:
                         # EAGLE-1/2: 只用最后一层 (4096 dim)
                         raw_tree_hidden = outputs[0]
                     
                     accepted_indices = retrieve_indices[best_candidate, :accept_length]
                     rest_hidden = raw_tree_hidden[:, accepted_indices, :].cpu()
                     step_hidden = torch.cat([first_hidden, rest_hidden], dim=1)
                 else:
                     step_hidden = first_hidden
                     
                 
                 if step_accepted_tokens.shape[0] > 1: # 只有产生了新 token 才收集
                     # 切掉 Root
                     new_tokens = step_accepted_tokens[1:]
                     new_hiddens = step_hidden[:, 1:, :]
                     
                     num_new = new_tokens.shape[0] # accept_length
                     
                     # 取出对应长度的 Logits
                     current_step_logits = logits[best_candidate, :num_new, :].unsqueeze(0).cpu()
                     
                     collected_tokens.append(new_tokens.cpu().unsqueeze(0))
                     collected_hiddens.append(new_hiddens)
                     collected_logits.append(current_step_logits)
                 
                 pass

            total_accept_length += accept_length
            total_steps += 1
            total_drafted_tokens += candidates.shape[1] 

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
            
            # 更新 last_step_hidden 用于下一次收集
            # 直接使用当前 step 拼接好的最后一个 hidden，确保维度一致 (4096)
            if is_collecting:
                last_step_hidden = step_hidden[:, -1:, :]

            # === 触发训练 ===
            if is_collecting and total_steps == warmup_steps:
                # 拼接所有数据
                # collected_tokens: List of [1, seq_len]
                full_tokens = torch.cat(collected_tokens, dim=1)
                full_hiddens = torch.cat(collected_hiddens, dim=1)
                full_logits = torch.cat(collected_logits, dim=1)
                
                # print(f"[Online Adaptation] Warmup done (steps={total_steps}). Training on {full_tokens.shape[1]} tokens...")
                
                adaptation_loss = self._perform_online_adaptation(
                    all_accepted_tokens=full_tokens,
                    all_hidden_states=full_hiddens,
                    all_target_logits=full_logits,
                    train_steps=1
                )
                
                # 停止收集
                is_collecting = False
                # print(f"[Online Adaptation] Training completed. Loss: {adaptation_loss:.4f}")

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
            print(f"[Stats] Total Iteration Steps: {total_steps}, Total Accepted: {total_accept_length}")        

        if use_adaptation:
             print(f"[Online Adaptation] Warmup loss={adaptation_loss:.4f}")

        run_stats = {
            "total_accept_length": total_accept_length,
            "total_drafted_tokens": total_drafted_tokens, 
            "total_steps": total_steps,
            "losses": [adaptation_loss] if use_adaptation else []
        }

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, run_stats
