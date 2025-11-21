"""
dbug
"""
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from safetensors.torch import load_file

def check_base_model(base_model_path):
    """直接检查base model"""
    print("=== Checking Base Model ===")
    print(f"Path: {base_model_path}")
    
    # 加载配置
    config = AutoConfig.from_pretrained(base_model_path)
    print(f"Model type: {config.model_type}")
    print(f"Config vocab size: {config.vocab_size}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"LM head shape: {model.lm_head.weight.shape}")
    print(f"Actual vocab size: {model.lm_head.weight.shape[0]}")
    
    # 检查tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    return model, config

def check_ea_weights(ea_model_path):
    """直接检查EA model权重文件"""
    print("\n=== Checking EA Model Weights ===")
    print(f"Path: {ea_model_path}")
    
    # 读取配置
    config_path = os.path.join(ea_model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            ea_config = json.load(f)
        print(f"EA Config: {json.dumps(ea_config, indent=2)}")
    
    # 尝试加载权重
    weights = None
    
    # 尝试加载pytorch格式
    try:
        pt_path = os.path.join(ea_model_path, "pytorch_model.bin")
        if os.path.exists(pt_path):
            weights = torch.load(pt_path, map_location='cpu')
            print("Loaded pytorch_model.bin")
    except:
        pass
    
    # 尝试加载safetensors格式
    if weights is None:
        try:
            st_path = os.path.join(ea_model_path, "model.safetensors")
            if os.path.exists(st_path):
                weights = load_file(st_path)
                print("Loaded model.safetensors")
        except:
            pass
    
    if weights:
        print("\n--- Weight shapes ---")
        for name, tensor in weights.items():
            if 'lm_head' in name or 'embed' in name:
                print(f"{name}: {tensor.shape}")
        
        # 特别检查lm_head
        lm_head_keys = [k for k in weights.keys() if 'lm_head' in k and 'weight' in k]
        if lm_head_keys:
            print(f"\nLM head weight shape: {weights[lm_head_keys[0]].shape}")
    
    return weights, ea_config if 'ea_config' in locals() else None

def check_cnets_module():
    """检查cnets模块的设置"""
    print("\n=== Checking cnets module ===")
    
    # 导入cnets看看它的模型定义
    try:
        from eagle.model.cnets import Model
        from eagle.model.configs import EConfig
        
        # 创建一个dummy config来检查
        dummy_config = EConfig(
            hidden_size=4096,
            vocab_size=32000,  # 检查这个默认值
            num_layers=1,
            num_attention_heads=32,
            num_groups=1,
            total_tokens=60,
            depth=7,
            top_k=10,
            threshold=1.0
        )
        
        print(f"EConfig default vocab_size: {dummy_config.vocab_size}")
        
    except Exception as e:
        print(f"Error loading cnets: {e}")

def compare_models(base_model_path, ea_model_path):
    """比较两个模型"""
    print("="*60)
    print("DIRECT MODEL COMPARISON")
    print("="*60)
    
    # 检查base model
    base_model, base_config = check_base_model(base_model_path)
    
    # 检查ea model权重
    ea_weights, ea_config = check_ea_weights(ea_model_path)
    
    # 检查cnets
    check_cnets_module()
    
    print("\n=== Summary ===")
    print(f"Base model vocab size: {base_model.lm_head.weight.shape[0]}")
    if ea_weights and any('lm_head.weight' in k for k in ea_weights.keys()):
        lm_head_key = [k for k in ea_weights.keys() if 'lm_head.weight' in k][0]
        print(f"EA model vocab size: {ea_weights[lm_head_key].shape[0]}")
    
    # 额外检查：看看EA model是否有词表映射
    if ea_weights:
        print("\n--- Checking for vocabulary mappings ---")
        for key in ea_weights.keys():
            if 'd2t' in key or 't2d' in key or 'vocab' in key.lower():
                print(f"Found: {key} with shape {ea_weights[key].shape}")

if __name__ == "__main__":
    base_model_path = "./model_weight/DeepSeek-R1-Distill-Llama-8B/"
    ea_model_path = "./model_weight/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B/"
    
    compare_models(base_model_path, ea_model_path)
