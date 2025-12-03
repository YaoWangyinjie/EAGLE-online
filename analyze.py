import json
import os
import statistics
from typing import List, Dict, Any

json_dir_name = "mt_bench_online"
json_file_name = "0-temperature-0.0.jsonl" 
report_root_dir = "analysis_report"

def load_data(json_dirname: str, json_filename: str) -> List[Dict[str, Any]]:
    current_dir = os.getcwd()
    filepath = os.path.join(current_dir, json_dirname, json_filename)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print(f"Current working directory: {current_dir}")
        return []
    
    try:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line: 
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
        print(f"Successfully loaded {len(data)} entries from {filepath}")
        return data
    except Exception as e:
        print(f"An error occurred while reading file: {e}")
        return []

def calculate_stats(data_list: List[float]) -> Dict[str, float]:
    if not data_list:
        return {"mean": 0, "median": 0, "max": 0, "min": 0, "sum": 0}
    
    return {
        "mean": statistics.mean(data_list),
        "median": statistics.median(data_list),
        "max": max(data_list),
        "min": min(data_list),
        "sum": sum(data_list)
    }

def get_output_filepath(json_filename: str, json_dirname: str, report_root: str) -> str:
    current_dir = os.getcwd()
    report_dir = os.path.join(current_dir, report_root, json_dirname)
    os.makedirs(report_dir, exist_ok=True)
    
    base_name = os.path.splitext(json_filename)[0]
    output_filename = f"{base_name}.txt"
    output_path = os.path.join(report_dir, output_filename)
    
    return output_path

def generate_analysis(data: List[Dict[str, Any]], output_path: str):
    if not data:
        return

    model_ids = set()
    metrics = {
        "total_tokens": [], 
        "total_time": [],   
        
        "turn1_tokens": [],
        "turn1_time": [],
        "turn1_tps": [],
        
        "turn2_tokens": [],
        "turn2_time": [],
        "turn2_tps": [],
        
        "overall_tps": []   
    }

    valid_entries = 0
    
    for entry in data:
        model_ids.add(entry.get("model_id", "unknown"))
        
        if "choices" not in entry or not entry["choices"]:
            continue
            
        choice = entry["choices"][0]
        
        new_tokens_list = choice.get("new_tokens", [])
        wall_time_list = choice.get("wall_time", [])
        
        if len(new_tokens_list) != len(wall_time_list):
            continue

        valid_entries += 1
        
        q_total_tokens = sum(new_tokens_list)
        q_total_time = sum(wall_time_list)
        metrics["total_tokens"].append(q_total_tokens)
        metrics["total_time"].append(q_total_time)
        
        for i in range(len(new_tokens_list)):
            t_tokens = new_tokens_list[i]
            t_time = wall_time_list[i]
            
            t_tps = t_tokens / t_time if t_time > 0 else 0
            
            metrics["overall_tps"].append(t_tps)
            
            if i == 0: # Turn 1
                metrics["turn1_tokens"].append(t_tokens)
                metrics["turn1_time"].append(t_time)
                metrics["turn1_tps"].append(t_tps)
            elif i == 1: # Turn 2
                metrics["turn2_tokens"].append(t_tokens)
                metrics["turn2_time"].append(t_time)
                metrics["turn2_tps"].append(t_tps)

    stats = {k: calculate_stats(v) for k, v in metrics.items()}
    
    report_lines = []
    sep = "=" * 60
    sub_sep = "-" * 60
    
    report_lines.append(sep)
    report_lines.append(f"Analysis Report for: {json_file_name}")
    report_lines.append(f"Directory Context: {json_dir_name}")
    report_lines.append(sep)
    
    report_lines.append(f"\n[General Information]")
    report_lines.append(f"Total Questions Processed: {valid_entries}")
    report_lines.append(f"Model ID(s) Found: {', '.join(list(model_ids))}")
    
    report_lines.append(f"\n[Token Generation Statistics]")
    report_lines.append(sub_sep)
    report_lines.append(f"Avg Tokens per Question (Total): {stats['total_tokens']['mean']:.2f}")
    report_lines.append(f"Max Tokens in a Question:        {stats['total_tokens']['max']:.2f}")
    report_lines.append(f"\nTurn 1 - Avg Tokens:             {stats['turn1_tokens']['mean']:.2f}")
    report_lines.append(f"Turn 2 - Avg Tokens:             {stats['turn2_tokens']['mean']:.2f}")
    
    report_lines.append(f"\n[Latency / Wall Time Statistics (Seconds)]")
    report_lines.append(sub_sep)
    report_lines.append(f"Avg Time per Question (Total):   {stats['total_time']['mean']:.4f} s")
    report_lines.append(f"\nTurn 1 - Avg Time:               {stats['turn1_time']['mean']:.4f} s")
    report_lines.append(f"Turn 1 - Max Time:               {stats['turn1_time']['max']:.4f} s")
    report_lines.append(f"\nTurn 2 - Avg Time:               {stats['turn2_time']['mean']:.4f} s")
    report_lines.append(f"Turn 2 - Max Time:               {stats['turn2_time']['max']:.4f} s")
    
    report_lines.append(f"\n[Speed / Throughput Statistics (Tokens/Second)]")
    report_lines.append(sub_sep)
    report_lines.append(f"Overall Avg TPS (All turns):     {stats['overall_tps']['mean']:.2f} tokens/s")
    report_lines.append(f"Overall Median TPS:              {stats['overall_tps']['median']:.2f} tokens/s")
    
    report_lines.append(f"\nTurn 1 - Avg TPS:                {stats['turn1_tps']['mean']:.2f} tokens/s")
    report_lines.append(f"Turn 1 - Median TPS:             {stats['turn1_tps']['median']:.2f} tokens/s")
    
    report_lines.append(f"\nTurn 2 - Avg TPS:                {stats['turn2_tps']['mean']:.2f} tokens/s")
    report_lines.append(f"Turn 2 - Median TPS:             {stats['turn2_tps']['median']:.2f} tokens/s")
    
    report_lines.append(sep)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\nSuccessfully generated analysis report at:")
        print(f"  {output_path}")
        print("-" * 60)
    except Exception as e:
        print(f"Error writing report file: {e}")

if __name__ == "__main__":
    print(f"Working Directory: {os.getcwd()}")
    print(f"JSON Directory: {json_dir_name}")
    print(f"Target JSON File: {json_file_name}")
    
    output_filepath = get_output_filepath(json_file_name, json_dir_name, report_root_dir)
    print(f"Output Report Path: {output_filepath}")
    print("-" * 60)
    
    raw_data = load_data(json_dir_name, json_file_name)
    if raw_data:
        generate_analysis(raw_data, output_filepath)
    else:
        print("No data loaded to analyze.")
