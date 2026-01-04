"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import numpy as np
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *



def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    # if use_ray:
    #     ray.get(ans_handles)

    all_stats_collected = []
    
    if use_ray:
        results = ray.get(ans_handles) # 获取所有进程返回的 list
        for stats_list in results:
            all_stats_collected.extend(stats_list)
    else:
        # 单卡模式下 ans_handles 已经是 list of list
        for stats_list in ans_handles:
            all_stats_collected.extend(stats_list)

    # 构造统计文件名：{model_id}_stat.txt
    stat_file = f"{os.path.dirname(answer_file)}/{args.model_id}_stat.txt"
    save_statistics(all_stats_collected, stat_file)

def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # temperature = 0.0

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )

    if args.enable_online_adaptation:
        model.setup_online_adaptation(
            adaptation_lr=args.adaptation_lr,
            adaptation_temperature=args.adaptation_temperature
        )
        print(f"Online adaptation enabled with lr={args.adaptation_lr}")

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        messages = []
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(1):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, warmup_stats = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
                is_llama3=True,
                enable_adaptation=args.enable_online_adaptation,  # online
                adaptation_lr=args.adaptation_lr,
                adaptation_temperature=args.adaptation_temperature
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            if stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            # stop_str = "</s>"
            # if stop_str and output.find(stop_str) > 0:
            #     output = output[: output.find(stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()



            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            messages.append({
                "role": "assistant",
                "content": output
            })
    print('Warmup done')

    local_stats = [] 

    # questions=questions[6:]
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = []
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(1):
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

                # try:
                torch.cuda.synchronize()
                start_time = time.time()

                output_ids, new_token, idx, run_stats = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                    enable_adaptation=args.enable_online_adaptation,  # online
                    adaptation_lr=args.adaptation_lr,
                    adaptation_temperature=args.adaptation_temperature
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time

                r_accept_len = run_stats['total_accept_length']
                if hasattr(r_accept_len, 'item'):
                    r_accept_len = r_accept_len.item()
                    
                r_drafted = run_stats['total_drafted_tokens']
                if hasattr(r_drafted, 'item'):
                    r_drafted = r_drafted.item()
                    
                r_steps = run_stats['total_steps']
                if hasattr(r_steps, 'item'):
                    r_steps = r_steps.item()

                avg_accept = r_accept_len / r_steps if r_steps > 0 else 0
                if r_drafted > 0:
                    acc_rate = r_accept_len / r_drafted
                else:
                    acc_rate = 0.0

                losses = run_stats['losses']
                avg_loss = sum(losses) / len(losses) if len(losses) > 0 else None
                
                speed = int(new_token) / total_time if total_time > 0 else 0

                local_stats.append({
                    "qid": question["question_id"],
                    "speed": speed,
                    "acc_rate": acc_rate,          # 这里的 acc_rate 已经是 float 了
                    "avg_accept_len": avg_accept,  # float
                    "total_accept_tokens": r_accept_len, # int/float
                    "avg_loss": avg_loss,
                    "new_tokens": int(new_token),
                    "time": total_time
                })

                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                # stop_str = "</s>"
                # if stop_str and output.find(stop_str) > 0:
                #     output = output[: output.find(stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

    return local_stats


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

def save_statistics(all_stats, stat_file):
    """统计并保存详细性能数据"""
    if not all_stats:
        return

    # 提取各列数据
    speeds = [s['speed'] for s in all_stats]
    acc_rates = [s['acc_rate'] for s in all_stats] # 接受率
    avg_accepts = [s['avg_accept_len'] for s in all_stats] # 平均接受长度
    valid_losses = [s['avg_loss'] for s in all_stats if s['avg_loss'] is not None]

    os.makedirs(os.path.dirname(stat_file), exist_ok=True)
    
    with open(stat_file, "w") as f:
        # === 1. 总体统计 (Global Stats) ===
        f.write("="*80 + "\n")
        f.write(f"Global Statistics (Total Questions: {len(all_stats)})\n")
        f.write("="*80 + "\n\n")

        # 辅助打印函数
        def write_metric(name, data, is_percent=False):
            if not data:
                f.write(f"[{name}]\n  No data.\n\n")
                return
            fmt = ".2%" if is_percent else ".4f"
            f.write(f"[{name}]\n")
            f.write(f"  Mean:   {np.mean(data):{fmt}}\n")
            f.write(f"  Median: {np.median(data):{fmt}}\n")
            f.write(f"  Max:    {np.max(data):{fmt}}\n")
            f.write(f"  Min:    {np.min(data):{fmt}}\n\n")

        write_metric("Speed (tokens/s)", speeds)
        write_metric("Acceptance Rate (Accepted/Drafted)", acc_rates, is_percent=True)
        write_metric("Avg Accept Length (per step)", avg_accepts)
        
        if valid_losses:
            f.write(f"[Online Loss]\n")
            f.write(f"  Mean:   {np.mean(valid_losses):.6f}\n")
            f.write(f"  Median: {np.median(valid_losses):.6f}\n")
            f.write(f"  Max:    {np.max(valid_losses):.6f}\n")
            f.write(f"  Min:    {np.min(valid_losses):.6f}\n\n")
        else:
            f.write(f"[Online Loss]\n  No loss data recorded.\n\n")

        # === 2. 详细统计 (Per Question) ===
        f.write("="*100 + "\n")
        f.write("Per Question Details\n")
        f.write("="*100 + "\n")
        
        # 表头：增加了 Acc Rate 和 Total Acc Tokens
        header = f"{'Question ID':<12} | {'Speed(t/s)':<10} | {'Acc Rate':<10} | {'Avg Acc Len':<12} | {'Tot Acc Tok':<12} | {'Avg Loss':<10} | {'Time(s)':<8}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for s in all_stats:
            loss_str = f"{s['avg_loss']:.4f}" if s['avg_loss'] is not None else "N/A"
            line = (f"{s['qid']:<12} | {s['speed']:<10.2f} | {s['acc_rate']:<10.2%} | "
                    f"{s['avg_accept_len']:<12.2f} | {s['total_accept_tokens']:<12} | "
                    f"{loss_str:<10} | {s['time']:<8.2f}")
            f.write(line + "\n")
    
    print(f"Statistics saved to {stat_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/home/lyh/weights/hf/eagle3/DSL/8B3/",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/DSL/8B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="llama38b2_40")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )

    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument("--enable-online-adaptation", action="store_true")
    parser.add_argument("--adaptation-lr", type=float, default=5e-5)
    parser.add_argument("--adaptation-temperature", type=float, default=1.0)

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}_online/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
