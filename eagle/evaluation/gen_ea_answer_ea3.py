import argparse, json, os, tempfile, time
from pathlib import Path
import torch
from accelerate.utils import set_seed
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm
from ..model.ea_model_3 import EaModel
from .gen_ea_answer_ds import _json_safe, _atomic_write_jsonl, _output_paths, _decode_output


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def main():
    p = argparse.ArgumentParser(description='EA3 persistent long-term adaptation evaluator')
    p.add_argument('--base-model-path', required=True); p.add_argument('--ea-model-path', required=True)
    p.add_argument('--question-file', default='eagle/data/mt_bench/question.jsonl'); p.add_argument('--answer-file', required=True); p.add_argument('--model-id', required=True)
    p.add_argument('--max-new-token', type=int, default=1024); p.add_argument('--max-length', type=int, default=2048); p.add_argument('--warmup-steps', type=int, default=5); p.add_argument('--total-token', type=int, default=60); p.add_argument('--depth', type=int, default=5); p.add_argument('--top-k', type=int, default=10); p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--adaptation-lr', type=float, default=1e-6); p.add_argument('--adaptation-temperature', type=float, default=1.0); p.add_argument('--scope', default='head_only', choices=['head_only','default','fc_only','midlayer_only','norm_only','lm_head_only'])
    p.add_argument('--replay-max-fragments', type=int, default=64); p.add_argument('--replay-max-tokens', type=int, default=8192); p.add_argument('--replay-fraction', type=float, default=.5); p.add_argument('--anchor-weight', type=float, default=1e-3); p.add_argument('--max-relative-drift', type=float, default=.02); p.add_argument('--min-valid-vocab-ratio', type=float, default=.8); p.add_argument('--gradient-clip-norm', type=float, default=.2)
    p.add_argument('--experiment-name', required=True); p.add_argument('--question-begin', type=int); p.add_argument('--question-end', type=int)
    p.add_argument('--manifest', type=str); p.add_argument('--no-update', action='store_true')
    args = p.parse_args(); set_seed(0)
    model = EaModel.from_pretrained(base_model_path=args.base_model_path, ea_model_path=args.ea_model_path, total_token=args.total_token, depth=args.depth, top_k=args.top_k, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
    if not args.no_update:
        model.setup_long_term_adaptation(adaptation_lr=args.adaptation_lr, adaptation_temperature=args.adaptation_temperature, scope=args.scope, replay_max_fragments=args.replay_max_fragments, replay_max_tokens=args.replay_max_tokens, replay_fraction=args.replay_fraction, anchor_weight=args.anchor_weight, max_relative_drift=args.max_relative_drift, min_valid_vocab_ratio=args.min_valid_vocab_ratio, gradient_clip_norm=args.gradient_clip_norm)
    tok = model.get_tokenizer(); rows=[]; answers=[]; failures=[]
    manifest = Path(args.manifest or str(args.answer_file) + '.manifest.json')
    _atomic_write_text(manifest, json.dumps({'experiment': args.experiment_name, 'question_file': args.question_file, 'base_model_path': args.base_model_path, 'ea_model_path': args.ea_model_path, 'config': vars(args), 'strategy': 'no_update' if args.no_update else 'persistent_sliding_window_anchor_guard'}, indent=2, ensure_ascii=False))
    for q in tqdm(load_questions(args.question_file, args.question_begin, args.question_end)):
        messages=[]; turns=[]; idxs=[]; nts=[]; times=[]; failed=False
        for turn, text in enumerate(q.get('turns', [])):
            messages.append({'role':'user','content':text}); prompt=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True); ids=tok([prompt], add_special_tokens=False).input_ids
            if len(ids[0]) >= args.max_length:
                failure={'qid':q['question_id'],'turn':turn,'prompt_tokens':len(ids[0]),'reason':'prompt_exceeds_cache_capacity','status':'skipped','experiment':args.experiment_name}
                failures.append(_json_safe(failure)); rows.append(_json_safe(failure)); failed=True; break
            start=time.time()
            try:
                out,n,idx,run=model.eagenerate(torch.as_tensor(ids).cuda(), temperature=args.temperature, max_new_tokens=args.max_new_token, max_length=args.max_length, warmup_steps=args.warmup_steps, log=True, is_llama3=True, enable_adaptation=not args.no_update, adaptation_lr=args.adaptation_lr, adaptation_temperature=args.adaptation_temperature, diagnostics=True)
            except RuntimeError as exc:
                failure={'qid':q['question_id'],'turn':turn,'prompt_tokens':len(ids[0]),'reason':'runtime_error','error':str(exc),'status':'skipped','experiment':args.experiment_name}
                failures.append(_json_safe(failure)); rows.append(_json_safe(failure)); failed=True; break
            elapsed=time.time()-start
            d=run.get('diagnostics', {}) or {}; ar=d.get('adaptation', {}) or {}
            if d.get('cache_stop'):
                failure={'qid':q['question_id'],'turn':turn,'prompt_tokens':len(ids[0]),'reason':d['cache_stop'],'status':'skipped','experiment':args.experiment_name}
                failures.append(_json_safe(failure)); rows.append(_json_safe(failure)); failed=True; break
            drafted = int(run.get('total_drafted_tokens', 0)); accepted = int(run.get('total_accept_length', 0));
            if drafted > 0 and not args.no_update:
                model.record_acceptance(accepted / drafted, accepted / max(1, int(run.get('total_steps', 1))))
            state = model.get_adaptation_state()
            stat={'qid':q['question_id'],'category':q.get('category','unknown'),'turn':turn,'experiment':args.experiment_name,'dataset':Path(args.question_file).parent.name,'prompt_tokens':len(ids[0]),'new_tokens':int(n),'time':elapsed,'total_accept_length':run.get('total_accept_length',0),'total_drafted_tokens':run.get('total_drafted_tokens',0),'total_steps':run.get('total_steps',0),'loss':ar.get('loss'),'distill_loss':ar.get('distill_loss'),'anchor_loss':ar.get('anchor_loss'),'total_loss':ar.get('total_loss'),'finite_position_ratio':ar.get('finite_position_ratio'),'mapping_valid_ratio':ar.get('mapping_valid_ratio'),'adaptation_status':ar.get('status'),'replay_fragments':state.get('replay_fragments',0),'replay_tokens':state.get('replay_tokens',0),'paused':state.get('paused',False),'pause_reason':state.get('pause_reason'),'ema_loss':state.get('ema_loss'),'ema_acceptance':state.get('ema_acceptance'),'weight_version':state.get('weight_version',getattr(model,'weight_version',0)),'cache_epoch':state.get('cache_epoch',getattr(model,'cache_epoch',0)),'adaptation_state':_json_safe(state)}; rows.append(_json_safe(stat)); answer=_decode_output(tok,out,len(ids[0])); turns.append(answer); idxs.append(int(idx)); nts.append(int(n)); times.append(elapsed); messages.append({'role':'assistant','content':answer})
        if not failed:
            answers.append({'question_id':q['question_id'],'answer_id':str(q['question_id'])+'-'+str(time.time_ns()),'model_id':args.model_id,'choices':[{'index':0,'turns':turns,'idxs':idxs,'new_tokens':nts,'wall_time':times}],'tstamp':time.time()})
    answer,statfile,machine=_output_paths(args.answer_file); _atomic_write_jsonl(answer,answers); _atomic_write_jsonl(machine,rows)
    failure_path=str(Path(answer).with_name(Path(answer).stem+'_failures.jsonl')); _atomic_write_jsonl(failure_path,failures)
    _atomic_write_text(statfile, json.dumps({'experiment':args.experiment_name,'rows':len(rows),'failures':len(failures),'strategy':'no_update' if args.no_update else 'persistent_sliding_window_anchor_guard','learning_rate':args.adaptation_lr},indent=2)); print('Saved',answer,machine,failure_path)

if __name__=='__main__': main()
