import argparse
import contextlib
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_aug.augment_ent_link_thinking import build_prompt
from el.sft import (
    _apply_cot_candidate_constraint,
    _apply_eval_table_compression,
    _has_think_tags,
    _normalize_description_marker,
    _normalize_pred_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CoT checkpoint with optional DDP.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/khli/tableLlama/result_CoT/checkpoint-933",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.json",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/home/khli/tableLlama/result_CoT/checkpoint-933_mammotab_first10_predictions.jsonl",
    )
    parser.add_argument(
        "--eval_data_limit",
        type=int,
        default=10,
        help="How many samples to evaluate. <=0 means use full dataset.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--min_new_tokens", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing shard file(s) (<out_path stem>.rank<N><suffix>) if present: "
            "reads ALL existing shard files (regardless of which rank produced them) to find "
            "the globally completed idx, then re-balances only the still-missing samples across "
            "the CURRENT run's ranks with the same LPT split. This means the missing work is "
            "spread evenly over every GPU this time, instead of only landing back on whichever "
            "rank(s) happened to be slow last time. --data_path / --eval_data_limit should still "
            "match the run being resumed (they determine the sample pool), but --world_size no "
            "longer needs to match."
        ),
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=0,
        help=(
            "Optional hard cap on prompt tokens. 0 disables truncation and matches "
            "CoTValidationGenerativeEvalCallback (only tail-truncate when "
            "seq_len + max_new_tokens exceeds model max_position_embeddings)."
        ),
    )
    return parser.parse_args()


def setup_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            # 显式把设备绑定到进程组，避免 NCCL "device unknown" 警告以及收尾时挂死。
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device(f"cuda:{local_rank}"),
                timeout=timedelta(hours=6),
            )
        else:
            dist.init_process_group(backend="gloo", timeout=timedelta(hours=6))

    return is_distributed, world_size, rank, local_rank


def _align_eval_example(ex: dict) -> dict:
    """Match CoTValidationGenerativeEvalCallback preprocessing."""
    aligned = _apply_eval_table_compression(ex)
    for key in ("question", "output"):
        if key in aligned:
            aligned[key] = _normalize_description_marker(aligned[key])
    return aligned


def _tokenize_prompt(
    tokenizer,
    prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    max_position_embeddings: int,
    device,
):
    """Match CoTValidationGenerativeEvalCallback input handling."""
    if max_input_length > 0:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    seq_len = inputs["input_ids"].shape[1]
    max_pos = max_position_embeddings
    if max_input_length <= 0 and seq_len + max_new_tokens > max_pos:
        keep = max_pos - max_new_tokens
        inputs = {k: v[:, -keep:] for k, v in inputs.items()}
    return {k: v.to(device) for k, v in inputs.items()}


def main():
    args = parse_args()
    is_distributed, world_size, rank, local_rank = setup_dist()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_path = Path(args.data_path)
    out_path = Path(args.out_path)

    def log(msg: str):
        if rank == 0:
            print(msg)

    # ====== load base model name from adapter config ======
    adapter_cfg = json.loads((checkpoint_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_model_name = adapter_cfg["base_model_name_or_path"]
    log(f"Base model: {base_model_name}")
    log(f"Checkpoint : {checkpoint_dir}")
    log(f"WORLD_SIZE : {world_size}")

    # ====== tokenizer ======
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # ====== model ======
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if use_cuda else None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if use_cuda else "eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
        )

    # Ensure embedding size aligns with tokenizer used in LoRA training
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model.to(device)
    model.eval()
    log(f"Device(rank0): {device}")
    max_position_embeddings = getattr(model.config, "max_position_embeddings", 131072)

    # ====== load eval samples ======
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if args.eval_data_limit > 0:
        raw_samples = data[: args.eval_data_limit]
    else:
        raw_samples = data
    samples = [_align_eval_example(ex) for ex in raw_samples]
    log(f"Loaded {len(samples)} samples.")

    def _est_cost(ex):
        try:
            return len(json.dumps(ex, ensure_ascii=False))
        except Exception:
            return len(str(ex))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rank_out_path = out_path.parent / f"{out_path.stem}.rank{rank}{out_path.suffix}"

    # ---- resume: 汇总*所有*已存在的分片文件（不管是不是本 rank 之前写的），
    # 得到全局已完成的 idx 集合。这样才能正确处理"上次几块卡跑得快、几块跑得慢，
    # 中途分片被合并/删除"之后的续跑场景。
    done_idx: set = set()
    prev_count = 0
    if args.resume:
        existing_shard_paths = sorted(out_path.parent.glob(f"{out_path.stem}.rank*{out_path.suffix}"))
        for p in existing_shard_paths:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # 中断可能发生在写行/flush 中间，留下半行 JSON；丢弃即可，
                        # 对应样本会被当作未完成重新计算。
                        continue
                    idx = row.get("idx")
                    if idx is None or idx in done_idx:
                        continue
                    done_idx.add(idx)
                    if p == rank_out_path:
                        prev_count += 1
        log(
            f"resume: {len(done_idx)} previously completed rows found across "
            f"{len(existing_shard_paths)} existing shard file(s)"
        )

    # 按预估长度做贪心(LPT)均衡切分，避免像 index striding 那样某个 rank 早早跑完、
    # 却要在 barrier 上空等另一个 rank 数小时。只对*还没完成*的样本重新切分，这样
    # 续跑时缺口会被摊平到本次的所有 rank 上，而不是只落回上次跑得慢的那几个 rank。
    pending_indices = [i for i in range(len(samples)) if i not in done_idx]
    order = sorted(pending_indices, key=lambda i: _est_cost(samples[i]), reverse=True)
    rank_loads = [0] * world_size
    rank_assign = [[] for _ in range(world_size)]
    for i in order:
        r = min(range(world_size), key=lambda k: rank_loads[k])
        rank_assign[r].append(i)
        rank_loads[r] += _est_cost(samples[i])
    rank_items = [(idx, samples[idx]) for idx in sorted(rank_assign[rank])]
    if args.resume:
        log(
            f"resume: {len(pending_indices)} samples still pending, re-balanced across "
            f"{world_size} rank(s) (this rank gets {len(rank_items)})"
        )
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    resuming = args.resume and rank_out_path.exists()
    if resuming:
        print(
            f"[rank={rank}] resume: {prev_count} previously completed rows found in "
            f"{rank_out_path}; {len(rank_items)} samples assigned to this rank this run",
            flush=True,
        )

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    # 逐条算完立刻写盘并 flush/fsync，这样即使评估中途被打断（Ctrl+C、OOM、
    # 被 kill 等），已完成的样本也不会丢，下次 --resume 才能真正跳过它们。
    new_count = 0
    with rank_out_path.open("a" if resuming else "w", encoding="utf-8") as out_f:
        for global_idx, ex in rank_items:
            prompt = build_prompt(ex)
            inputs = _tokenize_prompt(
                tokenizer,
                prompt,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                max_position_embeddings=max_position_embeddings,
                device=device,
            )

            with torch.no_grad():
                with autocast_ctx:
                    # 与 CoTValidationGenerativeEvalCallback 一致：纯 greedy，不加 repetition_penalty /
                    # no_repeat_ngram_size，避免抑制照抄候选导致格式崩坏。
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=args.min_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=int(eos_token_id) if eos_token_id is not None else None,
                        pad_token_id=pad_token_id,
                    )

            new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            raw_pred = tokenizer.decode(new_ids, skip_special_tokens=True)
            # 优先提取 </think> 之后的实体；无闭合 tag 时回退到直接解析整段原始输出。
            # 是否有 <think> 只作为统计量（has_think），不影响 correct 的判定。
            prediction = _apply_cot_candidate_constraint(ex.get("question", ""), raw_pred)
            gold = _normalize_pred_text(ex.get("output") or "")
            has_think = int(_has_think_tags(raw_pred))

            row = {
                "idx": global_idx,
                "prompt": prompt,
                "raw_prediction": raw_pred,
                "prediction": prediction,
                "gold": gold,
                "has_think": has_think,
                "correct": int(prediction == gold),
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            new_count += 1
            print(f"[rank={rank}] sample_idx={global_idx} done")

    print(
        f"[rank={rank}] wrote {new_count} new rows (total {prev_count + new_count}) to {rank_out_path}",
        flush=True,
    )

    if is_distributed:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    if rank == 0:
        # glob 而不是固定 range(world_size)：resume 时分片文件数量可能来自不同的
        # world_size 历史，用 glob 收集当前实际存在的所有分片，避免因为本次
        # world_size 和某次历史 run 不一致而漏掉数据或误报缺文件。
        merge_paths = sorted(out_path.parent.glob(f"{out_path.stem}.rank*{out_path.suffix}"))
        if not merge_paths:
            raise FileNotFoundError(
                f"No rank shard files found matching {out_path.stem}.rank*{out_path.suffix}"
            )
        results_by_idx: dict = {}
        for p in merge_paths:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    results_by_idx[row["idx"]] = row
        results = [results_by_idx[k] for k in sorted(results_by_idx)]

        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        matched = sum(int(row.get("correct", 0)) for row in results)
        total = len(results)
        n_has_think = sum(int(row.get("has_think", 0)) for row in results)
        accuracy = (matched / total) if total else 0.0

        print("=" * 80, flush=True)
        print(f"Saved {len(results)} results to: {out_path}", flush=True)
        print(
            f"Accuracy(prediction == gold, train-val aligned parsing): {accuracy:.4f} "
            f"({matched}/{total}); has_think={n_has_think}/{total}",
            flush=True,
        )

    sys.stdout.flush()
    sys.stderr.flush()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()