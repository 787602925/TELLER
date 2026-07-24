#!/usr/bin/env python3
"""
对 rl/gen_dpo_data.py 产出的 DPO 数据里，reject 结构不完整（<think>...</think>\\n<entity>
解析不出来，几乎全部因为当时本地生成 --max_new_tokens 只给了 256、被硬截断）的样本，
用更大的 max_new_tokens 重新生成一次 reject：

  - 用同一个 checkpoint、同一条 prompt，重新 greedy 生成（默认 --max_new_tokens 512）
  - 解析新生成结果的最终实体：
      * 若与 gold 匹配 -> 说明这条“错误样本”其实是被截断误判的，整条 pair 直接丢弃
      * 若仍不匹配   -> 用新的完整生成替换原 reject（并同步更新 model_prediction）
  - 其余（reject 本来就结构完整的）样本原样保留，不做改动

原始输入文件 --in_path 只读，不会被修改；处理结果（改动 + 未改动的样本）写到新的
--out_path 文件里。

支持多卡：每个 round（= --gen_batch_size 条待重跑样本）轮转分配给各 rank 并行处理，
各 rank 把自己处理到的“增量”（更新后的记录，或 {"idx":.., "__drop__": true} 丢弃标记）
写到各自的 rank shard 文件，最后由 rank0 汇总 baseline（全量原始记录）+ 各 rank 的增量
（按 idx 覆盖、丢弃标记会被过滤掉）合并写出最终 --out_path，与 rl/gen_dpo_data.py 的
分片/合并逻辑一致，也支持 --resume 断点续跑。

用法示例：
  conda activate tablellama-fa

  # 先 dry-run 看一下要重新生成多少条（不加载模型，几秒钟出结果）
  python rl/regen_incomplete_reject.py \\
    --in_path "/DATA1/khli/t&m/rl_checkpoint-350_10000.jsonl" \\
    --dry_run

  # 单卡跑
  python rl/regen_incomplete_reject.py \\
    --in_path "/DATA1/khli/t&m/rl_checkpoint-350_10000.jsonl" \\
    --out_path "/DATA1/khli/t&m/rl_checkpoint-350_10000_reject_regen.jsonl" \\
    --max_new_tokens 512

  # 多卡（每卡 1 进程，与 gen_dpo_data.py 一致）
  torchrun --nproc_per_node=2 rl/regen_incomplete_reject.py \\
    --in_path "/DATA1/khli/t&m/rl_checkpoint-350_10000.jsonl" \\
    --out_path "/DATA1/khli/t&m/rl_checkpoint-350_10000_reject_regen.jsonl" \\
    --max_new_tokens 512
  # 断点续跑（读各 rank 分片跳过已处理的 idx）：
  torchrun --nproc_per_node=2 rl/regen_incomplete_reject.py \\
    --in_path "/DATA1/khli/t&m/rl_checkpoint-350_10000.jsonl" \\
    --out_path "/DATA1/khli/t&m/rl_checkpoint-350_10000_reject_regen.jsonl" \\
    --max_new_tokens 512 --resume
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_DATA_CLEAN_DIR = _PROJECT_ROOT / "data_clean"
if str(_DATA_CLEAN_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_CLEAN_DIR))

from compress_long_cot import parse_output  # noqa: E402  (复用同款 <think>/entity 结构校验)
from el.sft import _apply_cot_candidate_constraint, _normalize_pred_text  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

DEFAULT_IN_PATH = "/DATA1/khli/t&m/rl_checkpoint-350_10000.jsonl"
# 与生成这批数据时用的本地 checkpoint 保持一致（见 rl/data_generate.log）。
DEFAULT_MODEL = "result_CoT_filtered_compressed_data_lora/checkpoint-350"

# 与 el/sft.build_prompt 拼接格式一致："### Question:\n{question}\n\n### Response:"
_QUESTION_RE = re.compile(r"### Question:\n(.*?)\n\n### Response:", flags=re.S)


def extract_question(prompt: str) -> str:
    m = _QUESTION_RE.search(prompt or "")
    return m.group(1).strip() if m else ""


def is_reject_incomplete(reject: str) -> bool:
    """reject 是否解析不出合法的 <think>...</think>\\n<entity> 结构。"""
    return parse_output(reject) is None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate structurally-incomplete reject with a larger max_new_tokens budget."
    )
    p.add_argument("--in_path", type=str, default=DEFAULT_IN_PATH,
                   help="只读，不会被修改")
    p.add_argument("--out_path", type=str, default="", help="缺省: <in_path stem>_reject_regen.jsonl")
    p.add_argument("--report_path", type=str, default="", help="缺省: <out_path>.report.json")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help="本地 LoRA checkpoint 目录（相对路径相对项目根）")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--min_new_tokens", type=int, default=2)
    p.add_argument("--max_input_length", type=int, default=2560)
    p.add_argument("--gen_batch_size", type=int, default=8,
                   help="每个 round 的样本数，多卡时按 round 轮转分配给各 rank")
    p.add_argument("--resume", action="store_true",
                   help="断点续跑：读取各 rank 分片，跳过已处理过的 idx")
    p.add_argument("--dry_run", action="store_true",
                   help="只扫描统计需要重新生成的条数，不加载模型、不生成、不涉及分布式")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path).expanduser().resolve()
    out_path = (
        Path(args.out_path).expanduser().resolve()
        if args.out_path.strip()
        else in_path.with_name(in_path.stem + "_reject_regen" + in_path.suffix)
    )
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path.strip()
        else Path(str(out_path) + ".report.json")
    )

    records = load_jsonl(in_path)
    todo_idx_values = [r["idx"] for r in records if is_reject_incomplete(r.get("reject", ""))]

    if args.dry_run:
        print(f"[load] {len(records)} records from {in_path}")
        print(f"[scan] structurally-incomplete reject: {len(todo_idx_values)}/{len(records)}")
        print("[dry-run] exiting without loading model / distributed init.")
        return

    # 延迟到这里才 import torch / 初始化分布式 / 加载模型，让 --dry_run 秒出结果。
    import gen_dpo_data as gdd  # 同目录模块，复用 dist / 模型加载 / 批量生成 / 分片合并逻辑

    is_distributed, world_size, rank, local_rank = gdd.setup_dist()

    def log(msg: str) -> None:
        if rank == 0:
            print(msg)

    log(f"[load] {len(records)} records from {in_path}")
    log(f"[scan] structurally-incomplete reject: {len(todo_idx_values)}/{len(records)}")
    log(f"[dist] WORLD_SIZE={world_size}, rank={rank}, local_rank={local_rank}")

    by_idx: Dict[int, Dict[str, Any]] = {r["idx"]: r for r in records}

    checkpoint_dir = Path(args.model)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (_PROJECT_ROOT / checkpoint_dir).resolve()
    if not (checkpoint_dir / "adapter_config.json").is_file():
        raise SystemExit(f"adapter_config.json not found under: {checkpoint_dir}")

    tokenizer, model, device = gdd.load_local_model(
        checkpoint_dir, local_rank=local_rank, is_distributed=is_distributed, log=log
    )

    shard_path = gdd.rank_shard_path(out_path, rank)
    already_done: Set[int] = set()
    if args.resume and shard_path.exists():
        already_done = {
            rec["idx"] for rec in gdd.load_jsonl_records(shard_path) if isinstance(rec.get("idx"), int)
        }
        out_mode = "a"
        log(f"[resume] rank{rank} shard already has {len(already_done)} processed idx")
    else:
        if shard_path.exists():
            print(f"Warning: {shard_path} exists and will be overwritten (no --resume).", file=sys.stderr)
        out_mode = "w"

    batch_size = max(1, args.gen_batch_size)
    all_batches = [
        todo_idx_values[i:i + batch_size] for i in range(0, len(todo_idx_values), batch_size)
    ]
    # 按 round 轮转分配给各 rank（与 gen_dpo_data.py 一致），round 间各 rank 完全独立并行。
    my_batches = [
        [idx for idx in batch if idx not in already_done]
        for b_idx, batch in enumerate(all_batches)
        if b_idx % world_size == rank
    ]
    my_batches = [b for b in my_batches if b]
    my_total = sum(len(b) for b in my_batches)
    log(f"[assign] total round={len(all_batches)}, world_size={world_size} "
        f"-> this run will process {my_total} idx across ranks (rank{rank} handles its own share)")

    dropped_now_correct = 0
    updated_still_wrong = 0

    iterator = my_batches
    if tqdm is not None and rank == 0:
        iterator = tqdm(my_batches, desc="regen-reject", unit="batch")

    with shard_path.open(out_mode, encoding="utf-8") as out_f:
        for batch_idx_values in iterator:
            prompts = [by_idx[idx]["prompt"] for idx in batch_idx_values]
            raw_preds = gdd.local_generate_batch(tokenizer, model, device, prompts, args)
            for idx, raw_pred in zip(batch_idx_values, raw_preds):
                rec = by_idx[idx]
                question = extract_question(rec.get("prompt", ""))
                gold_norm = _normalize_pred_text((rec.get("gold") or "").strip())
                new_reject = gdd._strip_trailing_eos(raw_pred)
                pred_norm = _apply_cot_candidate_constraint(question, new_reject)

                if gold_norm and pred_norm == gold_norm:
                    delta: Dict[str, Any] = {"idx": idx, "__drop__": True}
                    dropped_now_correct += 1
                else:
                    delta = dict(rec)
                    delta["reject_before_regen"] = rec.get("reject", "")
                    delta["reject"] = new_reject
                    delta["model_prediction"] = pred_norm
                    delta["reject_regenerated_max_new_tokens"] = args.max_new_tokens
                    updated_still_wrong += 1

                out_f.write(json.dumps(delta, ensure_ascii=False) + "\n")
                out_f.flush()

    if is_distributed:
        gdd.dist_barrier(is_distributed=is_distributed, local_rank=local_rank)

    total_dropped = gdd.all_reduce_sum(dropped_now_correct, device, is_distributed)
    total_updated = gdd.all_reduce_sum(updated_still_wrong, device, is_distributed)

    if rank == 0:
        merged = gdd.merge_rank_outputs(out_path, world_size, existing_records=records)
        final_records = [r for r in merged if not r.get("__drop__")]

        with out_path.open("w", encoding="utf-8") as f:
            for r in final_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        report = {
            "in_path": str(in_path),
            "out_path": str(out_path),
            "model": str(checkpoint_dir),
            "max_new_tokens": args.max_new_tokens,
            "world_size": world_size,
            "total_in": len(records),
            "total_out": len(final_records),
            "incomplete_reject_found": len(todo_idx_values),
            "dropped_now_correct": total_dropped,
            "updated_still_wrong": total_updated,
            "unchanged_already_complete": len(records) - len(todo_idx_values),
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("=" * 80)
        for k, v in report.items():
            print(f"{k}: {v}")
        print(f"saved  : {out_path}")
        print(f"report : {report_path}")
        if world_size > 1:
            print(f"shards : {out_path.stem}.rank*{out_path.suffix} (kept for --resume, safe to delete)")

    if is_distributed:
        gdd.dist.destroy_process_group()


if __name__ == "__main__":
    main()
