#!/usr/bin/env python3
"""
从已经合并好的主 jsonl，反推出 eval_CoT_checkpoint.py 需要的每个 rank 的分片文件
(<out_path stem>.rank<r><suffix>)，这样重新跑 --resume 时才能正确识别出"这个 rank
已经算过哪些 idx"，只补算缺失的样本，而不是把所有样本重新算一遍。

背景：eval_CoT_checkpoint.py 的 --resume 逻辑只看 <out_path>.rank{rank}{suffix}
这个分片文件里已有的 idx，不看合并后的主文件。分片文件被删掉之后，直接加
--resume 重新跑，每个 rank 都会判定为"没有可 resume 的文件"，从而把分配给它的
全部样本重新计算一遍(虽然最终结果仍然完整，但浪费已经算过的算力/时间)。

用法示例：
  python el/rebuild_rank_shards.py \\
      --data_path "/home/khli/DATA/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.json" \\
      --eval_data_limit -1 \\
      --world_size 4 \\
      --merged_path /home/khli/tableLlama/result_dpo_v2_norpo/checkpoint-150_mam_val_predictions.jsonl \\
      --out_path /home/khli/tableLlama/result_dpo_v2_norpo/checkpoint-150_mam_val_predictions.jsonl

注意：--data_path / --eval_data_limit / --world_size 必须和当年那次生成用的完全
一致，因为样本到 rank 的分配(LPT 贪心均衡切分)是由这三者决定的，和
eval_CoT_checkpoint.py 里的算法逐字保持一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from el.sft import _apply_eval_table_compression, _normalize_description_marker


def _align_eval_example(ex: dict) -> dict:
    """Must stay identical to eval_CoT_checkpoint.py's _align_eval_example."""
    aligned = _apply_eval_table_compression(ex)
    for key in ("question", "output"):
        if key in aligned:
            aligned[key] = _normalize_description_marker(aligned[key])
    return aligned


def _est_cost(ex: dict) -> int:
    try:
        return len(json.dumps(ex, ensure_ascii=False))
    except Exception:
        return len(str(ex))


def compute_rank_assign(samples: List[dict], world_size: int) -> List[List[int]]:
    """Reproduce eval_CoT_checkpoint.py's LPT greedy split exactly."""
    order = sorted(range(len(samples)), key=lambda i: _est_cost(samples[i]), reverse=True)
    rank_loads = [0] * world_size
    rank_assign: List[List[int]] = [[] for _ in range(world_size)]
    for i in order:
        r = min(range(world_size), key=lambda k: rank_loads[k])
        rank_assign[r].append(i)
        rank_loads[r] += _est_cost(samples[i])
    return rank_assign


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument(
        "--eval_data_limit",
        type=int,
        default=-1,
        help="必须和原来那次生成用的 --eval_data_limit 一致",
    )
    p.add_argument(
        "--world_size",
        type=int,
        required=True,
        help="必须和原来那次 torchrun --nproc_per_node 一致",
    )
    p.add_argument("--merged_path", type=str, required=True, help="已经合并好的主 jsonl")
    p.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="等号 eval_CoT_checkpoint.py 的 --out_path，用来决定分片文件名",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    merged_path = Path(args.merged_path)
    out_path = Path(args.out_path)

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_samples = data[: args.eval_data_limit] if args.eval_data_limit > 0 else data
    samples = [_align_eval_example(ex) for ex in raw_samples]
    print(f"[info] loaded {len(samples)} samples from {data_path}")

    rank_assign = compute_rank_assign(samples, args.world_size)
    for r, idxs in enumerate(rank_assign):
        print(f"[info] rank{r}: assigned {len(idxs)} samples")

    merged_by_idx: Dict[int, Any] = {}
    with merged_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] skip bad json in {merged_path} line {line_no}: {e}")
                continue
            merged_by_idx[row["idx"]] = row
    print(f"[info] loaded {len(merged_by_idx)} completed rows from {merged_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_assigned = 0
    total_done = 0
    total_missing = 0
    for r, idxs in enumerate(rank_assign):
        rank_out_path = out_path.parent / f"{out_path.stem}.rank{r}{out_path.suffix}"
        rows = [merged_by_idx[i] for i in idxs if i in merged_by_idx]
        missing = [i for i in idxs if i not in merged_by_idx]
        total_assigned += len(idxs)
        total_done += len(rows)
        total_missing += len(missing)
        with rank_out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[done] rank{r}: wrote {len(rows)} completed rows -> {rank_out_path}; "
            f"{len(missing)} still missing (will be (re)computed on --resume)"
        )

    print("=" * 60)
    print(f"total assigned={total_assigned}, already done={total_done}, missing={total_missing}")
    print(
        "Now rerun the original torchrun command with --resume unchanged; "
        "each rank will only recompute its missing idx."
    )


if __name__ == "__main__":
    main()
