#!/usr/bin/env python3
"""
把多卡 DPO 生成产生的 rank 分片合并进主 jsonl。

默认行为（与 gen_dpo_data.py 的 resume 语义一致，但主文件优先）：
  - 先读主文件里已有记录（不会丢、也不会被分片覆盖）
  - 再读各 .rank*.jsonl，只把主文件里还没有的 idx 追加进去
  - 最终按 idx 排序写回主文件

用法示例：
  python rl/merge_rank_shards.py
  python rl/merge_rank_shards.py \\
      --main /DATA1/khli/t\\&m/rl_checkpoint-350_9000.jsonl \\
      --shards /DATA1/khli/t\\&m/rl_checkpoint-350_9000.rank0.jsonl \\
              /DATA1/khli/t\\&m/rl_checkpoint-350_9000.rank1.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_MAIN = "/home/khli/tableLlama/result_dpo_v2_norpo/checkpoint-150_mam_val_predictions.jsonl"
DEFAULT_SHARDS = [
    "/home/khli/tableLlama/result_dpo_v2_norpo/checkpoint-150_mam_val_predictions.rank0.jsonl",
    "/home/khli/tableLlama/result_dpo_v2_norpo/checkpoint-150_mam_val_predictions.rank1.jsonl",
    "/home/khli/tableLlama/result_dpo_v2_norpo/checkpoint-150_mam_val_predictions.rank2.jsonl",
    "/home/khli/tableLlama/result_dpo_v2_norpo/checkpoint-150_mam_val_predictions.rank3.jsonl"
]


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad += 1
                print(f"[warn] skip bad json in {path} line {line_no}: {e}")
    if bad:
        print(f"[warn] {path}: skipped {bad} malformed line(s)")
    return records


def merge_keep_main(
    main_records: List[Dict[str, Any]],
    shard_records_list: List[Tuple[Path, List[Dict[str, Any]]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    主文件优先：同 idx 时保留主文件原记录，分片里的同 idx 丢弃不覆盖。
    返回 (按 idx 排序后的合并列表, 统计信息)。
    """
    by_idx: Dict[int, Dict[str, Any]] = {}
    main_kept = 0
    main_no_idx = 0
    for rec in main_records:
        idx = rec.get("idx")
        if isinstance(idx, int):
            by_idx[idx] = rec
            main_kept += 1
        else:
            main_no_idx += 1

    added = 0
    skipped_dup = 0
    shard_no_idx = 0
    for _shard_path, shard_records in shard_records_list:
        for rec in shard_records:
            idx = rec.get("idx")
            if not isinstance(idx, int):
                shard_no_idx += 1
                continue
            if idx in by_idx:
                skipped_dup += 1
                continue
            by_idx[idx] = rec
            added += 1

    merged = [by_idx[k] for k in sorted(by_idx)]
    stats = {
        "main_kept": main_kept,
        "main_no_idx": main_no_idx,
        "added_from_shards": added,
        "skipped_dup_idx": skipped_dup,
        "shard_no_idx": shard_no_idx,
        "merged_total": len(merged),
    }
    return merged, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge rank*.jsonl shards into main DPO jsonl without overwriting existing main records."
    )
    p.add_argument(
        "--main",
        type=str,
        default=DEFAULT_MAIN,
        help=f"主 jsonl 路径（已有数据保留不被覆盖，默认 {DEFAULT_MAIN}）",
    )
    p.add_argument(
        "--shards",
        type=str,
        nargs="+",
        default=DEFAULT_SHARDS,
        help="要合并进来的 rank 分片路径（可多个）",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印合并统计，不写回主文件",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    main_path = Path(args.main).expanduser().resolve()
    shard_paths = [Path(s).expanduser().resolve() for s in args.shards]

    if not main_path.exists():
        print(f"[info] main file not found, will create: {main_path}")
        main_records: List[Dict[str, Any]] = []
    else:
        main_records = load_jsonl_records(main_path)
        print(f"[main] {main_path}: {len(main_records)} records")

    shard_records_list: List[Tuple[Path, List[Dict[str, Any]]]] = []
    for sp in shard_paths:
        if not sp.exists():
            raise SystemExit(f"shard not found: {sp}")
        recs = load_jsonl_records(sp)
        print(f"[shard] {sp}: {len(recs)} records")
        shard_records_list.append((sp, recs))

    merged, stats = merge_keep_main(main_records, shard_records_list)
    print("=" * 60)
    print(f"main kept (by idx)     : {stats['main_kept']}")
    print(f"added from shards      : {stats['added_from_shards']}")
    print(f"skipped dup idx        : {stats['skipped_dup_idx']} (kept main version)")
    if stats["main_no_idx"] or stats["shard_no_idx"]:
        print(f"records without idx    : main={stats['main_no_idx']}, shards={stats['shard_no_idx']}")
    print(f"merged total           : {stats['merged_total']}")

    if args.dry_run:
        print("[dry_run] not writing")
        return

    main_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = main_path.with_suffix(main_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(main_path)
    print(f"[done] wrote {stats['merged_total']} records -> {main_path}")


if __name__ == "__main__":
    main()
