#!/usr/bin/env python3
"""
Filter & process RL (DPO) 数据，把 rl/gen_dpo_data.py（或 rl/regen_incomplete_reject.py）
产出的 accept/reject pair 数据清洗成可以直接用于 RL 训练的版本。全程只用 CPU（结构解析用
正则、质量打分用规则、长度压缩只需要本地 tokenizer 文件），不需要 GPU / 加载模型。

三步流水线（复用现成脚本里的核心函数，不重复实现）：

  Step 1 - 结构过滤（复用 data_clean/compress_long_cot.py 的 parse_output）：
    * accept 必须能 parse 出 <think>...</think>\\n<entity>，parse 不出来整条丢
      （对应"accept 完全没有推理"的问题，约 16.2%）。
    * 顺带做 gold 一致性 sanity check：parse 出来的最终实体 normalize 后必须等于 gold，
      不等就丢（约 16 条异常）。
    * reject 不做结构完整性要求（截断/重复循环本身是有信息量的负样本，见此前讨论），
      只清理"实体后又冒出字面 </s> 再续写一段文字"这种格式崩坏（约 10 条）。

  Step 2 - 质量过滤（直接复用 data_clean/filter_low_info_cot.py 的打分函数 _analyze_one，
    等价于对 accept 跑一次 `filter_low_info_cot.py --drop-level all`）：
    去掉 rambling / answer_leakage / copies_candidate_list / no_justification 等
    低信息量或残留泄露痕迹的 accept。

  Step 3 - 长度压缩（复用 data_clean/compress_long_cot.py 的 compress_one）：
    对通过 1/2 步的 accept，若 token 数超过 --compress-threshold-tokens，抽取式压缩到
    --target-output-tokens 以下；压缩失败（unparsable / entity 改变 / 仍超长 / 重复率高）
    的样本整条丢弃。

输入文件只读，不会被修改；结果（含未改动的样本）写到新的 --out_path。

用法示例：
  conda activate tablellama-fa   # 或任何装了 transformers 的环境，不需要 GPU

  # 先 dry-run 看每一步的漏斗统计，不写最终文件（几十秒，只需加载一次 tokenizer）
  python rl/filter_process_rl_data.py \\
    --in_path "/DATA1/khli/t&m/rl_checkpoint-350_10000.jsonl" \\
    --dry_run

  # 正式跑，写出最终过滤后的数据集
  python rl/filter_process_rl_data.py \\
    --in_path "/DATA1/khli/t&m/rl_checkpoint-350_10000.jsonl" \\
    --out_path "/DATA1/khli/t&m/rl_checkpoint-350_10000_filtered.jsonl"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_DATA_CLEAN_DIR = _PROJECT_ROOT / "data_clean"
if str(_DATA_CLEAN_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_CLEAN_DIR))

from compress_long_cot import TokenCounter, compress_one, normalize_entity, parse_output  # noqa: E402
from filter_low_info_cot import _analyze_one  # noqa: E402

DEFAULT_IN_PATH = "/DATA1/khli/t&m/rl_checkpoint-350_10000.jsonl"
DEFAULT_OUT_PATH = "/DATA1/khli/t&m/rl_checkpoint-350_10000_filtered.jsonl"
# 只用来测 token 数，跟训练/生成这批数据用的是同一个 tokenizer，不需要 GPU。
DEFAULT_TOKENIZER = "result_CoT_filtered_compressed_data_lora/checkpoint-350"

_FAKE_EOS_RE = re.compile(r"</s>", flags=re.I)


def has_fake_eos_continuation(reject: str) -> bool:
    """检测 reject 里"实体后又冒出字面 </s> 再续写一段文字"这种格式崩坏。

    跟"截断/重复循环没写完"不是一类问题：截断类本身是模型真实的失败模式（啰嗦/
    不收敛），保留下来对 RL 是有信息量的负样本；这里只清理生成异常导致的乱码续写。
    """
    if not isinstance(reject, str):
        return False
    matches = list(_FAKE_EOS_RE.finditer(reject))
    if not matches:
        return False
    trailing = reject[matches[-1].end():]
    return bool(trailing.strip())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter & process RL DPO accept/reject data.")
    p.add_argument("--in_path", type=str, default=DEFAULT_IN_PATH, help="只读，不会被修改")
    p.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH)
    p.add_argument("--report_path", type=str, default="", help="缺省: <out_path>.report.json")

    # Step 2 (filter_low_info_cot.py 同款默认值)
    p.add_argument("--think_min_chars", type=int, default=80)
    p.add_argument("--drop_score_threshold", type=int, default=3)

    # Step 3 (compress_long_cot.py 同款默认值，按你 SFT 768 上限留 margin)
    p.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    p.add_argument("--compress_threshold_tokens", type=int, default=700)
    p.add_argument("--target_output_tokens", type=int, default=650)
    p.add_argument("--min_think_sentences", type=int, default=1)
    p.add_argument("--repeat_ngram", type=int, default=8)
    p.add_argument("--max_repeat_ratio", type=float, default=0.3)

    p.add_argument("--dry_run", action="store_true",
                   help="只打印每一步的漏斗统计和 reason 分布，不写最终数据集")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path.strip()
        else Path(str(out_path) + ".report.json")
    )

    tokenizer_path = args.tokenizer
    tok_path_obj = Path(tokenizer_path)
    if not tok_path_obj.is_absolute():
        tok_path_obj = (_PROJECT_ROOT / tokenizer_path).resolve()
    tokenizer_path = str(tok_path_obj)

    records = load_jsonl(in_path)
    total = len(records)
    print(f"[load] {total} records from {in_path}")

    drop_reason_counter: Counter[str] = Counter()
    step_counts = {
        "step1_dropped": 0,
        "step2_dropped": 0,
        "step3_compressed": 0,
        "step3_dropped": 0,
        "kept": 0,
    }

    # ---- Step 1: 结构过滤（accept parse + gold sanity check + reject 格式崩坏清理） ----
    step1_survivors: List[Dict[str, Any]] = []
    for rec in records:
        accept = rec.get("accept", "") or ""
        reject = rec.get("reject", "") or ""
        gold = (rec.get("gold") or "").strip()

        parsed = parse_output(accept)
        if parsed is None:
            drop_reason_counter["step1_accept_unparsable"] += 1
            step_counts["step1_dropped"] += 1
            continue
        if gold and normalize_entity(parsed.final_entity) != normalize_entity(gold):
            drop_reason_counter["step1_gold_mismatch"] += 1
            step_counts["step1_dropped"] += 1
            continue
        if has_fake_eos_continuation(reject):
            drop_reason_counter["step1_reject_format_collapse"] += 1
            step_counts["step1_dropped"] += 1
            continue

        step1_survivors.append(rec)

    print(f"[step1] structural filter: kept {len(step1_survivors)}/{total}, "
          f"dropped {step_counts['step1_dropped']}")

    # ---- Step 2: 质量过滤（复用 filter_low_info_cot.py 的打分逻辑，等价于跑
    #      filter_low_info_cot.py --drop-level all，只是直接对 accept 内联调用） ----
    step2_survivors: List[Dict[str, Any]] = []
    for rec in step1_survivors:
        accept = rec.get("accept", "") or ""
        decision = _analyze_one(
            {"output": accept},
            think_min_chars=args.think_min_chars,
            drop_score_threshold=args.drop_score_threshold,
        )
        if decision.tag != "keep":
            drop_reason_counter[f"step2_{decision.tag}"] += 1
            for reason in decision.reasons:
                drop_reason_counter[f"step2_reason:{reason}"] += 1
            step_counts["step2_dropped"] += 1
            continue
        step2_survivors.append(rec)

    print(f"[step2] low-info quality filter: kept {len(step2_survivors)}/{len(step1_survivors)}, "
          f"dropped {step_counts['step2_dropped']}")

    # ---- Step 3: 长度压缩 ----
    counter = TokenCounter(tokenizer_path)
    final_records: List[Dict[str, Any]] = []
    compressed_before_sum = 0
    compressed_after_sum = 0
    for rec in step2_survivors:
        accept = rec.get("accept", "") or ""
        out_tokens = counter.count(accept)
        if out_tokens <= args.compress_threshold_tokens:
            final_records.append(rec)
            step_counts["kept"] += 1
            continue

        result = compress_one(
            accept,
            counter,
            target_tokens=args.target_output_tokens,
            min_sentences=args.min_think_sentences,
            drop_final_answer=True,
            repeat_ngram=args.repeat_ngram,
            max_repeat_ratio=args.max_repeat_ratio,
        )
        if not result.ok:
            drop_reason_counter[f"step3_compress_failed:{result.reason}"] += 1
            step_counts["step3_dropped"] += 1
            continue

        new_rec = dict(rec)
        new_rec["accept"] = result.output
        new_rec["accept_compressed_from_tokens"] = result.before_tokens
        new_rec["accept_compressed_to_tokens"] = result.after_tokens
        final_records.append(new_rec)
        step_counts["step3_compressed"] += 1
        step_counts["kept"] += 1
        compressed_before_sum += result.before_tokens
        compressed_after_sum += result.after_tokens

    print(f"[step3] length compression: kept {step_counts['kept']}/{len(step2_survivors)}, "
          f"compressed {step_counts['step3_compressed']}, "
          f"dropped(compress-failed) {step_counts['step3_dropped']}")

    report = {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "tokenizer": tokenizer_path,
        "total_in": total,
        "total_out": len(final_records),
        "kept_ratio": round(len(final_records) / total, 4) if total else 0.0,
        "step_counts": step_counts,
        "drop_reasons": dict(drop_reason_counter.most_common()),
        "compress_avg_tokens_before": round(compressed_before_sum / step_counts["step3_compressed"], 1)
        if step_counts["step3_compressed"] else 0.0,
        "compress_avg_tokens_after": round(compressed_after_sum / step_counts["step3_compressed"], 1)
        if step_counts["step3_compressed"] else 0.0,
        "thresholds": {
            "think_min_chars": args.think_min_chars,
            "drop_score_threshold": args.drop_score_threshold,
            "compress_threshold_tokens": args.compress_threshold_tokens,
            "target_output_tokens": args.target_output_tokens,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"total_in={total}, total_out={len(final_records)} "
          f"({len(final_records) / total:.2%} kept)")
    print(f"step_counts: {step_counts}")
    print(f"top drop reasons: {drop_reason_counter.most_common(10)}")
    print(f"report: {report_path}")

    if args.dry_run:
        print("[dry-run] not writing filtered dataset.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"filtered_data: {out_path}")


if __name__ == "__main__":
    main()
