"""
合并两段 CoT-SFT 训练数据：
1. 从 mammotab stage4(带 think) 文件取前 m 条
2. 从 ent_link single_table(带 think) 文件取前 n 条
3. 合并后写入 /DATA1/khli/t&m/merged_CoTSFT_{m//1000}k&{n//1000}k.json

默认输入：
  mammotab: /DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified_with_think_correct_pro_30000.jsonl
  ent_link: /DATA1/khli/tablellama/ent_link_train_single_table_with_think_correct_gpt-5.2.jsonl

用法：
  python3 -m data_clean.data_merge4CoTSFT --m 5000 --n 3000
  python3 data_clean/data_merge4CoTSFT.py --m 20000 --n 10000
  python3 data_clean/data_merge4CoTSFT.py  # 不指定 m/n 时，两个文件都提取全部可用样本
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_clean.single_data_train import _iter_json_items  # noqa: E402

DEFAULT_MAMMOTAB = (
    "/DATA1/khli/mammotab/modified_mammotab/"
    "stage4_single_shards_merge_simplified_with_think_correct_pro_30000.jsonl"
)
DEFAULT_ENT_LINK = "/DATA1/khli/tablellama/ent_link_train_single_table_with_think_correct_gpt-5.2.jsonl"
DEFAULT_OUT_DIR = "/DATA1/khli/t&m"


def _iter_jsonl_items(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if not isinstance(item, dict):
                raise ValueError(f"jsonl line {line_no} is not an object")
            yield item


def _iter_mixed_items(path: str) -> Iterator[Dict[str, Any]]:
    if Path(path).suffix.lower() == ".jsonl":
        yield from _iter_jsonl_items(path)
        return
    yield from _iter_json_items(path)


_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.S | re.I)


def _extract_output_text(item: Dict[str, Any]) -> str:
    # CoT-SFT 样本通常将目标输出放在 output 字段，兼容少量变体命名。
    for key in ("output", "response", "target", "answer", "completion"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _has_extractable_think(item: Dict[str, Any]) -> bool:
    text = _extract_output_text(item)
    if not text:
        return False
    m = _THINK_RE.search(text)
    if not m:
        return False
    return bool((m.group(1) or "").strip())


def _take_with_think_filter(
    items: Iterable[Dict[str, Any]], n: int | None
) -> tuple[list[Dict[str, Any]], int]:
    if n is not None and n <= 0:
        return [], 0
    kept: list[Dict[str, Any]] = []
    filtered = 0
    for item in items:
        if _has_extractable_think(item):
            kept.append(item)
            if n is not None and len(kept) >= n:
                break
        else:
            filtered += 1
    if n is not None and len(kept) < n:
        raise ValueError(
            f"cannot collect enough samples with extractable <think>: need {n}, got {len(kept)}"
        )
    return kept, filtered


def _count_tag(value: int | None) -> str:
    if value is None:
        return "all"
    return f"{value // 1000}k"


def _merged_output_name(m: int | None, n: int | None) -> str:
    return f"merged_CoTSFT_{_count_tag(m)}&{_count_tag(n)}.json"


def _merge_items_with_stats(
    mammotab_path: str, m: int | None, ent_link_path: str, n: int | None
) -> tuple[list[Dict[str, Any]], int, int, int, int]:
    mammotab_items, mammotab_filtered = _take_with_think_filter(_iter_mixed_items(mammotab_path), m)
    ent_link_items, ent_link_filtered = _take_with_think_filter(_iter_mixed_items(ent_link_path), n)
    return (
        mammotab_items + ent_link_items,
        len(mammotab_items),
        len(ent_link_items),
        mammotab_filtered,
        ent_link_filtered,
    )


def _write_json(items: Iterable[Dict[str, Any]], out_path: str) -> int:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_file.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for item in items:
            if not first:
                f.write(",\n")
            else:
                first = False
            f.write(json.dumps(item, ensure_ascii=False))
            total += 1
        f.write("\n]\n")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge mammotab and ent_link CoT-SFT data for training.")
    parser.add_argument("--m", type=int, help="从 mammotab 文件提取的前 m 条；不指定则提取全部可用样本")
    parser.add_argument("--n", type=int, help="从 ent_link 文件提取的前 n 条；不指定则提取全部可用样本")
    parser.add_argument("--mammotab", default=DEFAULT_MAMMOTAB, help="mammotab 输入路径（json/jsonl）")
    parser.add_argument("--ent-link", default=DEFAULT_ENT_LINK, help="ent_link 输入路径（json/jsonl）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    args = parser.parse_args()

    if (args.m is not None and args.m < 0) or (args.n is not None and args.n < 0):
        raise SystemExit("--m and --n must be non-negative")

    out_dir = Path(args.out_dir)
    out_path = out_dir / _merged_output_name(args.m, args.n)
    (
        merged_items,
        mammotab_kept,
        ent_link_kept,
        mammotab_filtered,
        ent_link_filtered,
    ) = _merge_items_with_stats(
        args.mammotab,
        args.m,
        args.ent_link,
        args.n,
    )
    total = _write_json(merged_items, str(out_path))
    expected_total = mammotab_kept + ent_link_kept
    if total != expected_total:
        raise RuntimeError(f"output size mismatch: expected {expected_total}, wrote {total}")

    print(out_path)
    print(
        f"mammotab extracted(after filter): {mammotab_kept} "
        f"(requested={'all' if args.m is None else args.m})"
    )
    print(
        f"ent_link extracted(after filter): {ent_link_kept} "
        f"(requested={'all' if args.n is None else args.n})"
    )
    print(f"wrote {total} items")
    print(f"filtered from mammotab: {mammotab_filtered}")
    print(f"filtered from ent_link: {ent_link_filtered}")


if __name__ == "__main__":
    main()
