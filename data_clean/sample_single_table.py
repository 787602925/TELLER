"""
从 ent_link_train_simplified.json 中按表格随机抽样：
- 表格定义：input_seg 中的 (Wikipedia page, Wikipedia section)
- 每个表格随机保留 1 条（reservoir sampling）

默认：
  输入：/DATA1/khli/tablellama/ent_link_train_simplified.json
  输出：/DATA1/khli/tablellama/ent_link_train_single_table.json

用法：
  python3 -m data_clean.sample_single_table
  python3 data_clean/sample_single_table.py
  python3 data_clean/sample_single_table.py --seed 42
  python3 data_clean/sample_single_table.py --in /path/in.json --out /path/out.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

from data_clean.single_data_train import _iter_json_items


DEFAULT_IN = "/DATA1/khli/tablellama/ent_link_train_simplified.json"
DEFAULT_OUT = "/DATA1/khli/tablellama/ent_link_train_single_table.json"


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _extract_table_key(input_seg: Any) -> Tuple[str, str]:
    if not isinstance(input_seg, str):
        return ("", "")

    m_page = re.search(r"The Wikipedia page is about\s+(.*?)\.", input_seg, flags=re.I | re.S)
    m_section = re.search(
        r"The Wikipedia section is about\s+(.*?)\.", input_seg, flags=re.I | re.S
    )
    page = _normalize_text(m_page.group(1)) if m_page else ""
    section = _normalize_text(m_section.group(1)) if m_section else ""
    return (page, section)


def _sample_one_per_table(items: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    # 每个 key 单独进行 reservoir sampling，保证该表任意一条被选中的概率一致
    samples: Dict[Tuple[str, str], Dict[str, Any]] = {}
    counts: Dict[Tuple[str, str], int] = {}
    table_order: list[Tuple[str, str]] = []

    for item in items:
        key = _extract_table_key(item.get("input_seg", ""))
        if key not in samples:
            samples[key] = item
            counts[key] = 1
            table_order.append(key)
            continue

        counts[key] += 1
        if random.randint(1, counts[key]) == 1:
            samples[key] = item

    for key in table_order:
        yield samples[key]


def _write_json_array(items: Iterable[Dict[str, Any]], out_path: str) -> int:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_file.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for item in items:
            if not first:
                f.write(",\n")
            first = False
            f.write(json.dumps(item, ensure_ascii=False, indent=2))
            n += 1
        f.write("\n]\n")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample one random record per table.")
    parser.add_argument("--in", dest="in_path", default=DEFAULT_IN, help="输入 JSON 数组路径")
    parser.add_argument("--out", dest="out_path", default=DEFAULT_OUT, help="输出 JSON 数组路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    items_iter = _iter_json_items(args.in_path)
    sampled_iter = _sample_one_per_table(items_iter)
    n = _write_json_array(sampled_iter, args.out_path)
    print(args.out_path)
    print(f"wrote {n} items")


if __name__ == "__main__":
    main()
