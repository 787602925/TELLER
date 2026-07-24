"""
合并两段 SFT 训练数据：
1. 从 mammotab stage4 文件取前 m 条
2. 从 ent_link single_table 文件取前 n 条
3. 合并后写入 /DATA1/khli/t&m/merged_SFT_{m//1000}k&{n//1000}k.json

默认输入：
  mammotab: /DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified.json
  ent_link: /DATA1/khli/tablellama/ent_link_train_single_table.json

用法：
  python3 -m data_clean.data_merge4SFT --m 5000 --n 3000
  python3 data_clean/data_merge4SFT.py --m 70000 --n 35000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_clean.sample_single_table import _write_json_array  # noqa: E402
from data_clean.single_data_train import _iter_json_items  # noqa: E402

DEFAULT_MAMMOTAB = (
    "/DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified.json"
)
DEFAULT_ENT_LINK = "/DATA1/khli/tablellama/ent_link_train_single_table.json"
DEFAULT_OUT_DIR = "/DATA1/khli/t&m"


def _take_first(items: Iterable[Dict[str, Any]], n: int) -> Iterator[Dict[str, Any]]:
    if n <= 0:
        return
    count = 0
    for item in items:
        yield item
        count += 1
        if count >= n:
            break


def _merged_output_name(m: int, n: int) -> str:
    return f"merged_SFT_{m // 1000}k&{n // 1000}k.json"


def _merge_items(mammotab_path: str, m: int, ent_link_path: str, n: int) -> Iterator[Dict[str, Any]]:
    yield from _take_first(_iter_json_items(mammotab_path), m)
    yield from _take_first(_iter_json_items(ent_link_path), n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge mammotab and ent_link SFT data for training.")
    parser.add_argument("--m", type=int, required=True, help="从 mammotab 文件提取的前 m 条")
    parser.add_argument("--n", type=int, required=True, help="从 ent_link 文件提取的前 n 条")
    parser.add_argument("--mammotab", default=DEFAULT_MAMMOTAB, help="mammotab 输入 JSON 路径")
    parser.add_argument("--ent-link", default=DEFAULT_ENT_LINK, help="ent_link 输入 JSON 路径")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    args = parser.parse_args()

    if args.m < 0 or args.n < 0:
        raise SystemExit("--m and --n must be non-negative")

    out_dir = Path(args.out_dir)
    out_path = out_dir / _merged_output_name(args.m, args.n)

    merged_iter = _merge_items(args.mammotab, args.m, args.ent_link, args.n)
    total = _write_json_array(merged_iter, str(out_path))

    print(out_path)
    print(f"wrote {total} items (m={args.m}, n={args.n})")


if __name__ == "__main__":
    main()
