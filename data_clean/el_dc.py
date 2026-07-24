"""
批量处理实体链接数据：对每条样本应用 `single_data_train.process_item`，
输出与源文件相同的顶层 JSON 数组格式。

默认：
  输入  --type train: ~/DATA/tablellama/ent_link_train.json
        --type val:   ~/DATA/tablellama/ent_link_train.json
        --type eval:  ~/DATA/tablellama/ent_link_test.json
  输出  --type train: 未指定 --num 时为 ent_link_train_simplified.json；
                     指定 --num N 时为 ent_link_train_simplified_{N}.json
       --type val:   指定 --num N 时为 ent_link_val_{N}.json
       --type eval:  未指定 --num 时为 ent_link_test_simplified.json；
                     指定 --num N 时为 ent_link_test_first{N}.json
        （均在 ~/DATA/tablellama/ 下）

指定 --num 时按“表格（Wikipedia page + section）”去重：
- --type train：从前往后取前 N 个不同表格，每个表随机取 1 条。
- --type val：从后往前取前 N 个不同表格，每个表随机取 1 条。
- --type eval：直接从 ent_link_test.json 取前 N 条。
未指定 --num 时流式处理直至 EOF。

用法：
  python -m data_clean.el_dc --type train
  python -m data_clean.el_dc --type train --num 1000
  python -m data_clean.el_dc --type val --num 500
  python -m data_clean.el_dc --type eval
  python -m data_clean.el_dc --type eval --num 100
  python -m data_clean.el_dc --type mam_eval --num 50
  python /path/to/data_clean/el_dc.py --in ... --out ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, TextIO, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_clean.single_data_train import (  # noqa: E402
    _iter_json_items,
    get_infer_failure_count,
    process_item,
    reset_infer_failure_count,
)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _iter_with_progress(
    items_iter: Iterable[Dict[str, Any]],
    desc: str,
    unit: str = "item",
) -> Iterator[Dict[str, Any]]:
    """
    若安装了 tqdm 则使用标准进度条；
    否则退化为终端单行动态进度显示（条数 + 速率 + 动态条）。
    """
    if tqdm is not None:
        yield from tqdm(
            items_iter,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        return

    start = time.time()
    last_flush = start
    count = 0
    frames = (
        "[>         ]",
        "[=>        ]",
        "[==>       ]",
        "[===>      ]",
        "[====>     ]",
        "[=====>    ]",
        "[======>   ]",
        "[=======>  ]",
        "[========> ]",
        "[=========>]",
    )
    frame_idx = 0

    for item in items_iter:
        count += 1
        now = time.time()
        if now - last_flush >= 1.0:
            elapsed = max(now - start, 1e-9)
            rate = count / elapsed
            frame = frames[frame_idx % len(frames)]
            frame_idx += 1
            msg = f"\r{desc} {frame} {count} {unit}s ({rate:.1f} {unit}s/s)"
            print(msg, end="", file=sys.stderr, flush=True)
            last_flush = now
        yield item

    elapsed = max(time.time() - start, 1e-9)
    rate = count / elapsed
    print(
        f"\r{desc} [==========] {count} {unit}s ({rate:.1f} {unit}s/s)",
        file=sys.stderr,
    )


def _iter_json_items_up_to(path: str, max_items: Optional[int]) -> Iterator[Dict[str, Any]]:
    """
    从顶层 JSON 数组流式读取 object。max_items 为 None 时读到文件结束；
    为正整数时只解析并产出前 max_items 条，随后 close 底层生成器，停止读入并释放文件句柄
    （依赖 single_data_train._iter_json_items 的 ijson/生成器行为）。
    """
    gen = _iter_json_items(path)
    try:
        if max_items is None:
            yield from gen
        else:
            for _ in range(max_items):
                try:
                    yield next(gen)
                except StopIteration:
                    break
    finally:
        gen.close()


def _iter_jsonl_items_up_to(path: str, max_items: Optional[int]) -> Iterator[Dict[str, Any]]:
    """
    从 jsonl 文件读取 object。max_items 为 None 时读到文件结束；
    为正整数时仅产出前 max_items 条。
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if not isinstance(item, dict):
                raise ValueError(f"jsonl line {line_no} is not an object")
            yield item
            count += 1
            if max_items is not None and count >= max_items:
                break


def _adapt_mam_eval_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    mam_eval 源数据使用 input 字段；为复用 eval 同款 simplify 逻辑，
    临时补出 input_seg 给 process_item 使用。
    """
    adapted = dict(item)
    if "input_seg" not in adapted and "input" in adapted:
        adapted["input_seg"] = adapted.get("input", "")
    if "entity" not in adapted:
        inferred_entity = _infer_mam_entity_from_input_and_question(
            adapted.get("input_seg", ""),
            adapted.get("question", ""),
        )
        if inferred_entity is not None:
            adapted["entity"] = inferred_entity
    return adapted


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _extract_mam_row_cells(row_body: str) -> list[str]:
    parts = row_body.split("|")
    if len(parts) >= 2:
        return [c.strip() for c in parts[1:-1]]
    return [c.strip() for c in parts if c.strip()]


def _infer_mam_entity_from_input_and_question(
    input_seg: Any,
    question: Any,
) -> list[Any] | None:
    if not isinstance(input_seg, str) or not isinstance(question, str):
        return None

    m_mention = re.search(
        r"selected entity mention in the table cell is:\s*['\"]?(.*?)['\"]?\.\s*The column name",
        question,
        flags=re.I | re.S,
    )
    m_col = re.search(r"The column name for .*? is\s*([^\.]+)\.", question, flags=re.I | re.S)
    if not m_mention or not m_col:
        return None

    mention = _normalize_text(m_mention.group(1)).strip("'\"")
    col_name = _normalize_text(m_col.group(1))
    if not mention or not col_name:
        return None

    m_col_idx = re.fullmatch(r"col\s*(\d+)", col_name, flags=re.I)
    if not m_col_idx:
        return None
    col_idx = int(m_col_idx.group(1))

    mention_lower = mention.lower()
    for part in input_seg.split("[SEP]"):
        m_row = re.search(r"row\s+(\d+)\s*:\s*(.*)$", part.strip(), flags=re.I | re.S)
        if not m_row:
            continue
        row_num = int(m_row.group(1))
        cells = _extract_mam_row_cells(m_row.group(2))
        if col_idx >= len(cells):
            continue
        cell_text = _normalize_text(cells[col_idx]).lower()
        if mention_lower and mention_lower in cell_text:
            return [[row_num - 1, col_idx], mention]
    return None


def _extract_table_key(input_seg: Any) -> Tuple[str, str]:
    """
    从 input_seg 里解析表格标识：(wikipedia_page, wikipedia_section)。
    解析失败时退化为空字符串，避免抛异常中断流程。
    """
    if not isinstance(input_seg, str):
        return ("", "")

    m_page = re.search(r"The Wikipedia page is about\s+(.*?)\.", input_seg, flags=re.I | re.S)
    m_section = re.search(
        r"The Wikipedia section is about\s+(.*?)\.", input_seg, flags=re.I | re.S
    )
    page = _normalize_text(m_page.group(1)) if m_page else ""
    section = _normalize_text(m_section.group(1)) if m_section else ""
    return (page, section)


def _iter_distinct_tables(path: str, table_count: int) -> Iterator[Dict[str, Any]]:
    """
    流式读取数据，按“前 table_count 个不同表格”采样：
    - 每个表格随机保留 1 条（reservoir sampling）
    - 当遇到第 table_count+1 个新表格时立即停止读取
    """
    gen = _iter_with_progress(_iter_json_items(path), desc="Scanning distinct tables")
    try:
        samples: Dict[Tuple[str, str], Dict[str, Any]] = {}
        seen_counts: Dict[Tuple[str, str], int] = {}
        table_order: list[Tuple[str, str]] = []

        for item in gen:
            key = _extract_table_key(item.get("input_seg", ""))

            if key in samples:
                seen_counts[key] += 1
                # 每张表做一次 reservoir sampling，确保“任意一条”更均匀
                if random.randint(1, seen_counts[key]) == 1:
                    samples[key] = item
                continue

            if len(samples) >= table_count:
                break

            samples[key] = item
            seen_counts[key] = 1
            table_order.append(key)

        for key in table_order:
            yield samples[key]
    finally:
        gen.close()


def _iter_distinct_tables_from_end(path: str, table_count: int) -> Iterator[Dict[str, Any]]:
    """
    按“从后往前”的顺序取前 table_count 个不同表格，并为每个表随机保留 1 条：
    - 第 1 遍：流式统计每个表最后一次出现的位置
    - 第 2 遍：只在命中的后缀区间中做 reservoir sampling
    """
    # pass 1: 只记录 last index，内存占用与“不同表格数”线性相关
    last_pos: Dict[Tuple[str, str], int] = {}
    pass1_iter = _iter_with_progress(_iter_json_items(path), desc="Reverse mode pass 1")
    for idx, item in enumerate(pass1_iter):
        key = _extract_table_key(item.get("input_seg", ""))
        last_pos[key] = idx

    if not last_pos:
        return

    ordered = sorted(last_pos.items(), key=lambda x: x[1], reverse=True)
    selected_pairs = ordered[:table_count]
    selected_keys = {k for k, _ in selected_pairs}
    reverse_table_order = [k for k, _ in selected_pairs]
    cutoff = selected_pairs[-1][1]

    # pass 2: 在 suffix（idx >= cutoff）里对每个表随机保留 1 条
    samples: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen_counts: Dict[Tuple[str, str], int] = {}
    pass2_iter = _iter_with_progress(_iter_json_items(path), desc="Reverse mode pass 2")
    for idx, item in enumerate(pass2_iter):
        if idx < cutoff:
            continue
        key = _extract_table_key(item.get("input_seg", ""))
        if key not in selected_keys:
            continue
        prev = seen_counts.get(key, 0) + 1
        seen_counts[key] = prev
        if prev == 1 or random.randint(1, prev) == 1:
            samples[key] = item

    for key in reverse_table_order:
        if key in samples:
            yield samples[key]


def _iter_last_distinct_tables_from_end(path: str, table_count: int) -> Iterator[Dict[str, Any]]:
    """
    从后往前取前 table_count 个不同表格，每个表仅保留其“最后一次出现”的那条样本。
    输出顺序与“从后往前”一致（即最后出现位置更靠后的样本先输出）。
    """
    last_pos: Dict[Tuple[str, str], int] = {}
    pass1_iter = _iter_with_progress(_iter_json_items(path), desc="mam_val pass 1")
    for idx, item in enumerate(pass1_iter):
        key = _extract_table_key(item.get("input_seg", ""))
        last_pos[key] = idx

    if not last_pos:
        return

    selected_pairs = sorted(last_pos.items(), key=lambda x: x[1], reverse=True)[:table_count]
    selected_order = [k for k, _ in selected_pairs]
    selected_keys = set(selected_order)

    selected_items: Dict[Tuple[str, str], Dict[str, Any]] = {}
    pass2_iter = _iter_with_progress(_iter_json_items(path), desc="mam_val pass 2")
    for idx, item in enumerate(pass2_iter):
        key = _extract_table_key(item.get("input_seg", ""))
        if key in selected_keys and idx == last_pos.get(key):
            selected_items[key] = item

    for key in selected_order:
        if key in selected_items:
            yield selected_items[key]


def _write_json_array_streaming(
    items_iter,
    out_f: TextIO,
    *,
    indent: int = 2,
) -> int:
    """
    将迭代器中的 dict 写成 JSON 数组，流式写入，避免一次性加载全部数据。
    格式与 json.dump(list, indent=2) 的数组相近：每个元素缩进一层。
    返回写入的元素个数。
    """
    sep = " " * indent
    out_f.write("[\n")
    first = True
    count = 0
    write_iter = _iter_with_progress(items_iter, desc="Processing and writing")
    for item in write_iter:
        processed = process_item(item, drop_on_infer_failure=True)
        if processed is None:
            continue
        if not first:
            out_f.write(",\n")
        first = False
        block = json.dumps(processed, ensure_ascii=False, indent=indent)
        for line in block.splitlines():
            out_f.write(sep)
            out_f.write(line)
            out_f.write("\n")
        count += 1
    out_f.write("]\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simplify ent_link_train.json with single_data_train.process_item."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default="~/DATA/tablellama/ent_link_train.json",
        help="源 JSON 数组文件路径",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="",
        help="输出 JSON 数组文件路径；不填则按 --num 自动命名（见文档）",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        metavar="N",
        help="提取前 N 个不同表格（按 input_seg 的 Wikipedia page + section），每个表随机取 1 条；不填则处理全部",
    )
    parser.add_argument(
        "--type",
        choices=("train", "val", "eval", "mam_eval", "mam_val"),
        default="train",
        help=(
            "生成 train/val/eval 数据；val 表示从 ent_link_train.json 末尾反向取 N 个不同表格，"
            "每表随机 1 条；eval 表示从 ent_link_test.json 取前 N 条（未指定 --num 时处理全量）；"
            "mam_eval 表示从 mammotab_2024_prompts_50.jsonl 取数据并按 eval 规则 simplify；"
            "mam_val 表示从 stage4_single_shards_merge_simplified.json 末尾反向取 N 个不同表格"
        ),
    )
    args = parser.parse_args()

    if args.num is not None and args.num < 1:
        parser.error("--num 必须是正整数")
    if args.type == "val" and args.num is None:
        parser.error(f"--type {args.type} 必须指定 --num")
    if args.type == "mam_val" and args.num is None:
        parser.error(f"--type {args.type} 必须指定 --num")

    if args.type == "eval":
        in_path = os.path.expanduser("~/DATA/tablellama/ent_link_test.json")
    elif args.type == "mam_eval":
        in_path = "/DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.jsonl"
    elif args.type == "mam_val":
        in_path = "/DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified.json"
    elif args.type == "val":
        in_path = os.path.expanduser("~/DATA/tablellama/ent_link_train.json")
    else:
        in_path = os.path.expanduser(args.in_path)
    out_raw = (args.out_path or "").strip()
    if out_raw:
        out_path = os.path.expanduser(out_raw)
    elif args.num is not None:
        if args.type == "train":
            out_path = os.path.expanduser(
                f"~/DATA/tablellama/ent_link_train_simplified_{args.num}.json"
            )
        elif args.type == "val":
            out_path = os.path.expanduser(
                f"~/DATA/tablellama/ent_link_val_{args.num}.json"
            )
        elif args.type == "mam_eval":
            out_path = (
                "/DATA1/khli/mammotab/mammotab_dataset_semtab/"
                f"mammotab_2024_prompts_50_first{args.num}.json"
            )
        elif args.type == "mam_val":
            out_path = (
                "/DATA1/khli/mammotab/modified_mammotab/"
                f"stage4_single_shards_merge_simplified_mam_val_last{args.num}.json"
            )
        else:
            out_path = os.path.expanduser(
                f"~/DATA/tablellama/ent_link_test_first{args.num}.json"
            )
    else:
        if args.type == "train":
            out_path = os.path.expanduser("~/DATA/tablellama/ent_link_train_simplified.json")
        elif args.type == "val":
            out_path = os.path.expanduser("~/DATA/tablellama/ent_link_val_100.json")
        elif args.type == "mam_eval":
            out_path = (
                "/DATA1/khli/mammotab/mammotab_dataset_semtab/"
                "mammotab_2024_prompts_50_simplified.json"
            )
        elif args.type == "mam_val":
            out_path = (
                "/DATA1/khli/mammotab/modified_mammotab/"
                "stage4_single_shards_merge_simplified_mam_val.json"
            )
        else:
            out_path = os.path.expanduser("~/DATA/tablellama/ent_link_test_simplified.json")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    reset_infer_failure_count()

    if args.type == "mam_eval":
        base_iter = _iter_jsonl_items_up_to(in_path, args.num)
        items_iter = (_adapt_mam_eval_item(item) for item in base_iter)
    elif args.type == "mam_val":
        items_iter = _iter_last_distinct_tables_from_end(in_path, int(args.num))
    elif args.num is None:
        items_iter = _iter_json_items_up_to(in_path, None)
    else:
        if args.type == "train":
            items_iter = _iter_distinct_tables(in_path, args.num)
        elif args.type == "val":
            items_iter = _iter_distinct_tables_from_end(in_path, args.num)
        else:
            items_iter = _iter_json_items_up_to(in_path, args.num)

    with open(out_path, "w", encoding="utf-8") as out_f:
        n = _write_json_array_streaming(items_iter, out_f)
    print(out_path)
    print(f"wrote {n} items", file=sys.stderr)
    print(f"infer_failure_count={get_infer_failure_count()}", file=sys.stderr)


if __name__ == "__main__":
    main()
