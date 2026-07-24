"""
Stage 4: Simplify merged stage2 JSONL data and write JSON array output.

Goals:
1) Keep headers + full entity row + entity column values in other rows.
2) Keep only nearby rows around the target entity (default max 10 rows).
3) Output as .json (top-level JSON array).

Example:
  python -m mammotab2ti.stage4_simplify
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def _iter_jsonl_items(path: Path, max_items: int = 0) -> Iterator[Dict[str, Any]]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
                count += 1
                if max_items > 0 and count >= max_items:
                    break


def _split_prefix_and_rows_anyrow(input_text: str) -> Tuple[str, str]:
    if not isinstance(input_text, str) or not input_text:
        return ("", "")
    m = re.search(r"\brow\s+\d+\s*:", input_text, flags=re.I)
    if not m:
        return (input_text, "")
    idx = m.start()
    return (input_text[:idx].rstrip(), input_text[idx:].lstrip())


def _parse_header_cells(prefix_text: str) -> List[str]:
    if not prefix_text:
        return []
    marker = "[TAB]"
    idx = prefix_text.find(marker)
    if idx == -1:
        return []
    sub = prefix_text[idx:]
    col_idx = sub.find("col:")
    if col_idx == -1:
        return []
    after = sub[col_idx + len("col:") :]
    first_bar = after.find("|")
    if first_bar == -1:
        return []
    cells_raw = after[first_bar + 1 :]
    return [c.strip() for c in cells_raw.split("|") if c.strip()]


def _split_rows(rows_part: str) -> List[str]:
    if not isinstance(rows_part, str) or not rows_part.strip():
        return []
    rows = re.split(r"\s*\[SEP\]\s*", rows_part.strip())
    return [r.strip() for r in rows if r.strip()]


def _parse_row_cells(row_text: str) -> Tuple[Optional[int], List[str]]:
    if not isinstance(row_text, str):
        return (None, [])
    m = re.match(r"\s*row\s+(\d+)\s*:\s*(.*)\s*$", row_text, flags=re.I)
    if not m:
        return (None, [])
    row_number = int(m.group(1))
    payload = m.group(2).strip()
    if "|" in payload:
        cells = [c.strip() for c in payload.split("|") if c.strip()]
    elif payload:
        cells = [payload]
    else:
        cells = []
    return (row_number, cells)


def _infer_entity_position_from_input_and_question(
    input_seg: str,
    question: str,
) -> Optional[Tuple[int, int]]:
    if not input_seg or not question:
        return None

    m_mention = re.search(
        r"selected entity mention in the table cell is:\s*(.*?)\.\s*The column name",
        question,
        flags=re.I | re.S,
    )
    m_col = re.search(r"The column name for '.+?' is\s*([^\.]+)\.", question, flags=re.I)
    if not m_mention or not m_col:
        return None
    mention = _normalize_text(m_mention.group(1))
    col_name = _normalize_text(m_col.group(1))
    if not mention or not col_name:
        return None

    prefix, rows_part = _split_prefix_and_rows_anyrow(input_seg)
    header_cells = _parse_header_cells(prefix)
    if not header_cells:
        return None

    header_norm = [_normalize_text(x) for x in header_cells]
    try:
        col_index = header_norm.index(col_name)
    except ValueError:
        return None

    mention_lower = mention.lower()
    for row_text in _split_rows(rows_part):
        row_number, cells = _parse_row_cells(row_text)
        if row_number is None or not cells or col_index >= len(cells):
            continue
        if mention_lower in _normalize_text(cells[col_index]).lower():
            return (row_number, col_index)
    return None


def _simplify_input_seg(input_seg: str, entity_row_number: int, entity_col_index: int) -> str:
    prefix, rows_part = _split_prefix_and_rows_anyrow(input_seg)
    rows = _split_rows(rows_part)
    if not rows:
        return input_seg.replace("\n", " ")
    rows = _select_row_window(rows, entity_row_number, before=4, after=5, max_rows=10)

    new_rows: List[str] = []
    for row_text in rows:
        parsed_row_number, cells = _parse_row_cells(row_text)
        row_prefix_end = row_text.find(":")
        row_prefix = row_text[: row_prefix_end + 1].strip() if row_prefix_end != -1 else row_text.strip()

        if parsed_row_number == entity_row_number:
            new_rows.append(row_text.strip())
            continue

        if not cells or entity_col_index >= len(cells):
            new_rows.append(row_text.strip())
            continue

        new_rows.append(f"{row_prefix} {cells[entity_col_index]}".strip())

    rebuilt_rows = " [SEP] ".join(new_rows)
    result = f"{prefix} {rebuilt_rows}".strip() if prefix else rebuilt_rows
    return result.replace("\n", " ")


def _select_row_window(
    rows: List[str],
    entity_row_number: int,
    before: int = 4,
    after: int = 5,
    max_rows: int = 10,
) -> List[str]:
    if len(rows) <= max_rows:
        return rows

    entity_idx: Optional[int] = None
    for idx, row_text in enumerate(rows):
        parsed_row_number, _ = _parse_row_cells(row_text)
        if parsed_row_number == entity_row_number:
            entity_idx = idx
            break

    if entity_idx is None:
        fallback_idx = max(0, entity_row_number - 1)
        entity_idx = min(fallback_idx, len(rows) - 1)

    start = entity_idx - before
    min_start = 0
    max_start = max(0, len(rows) - max_rows)
    start = max(min_start, min(start, max_start))
    end = start + max_rows
    return rows[start:end]


def _process_item(item: Dict[str, Any], drop_on_infer_failure: bool) -> Optional[Dict[str, Any]]:
    question = str(item.get("question", ""))
    input_seg = str(item.get("input_seg", ""))
    inferred = _infer_entity_position_from_input_and_question(input_seg, question)
    if inferred is None:
        if drop_on_infer_failure:
            return None
        processed = dict(item)
        processed["input_seg"] = input_seg.replace("\n", " ")
        return processed

    row_number, col_index = inferred
    processed = dict(item)
    processed["input_seg"] = _simplify_input_seg(input_seg, row_number, col_index)
    return processed


def _write_json_array(items: Iterable[Dict[str, Any]], output_path: Path, indent: int = 2) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        first = True
        for item in items:
            if not first:
                out_f.write(",\n")
            first = False
            block = json.dumps(item, ensure_ascii=False, indent=indent)
            for line in block.splitlines():
                out_f.write(" " * indent + line + "\n")
            count += 1
        out_f.write("]\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simplify merged stage2 JSONL data and write JSON array output."
    )
    parser.add_argument(
        "--in_jsonl",
        type=str,
        default="/DATA1/khli/mammotab/modified_mammotab/stage3_single_shards_merge.jsonl",
        help="Input merged stage2 JSONL path",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="/DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified.json",
        help="Output JSON array path",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Only process first N input lines for quick tests (0 means all)",
    )
    parser.add_argument(
        "--keep_on_infer_failure",
        action="store_true",
        help="Keep sample when entity row/column inference fails (default: drop)",
    )
    args = parser.parse_args()

    in_path = Path(args.in_jsonl).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"input file not found: {in_path}")

    max_items = max(0, int(args.max_items))
    drop_on_infer_failure = not bool(args.keep_on_infer_failure)

    total = 0
    kept = 0
    dropped_infer_failure = 0

    def _processed_iter() -> Iterator[Dict[str, Any]]:
        nonlocal total, kept, dropped_infer_failure
        for item in _iter_jsonl_items(in_path, max_items=max_items):
            total += 1
            processed = _process_item(item, drop_on_infer_failure=drop_on_infer_failure)
            if processed is None:
                dropped_infer_failure += 1
                continue
            kept += 1
            if total % 5000 == 0:
                print(f"processed={total}, kept={kept}, dropped_infer={dropped_infer_failure}")
            yield processed

    wrote = _write_json_array(_processed_iter(), out_path, indent=2)
    print(f"done: total={total}, wrote={wrote}, kept={kept}, dropped_infer={dropped_infer_failure}")
    print(str(out_path))


if __name__ == "__main__":
    main()
