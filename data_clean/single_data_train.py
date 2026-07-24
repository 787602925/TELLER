"""
从实体链接训练数据中导出单条样本，并生成“处理后版本”。

输入：
1) JSON 文件路径（默认 ~/DATA/tablellama/ent_link_train.json）
2) 第几条数据（0-based index）

输出：一个 JSON 文件（默认写入 auxiliary/ 下），包含：
1) 原始样本内容
2) 处理后的样本内容（仅修改 input_seg 与 question 的 candidates；其他字段不变）

python -m data_clean.single_data_train \
  --data ~/DATA/tablellama/ent_link_train.json \
  --index 123 \
  --out /home/khli/tableLlama/auxiliary/ent_link_item_123.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from auxiliary.get_prompt_tablellama_test import _parse_row_cells, _split_rows  # noqa: E402

CANDIDATE_COUNT = 20
CANDIDATE_OTHER_COUNT = CANDIDATE_COUNT - 1
_INFER_FAILURE_COUNT = 0


def reset_infer_failure_count() -> None:
    global _INFER_FAILURE_COUNT
    _INFER_FAILURE_COUNT = 0


def get_infer_failure_count() -> int:
    return int(_INFER_FAILURE_COUNT)


def _iter_json_items(path: str) -> Iterable[Dict[str, Any]]:
    """
    流式迭代一个 JSON 列表文件的元素。
    优先使用 ijson；若不可用，则用轻量的字符级解析器逐个 yield 对象，
    避免 json.load 一次性加载大文件。
    """
    try:
        import ijson  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            yield from ijson.items(f, "item")
        return
    except Exception:
        pass

    # 退化实现：只支持顶层为 JSON array，元素为 object。
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        in_string = False
        escape = False
        depth = 0
        started_array = False
        current_obj: List[str] = []

        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            buf += chunk
            i = 0
            while i < len(buf):
                ch = buf[i]

                if not started_array:
                    if ch == "[":
                        started_array = True
                    i += 1
                    continue

                if not in_string:
                    if ch == "{":
                        if depth == 0:
                            current_obj = []
                        depth += 1
                        current_obj.append(ch)
                    elif ch == "}":
                        depth -= 1
                        current_obj.append(ch)
                        if depth == 0:
                            obj_str = "".join(current_obj)
                            current_obj = []
                            try:
                                item = json.loads(obj_str)
                            except json.JSONDecodeError:
                                item = None
                            if isinstance(item, dict):
                                yield item
                    elif depth > 0:
                        current_obj.append(ch)

                    if ch == '"':
                        in_string = True
                        escape = False
                else:
                    if depth > 0:
                        current_obj.append(ch)
                    if escape:
                        escape = False
                    else:
                        if ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                i += 1

            buf = ""


def _get_item_by_index(path: str, index: int) -> Dict[str, Any]:
    if index < 0:
        raise ValueError("index 必须 >= 0")
    for i, item in enumerate(_iter_json_items(path)):
        if i == index:
            return item
    raise IndexError(f"索引 {index} 超出范围（文件不足 {index + 1} 条）。")


def _parse_header_cells(prefix_text: str) -> List[str]:
    """
    从 input_seg 的前缀里解析表头（col: | ... | ... |）。
    """
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
    # 只取 col: 之后到结束
    after = sub[col_idx + len("col:") :]
    first_bar = after.find("|")
    if first_bar == -1:
        return []
    cells_raw = after[first_bar + 1 :]
    cells = [c.strip() for c in cells_raw.split("|") if c.strip() != ""]
    return cells


def _norm_cell(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _split_prefix_and_rows_anyrow(input_text: str) -> Tuple[str, str]:
    """
    将 input_seg 划分为前缀与行区块，行区块从首个 "row N:" 开始。
    兼容并非从 row 1 开始的切片表格（如 row 39:, row 1230:）。
    """
    if not isinstance(input_text, str) or not input_text:
        return ("", "")
    m = re.search(r"\brow\s+\d+\s*:", input_text, re.I)
    if not m:
        return (input_text, "")
    idx = m.start()
    return (input_text[:idx].rstrip(), input_text[idx:].lstrip())


def _infer_entity_position_from_input_and_question(
    input_seg: str, question: str
) -> Tuple[int, int] | None:
    """
    当样本里没有 entity 坐标时，尝试根据：
    - question 里的 mention 与 column name
    - input_seg 的表头与表体
    推断出 entity 所在 (row_number_1based, col_index_0based)。
    """
    if not input_seg or not question:
        return None

    m_mention = re.search(
        r"selected entity mention in the table cell is:\s*(.*?)\.\s*The column name",
        question,
        re.I | re.S,
    )
    m_col = re.search(r"The column name for '.+?' is\s*([^\.]+)\.", question, re.I)
    if not m_mention or not m_col:
        return None
    mention = _norm_cell(m_mention.group(1))
    col_name = _norm_cell(m_col.group(1))
    if not mention or not col_name:
        return None

    prefix, rows_part = _split_prefix_and_rows_anyrow(input_seg)
    header_cells = _parse_header_cells(prefix)
    if not header_cells:
        return None

    try:
        col_index = header_cells.index(col_name)
    except ValueError:
        # 某些表头可能有轻微空白差异，做一次弱匹配
        header_norm = [_norm_cell(h) for h in header_cells]
        if col_name in header_norm:
            col_index = header_norm.index(col_name)
        else:
            return None

    rows = _split_rows(rows_part)
    if not rows:
        return None

    mention_lower = mention.lower()
    for row_text in rows:
        row_number, cells = _parse_row_cells(row_text)
        if not row_number or not cells or col_index >= len(cells):
            continue
        # 仅在目标列做匹配，且允许 mention 是 cell 的子串。
        if mention_lower and mention_lower in _norm_cell(cells[col_index]).lower():
            return (row_number, col_index)
    return None


def _select_row_window(
    rows: List[str],
    entity_row_number: int,
    before: int = 4,
    after: int = 5,
    max_rows: int = 10,
) -> List[str]:
    """
    当总行数超过 max_rows 时，保留 entity 行附近窗口：
    默认上 before 行 + 本行 + 下 after 行。
    触边时向另一侧补齐，最终尽量保持 max_rows 行。
    """
    if len(rows) <= max_rows:
        return rows

    entity_idx: int | None = None
    for idx, row_text in enumerate(rows):
        parsed_row_number, _ = _parse_row_cells(row_text)
        if parsed_row_number == entity_row_number:
            entity_idx = idx
            break

    if entity_idx is None:
        fallback_idx = max(0, entity_row_number - 1)
        entity_idx = min(fallback_idx, len(rows) - 1)

    # 先按“上 before 行 + 本行 + 下 after 行”的中心窗口取，再做边界平移
    start = entity_idx - before
    min_start = 0
    max_start = max(0, len(rows) - max_rows)
    start = max(min_start, min(start, max_start))
    end = start + max_rows
    return rows[start:end]


def _simplify_input_seg(
    input_seg: str,
    entity: Any,
    question: str,
    *,
    drop_on_infer_failure: bool = False,
) -> str | None:
    """
    按 auxiliary/get_prompt_tablellama_train.py 的规则处理 input_seg：
    - 保留表格前缀（表名、表头）
    - entity 所在行保留整行
    - 其他行只保留与 entity 同列的单元格
    """
    if not isinstance(input_seg, str):
        return str(input_seg)

    entity_row_number: int | None = None
    entity_col_index: int | None = None

    if (
        isinstance(entity, list)
        and len(entity) >= 1
        and isinstance(entity[0], list)
        and len(entity[0]) == 2
    ):
        row_idx_raw, col_idx_raw = entity[0]
        try:
            row_idx = int(row_idx_raw)
            col_idx = int(col_idx_raw)
            entity_row_number = row_idx + 1  # input_seg 里行号从 1 开始
            entity_col_index = col_idx  # 0-based
        except (TypeError, ValueError):
            entity_row_number = None
            entity_col_index = None

    if entity_row_number is None or entity_col_index is None:
        inferred = _infer_entity_position_from_input_and_question(input_seg, question)
        if inferred is None:
            global _INFER_FAILURE_COUNT
            _INFER_FAILURE_COUNT += 1
            if drop_on_infer_failure:
                return None
            return input_seg.replace("\n", " ")
        entity_row_number, entity_col_index = inferred

    prefix, rows_part = _split_prefix_and_rows_anyrow(input_seg)
    rows = _split_rows(rows_part)
    if not rows:
        return input_seg.replace("\n", " ")
    rows = _select_row_window(rows, entity_row_number, before=4, after=5, max_rows=10)

    new_rows: List[str] = []
    for row_text in rows:
        parsed_row_number, cells = _parse_row_cells(row_text)

        row_prefix_end = row_text.find(":")
        if row_prefix_end != -1:
            row_prefix = row_text[: row_prefix_end + 1].strip()
        else:
            row_prefix = row_text.strip()

        if parsed_row_number == entity_row_number:
            new_rows.append(row_text.strip())
            continue

        if not cells or entity_col_index >= len(cells):
            new_rows.append(row_text.strip())
            continue

        target_cell = cells[entity_col_index]
        new_rows.append(f"{row_prefix} {target_cell}".strip())

    rebuilt_rows = " [SEP] ".join(new_rows)
    result = f"{prefix} {rebuilt_rows}".strip() if prefix else rebuilt_rows
    return result.replace("\n", " ")


@dataclass(frozen=True)
class _QuestionParts:
    prefix: str
    candidates: List[str]  # each like "<...>"
    suffix: str


def _split_question_candidates(question: str) -> _QuestionParts | None:
    """
    将 question 切成 prefix + candidates(list of <...>) + suffix。
    如果找不到候选段，返回 None。
    """
    if not isinstance(question, str) or not question:
        return None

    lower_q = question.lower()
    start_key = "the referent entity candidates are:"
    start_idx = lower_q.find(start_key)
    if start_idx == -1:
        return None
    start_idx += len(start_key)

    # candidates 段通常在 "What is" 之前（参考 data_analyse/ent_link_train_candidates_stats.py）
    end_key = "what is"
    end_idx = lower_q.find(end_key, start_idx)
    if end_idx == -1:
        end_idx = len(question)

    prefix = question[:start_idx]
    segment = question[start_idx:end_idx]
    suffix = question[end_idx:]

    candidates = re.findall(r"<[^>]+>", segment)
    if not candidates:
        return None
    return _QuestionParts(prefix=prefix, candidates=candidates, suffix=suffix)


def _text_ngrams(text: str, n: int) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip().lower())
    if not text:
        return []
    if len(text) <= n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _cosine_sim(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb:
            dot += float(va) * float(vb)
    na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _embed_counter(text: str) -> Counter[str]:
    # 字符 3-gram + 4-gram 的简单向量，适配中英混合且无需额外依赖
    inner = text.strip()
    if inner.startswith("<") and inner.endswith(">"):
        inner = inner[1:-1]
    feats: List[str] = []
    feats.extend(_text_ngrams(inner, 3))
    feats.extend(_text_ngrams(inner, 4))
    return Counter(feats)


def _pick_topk_by_similarity(
    candidates: List[str], gold: str, k_other: int = CANDIDATE_OTHER_COUNT
) -> List[str]:
    gold_norm = gold.strip()
    gold_vec = _embed_counter(gold_norm)

    uniq: List[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    scored: List[Tuple[float, str]] = []
    for c in uniq:
        if c.strip() == gold_norm:
            continue
        sim = _cosine_sim(_embed_counter(c), gold_vec)
        scored.append((sim, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    picked = [c for _, c in scored[: max(0, int(k_other))]]

    out: List[str] = []
    out.append(gold_norm)
    for c in picked:
        if c != gold_norm and c not in out:
            out.append(c)

    # 如果 gold 不在原 candidates 里，也照样保留 gold；总数最多 10
    return out[: 1 + max(0, int(k_other))]


def _rebuild_question_with_candidates(parts: _QuestionParts, new_candidates: List[str]) -> str:
    # 统一用 ", " 连接，格式与原数据一致
    mid = " " + ", ".join(new_candidates) + " "
    return f"{parts.prefix}{mid}{parts.suffix}"


def process_item(
    item: Dict[str, Any], *, drop_on_infer_failure: bool = False
) -> Dict[str, Any] | None:
    processed = dict(item)

    simplified_input_seg = _simplify_input_seg(
        item.get("input_seg", ""),
        item.get("entity"),
        item.get("question", ""),
        drop_on_infer_failure=drop_on_infer_failure,
    )
    if simplified_input_seg is None:
        return None
    processed["input_seg"] = simplified_input_seg

    question = item.get("question", "")
    gold = item.get("output", "")
    parts = _split_question_candidates(question)
    if parts and isinstance(gold, str) and gold.strip():
        new_candidates = _pick_topk_by_similarity(
            parts.candidates,
            gold.strip(),
            k_other=CANDIDATE_OTHER_COUNT,
        )
        if len(new_candidates) > 1:
            random.shuffle(new_candidates)
        processed["question"] = _rebuild_question_with_candidates(parts, new_candidates)

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick and process one ent_link sample.")
    parser.add_argument(
        "--data",
        default="~/DATA/tablellama/ent_link_train.json",
        help="JSON 数据路径（默认：~/DATA/tablellama/ent_link_train.json）",
    )
    parser.add_argument("--index", type=int, required=True, help="样本 index（0-based）")
    parser.add_argument(
        "--out",
        default="",
        help="输出 JSON 路径（默认：auxiliary/ent_link_item_{index}.json）",
    )
    args = parser.parse_args()

    data_path = os.path.expanduser(args.data)
    item = _get_item_by_index(data_path, int(args.index))
    processed = process_item(item)

    out_path = args.out.strip()
    if not out_path:
        out_path = str(Path(__file__).resolve().parent / f"ent_link_item_{int(args.index)}.json")
    out_path = os.path.expanduser(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "_meta": {"data_path": data_path, "index": int(args.index)},
        "original": item,
        "processed": processed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(out_path)


if __name__ == "__main__":
    main()

