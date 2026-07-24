"""
从实体链接训练数据中构造精简版表格 input。

只加载给定 JSON 文件中的前 100 条数据，根据传入的样本索引：
- 读取该样本的 "input" 和 "entity"
- 解析表格，定位 entity 所在的行和列
- 构造新的 input：
  - 保留原始标题和表头部分（[TLE] ... [TAB] col: ...）
  - 对于所有行：
    - entity 所在的那一行：保留整行（含所有列）
    - 其他行：只保留与 entity 同一列的单元格内容
  - 行之间仍然用 "[SEP]" 分隔
最终返回的是单行字符串（不包含换行）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_mention_from_question(question: str) -> str | None:
    """
    从 question 字段中抽取需要做 entity linking 的实体 mention。

    期望格式示例：
    "The selected entity mention in the table cell is: Richard Dyer. The column name ..."
    """
    if not isinstance(question, str) or not question:
        return None

    marker = "The selected entity mention in the table cell is:"
    idx = question.find(marker)
    if idx == -1:
        return None

    start = idx + len(marker)
    remaining = question[start:].lstrip()
    if not remaining:
        return None

    # 取到下一个句号为止（通常是 ". The column name ..."）
    dot_idx = remaining.find(". The column name")
    if dot_idx != -1:
        mention = remaining[:dot_idx]
    else:
        mention = remaining

    mention = mention.strip()
    return mention or None


def _split_prefix_and_rows(input_text: str) -> Tuple[str, str]:
    """
    将原始 input 划分为前缀部分和从 "row 1:" 开始的行部分。
    """
    marker = "row 1:"
    idx = input_text.find(marker)
    if idx == -1:
        # 找不到行标记时，直接认为没有表格结构
        return input_text, ""
    prefix = input_text[:idx].rstrip()
    rows_part = input_text[idx:].lstrip()
    return prefix, rows_part


def _split_rows(rows_part: str) -> List[str]:
    """
    将行部分按 "[SEP]" 切分成若干行片段，保留原始顺序。
    不包含 "[SEP]" 本身。
    """
    if not rows_part:
        return []
    # 原始格式基本是 "... row 1: ... [SEP] row 2: ... [SEP] ..."
    parts = rows_part.split("[SEP]")
    return [p.strip() for p in parts if p.strip()]


def _parse_row_cells(row_text: str) -> Tuple[int | None, List[str]]:
    """
    从 row 文本中解析：
    - 行号（如果解析失败则为 None）
    - 按 '|' 切分后的各列（不包含行号前缀）
    """
    # 例子: "row 2: | 2006–07 | 12,927 | ... | Melbourne Victory | ..."
    row_prefix_end = row_text.find(":")
    row_idx = None
    if row_prefix_end != -1:
        prefix = row_text[:row_prefix_end]
        # "row 2"
        tokens = prefix.strip().split()
        if len(tokens) == 2 and tokens[0].lower() == "row":
            try:
                row_idx = int(tokens[1])
            except ValueError:
                row_idx = None

    # 查找第一根管道，后面才是真正的表格列
    first_bar = row_text.find("|")
    if first_bar == -1:
        return row_idx, []
    cells_raw = row_text[first_bar + 1 :]
    # 用 '|' 切分，并去掉首尾空白
    cells = [c.strip() for c in cells_raw.split("|") if c.strip() != ""]
    return row_idx, cells


def _load_first_n_items(path: Path, n: int) -> List[Dict[str, Any]]:
    """
    只从一个形如 [ {...}, {...}, ... ] 的 JSON 列表文件中流式解析前 n 条数据。
    不会把整个大文件一次性读入内存。
    """
    items: List[Dict[str, Any]] = []
    if n <= 0:
        return items

    with path.open("r", encoding="utf-8") as f:
        text_iter = iter(lambda: f.read(8192), "")
        buf = ""
        in_string = False
        escape = False
        depth = 0
        current_obj = []
        started_array = False

        for chunk in text_iter:
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
                            # 完整拿到一个对象
                            obj_str = "".join(current_obj)
                            try:
                                items.append(json.loads(obj_str))
                            except json.JSONDecodeError:
                                pass
                            if len(items) >= n:
                                return items
                            current_obj = []
                    elif depth > 0:
                        current_obj.append(ch)

                    if ch == '"':
                        in_string = True
                        escape = False
                else:
                    current_obj.append(ch) if depth > 0 else None
                    if escape:
                        escape = False
                    else:
                        if ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                i += 1

            # 为简单起见，处理完 chunk 后丢弃缓冲，因为我们只按字符流式处理
            buf = ""

    return items


def get_prompt_input(file_path: str | Path, index: int) -> str:
    """
    从给定 JSON 文件（只加载前 100 条）中取出第 index 条样本，
    按照描述规则构造并返回处理后的 input 文本。

    参数
    ----
    file_path: JSON 文件路径
    index: 要取的样本索引（0-based），仅在前 100 条之内有效

    返回
    ----
    处理后的单行 input 字符串（不包含换行）
    """
    if index < 0:
        raise ValueError("index 必须 >= 0")

    path = Path(file_path)
    # 真正只解析前 100 条，避免一次性加载超大文件
    data: List[Dict[str, Any]] = _load_first_n_items(path, 100)
    if index >= len(data):
        raise IndexError(f"索引 {index} 超出范围，数据长度为 {len(data)}（最多加载 100 条）")

    ex = data[index]
    input_text: str = ex.get("input_seg", "")
    entity = ex.get("entity")
    question: str = ex.get("question", "")

    prefix, rows_part = _split_prefix_and_rows(input_text)
    rows = _split_rows(rows_part)

    if not rows:
        # 没有识别到行结构，同样直接返回原始 input
        return input_text.replace("\n", " ")

    # -----------------------------
    # 1) 优先从 question 中抽取 mention，并在表格中找到对应的单元格和行
    # -----------------------------
    entity_row_number: int | None = None  # 表格里 "row N" 的 N
    entity_row_pos: int | None = None  # rows 列表中的下标
    entity_col_index: int | None = None  # 0-based

    mention = _extract_mention_from_question(question)
    if mention:
        for i, row_text in enumerate(rows):
            parsed_row_number, cells = _parse_row_cells(row_text)
            for j, cell in enumerate(cells):
                # 只匹配完整单元格内容
                if cell == mention:
                    entity_row_number = parsed_row_number
                    entity_row_pos = i
                    entity_col_index = j
                    break
            if entity_row_pos is not None:
                break

        # 如果整张表的行数 > 10，并且找到了 mention 所在的行，
        # 只保留“前 4 行 + 本行 + 后 5 行”的窗口
        if len(rows) > 10 and entity_row_pos is not None:
            start = max(0, entity_row_pos - 4)
            end = min(len(rows), entity_row_pos + 1 + 5)  # 右开区间
            rows = rows[start:end]

    # -----------------------------
    # 2) 如果 still 没有拿到 entity 信息，则回退到原始 entity 字段
    # -----------------------------
    if entity_row_number is None or entity_col_index is None:
        # entity 期望形如 [[row_idx, col_idx], "surface"]
        if (
            not isinstance(entity, list)
            or len(entity) < 1
            or not isinstance(entity[0], list)
            or len(entity[0]) != 2
        ):
            # 格式异常时，直接返回原始 input
            return input_text.replace("\n", " ")

        row_idx_raw, col_idx_raw = entity[0]
        try:
            row_idx = int(row_idx_raw)
            col_idx = int(col_idx_raw)
        except (TypeError, ValueError):
            return input_text.replace("\n", " ")

        # entity 的 [3,3] 表示第 4 行第 4 列，因此这里假定是 0-based
        entity_row_number = row_idx + 1  # 转成以 1 开始的行号
        entity_col_index = col_idx  # 仍然是 0-based 列索引

    new_rows: List[str] = []

    for i, row_text in enumerate(rows):
        parsed_row_number, cells = _parse_row_cells(row_text)

        # 行前缀（例如 "row 2:"）
        row_prefix_end = row_text.find(":")
        if row_prefix_end != -1:
            row_prefix = row_text[: row_prefix_end + 1].strip()
        else:
            row_prefix = row_text.strip()

        # entity 所在行：优先用解析出来的行号判断；如果没有行号，则用位置下标判断
        is_entity_row = False
        if entity_row_number is not None and parsed_row_number == entity_row_number:
            is_entity_row = True
        elif entity_row_number is None and entity_row_pos is not None and i == entity_row_pos:
            is_entity_row = True

        if is_entity_row:
            # entity 所在行：保留整行原始内容
            new_rows.append(row_text.strip())
            continue

        # 其他行：只保留与 entity 同列的 cell
        if not cells or entity_col_index is None or entity_col_index >= len(cells):
            # 列数不够，保底直接保留整行
            new_rows.append(row_text.strip())
            continue

        target_cell = cells[entity_col_index]
        # 构造简化后的行，例如 "row 2: Melbourne Victory"
        simplified_row = f"{row_prefix} {target_cell}".strip()
        new_rows.append(simplified_row)

    # 重新拼接成单行字符串。行之间保留 "[SEP]" 分隔。
    # 注意最终不包含换行。
    rebuilt_rows = " [SEP] ".join(new_rows)
    if prefix:
        result = f"{prefix} {rebuilt_rows}"
    else:
        result = rebuilt_rows

    # 保证没有换行
    return result.replace("\n", " ")


if __name__ == "__main__":
    # 简单手动测试用例，方便从命令行快速查看效果：
    #   python auxiliary/get_prompt_tablellama_train.py
    test_path = Path.home() / "DATA" / "tablellama" / "ent_link_test.json"
    try:
        sample = get_prompt_input(test_path, 20)
        print(sample)
    except Exception as e:
        print("测试时出错：", e)

