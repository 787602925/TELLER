from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

DEFAULT_INSTRUCTION = (
    "This is an entity linking task. The goal for this task is to link the selected entity "
    "mention in the table cells to the entity in the knowledge base. You will be given a list "
    "of referent entities, with each one composed of an entity name, its description and its "
    "type. Please choose the correct one from the referent entity candidates. Note that the "
    "Wikipedia page, Wikipedia section and table caption (if any) provide important information "
    "for choosing the correct referent entity."
)


@dataclass
class MentionJob:
    job_id: str
    source_path: str
    source_file: str
    row_index: int
    col_index: int
    mention: str
    column_name: str
    page_title: str
    section_title: str
    table_caption: str
    gold_qid: str
    gold_wiki_title: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_path": self.source_path,
            "source_file": self.source_file,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "mention": self.mention,
            "column_name": self.column_name,
            "page_title": self.page_title,
            "section_title": self.section_title,
            "table_caption": self.table_caption,
            "gold_qid": self.gold_qid,
            "gold_wiki_title": self.gold_wiki_title,
        }


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _cell(matrix: List[List[Any]], row_idx: int, col_idx: int) -> str:
    if row_idx < 0 or row_idx >= len(matrix):
        return ""
    row = matrix[row_idx]
    if col_idx < 0 or col_idx >= len(row):
        return ""
    return _clean_text(row[col_idx])


def load_table_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_text_matrix(table: Dict[str, Any]) -> List[List[str]]:
    text = table.get("text")
    if isinstance(text, list) and text:
        out: List[List[str]] = []
        for row in text:
            if not isinstance(row, list):
                continue
            out.append([_clean_text(x) for x in row])
        if out:
            return out

    cells = table.get("cells")
    header = table.get("header")
    if not isinstance(cells, list) or not cells:
        return []
    header_width = 0
    if isinstance(header, list) and header and isinstance(header[0], list):
        header_width = len(header[0])
    out2: List[List[str]] = []
    for row in cells:
        if not isinstance(row, list):
            continue
        cur = [_clean_text(x) for x in row]
        if header_width > 0:
            cur = cur[:header_width]
        out2.append(cur)
    return out2


def get_page_title(table: Dict[str, Any]) -> str:
    external = table.get("external_context", {})
    if not isinstance(external, dict):
        return ""
    null_text = str(external.get("null", ""))
    m = re.search(r"<title>(.*?)</title>", null_text, flags=re.S | re.I)
    return _clean_text(m.group(1)) if m else ""


def infer_section_title(table: Dict[str, Any], mention: str, link_title: str) -> str:
    external = table.get("external_context", {})
    if not isinstance(external, dict):
        return ""

    candidates: List[tuple[float, str]] = []
    mention_low = _clean_text(mention).lower()
    link_low = _clean_text(link_title).replace("_", " ").lower()

    for key, value in external.items():
        if key == "null":
            continue
        key_clean = _clean_text(key)
        text = str(value).lower()
        score = 0.0
        if mention_low and mention_low in text:
            score += 2.0
        if link_low and link_low in text:
            score += 3.0
        if "{|" in str(value):
            score += 0.2
        if key_clean:
            candidates.append((score, key_clean))

    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_key = candidates[0]
    if best_score <= 0.0:
        return ""
    return best_key


def extract_jobs_from_table(table: Dict[str, Any], source_path: str) -> Iterable[MentionJob]:
    text = get_text_matrix(table)
    if len(text) < 2:
        return []
    links = table.get("link", [])
    entities = table.get("entity", [])
    if not isinstance(links, list):
        links = []
    if not isinstance(entities, list):
        entities = []

    page_title = get_page_title(table)
    caption = _clean_text(table.get("caption", ""))
    if caption.lower() == "none":
        caption = ""

    header = text[0]
    jobs: List[MentionJob] = []
    file_name = Path(source_path).name

    for r in range(1, len(text)):
        row = text[r]
        for c in range(min(len(row), len(header))):
            mention = _clean_text(row[c])
            if not mention:
                continue
            link_title = _cell(links, r, c)
            gold_qid = _cell(entities, r, c)
            if not link_title and not gold_qid:
                continue

            section = infer_section_title(table, mention=mention, link_title=link_title)
            col_name = _clean_text(header[c]) if c < len(header) else f"col_{c}"
            job = MentionJob(
                job_id=f"{file_name}#{r}:{c}",
                source_path=str(Path(source_path).resolve()),
                source_file=file_name,
                row_index=r,
                col_index=c,
                mention=mention,
                column_name=col_name,
                page_title=page_title,
                section_title=section,
                table_caption=caption,
                gold_qid=gold_qid,
                gold_wiki_title=link_title,
            )
            jobs.append(job)
    return jobs


def build_input_seg(table: Dict[str, Any], row_index: int, col_index: int) -> str:
    cached_text = table.get("__mt2ti_cached_text")
    text: List[List[str]]
    if isinstance(cached_text, list):
        text = cached_text
    else:
        text = get_text_matrix(table)
        table["__mt2ti_cached_text"] = text
    if len(text) < 2:
        return "[TLE] The Wikipedia page is about . The Wikipedia section is about . [TAB] col: |"

    header = text[0]
    page_title = str(table.get("__mt2ti_cached_page_title", ""))
    if not page_title:
        page_title = get_page_title(table) or ""
        table["__mt2ti_cached_page_title"] = page_title

    section = infer_section_title(table, mention=_cell(text, row_index, col_index), link_title="")
    caption = str(table.get("__mt2ti_cached_caption", ""))
    if caption == "":
        cap = _clean_text(table.get("caption", ""))
        if cap.lower() == "none":
            cap = ""
        caption = cap
        table["__mt2ti_cached_caption"] = caption

    prefix = f"[TLE] The Wikipedia page is about {page_title}. The Wikipedia section is about {section}."
    if caption:
        prefix += f" The table caption is about {caption}."

    col_line = str(table.get("__mt2ti_cached_col_line", ""))
    rows_joined = str(table.get("__mt2ti_cached_rows_joined", ""))
    if not col_line or not rows_joined:
        col_line = " | ".join(header)
        start = 1
        end = len(text) - 1
        rows_str: List[str] = []
        for r in range(start, end + 1):
            row = text[r]
            row_vals = row[: len(header)]
            if len(row_vals) < len(header):
                row_vals = row_vals + [""] * (len(header) - len(row_vals))
            content = "| " + " | ".join(row_vals) + " |"
            rows_str.append(f"row {r}: {content}")
        rows_joined = " [SEP] ".join(rows_str)
        table["__mt2ti_cached_col_line"] = col_line
        table["__mt2ti_cached_rows_joined"] = rows_joined

    return f"{prefix} [TAB] col: | {col_line} | {rows_joined}"

