#!/usr/bin/env python3
"""将 MammoTab JSONL 测试数据转换为 JSON 数组格式。

用法：
  python3 -m el.mammotab_testdata_reformat --input /DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.jsonl --output /DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT = Path(
    "/DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.jsonl"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def jsonl_to_json(input_path: Path, output_path: Path | None = None) -> Path:
    input_path = input_path.expanduser().resolve()
    if output_path is None:
        output_path = input_path.with_suffix(".json")
    else:
        output_path = output_path.expanduser().resolve()

    records = load_jsonl(input_path)
    save_json(records, output_path)
    print(f"Converted {len(records)} records: {input_path} -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MammoTab JSONL to JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入 JSONL 路径（默认: {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSON 路径（默认: 与输入同目录，扩展名改为 .json）",
    )
    args = parser.parse_args()
    jsonl_to_json(args.input, args.output)


if __name__ == "__main__":
    main()
