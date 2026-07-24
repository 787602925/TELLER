"""
Stage 1: Scan Mammotab JSON files and extract mention-level jobs.

Example:
  python -m mammotab2ti.stage1_extract_jobs \
    --input_dir /path/to/mammotab_jsons \
    --output_jobs /path/to/mammotab_jobs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from mammotab2ti.mammotab_job_utils import extract_jobs_from_table, load_table_json


def _iter_json_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.json")):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract mention jobs from Mammotab JSON files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="~/DATA/mammotab/mammotab_dataset_semtab/json",
        help="Directory containing Mammotab JSON files",
    )
    parser.add_argument(
        "--output_jobs",
        type=str,
        default="~/DATA/mammotab/modified_mammotab/mammotab_jobs.jsonl",
        help="Output JSONL path for extracted jobs",
    )
    parser.add_argument("--max_files", type=int, default=0, help="Only process first N files (0 means all)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_jobs = Path(args.output_jobs).expanduser().resolve()
    output_jobs.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir not found or not a directory: {input_dir}")

    file_count = 0
    job_count = 0
    bad_count = 0

    with open(output_jobs, "w", encoding="utf-8") as out_f:
        for file_path in _iter_json_files(input_dir):
            if args.max_files > 0 and file_count >= args.max_files:
                break
            file_count += 1

            try:
                table = load_table_json(str(file_path))
                jobs = extract_jobs_from_table(table, str(file_path))
            except Exception:
                bad_count += 1
                continue

            for job in jobs:
                out_f.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")
                job_count += 1

            if file_count % 2000 == 0:
                print(f"processed_files={file_count}, jobs={job_count}, bad_files={bad_count}")

    print(f"done: files={file_count}, jobs={job_count}, bad_files={bad_count}")
    print(str(output_jobs))


if __name__ == "__main__":
    main()

