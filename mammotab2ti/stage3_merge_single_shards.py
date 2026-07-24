"""
Merge stage2 single-shard outputs into two JSONL files.

Example:
  python -m mammotab2ti.stage3_merge_single_shards
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Iterable


def _iter_shard_paths(
    shard_dir: Path,
    pattern: str,
    num_shards: int,
) -> Iterable[Path]:
    for shard_id in range(num_shards):
        yield shard_dir / pattern.format(shard_id=shard_id)


class _ProgressBar:
    def __init__(self, total: int, label: str) -> None:
        self.total = max(total, 1)
        self.label = label
        self.current = 0
        self._last_draw_time = 0.0
        self._draw(force=True)

    def update(self, increment: int = 1) -> None:
        self.current += increment
        now = time.time()
        if now - self._last_draw_time >= 0.1 or self.current >= self.total:
            self._draw(force=False)

    def close(self) -> None:
        self.current = self.total
        self._draw(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _draw(self, force: bool) -> None:
        self._last_draw_time = time.time()
        ratio = min(self.current / self.total, 1.0)
        width = 30
        done = int(width * ratio)
        bar = "#" * done + "-" * (width - done)
        percent = ratio * 100
        sys.stderr.write(
            f"\r{self.label}: [{bar}] {percent:6.2f}% ({self.current}/{self.total})"
        )
        if force:
            sys.stderr.flush()


_CANDIDATE_PREFIX = "The referent entity candidates are: "
_QUESTION_SUFFIX = " What is the correct referent entity for the entity mention"
_CANDIDATE_TEXT_PATTERN = re.compile(
    r"<.*? \[DESCRIPTION\] .*? \[TYPE\] .*?>",
    flags=re.DOTALL,
)


def _count_non_empty_lines(input_paths: Iterable[Path]) -> int:
    line_count = 0
    for in_path in input_paths:
        if not in_path.exists():
            raise FileNotFoundError(f"missing shard file: {in_path}")
        with open(in_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if line.strip():
                    line_count += 1
    return line_count


def _iter_non_empty_jsonl(path: Path) -> Iterable[tuple[int, str, dict]]:
    with open(path, "r", encoding="utf-8") as in_f:
        for line_no, line in enumerate(in_f, 1):
            if not line.strip():
                continue
            yield line_no, line.rstrip("\n"), json.loads(line)


def _expected_candidate_count(audit_obj: dict) -> int:
    raw_count = audit_obj.get("candidate_count")
    if isinstance(raw_count, int) and raw_count >= 0:
        return raw_count
    candidate_qids = audit_obj.get("candidate_qids")
    if isinstance(candidate_qids, list):
        return len(candidate_qids)
    return -1


def _extract_candidates_by_expected_count(
    question: str, expected_count: int
) -> tuple[int, int, list[str], bool]:
    prefix_idx = question.find(_CANDIDATE_PREFIX)
    if prefix_idx < 0:
        return -1, -1, [], False
    start_idx = prefix_idx + len(_CANDIDATE_PREFIX)
    end_idx = question.find(_QUESTION_SUFFIX, start_idx)
    if end_idx < 0:
        return -1, -1, [], False

    segment = question[start_idx:end_idx]
    if expected_count == 0:
        return start_idx, end_idx, [], segment.strip() == ""

    candidates = [m.group(0).strip() for m in _CANDIDATE_TEXT_PATTERN.finditer(segment)]
    if expected_count > 0:
        return start_idx, end_idx, candidates, len(candidates) == expected_count
    return start_idx, end_idx, candidates, len(candidates) > 0 or segment.strip() == ""


def _rewrite_candidates_for_record(
    record: dict, audit_obj: dict, rng: random.Random
) -> tuple[dict, bool, bool, bool, bool]:
    question = record.get("question")
    gold = record.get("output")
    if not isinstance(question, str):
        return record, False, False, False, False

    expected_count = _expected_candidate_count(audit_obj)
    start_idx, end_idx, candidates, parsed_ok = _extract_candidates_by_expected_count(
        question, expected_count
    )
    if not parsed_ok:
        return record, False, False, False, False

    added_gold = False
    normalized_gold = gold.strip() if isinstance(gold, str) else ""
    should_add_gold = bool(normalized_gold) and normalized_gold.upper() != "NIL"
    if should_add_gold and normalized_gold not in candidates:
        candidates.append(normalized_gold)
        added_gold = True

    shuffled = False
    if len(candidates) > 1:
        rng.shuffle(candidates)
        shuffled = True

    record["question"] = question[:start_idx] + ", ".join(candidates) + question[end_idx:]
    over_limit = len(candidates) > 21
    return record, added_gold, shuffled, True, over_limit


def _merge_data_and_audit_jsonl_files(
    data_paths: Iterable[Path],
    audit_paths: Iterable[Path],
    output_data: Path,
    output_audit: Path,
    seed: int,
) -> tuple[int, int, int, int, int]:
    data_path_list = list(data_paths)
    audit_path_list = list(audit_paths)
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)

    total_lines = _count_non_empty_lines(data_path_list)
    progress = _ProgressBar(total_lines, "merge data/audit")
    rng = random.Random(seed)

    line_count = 0
    added_gold_count = 0
    shuffled_count = 0
    parse_failed_count = 0
    over_limit_count = 0

    with open(output_data, "w", encoding="utf-8") as out_data_f, open(
        output_audit, "w", encoding="utf-8"
    ) as out_audit_f:
        for data_path, audit_path in zip(data_path_list, audit_path_list):
            if not data_path.exists():
                raise FileNotFoundError(f"missing shard file: {data_path}")
            if not audit_path.exists():
                raise FileNotFoundError(f"missing shard file: {audit_path}")

            data_iter = iter(_iter_non_empty_jsonl(data_path))
            audit_iter = iter(_iter_non_empty_jsonl(audit_path))
            while True:
                data_item = next(data_iter, None)
                audit_item = next(audit_iter, None)
                if data_item is None and audit_item is None:
                    break
                if data_item is None or audit_item is None:
                    raise ValueError(
                        f"line mismatch between shards: data={data_path}, audit={audit_path}"
                    )

                _, _, data_obj = data_item
                _, audit_raw, audit_obj = audit_item

                data_obj, added_gold, shuffled, parsed_ok, over_limit = (
                    _rewrite_candidates_for_record(data_obj, audit_obj, rng)
                )
                if added_gold:
                    added_gold_count += 1
                if shuffled:
                    shuffled_count += 1
                if not parsed_ok:
                    parse_failed_count += 1
                if over_limit:
                    over_limit_count += 1

                out_data_f.write(json.dumps(data_obj, ensure_ascii=False) + "\n")
                out_audit_f.write(audit_raw + "\n")
                line_count += 1
                progress.update()

    progress.close()
    return (
        line_count,
        added_gold_count,
        shuffled_count,
        parse_failed_count,
        over_limit_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge stage2 data/audit shard JSONL files into two final JSONL files."
    )
    parser.add_argument(
        "--shard_dir",
        type=str,
        default="/DATA1/khli/mammotab/modified_mammotab/stage2_single_shards",
        help="Directory containing stage2 shard JSONL files",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        default="/DATA1/khli/mammotab/modified_mammotab/stage3_single_shards_merge.jsonl",
        help="Merged data JSONL output path",
    )
    parser.add_argument(
        "--output_audit",
        type=str,
        default="/DATA1/khli/mammotab/modified_mammotab/stage3_single_shards_audit_merge.jsonl",
        help="Merged audit JSONL output path",
    )
    parser.add_argument("--num_shards", type=int, default=24, help="Total shard count")
    parser.add_argument(
        "--data_pattern",
        type=str,
        default="ent_link_train_generated.shard_{shard_id}.jsonl",
        help="Filename pattern for data shard files",
    )
    parser.add_argument(
        "--audit_pattern",
        type=str,
        default="ent_link_train_generated.audit.shard_{shard_id}.jsonl",
        help="Filename pattern for audit shard files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle question.candidates in data records",
    )
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir).expanduser().resolve()
    output_data = Path(args.output_data).expanduser().resolve()
    output_audit = Path(args.output_audit).expanduser().resolve()

    if not shard_dir.exists() or not shard_dir.is_dir():
        raise ValueError(f"shard_dir not found or not a directory: {shard_dir}")
    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {args.num_shards}")

    data_paths = list(_iter_shard_paths(shard_dir, args.data_pattern, args.num_shards))
    audit_paths = list(_iter_shard_paths(shard_dir, args.audit_pattern, args.num_shards))

    (
        data_lines,
        added_gold_count,
        shuffled_count,
        parse_failed_count,
        over_limit_count,
    ) = _merge_data_and_audit_jsonl_files(
        data_paths, audit_paths, output_data, output_audit, args.seed
    )
    audit_lines = data_lines

    print(f"merged data lines: {data_lines}")
    print(f"added output-to-candidates count: {added_gold_count}")
    print(f"shuffled candidates count: {shuffled_count}")
    print(f"question parse failed count: {parse_failed_count}")
    print(f"candidates > 21 count: {over_limit_count}")
    print(f"merged audit lines: {audit_lines}")
    print(f"data output: {output_data}")
    print(f"audit output: {output_audit}")


if __name__ == "__main__":
    main()
