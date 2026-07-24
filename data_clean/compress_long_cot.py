"""
Compress over-long CoT outputs (rule-based, extractive) so that valuable but
lengthy samples can be kept for training instead of being dropped by the
`--max_output_tokens` filter in `el/sft_CoT.py`.

What it does
------------
For each sample whose `output` token length exceeds `--compress-threshold-tokens`
(default 500), it rewrites ONLY the reasoning inside `<think>...</think>` by
selecting a subset of the ORIGINAL sentences (no paraphrasing, no new text),
dropping candidate-list dumps and redundant/duplicated sentences, until the
whole output fits under `--target-output-tokens`. The final entity emitted after
`</think>` is preserved verbatim.

Samples at or below the threshold are copied through unchanged.

Every rewrite must pass quality gates (final entity unchanged, well-formed
`<think>...</think>\n<entity>` structure, length under budget, low repetition).
If a sample cannot be compressed safely, it is handled per `--on-fail`
(default: drop it, since the training filter would drop it anyway).

Usage
-----
  # dry-run: only report + samples, no dataset written
  python3 data_clean/compress_long_cot.py \
    --in-path "/DATA1/khli/t&m/merged_CoTSFT_all&all_filtered.json" \
    --dry-run

  # write the compressed dataset
  python3 data_clean/compress_long_cot.py \
    --in-path "/DATA1/khli/t&m/merged_CoTSFT_all&all_filtered.json" \
    --out-path "/DATA1/khli/t&m/merged_CoTSFT_all&all_filtered_cot_compressed.json"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------- regex ----------

_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", flags=re.I)
_THINK_CLOSE = "</think>"
_ENTITY_RE = re.compile(
    r"<\s*[^<>]+?\s*\[(?:DESC|DESCRIPTION)\]\s*[^<>]*?\s*\[TYPE\]\s*[^<>]*?>",
    flags=re.I | re.S,
)
_FINAL_ANSWER_LINE_RE = re.compile(r"^\s*(final answer|final|answer|答案|最终答案)\s*[:：]", flags=re.I)

_EVIDENCE_PATTERNS = [
    re.compile(r"\brow\b", flags=re.I),
    re.compile(r"\bcolumn\b", flags=re.I),
    re.compile(r"\bcaption\b", flags=re.I),
    re.compile(r"\bcontext\b", flags=re.I),
    re.compile(r"\bsection\b", flags=re.I),
    re.compile(r"\bpage\b", flags=re.I),
    re.compile(r"\btype\b", flags=re.I),
    re.compile(r"\bdescription\b", flags=re.I),
    re.compile(r"\bmatch(es|ed|ing)?\b", flags=re.I),
    re.compile(r"\bcontext\b", flags=re.I),
]


# ---------- token counting ----------


class TokenCounter:
    """Count tokens with the training tokenizer; fall back to a word estimate."""

    def __init__(self, tokenizer_path: str) -> None:
        self._tok = None
        self._encode: Callable[[str], List[int]]
        self._decode: Optional[Callable[[List[int]], str]] = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
                self._encode = lambda t: self._tok.encode(t, add_special_tokens=False)
                self._decode = lambda ids: self._tok.decode(ids, skip_special_tokens=True)
                print(f"[tokenizer] loaded {tokenizer_path}")
                return
            except Exception as e:  # pragma: no cover - environment dependent
                print(f"[tokenizer] failed to load {tokenizer_path} ({e}); using word estimate.")
        # Fallback: rough estimate (~1.3 tokens/word). Truncation disabled.
        self._encode = lambda t: [0] * max(1, int(len(t.split()) * 1.3)) if t.strip() else []

    def count(self, text: str) -> int:
        return len(self._encode(text or ""))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Keep only the first `max_tokens` tokens of `text` (best effort)."""
        if self._tok is None or self._decode is None:
            # No real tokenizer: approximate by words.
            words = (text or "").split()
            keep = max(1, int(max_tokens / 1.3))
            return " ".join(words[:keep])
        ids = self._encode(text or "")
        if len(ids) <= max_tokens:
            return text
        return self._decode(ids[:max_tokens]).strip()


# ---------- parsing ----------


@dataclass
class ParsedOutput:
    think_inner: str
    final_entity: str


def parse_output(output: str) -> Optional[ParsedOutput]:
    """Split output into (think reasoning, final entity). Return None if malformed."""
    if not isinstance(output, str) or not output.strip():
        return None
    lower = output.lower()
    close_idx = lower.rfind(_THINK_CLOSE)
    if close_idx == -1:
        return None

    post = output[close_idx + len(_THINK_CLOSE):]
    ent_match = _ENTITY_RE.search(post)
    if not ent_match:
        return None
    final_entity = ent_match.group(0).strip()

    pre = output[:close_idx]
    open_match = _THINK_OPEN_RE.search(pre)
    think_inner = pre[open_match.end():] if open_match else pre
    return ParsedOutput(think_inner=think_inner.strip(), final_entity=final_entity)


def build_output(think_inner: str, final_entity: str) -> str:
    think_inner = (think_inner or "").strip()
    return f"<think>\n{think_inner}\n</think>\n{final_entity}"


# ---------- sentence handling ----------


def split_sentences(text: str) -> List[str]:
    # Split on sentence terminators and newlines, keep non-empty trimmed pieces.
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


_LIST_MARKER_RE = re.compile(r"^\s*[-*•]\s")


def is_candidate_dump(sentence: str) -> bool:
    """A sentence that quotes/enumerates candidates rather than reasoning.

    We drop any sentence that copies a full ``<Name [DESC/DESCRIPTION] ... [TYPE] ...>``
    candidate verbatim, since keeping such enumerations both inflates length and
    teaches the model to list candidates (the exact degenerate behavior we want to
    suppress). The chosen final entity is preserved separately, after </think>.
    """
    if _ENTITY_RE.search(sentence):
        return True
    marker_hits = len(re.findall(r"\[(?:DESC|DESCRIPTION|TYPE)\]", sentence, flags=re.I))
    if marker_hits >= 2:
        return True
    if _LIST_MARKER_RE.match(sentence) and "<" in sentence:
        return True
    return False


def is_final_answer_line(sentence: str) -> bool:
    return bool(_FINAL_ANSWER_LINE_RE.search(sentence))


def evidence_score(sentence: str) -> int:
    return sum(1 for p in _EVIDENCE_PATTERNS if p.search(sentence))


def _norm_key(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.lower()).strip()


def dedupe_sentences(sentences: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in sentences:
        key = _norm_key(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def repeat_ratio(text: str, n: int) -> float:
    words = (text or "").split()
    if len(words) <= n:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


# ---------- compression ----------


@dataclass
class CompressResult:
    output: str
    ok: bool
    reason: str
    before_tokens: int
    after_tokens: int


def normalize_entity(entity: str) -> str:
    s = re.sub(r"\[\s*description\s*\]", "[DESC]", entity or "", flags=re.I)
    s = re.sub(r"\[\s*desc\s*\]", "[DESC]", s, flags=re.I)
    s = re.sub(r"\[\s*type\s*\]", "[TYPE]", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip().strip("<>").strip()


def compress_one(
    output: str,
    counter: TokenCounter,
    *,
    target_tokens: int,
    min_sentences: int,
    drop_final_answer: bool,
    repeat_ngram: int,
    max_repeat_ratio: float,
) -> CompressResult:
    before_tokens = counter.count(output)
    parsed = parse_output(output)
    if parsed is None:
        return CompressResult(output, False, "unparsable", before_tokens, before_tokens)

    entity = parsed.final_entity
    # Fixed cost of the entity + <think></think> scaffolding. Leave a small
    # margin because re-tokenizing the joined string can add a few boundary tokens.
    scaffold_tokens = counter.count(build_output("", entity))
    reasoning_budget = max(16, target_tokens - scaffold_tokens - 8)

    sentences = split_sentences(parsed.think_inner)
    kept_pool: List[str] = []
    for s in sentences:
        if is_candidate_dump(s):
            continue
        if drop_final_answer and is_final_answer_line(s):
            continue
        kept_pool.append(s)
    kept_pool = dedupe_sentences(kept_pool)

    # Prefer higher-evidence sentences but preserve original order for coherence.
    if kept_pool:
        indexed = list(enumerate(kept_pool))
        ranked = sorted(indexed, key=lambda t: (-evidence_score(t[1]), t[0]))
    else:
        # Nothing survived filtering: fall back to first original sentence(s).
        fallback = dedupe_sentences(sentences) or [parsed.think_inner]
        indexed = list(enumerate(fallback))
        ranked = indexed

    # Greedily select by rank, then re-sort selected by original order.
    selected_idx: List[int] = []
    running = ""
    for orig_i, sent in ranked:
        candidate_sents = sorted(selected_idx + [orig_i])
        pool = kept_pool if kept_pool else (dedupe_sentences(sentences) or [parsed.think_inner])
        trial_think = " ".join(pool[i] for i in candidate_sents)
        if counter.count(trial_think) <= reasoning_budget or not selected_idx:
            selected_idx.append(orig_i)
            running = trial_think
        if len(selected_idx) >= min_sentences and counter.count(running) >= reasoning_budget:
            break

    pool = kept_pool if kept_pool else (dedupe_sentences(sentences) or [parsed.think_inner])
    selected_idx = sorted(set(selected_idx))
    compressed_think = " ".join(pool[i] for i in selected_idx).strip()

    # Hard cap: if a single long sentence still blows the budget, truncate tokens.
    if counter.count(compressed_think) > reasoning_budget:
        compressed_think = counter.truncate(compressed_think, reasoning_budget)

    new_output = build_output(compressed_think, entity)
    after_tokens = counter.count(new_output)

    # ----- quality gates -----
    reparsed = parse_output(new_output)
    if reparsed is None:
        return CompressResult(output, False, "rebuild_malformed", before_tokens, after_tokens)
    if normalize_entity(reparsed.final_entity) != normalize_entity(entity):
        return CompressResult(output, False, "entity_changed", before_tokens, after_tokens)
    if not compressed_think.strip():
        return CompressResult(output, False, "empty_reasoning", before_tokens, after_tokens)
    if after_tokens > target_tokens:
        return CompressResult(output, False, "still_too_long", before_tokens, after_tokens)
    if repeat_ratio(compressed_think, repeat_ngram) > max_repeat_ratio:
        return CompressResult(output, False, "repetitive", before_tokens, after_tokens)

    return CompressResult(new_output, True, "ok", before_tokens, after_tokens)


# ---------- io helpers ----------


def _read_items(path: str) -> List[Dict[str, Any]]:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"expected top-level list, got {type(obj).__name__}")
    return obj


def _default_path(in_path: str, suffix: str) -> str:
    p = Path(os.path.expanduser(in_path))
    return str(p.with_name(f"{p.stem}{suffix}"))


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Extractively compress over-long CoT outputs.")
    parser.add_argument("--in-path", required=True, help="Input dataset (json list).")
    parser.add_argument("--out-path", default="", help="Output dataset path (json).")
    parser.add_argument("--report-path", default="", help="Report path (json).")
    parser.add_argument("--sample-path", default="", help="Before/after samples path (jsonl).")
    parser.add_argument(
        "--tokenizer",
        default="/home/khli/tableLlama/result_CoT/checkpoint-1000",
        help="Tokenizer dir/name used to measure output tokens (matches training).",
    )
    parser.add_argument(
        "--compress-threshold-tokens",
        type=int,
        default=500,
        help="Only compress samples whose output token length exceeds this.",
    )
    parser.add_argument(
        "--target-output-tokens",
        type=int,
        default=400,
        help="Aim to bring compressed output at or below this token length.",
    )
    parser.add_argument("--min-think-sentences", type=int, default=1)
    parser.add_argument(
        "--keep-final-answer-line",
        action="store_true",
        help="Keep 'Final answer: ...' lines inside <think> (default: drop them).",
    )
    parser.add_argument("--repeat-ngram", type=int, default=8)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.3)
    parser.add_argument(
        "--on-fail",
        choices=("drop", "keep-original"),
        default="drop",
        help="What to do with over-long samples that fail compression gates.",
    )
    parser.add_argument("--max-sample-per-tag", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.target_output_tokens > args.compress_threshold_tokens:
        raise SystemExit("--target-output-tokens must be <= --compress-threshold-tokens")

    in_path = os.path.expanduser(args.in_path)
    out_path = os.path.expanduser(args.out_path) if args.out_path else _default_path(in_path, "_cot_compressed.json")
    report_path = os.path.expanduser(args.report_path) if args.report_path else _default_path(in_path, "_cot_compressed_report.json")
    sample_path = os.path.expanduser(args.sample_path) if args.sample_path else _default_path(in_path, "_cot_compressed_samples.jsonl")

    items = _read_items(in_path)
    total = len(items)
    if total == 0:
        raise SystemExit("empty dataset")

    counter = TokenCounter(args.tokenizer)

    tag_counter: Counter[str] = Counter()
    fail_reason_counter: Counter[str] = Counter()
    saved_before = 0
    saved_after = 0
    out_items: List[Dict[str, Any]] = []
    samples: Dict[str, List[Dict[str, Any]]] = {"compressed": [], "failed": []}

    for idx, item in enumerate(items):
        output = item.get("output")
        out_tokens = counter.count(output if isinstance(output, str) else "")

        if out_tokens <= args.compress_threshold_tokens:
            tag_counter["unchanged"] += 1
            out_items.append(item)
            continue

        result = compress_one(
            output,
            counter,
            target_tokens=args.target_output_tokens,
            min_sentences=args.min_think_sentences,
            drop_final_answer=not args.keep_final_answer_line,
            repeat_ngram=args.repeat_ngram,
            max_repeat_ratio=args.max_repeat_ratio,
        )

        if result.ok:
            tag_counter["compressed"] += 1
            saved_before += result.before_tokens
            saved_after += result.after_tokens
            new_item = dict(item)
            new_item["output"] = result.output
            new_item["output_compressed_from_tokens"] = result.before_tokens
            new_item["output_compressed_to_tokens"] = result.after_tokens
            out_items.append(new_item)
            if len(samples["compressed"]) < args.max_sample_per_tag:
                samples["compressed"].append(
                    {
                        "index": idx,
                        "before_tokens": result.before_tokens,
                        "after_tokens": result.after_tokens,
                        "before_output": output[:1600],
                        "after_output": result.output,
                    }
                )
        else:
            tag_counter["failed"] += 1
            fail_reason_counter[result.reason] += 1
            if len(samples["failed"]) < args.max_sample_per_tag:
                samples["failed"].append(
                    {
                        "index": idx,
                        "reason": result.reason,
                        "before_tokens": result.before_tokens,
                        "after_tokens": result.after_tokens,
                        "before_output": (output or "")[:1600],
                    }
                )
            if args.on_fail == "keep-original":
                out_items.append(item)
            # else: drop (do not append)

        if (idx + 1) % 2000 == 0:
            print(f"[progress] {idx + 1}/{total} scanned; tags={dict(tag_counter)}")

    dropped = total - len(out_items)
    avg_before = (saved_before / tag_counter["compressed"]) if tag_counter["compressed"] else 0.0
    avg_after = (saved_after / tag_counter["compressed"]) if tag_counter["compressed"] else 0.0

    report = {
        "input_path": in_path,
        "output_path": out_path,
        "total": total,
        "kept": len(out_items),
        "dropped": dropped,
        "tag_counts": dict(tag_counter),
        "fail_reasons": dict(fail_reason_counter),
        "compress_threshold_tokens": args.compress_threshold_tokens,
        "target_output_tokens": args.target_output_tokens,
        "on_fail": args.on_fail,
        "compressed_avg_tokens_before": round(avg_before, 1),
        "compressed_avg_tokens_after": round(avg_after, 1),
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as f:
        for tag in ("compressed", "failed"):
            for row in samples[tag]:
                f.write(json.dumps({"tag": tag, **row}, ensure_ascii=False) + "\n")

    print(f"report: {report_path}")
    print(f"samples: {sample_path}")
    print(
        "summary: "
        f"total={total}, kept={len(out_items)}, dropped={dropped}, "
        f"tags={dict(tag_counter)}, "
        f"avg_tokens {avg_before:.0f} -> {avg_after:.0f}"
    )
    if fail_reason_counter:
        print(f"fail_reasons: {dict(fail_reason_counter)}")

    if not args.dry_run:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_items, f, ensure_ascii=False)
        print(f"compressed_data: {out_path}")


if __name__ == "__main__":
    main()
