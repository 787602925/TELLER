"""
Filter low-information CoT samples so that the reasoning (<think>...</think>)
kept for SFT is genuinely useful.

This script ONLY removes low-information / low-quality reasoning. It does NOT
touch overly-long samples (token-length trimming / compression is handled by a
separate script), so length alone is never a reason to drop a sample here.

Quality model
-------------
For an entity-linking CoT sample the useful reasoning is the one that grounds
the choice in *context evidence* (page/section/caption, row/column, year, type,
candidate description) and *discriminates* the correct candidate from the close
distractors. We therefore score each <think> along two axes:

  * positive credits  : contrastive/discriminative reasoning + concrete evidence
  * low-info penalties : filler-only reasoning, no evidence, candidate-list
                         copying, answer leakage, rambling/circular text, and
                         reasoning whose own final entity contradicts the output

net_score = penalties - credits. Higher = worse. Samples above the drop
threshold (or matching a hard-drop rule) are removed.

Usage examples:
  # 1) Dry-run only (report + samples; no filtered dataset written)
  python3 data_clean/filter_low_info_cot.py \
    --in-path "/DATA1/khli/t&m/merged_CoTSFT_all&all.json" \
    --dry-run

  # 2) Write filtered dataset (drop hard + low-info samples)
  python3 data_clean/filter_low_info_cot.py \
    --in-path "/DATA1/khli/t&m/merged_CoTSFT_all&all.json" \
    --out-path "/DATA1/khli/t&m/merged_CoTSFT_all&all_filtered.json" \
    --drop-level all
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.I | re.S)
# Candidate list lives in the question; a candidate looks like
#   <Name [DESCRIPTION] ... [TYPE] ...>. We match the "[DESC]" / "[DESCRIPTION]"
# marker (both spellings appear after normalization) to count candidate copying.
_CANDIDATE_RE = re.compile(r"<\s*([^<>\[]+?)\s*\[(?:DESC|DESCRIPTION)\]", flags=re.I)
_DESC_MARKER_RE = re.compile(r"\[(?:DESC|DESCRIPTION)\]", flags=re.I)
# A well-formed final entity token: <Name [DESC(RIPTION)] ... [TYPE] ...>.
_ENTITY_RE = re.compile(
    r"<\s*[^<>]+?\s*\[(?:DESC|DESCRIPTION)\]\s*[^<>]*?\s*\[TYPE\]\s*[^<>]*?>",
    flags=re.I | re.S,
)

# Filler / template phrases: reasoning that only announces it will answer
# without actually reasoning. Strong low-info signal.
_TEMPLATE_PATTERNS = [
    re.compile(r"now output after reasoning", flags=re.I),
    re.compile(r"output reasoning within", flags=re.I),
    re.compile(r"we('| wi)ll output (that|it|the)", flags=re.I),
    re.compile(r"we should output that (one|entity)", flags=re.I),
    re.compile(r"thus,?\s+the output should be that entity", flags=re.I),
    re.compile(r"i'?ll (just )?(select|choose|output|pick) (that|it|this)( candidate| entity| one)?\.?\s*$", flags=re.I),
    re.compile(r"looking at candidates\b", flags=re.I),
    re.compile(r"因此选择该实体"),
    re.compile(r"下面给出答案"),
]

# Concrete context-evidence terms. Grounded reasoning references these.
_EVIDENCE_PATTERNS = [
    re.compile(r"\bcolumn\b", flags=re.I),
    re.compile(r"\brow\b", flags=re.I),
    re.compile(r"\btable\b", flags=re.I),
    re.compile(r"\bcontext\b", flags=re.I),
    re.compile(r"\bsection\b", flags=re.I),
    re.compile(r"\bpage\b", flags=re.I),
    re.compile(r"\bcaption\b", flags=re.I),
    re.compile(r"\byear\b", flags=re.I),
    re.compile(r"\btype\b", flags=re.I),
    re.compile(r"\bdescription\b", flags=re.I),
    re.compile(r"候选"),
    re.compile(r"列"),
    re.compile(r"行"),
    re.compile(r"表格"),
    re.compile(r"上下文"),
    re.compile(r"章节|页面|年份|类型|描述"),
]

# Justification links: the reasoning connects evidence to the chosen entity
# ("... so it refers to ...", "matches the context", "because ..."). Their
# presence is what separates a real explanation from a bare restatement/answer.
_JUSTIFY_PATTERNS = [
    re.compile(r"\bbecause\b", flags=re.I),
    re.compile(r"\bsince\b", flags=re.I),
    re.compile(r"\bso\b", flags=re.I),
    re.compile(r"\bthus\b", flags=re.I),
    re.compile(r"\btherefore\b", flags=re.I),
    re.compile(r"refers?\s+to", flags=re.I),
    re.compile(r"\bindicat(e|es|ing)\b", flags=re.I),
    re.compile(r"\bcorrespond(s|ing)?\b", flags=re.I),
    re.compile(r"consistent with", flags=re.I),
    re.compile(r"\bmatch(es|ing)?\b", flags=re.I),
    re.compile(r"\balign(s|ed)?\b", flags=re.I),
    re.compile(r"\bfits?\b", flags=re.I),
    re.compile(r"points? to", flags=re.I),
    re.compile(r"\bsuggests?\b", flags=re.I),
    re.compile(r"given that", flags=re.I),
    re.compile(r"因为|所以|因此|对应|匹配|表明|说明"),
]

# Explicit conclusion cues; used to precisely detect answer inconsistency
# (reasoning concludes one entity but the sample outputs another).
_CONCLUSION_CUE_RE = re.compile(
    r"(?:final answer|the (?:correct |right )?(?:answer|entity|referent)(?:\s+(?:is|should be))|"
    r"so (?:the )?(?:answer|correct)(?:\s+is)?|therefore[, ]|thus[, ]|i(?:'| a)?ll (?:choose|select|go with)|"
    r"最终答案|正确(?:答案|实体)(?:是|为)?)",
    flags=re.I,
)

# Contrastive / discriminative reasoning: comparing candidates, ruling out
# distractors, justifying the pick. This is the hallmark of the *best* reasoning.
_CONTRAST_PATTERNS = [
    re.compile(r"other candidate", flags=re.I),
    re.compile(r"\bunlike\b", flags=re.I),
    re.compile(r"\bwhereas\b", flags=re.I),
    re.compile(r"rather than", flags=re.I),
    re.compile(r"instead of", flags=re.I),
    re.compile(r"distinguish", flags=re.I),
    re.compile(r"rule[sd]?\s+out", flags=re.I),
    re.compile(r"eliminat(e|es|ed|ing)", flags=re.I),
    re.compile(r"exclud(e|es|ed|ing)", flags=re.I),
    re.compile(r"not relevant", flags=re.I),
    re.compile(r"(does not|doesn'?t)\s+(match|fit|refer|apply)", flags=re.I),
    re.compile(r"(are|is)\s+not\s+(consistent|relevant|correct)", flags=re.I),
    re.compile(r"(most|more)\s+specific", flags=re.I),
    re.compile(r"best\s+match", flags=re.I),
    re.compile(r"the only candidate", flags=re.I),
    re.compile(r"only one (that )?(match|fit)", flags=re.I),
    re.compile(r"narrow", flags=re.I),
    re.compile(r"排除|而不是|区别|相比|只有一个"),
]

# Reasoning that references being handed / checking against the answer. Such
# post-hoc rationalization is not genuine reasoning we want to imitate.
_LEAKAGE_PATTERNS = [
    re.compile(r"\bverif(y|ies|ied|ication)\b", flags=re.I),
    re.compile(r"the solution (says|check|is)", flags=re.I),
    re.compile(r"must construct reasoning", flags=re.I),
    re.compile(r"reasoning that leads to (it|that|the)", flags=re.I),
    re.compile(r"given the (verif|solution|answer)", flags=re.I),
    re.compile(r"the (verified|provided|given|gold|correct)\s+(answer|entity|output|label)", flags=re.I),
    re.compile(r"i must comply", flags=re.I),
    re.compile(r"predetermined", flags=re.I),
    re.compile(r"answer key", flags=re.I),
    re.compile(r"as per the (solution|verification|answer)", flags=re.I),
]

# Circular / rambling self-correction markers.
_RAMBLE_PATTERNS = [
    re.compile(r"\bbut wait\b", flags=re.I),
    re.compile(r"\blet me reconsider\b", flags=re.I),
    re.compile(r"\blet'?s reconsider\b", flags=re.I),
    re.compile(r"\breconsider\b", flags=re.I),
    re.compile(r"\bon second thought\b", flags=re.I),
    re.compile(r"\bhold on\b", flags=re.I),
    re.compile(r"\bhmm+\b", flags=re.I),
    re.compile(r"\bwait,\s", flags=re.I),
    re.compile(r"\b(i'?m|i am) not sure\b", flags=re.I),
]

_REFUSAL_PATTERNS = [
    re.compile(r"^\s*(sorry|i am sorry|as an ai language model)", flags=re.I),
    re.compile(r"\b(can[ ]?not|can't)\s+(assist|help)\b", flags=re.I),
]


def _normalize_entity(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


@dataclass
class Decision:
    score: int
    tag: str  # keep | low_info | hard_drop
    reasons: List[str]
    features: Dict[str, Any]


def _read_items(path: str) -> List[Dict[str, Any]]:
    path = os.path.expanduser(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                if not isinstance(item, dict):
                    raise ValueError(f"jsonl line {line_no} is not an object")
                items.append(item)
        return items

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"expected top-level list, got {type(obj).__name__}")
    for i, x in enumerate(obj):
        if not isinstance(x, dict):
            raise ValueError(f"item {i} is not an object")
    return obj


def _extract_output(item: Dict[str, Any]) -> str:
    for key in ("output", "response", "answer", "chosen", "text"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_question(item: Dict[str, Any]) -> str:
    for key in ("question", "instruction", "prompt", "input"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _split_think(text: str) -> Tuple[str, str, bool]:
    """Return (think_body, post_think_text, has_think_block)."""
    m = _THINK_RE.search(text or "")
    if not m:
        return "", (text or "").strip(), False
    think = (m.group(1) or "").strip()
    post = (text or "")[m.end():].strip()
    return think, post, True


def _extract_candidate_names(question: str) -> List[str]:
    names = []
    for m in _CANDIDATE_RE.finditer(question or ""):
        name = (m.group(1) or "").strip()
        if len(name) >= 2:
            names.append(name)
    return names


def _count_pattern_hits(text: str, patterns: Iterable[re.Pattern[str]]) -> int:
    return sum(1 for p in patterns if p.search(text or ""))


def _final_output_entity(post_think: str, output_text: str) -> str:
    """The entity the sample actually outputs (the SFT target answer)."""
    m = _ENTITY_RE.search(post_think or "")
    if m:
        return m.group(0)
    # Fallback: last entity token anywhere in the output.
    matches = _ENTITY_RE.findall(output_text or "")
    return matches[-1] if matches else ""


def _concluded_entity(think_text: str) -> str:
    """Entity the reasoning explicitly concludes with.

    We anchor on the *last* conclusion cue ("final answer", "the correct entity
    is", ...) and take the first well-formed entity token after it. This avoids
    the false positives of "last entity token in think" when the reasoning is
    merely enumerating candidate distractors.
    """
    cues = list(_CONCLUSION_CUE_RE.finditer(think_text or ""))
    if not cues:
        return ""
    tail = think_text[cues[-1].start():]
    m = _ENTITY_RE.search(tail)
    return m.group(0) if m else ""


def _analyze_one(
    item: Dict[str, Any],
    *,
    think_min_chars: int,
    drop_score_threshold: int,
) -> Decision:
    output_text = _extract_output(item)
    think_text, post_think, has_think = _split_think(output_text)
    question = _extract_question(item)

    think_chars = len(think_text)

    template_hits = _count_pattern_hits(think_text, _TEMPLATE_PATTERNS)
    evidence_hits = _count_pattern_hits(think_text, _EVIDENCE_PATTERNS)
    justify_hits = _count_pattern_hits(think_text, _JUSTIFY_PATTERNS)
    contrast_hits = _count_pattern_hits(think_text, _CONTRAST_PATTERNS)
    leakage_hits = _count_pattern_hits(think_text, _LEAKAGE_PATTERNS)
    ramble_hits = _count_pattern_hits(think_text, _RAMBLE_PATTERNS)
    refusal_hits = _count_pattern_hits(output_text, _REFUSAL_PATTERNS)

    # Candidate-list copying: the prompt forbids restating candidates. Counting
    # the "[DESC]" markers inside think detects verbatim candidate dumps.
    desc_copy_count = len(_DESC_MARKER_RE.findall(think_text))

    # "Meaningful" reasoning = grounded in context evidence AND contains an
    # explicit justification linking that evidence to the choice. Bare
    # restatements ("we need to identify X") and bare answers ("thus the entity
    # is <...>") fail one or both and are the core low-info case.
    is_grounded = evidence_hits >= 1
    is_justified = justify_hits >= 1 or contrast_hits >= 1
    is_meaningful = is_grounded and is_justified

    # Answer consistency, anchored on an explicit conclusion cue to avoid
    # flagging reasoning that merely enumerates candidate distractors.
    final_entity = _final_output_entity(post_think, output_text)
    concluded = _concluded_entity(think_text)
    answer_inconsistent = bool(
        final_entity
        and concluded
        and _normalize_entity(concluded) != _normalize_entity(final_entity)
    )

    score = 0
    reasons: List[str] = []

    # ---- low-info penalties (higher score = worse) --------------------------
    if not is_grounded:
        score += 3
        reasons.append("no_evidence_terms")
    if not is_justified:
        score += 3
        reasons.append("no_justification")
    if think_chars < think_min_chars:
        score += 2
        reasons.append(f"short_think<{think_min_chars}")
    if template_hits > 0:
        score += min(4, 2 + template_hits - 1)
        reasons.append("template_phrases")
    if desc_copy_count >= 3:
        score += 2 + (1 if desc_copy_count >= 5 else 0)
        reasons.append("copies_candidate_list")
    if leakage_hits > 0:
        score += min(2, leakage_hits)
        reasons.append("answer_leakage")
    if ramble_hits >= 2:
        score += 2 + (1 if ramble_hits >= 4 else 0)
        reasons.append("rambling")
    if answer_inconsistent:
        score += 4
        reasons.append("answer_inconsistent")

    # ---- positive credits (lower score = better) ---------------------------
    if is_meaningful:
        score -= 2
        reasons.append("grounded_and_justified")
    if contrast_hits > 0:
        score -= min(3, 1 + contrast_hits)
        reasons.append("has_contrast")
    if evidence_hits >= 3:
        score -= 1
        reasons.append("rich_evidence")

    # ---- hard drops (unconditionally removed) -------------------------------
    hard_drop = (
        (not has_think)
        or (think_chars == 0)
        or refusal_hits > 0
        or (think_chars < 40 and not is_grounded)
        or (template_hits > 0 and not is_grounded and contrast_hits == 0)
    )

    if hard_drop:
        tag = "hard_drop"
    elif score >= drop_score_threshold:
        tag = "low_info"
    else:
        tag = "keep"

    return Decision(
        score=score,
        tag=tag,
        reasons=reasons,
        features={
            "think_chars": think_chars,
            "evidence_hits": evidence_hits,
            "justify_hits": justify_hits,
            "contrast_hits": contrast_hits,
            "leakage_hits": leakage_hits,
            "ramble_hits": ramble_hits,
            "desc_copy_count": desc_copy_count,
            "is_meaningful": is_meaningful,
            "answer_inconsistent": answer_inconsistent,
            "refusal_hits": refusal_hits,
            "has_think": has_think,
        },
    )


def _default_report_path(in_path: str) -> str:
    p = Path(os.path.expanduser(in_path))
    stem = p.stem
    return str(p.with_name(f"{stem}.low_info_report.json"))


def _default_sample_path(in_path: str) -> str:
    p = Path(os.path.expanduser(in_path))
    stem = p.stem
    return str(p.with_name(f"{stem}.low_info_samples.jsonl"))


def _default_out_path(in_path: str) -> str:
    p = Path(os.path.expanduser(in_path))
    stem = p.stem
    return str(p.with_name(f"{stem}.filtered.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter low-information CoT outputs from JSON/JSONL.")
    parser.add_argument("--in-path", required=True, help="Input dataset path (json/jsonl).")
    parser.add_argument("--out-path", default="", help="Filtered output path (json).")
    parser.add_argument("--report-path", default="", help="Report path (json).")
    parser.add_argument("--sample-path", default="", help="Sample records path (jsonl).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate report and samples; do not write filtered dataset.",
    )
    parser.add_argument(
        "--drop-level",
        choices=("hard", "all"),
        default="all",
        help="hard: drop only hard_drop; all: drop hard_drop + low_info.",
    )
    parser.add_argument(
        "--think-min-chars",
        type=int,
        default=80,
        help="Reasoning shorter than this (chars) is penalized as low-info.",
    )
    parser.add_argument(
        "--drop-score-threshold",
        type=int,
        default=3,
        help="net_score (penalties - credits) at/above which a sample is tagged low_info.",
    )
    parser.add_argument(
        "--max-sample-per-tag",
        type=int,
        default=80,
        help="Max sampled examples per decision tag.",
    )
    args = parser.parse_args()

    if args.think_min_chars < 1:
        raise SystemExit("--think-min-chars must be a positive integer")

    in_path = os.path.expanduser(args.in_path)
    out_path = os.path.expanduser(args.out_path) if args.out_path else _default_out_path(in_path)
    report_path = (
        os.path.expanduser(args.report_path) if args.report_path else _default_report_path(in_path)
    )
    sample_path = (
        os.path.expanduser(args.sample_path) if args.sample_path else _default_sample_path(in_path)
    )

    items = _read_items(in_path)
    total = len(items)
    if total == 0:
        raise SystemExit("empty dataset")

    tag_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    scored: List[Tuple[Dict[str, Any], Decision, int]] = []
    samples: Dict[str, List[Dict[str, Any]]] = {"keep": [], "low_info": [], "hard_drop": []}

    for idx, item in enumerate(items):
        decision = _analyze_one(
            item,
            think_min_chars=args.think_min_chars,
            drop_score_threshold=args.drop_score_threshold,
        )
        tag_counter[decision.tag] += 1
        reason_counter.update(decision.reasons)
        scored.append((item, decision, idx))

        if len(samples[decision.tag]) < args.max_sample_per_tag:
            samples[decision.tag].append(
                {
                    "index": idx,
                    "tag": decision.tag,
                    "score": decision.score,
                    "reasons": decision.reasons,
                    "features": decision.features,
                    "question": _extract_question(item)[:400],
                    "output_preview": _extract_output(item)[:800],
                }
            )

    keep_items: List[Dict[str, Any]] = []
    if args.drop_level == "hard":
        for item, decision, _ in scored:
            if decision.tag != "hard_drop":
                keep_items.append(item)
    else:
        for item, decision, _ in scored:
            if decision.tag == "keep":
                keep_items.append(item)

    dropped = total - len(keep_items)
    report = {
        "input_path": in_path,
        "total": total,
        "drop_level": args.drop_level,
        "kept": len(keep_items),
        "dropped": dropped,
        "dropped_ratio": dropped / total,
        "tag_counts": dict(tag_counter),
        "tag_ratios": {k: v / total for k, v in tag_counter.items()},
        "top_reasons": reason_counter.most_common(20),
        "thresholds": {
            "think_min_chars": args.think_min_chars,
            "drop_score_threshold": args.drop_score_threshold,
        },
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as f:
        for tag in ("hard_drop", "low_info", "keep"):
            for row in samples[tag]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"report: {report_path}")
    print(f"samples: {sample_path}")
    print(
        "summary: "
        f"total={total}, kept={len(keep_items)}, dropped={dropped} ({(dropped / total):.2%}), "
        f"tag_counts={dict(tag_counter)}"
    )

    if not args.dry_run:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(keep_items, f, ensure_ascii=False)
        print(f"filtered_data: {out_path}")


if __name__ == "__main__":
    main()
