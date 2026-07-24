"""
Stage 2: Generate tableInstruct-like EL samples from Stage-1 jobs using local Wikidata index.

Important:
- This script DOES NOT force-inject gold QID into top-k candidates.
- It writes an audit jsonl file for later "gold not in top20" analysis.

Example:
  python -m mammotab2ti.stage2_generate_candidates_offline \
    --jobs /path/to/mammotab_jobs.jsonl \
    --wikidata_db ~/DATA/wikidata/wikidata_candidates.sqlite \
    --output /path/to/ent_link_train_generated.json \
    --audit_output /path/to/ent_link_train_generated.audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mammotab2ti.local_wikidata_retriever import LocalWikidataRetriever, normalize_text
from mammotab2ti.mammotab_job_utils import DEFAULT_INSTRUCTION, get_text_matrix, load_table_json


DEFAULT_PRIOR_CACHE_PATH = Path("/DATA1/khli/mammotab/modified_mammotab/mention_qid_prior_single.pkl")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_gold_output(
    retriever: LocalWikidataRetriever,
    gold_qid: str,
    gold_wiki_title: str,
    mention: str,
) -> str:
    gold_qid = (gold_qid or "").strip()
    if gold_qid:
        ent = retriever.get_entity_by_qid(gold_qid)
        if ent is not None:
            return ent.to_candidate()["text"]

    fallback_title = (gold_wiki_title or "").replace("_", " ").strip()
    if fallback_title:
        return f"<{fallback_title} [DESCRIPTION] None [TYPE] None>"
    return f"<{mention} [DESCRIPTION] None [TYPE] None>"


def _iter_jobs(
    path: Path,
    skip_jobs: int = 0,
    max_jobs: int = 0,
    num_shards: int = 1,
    shard_id: int = 0,
) -> Iterable[Dict[str, Any]]:
    yielded = 0
    skipped = 0
    nshards = max(1, int(num_shards))
    sid = int(shard_id)
    if sid < 0 or sid >= nshards:
        raise ValueError(f"invalid shard_id={sid}, expected [0, {nshards - 1}]")
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if nshards > 1 and (line_idx % nshards) != sid:
                continue
            if skipped < skip_jobs:
                skipped += 1
                continue
            yield json.loads(line)
            yielded += 1
            if max_jobs > 0 and yielded >= max_jobs:
                break


def _build_question(mention: str, column_name: str, candidate_texts: List[str]) -> str:
    cand_str = ", ".join(candidate_texts)
    return (
        f"The selected entity mention in the table cell is: {mention}. "
        f"The column name for '{mention}' is {column_name}. "
        f"The referent entity candidates are: {cand_str} "
        f"What is the correct referent entity for the entity mention '{mention}'?"
    )


def _build_mention_qid_prior(jobs_path: Path) -> Dict[str, Dict[str, int]]:
    prior: Dict[str, Dict[str, int]] = {}
    for job in _iter_jobs(jobs_path):
        mention = normalize_text(str(job.get("mention", "")))
        qid = str(job.get("gold_qid", "")).strip()
        if not mention or not qid.startswith("Q"):
            continue
        qfreq = prior.setdefault(mention, {})
        qfreq[qid] = qfreq.get(qid, 0) + 1
    return prior


def _load_mention_qid_prior_cache(cache_path: Path) -> Dict[str, Dict[str, int]]:
    with open(cache_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"invalid prior cache format in {cache_path}")
    return obj


def _save_mention_qid_prior_cache(cache_path: Path, prior: Dict[str, Dict[str, int]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(prior, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)


def _count_non_empty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cnt += 1
    return cnt


def _row_context(table: Dict[str, Any], row_index: int) -> str:
    text = table.get("__mt2ti_cached_text")
    if not isinstance(text, list):
        text = get_text_matrix(table)
        table["__mt2ti_cached_text"] = text
    if row_index <= 0 or row_index >= len(text):
        return ""
    row = text[row_index]
    return " | ".join(x for x in row if x)


def _build_input_seg_fast(
    table: Dict[str, Any],
    page_title: str,
    section_title: str,
    table_caption: str,
    row_index: int,
    max_context_rows: int,
) -> str:
    text = table.get("__mt2ti_cached_text")
    if not isinstance(text, list):
        text = get_text_matrix(table)
        table["__mt2ti_cached_text"] = text
    if len(text) < 2:
        return "[TLE] The Wikipedia page is about . The Wikipedia section is about . [TAB] col: |"

    header = text[0]
    col_line = str(table.get("__mt2ti_cached_col_line", ""))
    if not col_line:
        col_line = " | ".join(header)
        table["__mt2ti_cached_col_line"] = col_line

    if max_context_rows > 0:
        # Keep a bounded local window around target row for much faster generation.
        data_row_count = max(0, len(text) - 1)
        center = min(max(1, row_index), max(1, data_row_count))
        half = max_context_rows // 2
        start = max(1, center - half)
        end = min(len(text) - 1, start + max_context_rows - 1)
        start = max(1, end - max_context_rows + 1)
        rows_str: List[str] = []
        for r in range(start, end + 1):
            row = text[r]
            row_vals = row[: len(header)]
            if len(row_vals) < len(header):
                row_vals = row_vals + [""] * (len(header) - len(row_vals))
            content = "| " + " | ".join(row_vals) + " |"
            rows_str.append(f"row {r}: {content}")
        rows_joined = " [SEP] ".join(rows_str)
    else:
        rows_joined = str(table.get("__mt2ti_cached_rows_joined", ""))
        if not rows_joined:
            rows_str: List[str] = []
            for r in range(1, len(text)):
                row = text[r]
                row_vals = row[: len(header)]
                if len(row_vals) < len(header):
                    row_vals = row_vals + [""] * (len(header) - len(row_vals))
                content = "| " + " | ".join(row_vals) + " |"
                rows_str.append(f"row {r}: {content}")
            rows_joined = " [SEP] ".join(rows_str)
            table["__mt2ti_cached_rows_joined"] = rows_joined

    page = re.sub(r"\s+", " ", str(page_title or "")).strip()
    section = re.sub(r"\s+", " ", str(section_title or "")).strip()
    caption = re.sub(r"\s+", " ", str(table_caption or "")).strip()
    if caption.lower() == "none":
        caption = ""

    prefix = f"[TLE] The Wikipedia page is about {page}. The Wikipedia section is about {section}."
    if caption:
        prefix += f" The table caption is about {caption}."
    return f"{prefix} [TAB] col: | {col_line} | {rows_joined}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tableInstruct-like samples from stage-1 jobs.")
    parser.add_argument("--jobs", type=str, default="~/DATA/mammotab/modified_mammotab/mammotab_jobs.jsonl", help="Stage-1 jobs jsonl")
    parser.add_argument("--wikidata_db", type=str, default="~/DATA/wikidata/wikidata_candidates.sqlite", help="Offline wikidata sqlite index")
    parser.add_argument("--output", type=str, default=None, help="Output JSON/JSONL path")
    parser.add_argument("--audit_output", type=str, default=None, help="Audit JSONL path for analysis")
    parser.add_argument("--num", type=int, default=0, help="Limit number of output samples (0 means no limit)")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k candidates")
    parser.add_argument("--max_jobs", type=int, default=0, help="Only process first N jobs (0 means all)")
    parser.add_argument("--skip_jobs", type=int, default=0, help="Skip first N jobs from jobs file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing jsonl output and audit files")
    parser.add_argument(
        "--output_format",
        type=str,
        default="auto",
        choices=["auto", "json", "jsonl"],
        help="Output format: auto infer from extension",
    )
    parser.add_argument("--no_prior", action="store_true", help="Disable mention->qid prior build for speed")
    parser.add_argument(
        "--prior_cache_path",
        type=str,
        default=str(DEFAULT_PRIOR_CACHE_PATH),
        help="Cache path for mention->qid prior (load if exists, build+save if missing)",
    )
    parser.add_argument("--table_cache_size", type=int, default=256, help="LRU cache size for loaded table jsons")
    parser.add_argument(
        "--max_context_rows",
        type=int,
        default=-1,
        help="Rows kept in input_seg around target row (-1 keeps full table, faster values: 1/3/5)",
    )
    parser.add_argument("--use_prefix", action="store_true", help="Enable prefix label lookup (slower)")
    parser.add_argument("--disable_fts", action="store_true", help="Disable FTS retrieval branch for speed")
    parser.add_argument("--disable_bm25", action="store_true", help="Disable bm25 scoring feature")
    parser.add_argument("--exact_limit", type=int, default=80, help="Candidate limit from exact alias lookup")
    parser.add_argument("--prefix_limit", type=int, default=80, help="Candidate limit from prefix lookup")
    parser.add_argument("--fts_limit", type=int, default=120, help="Candidate limit from FTS lookup")
    parser.add_argument("--max_entities", type=int, default=260, help="Max candidate entities loaded per mention")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of parallel shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard id in [0, num_shards-1]")
    args = parser.parse_args()

    if int(args.num) < 0:
        raise ValueError("--num must be >= 0")

    project_root = _project_root()
    use_project_root_default = int(args.num) > 0
    default_output = (
        project_root / "ent_link_train_generated.json"
        if use_project_root_default
        else Path("~/DATA/mammotab/modified_mammotab/ent_link_train_generated.json").expanduser()
    )
    default_audit = (
        project_root / "ent_link_train_generated.audit.jsonl"
        if use_project_root_default
        else Path("~/DATA/mammotab/modified_mammotab/ent_link_train_generated.audit.jsonl").expanduser()
    )

    jobs_path = Path(args.jobs).expanduser().resolve()
    wikidata_db = Path(args.wikidata_db).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output.resolve()
    audit_path = Path(args.audit_output).expanduser().resolve() if args.audit_output else default_audit.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if not wikidata_db.exists():
        raise ValueError(f"wikidata_db not found: {wikidata_db}")

    output_format = args.output_format
    if output_format == "auto":
        output_format = "jsonl" if output_path.suffix.lower() == ".jsonl" else "json"

    base_skip = max(0, int(args.skip_jobs))
    resume_skip = 0
    if args.resume:
        if output_format != "jsonl":
            raise ValueError("--resume currently supports only output_format=jsonl")
        out_done = _count_non_empty_lines(output_path)
        audit_done = _count_non_empty_lines(audit_path)
        resume_skip = min(out_done, audit_done)
        print(f"resume detected: output_lines={out_done}, audit_lines={audit_done}, resume_skip={resume_skip}")
    total_skip = base_skip + resume_skip

    mention_qid_prior: Dict[str, Dict[str, int]] = {}
    if args.no_prior:
        print("skip mention->qid prior build (--no_prior)")
    else:
        prior_cache_path = Path(args.prior_cache_path).expanduser()
        if prior_cache_path.exists():
            print(f"loading mention->qid prior cache from: {prior_cache_path}")
            mention_qid_prior = _load_mention_qid_prior_cache(prior_cache_path)
            print(f"prior cache loaded, entries: {len(mention_qid_prior)}")
        else:
            print("building mention->qid prior from jobs...")
            mention_qid_prior = _build_mention_qid_prior(jobs_path)
            print(f"saving mention->qid prior cache to: {prior_cache_path}")
            _save_mention_qid_prior_cache(prior_cache_path, mention_qid_prior)
        print(f"prior entries: {len(mention_qid_prior)}")

    retriever = LocalWikidataRetriever(str(wikidata_db))

    table_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _get_table(source_path: str) -> Dict[str, Any]:
        t = table_cache.get(source_path)
        if t is not None:
            table_cache.move_to_end(source_path)
            return t
        loaded = load_table_json(source_path)
        table_cache[source_path] = loaded
        if len(table_cache) > max(1, int(args.table_cache_size)):
            table_cache.popitem(last=False)
        return loaded

    processed = 0
    num_limit = max(0, int(args.num))
    start_ts = time.time()

    out_mode = "a" if (args.resume and output_format == "jsonl") else "w"
    with open(output_path, out_mode, encoding="utf-8") as out_f, open(audit_path, out_mode, encoding="utf-8") as audit_f:
        if output_format == "json":
            out_f.write("[\n")
        first = True
        for job in _iter_jobs(
            jobs_path,
            skip_jobs=total_skip,
            max_jobs=max(0, int(args.max_jobs)),
            num_shards=max(1, int(args.num_shards)),
            shard_id=int(args.shard_id),
        ):
            source_path = str(job.get("source_path", ""))
            cur_table = _get_table(source_path)

            mention = str(job.get("mention", "")).strip()
            if not mention:
                continue
            top_k = max(1, min(int(args.top_k), 20))

            candidates = retriever.retrieve(
                mention=mention,
                column_name=str(job.get("column_name", "")),
                page_title=str(job.get("page_title", "")),
                section_title=str(job.get("section_title", "")),
                row_context_text=_row_context(cur_table, int(job.get("row_index", 0))),
                mention_qid_prior=mention_qid_prior,
                top_k=top_k,
                use_prefix=bool(args.use_prefix),
                use_fts=not bool(args.disable_fts),
                use_bm25=not bool(args.disable_bm25),
                exact_limit=int(args.exact_limit),
                prefix_limit=int(args.prefix_limit),
                fts_limit=int(args.fts_limit),
                max_entities=int(args.max_entities),
            )

            candidate_texts = [str(x.get("text", "")) for x in candidates if x.get("text")]
            candidate_qids = [str(x.get("qid", "")) for x in candidates if x.get("qid")]
            candidate_texts = candidate_texts[:top_k]
            candidate_qids = candidate_qids[:top_k]

            input_seg = _build_input_seg_fast(
                cur_table,
                page_title=str(job.get("page_title", "")),
                section_title=str(job.get("section_title", "")),
                table_caption=str(job.get("table_caption", "")),
                row_index=int(job.get("row_index", 0)),
                max_context_rows=int(args.max_context_rows),
            )
            column_name = str(job.get("column_name", "")).strip()
            question = _build_question(mention=mention, column_name=column_name, candidate_texts=candidate_texts)
            output = _resolve_gold_output(
                retriever,
                gold_qid=str(job.get("gold_qid", "")),
                gold_wiki_title=str(job.get("gold_wiki_title", "")),
                mention=mention,
            )

            sample = {
                "instruction": DEFAULT_INSTRUCTION,
                "input_seg": input_seg,
                "question": question,
                "output": output,
            }

            if output_format == "json":
                if not first:
                    out_f.write(",\n")
                first = False
                out_f.write("  " + json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            else:
                out_f.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")

            audit_obj = {
                "job_id": job.get("job_id", ""),
                "source_file": job.get("source_file", ""),
                "mention": mention,
                "gold_qid": job.get("gold_qid", ""),
                "candidate_qids": candidate_qids,
                "gold_in_topk": str(job.get("gold_qid", "")) in set(candidate_qids),
                "candidate_count": len(candidate_qids),
            }
            audit_f.write(json.dumps(audit_obj, ensure_ascii=False) + "\n")

            processed += 1
            if num_limit > 0 and processed >= num_limit:
                print(f"reached --num={num_limit}, stop generation")
                break
            if processed % 500 == 0:
                elapsed = max(1e-6, time.time() - start_ts)
                speed = processed / elapsed
                print(f"processed_jobs={processed}, speed={speed:.2f}/s, total_seen={total_skip + processed}")

        if output_format == "json":
            out_f.write("\n]\n")

    retriever.close()
    print(f"done: processed_jobs={processed}")
    print(str(output_path))
    print(str(audit_path))


if __name__ == "__main__":
    main()

