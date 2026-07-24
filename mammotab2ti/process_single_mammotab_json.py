"""
Process one Mammotab JSON file and preview generated tableInstruct-like samples.

Example:
  python -m mammotab2ti.process_single_mammotab_json \
    --input_file /path/to/000IIMXB.json \
    --output /path/to/preview_samples.json \
    --max_jobs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from mammotab2ti.local_wikidata_retriever import LocalWikidataRetriever, normalize_text
from mammotab2ti.mammotab_job_utils import DEFAULT_INSTRUCTION, build_input_seg, extract_jobs_from_table, load_table_json


def _build_question(mention: str, column_name: str, candidate_texts: List[str]) -> str:
    cand_str = ", ".join(candidate_texts)
    return (
        f"The selected entity mention in the table cell is: {mention}. "
        f"The column name for '{mention}' is {column_name}. "
        f"The referent entity candidates are: {cand_str} "
        f"What is the correct referent entity for the entity mention '{mention}'?"
    )


def _build_prior(jobs) -> Dict[str, Dict[str, int]]:
    prior: Dict[str, Dict[str, int]] = {}
    for j in jobs:
        mention_norm = normalize_text(j.mention)
        qid = (j.gold_qid or "").strip()
        if not mention_norm or not qid.startswith("Q"):
            continue
        p = prior.setdefault(mention_norm, {})
        p[qid] = p.get(qid, 0) + 1
    return prior


def _resolve_gold_output(retriever: LocalWikidataRetriever, gold_qid: str, gold_wiki_title: str, mention: str) -> str:
    qid = (gold_qid or "").strip()
    if qid.startswith("Q"):
        ent = retriever.get_entity_by_qid(qid)
        if ent is not None:
            return ent.to_candidate()["text"]
    title = (gold_wiki_title or "").replace("_", " ").strip()
    if title:
        return f"<{title} [DESCRIPTION] None [TYPE] None>"
    return f"<{mention} [DESCRIPTION] None [TYPE] None>"


def _build_samples_for_one_table(
    table: Dict[str, Any],
    source_path: str,
    retriever: LocalWikidataRetriever,
    top_k: int,
    max_jobs: int,
) -> List[Dict[str, Any]]:
    jobs = list(extract_jobs_from_table(table, source_path))
    if max_jobs > 0:
        jobs = jobs[:max_jobs]
    samples: List[Dict[str, Any]] = []
    prior = _build_prior(jobs)

    for job in jobs:
        mention = job.mention
        cands = retriever.retrieve(
            mention=mention,
            column_name=job.column_name,
            page_title=job.page_title,
            section_title=job.section_title,
            row_context_text="",
            mention_qid_prior=prior,
            top_k=max(1, min(top_k, 20)),
        )
        candidate_texts = [x.get("text", "") for x in cands if x.get("text")]
        candidate_qids = [x.get("qid", "") for x in cands if x.get("qid")]

        sample = {
            "instruction": DEFAULT_INSTRUCTION,
            "input_seg": build_input_seg(table, row_index=job.row_index, col_index=job.col_index),
            "question": _build_question(mention=mention, column_name=job.column_name, candidate_texts=candidate_texts),
            "output": _resolve_gold_output(
                retriever,
                gold_qid=job.gold_qid,
                gold_wiki_title=job.gold_wiki_title,
                mention=mention,
            ),
            "meta": {
                "job_id": job.job_id,
                "row_index": job.row_index,
                "col_index": job.col_index,
                "gold_qid": job.gold_qid,
                "gold_wiki_title": job.gold_wiki_title,
                "candidate_qids": candidate_qids,
                "candidate_count": len(candidate_qids),
            },
        }
        samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Process one Mammotab JSON file for quick preview.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to one Mammotab JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output path (.json)")
    parser.add_argument("--wikidata_db", type=str, default="~/DATA/wikidata/wikidata_candidates.sqlite", help="Offline wikidata sqlite index")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k candidates")
    parser.add_argument("--max_jobs", type=int, default=50, help="At most N mention jobs in this file")
    args = parser.parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    wikidata_db = Path(args.wikidata_db).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not wikidata_db.exists():
        raise ValueError(f"wikidata_db not found: {wikidata_db}")

    table = load_table_json(str(input_path))
    retriever = LocalWikidataRetriever(str(wikidata_db))
    samples = _build_samples_for_one_table(
        table=table,
        source_path=str(input_path),
        retriever=retriever,
        top_k=int(args.top_k),
        max_jobs=int(args.max_jobs),
    )
    retriever.close()

    payload = {
        "source_file": str(input_path),
        "sample_count": len(samples),
        "samples": samples,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(str(output_path))


if __name__ == "__main__":
    main()

