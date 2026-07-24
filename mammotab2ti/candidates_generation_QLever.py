"""
Generate Wikidata entity candidates for a mention via QLever API.

Usage:
    python -m mammotab2ti.candidates_generation_QLever --mention "Paris" --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List

QLEVER_ENDPOINT = "https://qlever-api.wikidata.dbis.rwth-aachen.de"
DEFAULT_LIMIT = 20


@dataclass
class Candidate:
    entity_id: str
    uri: str
    label: str
    description: str
    qlever_score: float
    sitelinks: int
    statement_count: int
    is_disambiguation: bool
    rank_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "uri": self.uri,
            "label": self.label,
            "description": self.description,
            "qlever_score": self.qlever_score,
            "sitelinks": self.sitelinks,
            "statement_count": self.statement_count,
            "is_disambiguation": self.is_disambiguation,
            "rank_score": self.rank_score,
        }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _make_request(query: str, timeout: int = 20, retries: int = 2) -> Dict[str, Any]:
    """
    Execute a SPARQL query against QLever and parse JSON response.
    Uses small retry with exponential backoff for transient failures.
    """
    url = f"{QLEVER_ENDPOINT}/?query={urllib.parse.quote(query)}"
    headers = {"Accept": "application/sparql-results+json"}
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            if attempt < retries:
                time.sleep(0.4 * (2**attempt))

    raise RuntimeError(f"QLever request failed: {last_error}") from last_error


def _extract_qid(uri: str) -> str:
    if "/" in uri:
        return uri.rsplit("/", 1)[-1]
    return uri


def _safe_float(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _ranking_score(
    mention: str,
    label: str,
    qlever_score: float,
    sitelinks: int,
    statement_count: int,
    is_disambiguation: bool,
) -> float:
    mention_norm = _normalize_text(mention)
    label_norm = _normalize_text(label)

    is_exact = 1.0 if label_norm == mention_norm and mention_norm else 0.0
    is_prefix = 1.0 if mention_norm and label_norm.startswith(mention_norm) else 0.0
    has_contains = 1.0 if mention_norm and mention_norm in label_norm else 0.0

    mention_tokens = set(mention_norm.split())
    label_tokens = set(label_norm.split())
    overlap = 0.0
    if mention_tokens and label_tokens:
        overlap = len(mention_tokens & label_tokens) / len(mention_tokens | label_tokens)

    # Weighted hybrid score:
    # exact/prefix/contains > token overlap > qlever text score > sitelinks prior.
    disamb_penalty = -400.0 if is_disambiguation else 0.0
    return (
        1000.0 * is_exact
        + 200.0 * is_prefix
        + 50.0 * has_contains
        + 25.0 * overlap
        + qlever_score
        + math.log1p(max(sitelinks, 0))
        + 8.0 * math.log1p(max(statement_count, 0))
        + disamb_penalty
    )


def _parse_bindings(mention: str, payload: Dict[str, Any]) -> List[Candidate]:
    bindings = payload.get("results", {}).get("bindings", [])
    candidates: List[Candidate] = []

    for row in bindings:
        uri = row.get("entity", {}).get("value", "")
        label = row.get("label", {}).get("value", "")
        description = row.get("description", {}).get("value", "")
        qlever_score = _safe_float(row.get("score", {}).get("value"))
        sitelinks = _safe_int(row.get("sitelinks", {}).get("value"))
        entity_type = row.get("type", {}).get("value", "")
        is_disambiguation = (
            entity_type == "http://www.wikidata.org/entity/Q4167410"
            or "disambiguation" in _normalize_text(description)
        )
        if not uri or not label:
            continue
        entity_id = _extract_qid(uri)
        rank_score = _ranking_score(
            mention,
            label,
            qlever_score,
            sitelinks,
            statement_count=0,
            is_disambiguation=is_disambiguation,
        )
        candidates.append(
            Candidate(
                entity_id=entity_id,
                uri=uri,
                label=label,
                description=description,
                qlever_score=qlever_score,
                sitelinks=sitelinks,
                statement_count=0,
                is_disambiguation=is_disambiguation,
                rank_score=rank_score,
            )
        )

    # Deduplicate by QID (keep highest rank item)
    best_by_qid: Dict[str, Candidate] = {}
    for cand in candidates:
        prev = best_by_qid.get(cand.entity_id)
        if prev is None or cand.rank_score > prev.rank_score:
            best_by_qid[cand.entity_id] = cand

    deduped = list(best_by_qid.values())
    deduped.sort(
        key=lambda c: (
            -c.rank_score,
            -c.qlever_score,
            -c.sitelinks,
            len(c.label),
            c.entity_id,
        )
    )
    return deduped


def _statement_count_query(uris: List[str]) -> str:
    values = " ".join(f"<{u}>" for u in uris)
    return f"""
SELECT ?entity (COUNT(?p) AS ?statement_count)
WHERE {{
  VALUES ?entity {{ {values} }}
  ?entity ?p ?o .
  FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/"))
}}
GROUP BY ?entity
""".strip()


def _fetch_statement_counts(uris: List[str], chunk_size: int = 40) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for i in range(0, len(uris), chunk_size):
        chunk = uris[i : i + chunk_size]
        if not chunk:
            continue
        try:
            payload = _make_request(_statement_count_query(chunk), timeout=20, retries=1)
        except RuntimeError:
            continue

        for row in payload.get("results", {}).get("bindings", []):
            uri = row.get("entity", {}).get("value", "")
            cnt = _safe_int(row.get("statement_count", {}).get("value"))
            if uri:
                counts[uri] = cnt
    return counts


def _as_en_literal(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"@en'


def _exact_label_query(mentions: List[str], fetch_limit: int) -> str:
    values = " ".join(_as_en_literal(m) for m in mentions)
    return f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology-beta#>
SELECT ?entity ?label ?description ?sitelinks ?type
WHERE {{
  VALUES ?name {{ {values} }}
  ?entity rdfs:label ?name .
  BIND(?name AS ?label)
  FILTER(STRSTARTS(STR(?entity), "http://www.wikidata.org/entity/Q"))
  OPTIONAL {{
    ?entity schema:description ?description .
    FILTER(LANG(?description) = "en")
  }}
  OPTIONAL {{ ?entity <http://www.wikidata.org/prop/direct/P31> ?type }}
  OPTIONAL {{ ?entity wikibase:sitelinks ?sitelinks }}
}}
LIMIT {fetch_limit}
""".strip()


def _prefix_range_query(prefix: str, fetch_limit: int) -> str:
    lo = _as_en_literal(prefix)
    hi = _as_en_literal(prefix + "\uffff")
    return f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology-beta#>
SELECT ?entity ?label ?description ?sitelinks ?type
WHERE {{
  ?entity rdfs:label ?label .
  FILTER(?label >= {lo} && ?label < {hi})
  FILTER(STRSTARTS(STR(?entity), "http://www.wikidata.org/entity/Q"))
  OPTIONAL {{
    ?entity schema:description ?description .
    FILTER(LANG(?description) = "en")
  }}
  OPTIONAL {{ ?entity <http://www.wikidata.org/prop/direct/P31> ?type }}
  OPTIONAL {{ ?entity wikibase:sitelinks ?sitelinks }}
}}
LIMIT {fetch_limit}
""".strip()


def get_wikidata_candidates_via_qlever(
    mention: str,
    limit: int = DEFAULT_LIMIT,
    fetch_limit: int = 120,
) -> List[Dict[str, Any]]:
    """
    Return top-k Wikidata candidates for a mention via QLever API.

    Strategy:
    1) Exact matches on label/alias.
    2) Prefix-range search on labels (index-friendly on QLever).
    3) Deterministic hybrid re-ranking and top-k truncation.
    """
    mention = mention.strip()
    if not mention:
        return []

    # Variants improve recall while keeping queries index-friendly.
    mention_variants = [mention]
    title_case = mention.title()
    lower_case = mention.lower()
    for v in (title_case, lower_case):
        if v not in mention_variants:
            mention_variants.append(v)

    collected: List[Candidate] = []

    # 1) Exact matches on label/alias first (usually highest precision).
    exact_fetch_limit = max(fetch_limit * 8, 1000)
    try:
        exact_payload = _make_request(
            _exact_label_query(mention_variants, fetch_limit=exact_fetch_limit),
            timeout=25,
        )
        collected.extend(_parse_bindings(mention, exact_payload))
    except RuntimeError:
        pass

    # 2) Prefix-range on labels (index-friendly fallback).
    #    Stop early when we already have enough candidates.
    prefix_fetch = max(fetch_limit, limit * 4)
    for variant in mention_variants:
        prefix = variant.strip()
        if len(prefix) < 2:
            continue
        payload = _make_request(_prefix_range_query(prefix, fetch_limit=prefix_fetch))
        collected.extend(_parse_bindings(mention, payload))
        if len(collected) >= max(2 * limit, 40):
            break

    # De-duplicate and add a popularity prior from statement counts.
    best_by_qid: Dict[str, Candidate] = {}
    for cand in collected:
        prev = best_by_qid.get(cand.entity_id)
        if prev is None or cand.rank_score > prev.rank_score:
            best_by_qid[cand.entity_id] = cand

    candidates = list(best_by_qid.values())
    counts = _fetch_statement_counts([c.uri for c in candidates])
    for cand in candidates:
        cand.statement_count = counts.get(cand.uri, 0)
        cand.rank_score = _ranking_score(
            mention,
            cand.label,
            cand.qlever_score,
            cand.sitelinks,
            statement_count=cand.statement_count,
            is_disambiguation=cand.is_disambiguation,
        )
    candidates.sort(
        key=lambda c: (
            -c.rank_score,
            -c.statement_count,
            -c.qlever_score,
            -c.sitelinks,
            len(c.label),
            c.entity_id,
        )
    )

    return [c.to_dict() for c in candidates[: max(1, limit)]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Wikidata candidates from QLever API.")
    parser.add_argument("--mention", type=str, required=True, help="Entity mention string")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Top-k candidates to return")
    args = parser.parse_args()

    candidates = get_wikidata_candidates_via_qlever(args.mention, limit=max(1, args.limit))
    print(json.dumps(candidates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
