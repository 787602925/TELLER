"""
Generate Wikidata entity candidates via official Wikidata API.

Usage:
    python -m mammotab2ti.candidates_generation_wikidata_api --mention "Richard Dyer-Bennet" --limit 20
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
DEFAULT_LIMIT = 20
DEFAULT_TYPE_FALLBACK = "owl#Thing"


def _http_get_json(params: Dict[str, str], timeout: int = 20, retries: int = 2) -> Dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{WIKIDATA_API}?{query}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "tableLlama/1.0 (entity-linking script)",
    }
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

    raise RuntimeError(f"Wikidata API request failed: {last_error}") from last_error


def _extract_first_p31_qid(entity_payload: Dict[str, Any]) -> str | None:
    claims = entity_payload.get("claims", {})
    p31_claims = claims.get("P31", [])
    if not p31_claims:
        return None

    mainsnak = p31_claims[0].get("mainsnak", {})
    datavalue = mainsnak.get("datavalue", {})
    value = datavalue.get("value", {})
    if isinstance(value, dict):
        p31_qid = value.get("id")
        if isinstance(p31_qid, str):
            return p31_qid
    return None


def _batch_get_entities(ids: List[str], props: str, language: str = "en") -> Dict[str, Any]:
    if not ids:
        return {}
    payload = _http_get_json(
        {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(ids),
            "props": props,
            "languages": language,
        }
    )
    return payload.get("entities", {})


def get_wikidata_candidates_via_api(mention: str, limit: int = DEFAULT_LIMIT) -> List[str]:
    """
    Return top-k candidates in format:
    <LABEL [DESCRIPTION] DESCRIPTION [TYPE] INSTANCE_OF_LABEL>
    """
    mention = mention.strip()
    if not mention:
        return []

    top_k = max(1, min(limit, 20))
    search_payload = _http_get_json(
        {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "uselang": "en",
            "search": mention,
            "type": "item",
            "limit": str(top_k),
        }
    )

    ranked_items = search_payload.get("search", [])
    if not ranked_items:
        return []

    qids = [item.get("id", "") for item in ranked_items if item.get("id", "").startswith("Q")]
    entity_details = _batch_get_entities(qids, props="claims")

    p31_qids: List[str] = []
    for qid in qids:
        entity_payload = entity_details.get(qid, {})
        p31_qid = _extract_first_p31_qid(entity_payload)
        if p31_qid:
            p31_qids.append(p31_qid)

    type_label_map: Dict[str, str] = {}
    if p31_qids:
        unique_type_qids = sorted(set(p31_qids))
        type_entities = _batch_get_entities(unique_type_qids, props="labels")
        for t_qid, t_payload in type_entities.items():
            label_block = t_payload.get("labels", {}).get("en", {})
            label = label_block.get("value", "")
            if label:
                type_label_map[t_qid] = label

    output: List[str] = []
    for item in ranked_items:
        qid = item.get("id", "")
        if not qid.startswith("Q"):
            continue
        label = item.get("label") or qid
        description = item.get("description") or ""
        p31_qid = _extract_first_p31_qid(entity_details.get(qid, {}))
        type_label = type_label_map.get(p31_qid or "", DEFAULT_TYPE_FALLBACK)
        output.append(f"<{label} [DESCRIPTION] {description} [TYPE] {type_label}>")

    return output[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Wikidata candidates from official Wikidata API.")
    parser.add_argument("--mention", type=str, required=True, help="Entity mention string")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Top-k candidates to return (<=20)")
    args = parser.parse_args()

    candidates = get_wikidata_candidates_via_api(args.mention, limit=args.limit)
    print(json.dumps(candidates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
