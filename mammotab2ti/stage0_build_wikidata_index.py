"""
Build local Wikidata candidate index from dump (offline).

Supported dump formats:
- .json
- .json.gz
- .json.bz2
- .nt
- .nt.gz
- .nt.bz2

Expected entity line format: one JSON entity object per line in Wikidata dump array.

Example:
nohup python -m mammotab2ti.stage0_build_wikidata_index \
  --dump ~/DATA/wikidata/latest-truthy.nt.bz2 \
  --db ~/DATA/wikidata/wikidata_candidates.sqlite \
  > build_index.log 2>&1 &
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mammotab2ti.local_wikidata_retriever import normalize_text


def _open_text(path: Path):
    name = path.name.lower()
    if name.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_entities(dump_path: Path) -> Iterable[Dict[str, Any]]:
    with _open_text(dump_path) as f:
        for line in f:
            s = line.strip()
            if not s or s in ("[", "]"):
                continue
            if s.endswith(","):
                s = s[:-1]
            if not s or s[0] != "{":
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


_QID_RE = re.compile(r"/entity/(Q\d+)$")
_LANG_LITERAL_RE = re.compile(r'^"(.*)"@([a-zA-Z-]+)$')
_TYPED_LITERAL_RE = re.compile(r'^"(.*)"\^\^<[^>]+>$')
_PLAIN_LITERAL_RE = re.compile(r'^"(.*)"$')


def _qid_from_uri(uri: str) -> str:
    m = _QID_RE.search(uri)
    return m.group(1) if m else ""


def _decode_nt_literal(raw: str) -> str:
    # N-Triples escapes are close to unicode_escape for our use-case.
    try:
        return bytes(raw, "utf-8").decode("unicode_escape")
    except Exception:
        return raw


def _parse_nt_line(line: str) -> Optional[Tuple[str, str, str]]:
    s = line.strip()
    if not s.endswith(" .") or not s.startswith("<"):
        return None
    i1 = s.find("> <")
    if i1 <= 1:
        return None
    subj = s[1:i1]
    i2 = s.find("> ", i1 + 3)
    if i2 <= i1:
        return None
    pred = s[i1 + 3 : i2]
    obj = s[i2 + 2 : -2].strip()
    return subj, pred, obj


def _extract_literal_en(obj: str) -> str:
    m_lang = _LANG_LITERAL_RE.match(obj)
    if m_lang:
        lang = m_lang.group(2).lower()
        if lang == "en":
            return _decode_nt_literal(m_lang.group(1))
        return ""

    m_plain = _PLAIN_LITERAL_RE.match(obj)
    if m_plain:
        return _decode_nt_literal(m_plain.group(1))

    m_typed = _TYPED_LITERAL_RE.match(obj)
    if m_typed:
        return _decode_nt_literal(m_typed.group(1))
    return ""


def _flush_nt_entity(
    conn: sqlite3.Connection,
    qid: str,
    data: Dict[str, Any],
    with_fts: bool,
) -> None:
    if not qid:
        return
    label = str(data.get("label", ""))
    description = str(data.get("description", ""))
    aliases = list(data.get("aliases", []))
    p31 = str(data.get("p31", ""))
    enwiki_title = str(data.get("enwiki_title", ""))
    popularity = int(data.get("popularity", 0))
    statement_count = int(data.get("statement_count", 0))
    norm_label = normalize_text(label)
    norm_title = normalize_text(enwiki_title)

    conn.execute(
        """
        INSERT OR REPLACE INTO entities (
            qid, label, description, p31, p31_label, aliases_json, enwiki_title,
            popularity, statement_count, norm_label, norm_title
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            qid,
            label,
            description,
            p31,
            "",
            json.dumps(aliases, ensure_ascii=False),
            enwiki_title,
            popularity,
            statement_count,
            norm_label,
            norm_title,
        ),
    )

    alias_seen = set()
    for source, text in [("label", label), ("title", enwiki_title)] + [("alias", a) for a in aliases]:
        norm = normalize_text(text)
        if not norm:
            continue
        key = (norm, source)
        if key in alias_seen:
            continue
        alias_seen.add(key)
        conn.execute("INSERT INTO alias_index(alias_norm, qid, source) VALUES (?, ?, ?)", (norm, qid, source))

    if with_fts:
        aliases_text = " | ".join(aliases)
        try:
            conn.execute(
                "INSERT INTO fts_entities(qid, label, aliases, enwiki_title, description) VALUES (?, ?, ?, ?, ?)",
                (qid, label, aliases_text, enwiki_title, description),
            )
        except sqlite3.OperationalError:
            pass


def _iter_nt_entities(dump_path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    cur_qid = ""
    cur_data: Dict[str, Any] = {}

    with _open_text(dump_path) as f:
        for line in f:
            parsed = _parse_nt_line(line)
            if not parsed:
                continue
            subj, pred, obj = parsed
            qid = _qid_from_uri(subj)
            if not qid:
                continue

            if cur_qid and qid != cur_qid:
                yield cur_qid, cur_data
                cur_data = {}
            cur_qid = qid

            if pred.endswith("/prop/direct/P31") and obj.startswith("<"):
                p31_qid = _qid_from_uri(obj[1:-1])
                if p31_qid and not cur_data.get("p31"):
                    cur_data["p31"] = p31_qid
                    cur_data["statement_count"] = int(cur_data.get("statement_count", 0)) + 1
                continue

            if "/prop/direct/" in pred:
                cur_data["statement_count"] = int(cur_data.get("statement_count", 0)) + 1
                continue

            if pred.endswith("schema.org/name") or pred.endswith("rdf-schema#label"):
                val = _extract_literal_en(obj)
                if val and not cur_data.get("label"):
                    cur_data["label"] = val
                continue

            if pred.endswith("schema.org/description"):
                val = _extract_literal_en(obj)
                if val and not cur_data.get("description"):
                    cur_data["description"] = val
                continue

            if pred.endswith("skos/core#altLabel"):
                val = _extract_literal_en(obj)
                if val:
                    cur_data.setdefault("aliases", []).append(val)
                continue

            # Some truthy dumps contain sitelink-like info via schema:isPartOf + schema:name on wiki URL subjects,
            # but those are not straightforwardly keyed by wd:Q in one line, so we skip exact enwiki title extraction here.
            if pred.endswith("schema.org/about"):
                cur_data["popularity"] = int(cur_data.get("popularity", 0)) + 1
                continue

    if cur_qid:
        yield cur_qid, cur_data


def _extract_en(entity: Dict[str, Any], key: str) -> str:
    data = entity.get(key, {})
    if not isinstance(data, dict):
        return ""
    en = data.get("en", {})
    if not isinstance(en, dict):
        return ""
    val = en.get("value", "")
    return str(val or "")


def _extract_aliases_en(entity: Dict[str, Any]) -> List[str]:
    aliases = entity.get("aliases", {})
    if not isinstance(aliases, dict):
        return []
    en_list = aliases.get("en", [])
    if not isinstance(en_list, list):
        return []
    out: List[str] = []
    for obj in en_list:
        if isinstance(obj, dict):
            v = str(obj.get("value", "") or "")
            if v:
                out.append(v)
    return out


def _extract_first_p31(entity: Dict[str, Any]) -> str:
    claims = entity.get("claims", {})
    if not isinstance(claims, dict):
        return ""
    p31 = claims.get("P31", [])
    if not isinstance(p31, list) or not p31:
        return ""
    first = p31[0]
    if not isinstance(first, dict):
        return ""
    mainsnak = first.get("mainsnak", {})
    if not isinstance(mainsnak, dict):
        return ""
    datavalue = mainsnak.get("datavalue", {})
    if not isinstance(datavalue, dict):
        return ""
    value = datavalue.get("value", {})
    if not isinstance(value, dict):
        return ""
    qid = value.get("id", "")
    return str(qid or "")


def _ensure_schema(conn: sqlite3.Connection, with_fts: bool) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
            qid TEXT PRIMARY KEY,
            label TEXT,
            description TEXT,
            p31 TEXT,
            p31_label TEXT,
            aliases_json TEXT,
            enwiki_title TEXT,
            popularity INTEGER,
            statement_count INTEGER,
            norm_label TEXT,
            norm_title TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_norm_label ON entities(norm_label)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_norm_title ON entities(norm_title)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alias_index (
            alias_norm TEXT NOT NULL,
            qid TEXT NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alias_norm ON alias_index(alias_norm)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alias_qid ON alias_index(qid)")

    if with_fts:
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_entities USING fts5(
                    qid UNINDEXED,
                    label,
                    aliases,
                    enwiki_title,
                    description
                )
                """
            )
        except sqlite3.OperationalError:
            pass
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Wikidata index for offline candidate retrieval.")
    parser.add_argument("--dump", type=str, required=True, help="Path to Wikidata dump file")
    parser.add_argument("--db", type=str, default="~/DATA/wikidata/wikidata_candidates.sqlite", help="Output sqlite DB path")
    parser.add_argument("--max_entities", type=int, default=0, help="Only process first N entities (0 means all)")
    parser.add_argument("--commit_every", type=int, default=5000, help="Commit every N entities")
    parser.add_argument("--with_fts", action="store_true", help="Build sqlite FTS table for BM25 retrieval")
    args = parser.parse_args()

    dump_path = Path(args.dump).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not dump_path.exists():
        raise ValueError(f"dump not found: {dump_path}")

    conn = sqlite3.connect(str(db_path))
    _ensure_schema(conn, with_fts=args.with_fts)

    processed = 0
    kept = 0
    start_time = time.time()

    is_nt = dump_path.name.lower().endswith((".nt", ".nt.gz", ".nt.bz2"))
    if is_nt:
        entity_iter = _iter_nt_entities(dump_path)
        for qid, ent_data in entity_iter:
            if args.max_entities > 0 and processed >= args.max_entities:
                break
            processed += 1
            _flush_nt_entity(conn, qid=qid, data=ent_data, with_fts=args.with_fts)
            kept += 1
            if kept % max(1, args.commit_every) == 0:
                conn.commit()
                elapsed = max(1e-6, time.time() - start_time)
                print(f"processed={processed}, kept_qitems={kept}, speed={processed/elapsed:.1f}/s")
    else:
        for entity in _iter_entities(dump_path):
            if args.max_entities > 0 and processed >= args.max_entities:
                break
            processed += 1

            qid = str(entity.get("id", ""))
            if not qid.startswith("Q"):
                continue

            label = _extract_en(entity, "labels")
            description = _extract_en(entity, "descriptions")
            aliases = _extract_aliases_en(entity)
            p31 = _extract_first_p31(entity)

            sitelinks = entity.get("sitelinks", {})
            popularity = len(sitelinks) if isinstance(sitelinks, dict) else 0
            statement_count = len(entity.get("claims", {})) if isinstance(entity.get("claims", {}), dict) else 0
            enwiki_title = ""
            if isinstance(sitelinks, dict):
                enwiki_obj = sitelinks.get("enwiki", {})
                if isinstance(enwiki_obj, dict):
                    enwiki_title = str(enwiki_obj.get("title", "") or "")

            norm_label = normalize_text(label)
            norm_title = normalize_text(enwiki_title)

            conn.execute(
                """
                INSERT OR REPLACE INTO entities (
                    qid, label, description, p31, p31_label, aliases_json, enwiki_title,
                    popularity, statement_count, norm_label, norm_title
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    qid,
                    label,
                    description,
                    p31,
                    "",
                    json.dumps(aliases, ensure_ascii=False),
                    enwiki_title,
                    popularity,
                    statement_count,
                    norm_label,
                    norm_title,
                ),
            )

            alias_seen = set()
            for source, text in [("label", label), ("title", enwiki_title)] + [("alias", a) for a in aliases]:
                norm = normalize_text(text)
                if not norm:
                    continue
                key = (norm, source)
                if key in alias_seen:
                    continue
                alias_seen.add(key)
                conn.execute(
                    "INSERT INTO alias_index(alias_norm, qid, source) VALUES (?, ?, ?)",
                    (norm, qid, source),
                )

            if args.with_fts:
                aliases_text = " | ".join(aliases)
                try:
                    conn.execute(
                        "INSERT INTO fts_entities(qid, label, aliases, enwiki_title, description) VALUES (?, ?, ?, ?, ?)",
                        (qid, label, aliases_text, enwiki_title, description),
                    )
                except sqlite3.OperationalError:
                    pass

            kept += 1
            if kept % max(1, args.commit_every) == 0:
                conn.commit()
                elapsed = max(1e-6, time.time() - start_time)
                print(f"processed={processed}, kept_qitems={kept}, speed={processed/elapsed:.1f}/s")

    conn.commit()

    # Fill p31_label by joining entities table itself.
    conn.execute(
        """
        UPDATE entities
        SET p31_label = COALESCE(
            (SELECT e2.label FROM entities e2 WHERE e2.qid = entities.p31),
            ''
        )
        WHERE p31 IS NOT NULL AND p31 != ''
        """
    )
    conn.commit()
    conn.close()

    elapsed = max(1e-6, time.time() - start_time)
    print(f"done: processed={processed}, kept_qitems={kept}, elapsed={elapsed:.1f}s")
    print(str(db_path))


if __name__ == "__main__":
    main()

