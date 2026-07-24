from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple


def normalize_text(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    norm = normalize_text(text)
    if not norm:
        return []
    return [t for t in norm.split(" ") if t]


@lru_cache(maxsize=500000)
def _token_set_cached(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


@dataclass
class EntityRow:
    qid: str
    label: str
    description: str
    p31: str
    p31_label: str
    statement_count: int
    norm_label: str

    def to_candidate(self) -> Dict[str, str]:
        desc = self.description or "None"
        tpe = self.p31_label or "None"
        return {
            "qid": self.qid,
            "label": self.label or self.qid,
            "description": desc,
            "type": tpe,
            "text": f"<{self.label or self.qid} [DESCRIPTION] {desc} [TYPE] {tpe}>",
        }


class LocalWikidataRetriever:
    def __init__(self, db_path: str):
        # Prefer read-only mode for offline retrieval workloads.
        try:
            self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        except sqlite3.OperationalError:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._entity_cache: Dict[str, EntityRow] = {}
        try:
            self.conn.execute("PRAGMA temp_store=MEMORY")
            self.conn.execute("PRAGMA cache_size=-200000")
            self.conn.execute("PRAGMA mmap_size=30000000000")
            self.conn.execute("PRAGMA query_only=ON")
        except sqlite3.OperationalError:
            pass
        self.has_fts = self._detect_fts()

    def close(self) -> None:
        self.conn.close()

    def _detect_fts(self) -> bool:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_entities'"
        ).fetchone()
        return row is not None

    def _entity_from_row(self, row: sqlite3.Row) -> EntityRow:
        return EntityRow(
            qid=str(row["qid"] or ""),
            label=str(row["label"] or ""),
            description=str(row["description"] or ""),
            p31=str(row["p31"] or ""),
            p31_label=str(row["p31_label"] or ""),
            statement_count=int(row["statement_count"] or 0),
            norm_label=str(row["norm_label"] or ""),
        )

    def get_entity_by_qid(self, qid: str) -> EntityRow | None:
        row = self.conn.execute(
            """
            SELECT qid, label, description, p31, p31_label, statement_count, norm_label
            FROM entities WHERE qid = ?
            """,
            (qid,),
        ).fetchone()
        if row is None:
            return None
        return self._entity_from_row(row)

    def _get_exact_qids(self, mention_norm: str, limit: int = 300) -> List[str]:
        rows = self.conn.execute(
            "SELECT qid FROM alias_index WHERE alias_norm = ? LIMIT ?",
            (mention_norm, limit),
        ).fetchall()
        return [str(r["qid"]) for r in rows]

    def _get_prefix_qids(self, mention_norm: str, limit: int = 300) -> List[str]:
        rows = self.conn.execute(
            """
            SELECT qid FROM entities
            WHERE norm_label LIKE ?
            LIMIT ?
            """,
            (f"{mention_norm}%", limit),
        ).fetchall()
        return [str(r["qid"]) for r in rows]

    def _get_fts_qids(self, mention: str, limit: int = 300) -> List[Tuple[str, float]]:
        if not self.has_fts:
            return []
        terms = tokenize(mention)
        if not terms:
            return []
        match_query = " OR ".join(terms[:8])
        try:
            rows = self.conn.execute(
                """
                SELECT qid, bm25(fts_entities) AS bm25_score
                FROM fts_entities
                WHERE fts_entities MATCH ?
                ORDER BY bm25(fts_entities)
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
            out: List[Tuple[str, float]] = []
            for r in rows:
                qid = str(r["qid"])
                bm25_score = float(r["bm25_score"])
                out.append((qid, bm25_score))
            return out
        except sqlite3.OperationalError:
            return []

    def _load_entities(self, qids: Iterable[str]) -> List[EntityRow]:
        qid_list = list(dict.fromkeys(qids))
        if not qid_list:
            return []
        missing = [qid for qid in qid_list if qid not in self._entity_cache]
        if missing:
            placeholders = ",".join(["?"] * len(missing))
            rows = self.conn.execute(
                f"""
                SELECT qid, label, description, p31, p31_label, statement_count, norm_label
                FROM entities
                WHERE qid IN ({placeholders})
                """,
                missing,
            ).fetchall()
            for r in rows:
                ent = self._entity_from_row(r)
                self._entity_cache[ent.qid] = ent
        return [self._entity_cache[qid] for qid in qid_list if qid in self._entity_cache]

    def _type_match_score(self, column_name: str, p31_label: str) -> float:
        col = normalize_text(column_name)
        p31 = normalize_text(p31_label)
        if not col or not p31:
            return 0.0

        mappings = [
            ({"member", "athlete", "player", "person", "winner", "name"}, {"human", "person", "athlete"}),
            ({"city", "country", "state", "location", "place", "address"}, {"city", "country", "settlement", "location"}),
            ({"album", "song", "film", "book", "work"}, {"album", "song", "film", "book", "work"}),
            ({"team", "club"}, {"sports team", "club"}),
            ({"school", "university", "college"}, {"university", "college", "school"}),
        ]
        col_tokens = set(col.split())
        for col_keys, type_keys in mappings:
            if col_tokens & col_keys:
                for tk in type_keys:
                    if tk in p31:
                        return 1.0
        return 0.0

    def _context_overlap(self, description: str, context_text: str) -> float:
        dset = _token_set_cached(description)
        cset = _token_set_cached(context_text)
        if not dset or not cset:
            return 0.0
        return len(dset & cset) / max(1, len(dset | cset))

    def _sim(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        a_tokens = a.split()
        b_tokens = b.split()
        if not a_tokens or not b_tokens:
            return 0.0
        aset = set(a_tokens)
        bset = set(b_tokens)
        jaccard = len(aset & bset) / max(1, len(aset | bset))
        max_len = max(len(a), len(b))
        prefix = 0
        while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
            prefix += 1
        prefix_ratio = prefix / max(1, max_len)
        return 0.75 * jaccard + 0.25 * prefix_ratio

    def retrieve(
        self,
        mention: str,
        column_name: str,
        page_title: str,
        section_title: str,
        row_context_text: str,
        mention_qid_prior: Dict[str, Dict[str, int]],
        top_k: int = 20,
        use_bm25: bool = True,
        use_prefix: bool = False,
        use_fts: bool = True,
        exact_limit: int = 80,
        prefix_limit: int = 80,
        fts_limit: int = 120,
        max_entities: int = 260,
    ) -> List[Dict[str, Any]]:
        mention_norm = normalize_text(mention)
        if not mention_norm:
            return []

        qids: List[str] = []
        qids.extend(self._get_exact_qids(mention_norm, limit=max(1, int(exact_limit))))
        if use_prefix:
            qids.extend(self._get_prefix_qids(mention_norm, limit=max(1, int(prefix_limit))))
        fts_hits: List[Tuple[str, float]] = []
        if use_fts:
            fts_hits = self._get_fts_qids(mention, limit=max(1, int(fts_limit)))
        qids.extend(qid for qid, _ in fts_hits)
        qids = list(dict.fromkeys(qids))
        entities = self._load_entities(qids[: max(1, int(max_entities))])

        # FTS5 bm25 is lower-is-better; normalize to [0,1] with higher-is-better.
        fts_bm25_raw: Dict[str, float] = {}
        for qid, bm25_raw in fts_hits:
            prev = fts_bm25_raw.get(qid)
            if prev is None or bm25_raw < prev:
                fts_bm25_raw[qid] = bm25_raw
        fts_bm25_norm: Dict[str, float] = {}
        if use_bm25 and fts_bm25_raw:
            min_bm25 = min(fts_bm25_raw.values())
            max_bm25 = max(fts_bm25_raw.values())
            denom = max_bm25 - min_bm25
            if denom > 1e-12:
                for qid, bm25_raw in fts_bm25_raw.items():
                    fts_bm25_norm[qid] = (max_bm25 - bm25_raw) / denom
            else:
                for qid in fts_bm25_raw:
                    fts_bm25_norm[qid] = 0.0

        prior_for_mention = mention_qid_prior.get(mention_norm, {})
        prior_max = max(prior_for_mention.values()) if prior_for_mention else 0
        context_text = " ".join([page_title or "", section_title or "", row_context_text or ""]).strip()

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for ent in entities:
            exact = 1.0 if ent.norm_label == mention_norm else 0.0
            contains = 1.0 if mention_norm and mention_norm in ent.norm_label else 0.0
            sim = self._sim(mention_norm, ent.norm_label)
            prior_cnt = prior_for_mention.get(ent.qid, 0)
            prior_score = (prior_cnt / prior_max) if prior_max > 0 else 0.0
            type_score = self._type_match_score(column_name, ent.p31_label)
            ctx_score = self._context_overlap(ent.description, context_text)
            bm25_score = fts_bm25_norm.get(ent.qid, 0.0)

            score = (
                3.8 * exact
                + 1.3 * contains
                + 2.3 * sim
                + 1.8 * prior_score
                + 1.0 * type_score
                + 0.9 * ctx_score
                + 1.4 * bm25_score
            )
            cand = ent.to_candidate()
            cand["score"] = score
            cand["bm25_score"] = bm25_score
            cand["features"] = {
                "exact": exact,
                "contains": contains,
                "sim": sim,
                "prior_score": prior_score,
                "type_score": type_score,
                "ctx_score": ctx_score,
                "bm25_score": bm25_score,
                "use_bm25": float(1.0 if use_bm25 else 0.0),
            }
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[: max(1, min(top_k, 100))]]

