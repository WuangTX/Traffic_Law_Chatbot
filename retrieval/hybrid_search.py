"""
Hybrid Search: Direct Keyword + BM25 + FAISS + Query Expansion.
Priority layers for Vietnamese legal document retrieval.
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, List


# -- Query type configs (WITH diacritics - preserves semantic meaning) --
QUERY_CONFIGS = [
    {
        "trigger": ["nồng độ cồn", "uống rượu", "uống bia", "có cồn", "chất có cồn"],
        "keywords": ["nồng độ cồn", "miligam", "khí thở"],
        "amounts": ["2.000.000", "3.000.000", "4.000.000", "5.000.000",
                    "6.000.000", "8.000.000", "10.000.000",
                    "18.000.000", "20.000.000", "30.000.000", "40.000.000"],
    },
    {
        "trigger": ["đèn đỏ", "vượt đèn", "đèn tín hiệu", "tín hiệu giao thông",
                    "chấp hành hiệu lệnh", "hiệu lệnh"],
        "keywords": ["đèn tín hiệu giao thông", "hiệu lệnh", "không chấp hành"],
        "amounts": ["18.000.000", "20.000.000", "4.000.000", "6.000.000",
                    "800.000", "1.000.000", "100.000", "200.000"],
    },
    {
        "trigger": ["tốc độ", "chạy quá", "vượt tốc độ", "quá tốc độ"],
        "keywords": ["tốc độ", "km/h", "chạy quá"],
        "amounts": ["4.000.000", "6.000.000", "8.000.000", "10.000.000", "12.000.000",
                    "800.000", "1.000.000", "300.000", "400.000"],
    },
]

# Vehicle type markers (WITH diacritics for document matching)
VEHICLE_MARKERS = {
    "ô tô": ["điều 6", "xe ô tô", "xe chở người bốn"],
    "xe máy": ["điều 7", "xe mô tô", "xe gắn máy", "các loại"],
    "xe đạp": ["điều 9", "xe đạp", "xe thô sơ"],
}


class QueryExpander:
    """Query expansion with legal synonyms for BM25/FAISS fallback."""

    EXPANSIONS = {
        "uong ruou": ["nồng độ cồn", "có cồn", "miligam", "khí thở"],
        "nong do con": ["uống rượu", "uống bia", "có cồn", "miligam"],
        "co con": ["nồng độ cồn", "uống rượu"],
        "vuot den do": ["đèn tín hiệu", "chấp hành hiệu lệnh", "tín hiệu giao thông"],
        "den do": ["tín hiệu giao thông", "hiệu lệnh đèn", "chấp hành hiệu lệnh"],
        "toc do": ["chạy quá tốc độ", "vượt tốc độ quy định"],
        "phat bao nhieu": ["mức phạt", "hình phạt", "phạt tiền", "xử phạt"],
        "bi phat": ["xử phạt", "mức phạt", "hình phạt"],
    }

    @staticmethod
    def normalize(text: str) -> str:
        """Remove Vietnamese diacritics for keyword matching."""
        text = text.lower().strip()
        for ch in [('đ', 'd'), ('ơ', 'o'), ('ư', 'u')]:
            text = text.replace(ch[0], ch[1]).replace(ch[0].upper(), ch[1])
        nfd = unicodedata.normalize('NFD', text)
        return ' '.join(''.join(c for c in nfd if unicodedata.category(c) != 'Mn').split())

    @staticmethod
    def expand(query: str) -> List[str]:
        """Generate query variants with legal synonyms (max 6)."""
        variants = [query]
        norm = QueryExpander.normalize(query)
        for key, synonyms in QueryExpander.EXPANSIONS.items():
            if key in norm:
                for syn in synonyms:
                    v = query + " " + syn
                    if v not in variants:
                        variants.append(v)
        # Add legal penalty terms if user asks about fines
        if "phat" in norm:
            for t in ["xử phạt", "mức phạt", "hình phạt"]:
                v = query + " " + t
                if v not in variants:
                    variants.append(v)
        return variants[:6]


def _detect_vehicle(query_raw: str) -> str | None:
    """Detect vehicle type from query (checks both with & without diacritics)."""
    q = query_raw.lower()
    if any(t in q for t in ["ô tô", "o to", "xe ô tô", "xe o to", "xe con"]):
        return "ô tô"
    if any(t in q for t in ["xe máy", "xe may", "xe gắn máy", "xe gan may", "mô tô", "mo to"]):
        return "xe máy"
    if any(t in q for t in ["xe đạp", "xe dap", "xe thô sơ", "xe tho so"]):
        return "xe đạp"
    return None


class BM25Scorer:
    """BM25 keyword relevance scorer with pre-tokenized document vectors."""

    def __init__(self, corpus: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.idf = {}
        self._doc_tfs: List[Counter] = []
        self._doc_lens: List[int] = []
        self._build_idf()

    def _tokenize(self, text: str) -> List[str]:
        return [w for w in re.findall(r'\w+', text.lower()) if len(w) > 1]

    def _build_idf(self):
        doc_freq = Counter()
        self._doc_tfs = []
        self._doc_lens = []
        for doc in self.corpus:
            text = (doc.get("text", "") or "") + " " + (doc.get("text_for_display", "") or "")
            tokens = self._tokenize(text)
            self._doc_tfs.append(Counter(tokens))
            self._doc_lens.append(len(tokens))
            for word in set(tokens):
                doc_freq[word] += 1
        n = max(1, len(self.corpus))
        self.idf = {w: max(0.1, (n - f + 0.5) / (f + 0.5)) for w, f in doc_freq.items()}
        self.avg_len = sum(self._doc_lens) / max(1, n)

    def score(self, query: str, doc_idx: int) -> float:
        """Score a single doc (used for targeted re-ranking)."""
        if doc_idx >= len(self._doc_tfs):
            return 0.0
        qwords = self._tokenize(query)
        if not qwords:
            return 0.0
        tf = self._doc_tfs[doc_idx]
        doc_len = self._doc_lens[doc_idx]
        norm = 1 - self.b + self.b * (doc_len / self.avg_len)
        score = 0.0
        for w in qwords:
            if w in tf:
                idf = self.idf.get(w, 0.1)
                score += idf * ((self.k1 + 1) * tf[w]) / (self.k1 * norm + tf[w])
        return max(0.0, score)


class HybridSearcher:
    """3-layer search: Direct Keyword -> FAISS+BM25 fusion."""

    def __init__(self, faiss_retriever, corpus: List[Dict]):
        self.faiss = faiss_retriever
        self.corpus = corpus
        self.bm25 = BM25Scorer(corpus)

    def search(self, query: str, k: int = 3, alpha: float = 0.3) -> List[Dict]:
        """
        Hybrid search with 3 strategies:
        1. Definition question detection (là gì / định nghĩa)
        2. Direct keyword matching (alcohol, red light, speed)
        3. FAISS + BM25 fusion (fallback for other queries)
        """
        # Strategy 0: Definition question detection
        results = self._definition_search(query, k)
        if results:
            return results

        # Strategy 1: Direct keyword search
        results = self._keyword_search(query, k)
        if results:
            return results

        # Strategy 2: FAISS + BM25 fusion with query expansion
        return self._fusion_search(query, k, alpha)

    def _definition_search(self, query: str, k: int) -> List[Dict]:
        """Detect 'là gì' / 'định nghĩa' questions and search definition chunks."""
        # Extract key term from patterns like "X là gì", "định nghĩa X", "X là như thế nào"
        q_lower = query.lower()
        patterns = [
            r"(.+?)\s+(?:là\s+g[iì]|là\s+như\s+thế\s+nào|là\s+sao)",
            r"(?:định\s+ngh[ĩi]a|khái\s+niệm)\s+(.+)",
        ]
        term = None
        for pat in patterns:
            m = re.search(pat, q_lower)
            if m:
                term = m.group(1).strip()
                break

        if not term:
            return []

        # Search for definition chunks (ART2_ = Article 2 definition articles)
        candidates = []
        seen = set()
        for doc_id, doc in enumerate(self.corpus):
            chunk_id = doc.get("chunk_id", "")
            # Target: "ART2_" in chunk ID or "Giải thích từ ngữ" in text
            if "ART2_" not in chunk_id and "Giải thích" not in chunk_id:
                continue

            text = (doc.get("text_for_display") or doc.get("text", "")).lower()
            # The term must appear in the definition text
            if term not in text:
                continue

            key = chunk_id
            if key in seen:
                continue
            seen.add(key)

            candidates.append({
                "rank": 0, "id": doc_id, "score": 0.98,
                "source": doc.get("source", "N/A"), "page": doc.get("page", "N/A"),
                "text": doc.get("text_for_display") or doc.get("text", ""),
                "legal_refs": doc.get("metadata", {}).get("legal_refs", {}),
            })

        if candidates:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            top = candidates[:k]
            for i, item in enumerate(top, 1):
                item["rank"] = i
            return top
        return []

    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        """Direct keyword matching with vehicle type prioritization.

        Matching strategy (diacritics-first):
        1. Check triggers/config keywords against BOTH original text & normalized text
        2. Original (with-diacritics) matches get higher weight than normalized matches
        3. Fallback to normalized matching catches users typing without diacritics
        """
        q_norm = QueryExpander.normalize(query)
        q_lower = query.lower()
        vehicle = _detect_vehicle(query)

        # Find matching config (check both diacritics and normalized forms)
        config = None
        for cfg in QUERY_CONFIGS:
            cfg_triggers_norm = [QueryExpander.normalize(t) for t in cfg["trigger"]]
            if any(t in q_lower for t in cfg["trigger"]) or any(t in q_norm for t in cfg_triggers_norm):
                config = cfg
                break
        if not config:
            return []

        # Also check penalty intent in both forms
        penalty_words = ["phạt", "xử phạt", "mức phạt", "bao nhiêu",
                        "phat", "xu phat", "muc phat", "bao nhieu"]
        user_wants_penalty = any(w in q_lower or w in q_norm for w in penalty_words)

        candidates = []
        seen = set()

        for doc_id, doc in enumerate(self.corpus):
            doc_raw = (doc.get("text", "") or "") + " " + (doc.get("text_for_display", "") or "")
            doc_lower = doc_raw.lower()
            doc_norm = QueryExpander.normalize(doc_raw)

            # Keyword matching: diacritics (original) = higher weight
            kw_orig = sum(1 for kw in config["keywords"] if kw in doc_lower)
            kw_norm = sum(1 for kw in [QueryExpander.normalize(k) for k in config["keywords"]] if kw in doc_norm)

            # Total keyword hits: original has 2x weight (more precise semantically)
            kw_hits = kw_orig * 2 + kw_norm
            if kw_hits == 0:
                continue

            # Amount matching (amounts are digits + dots, no diacritics issue)
            amt_hits = sum(1 for a in config["amounts"] if a in doc_lower)
            has_penalty = amt_hits > 0

            # Score: diacritics match gets higher base
            if user_wants_penalty and has_penalty:
                score = 0.50 + 0.03 * min(kw_orig, 3) + 0.02 * min(kw_norm, 3) + 0.10 * min(amt_hits, 2)
            elif user_wants_penalty and not has_penalty:
                score = 0.25 + 0.03 * min(kw_orig, 3) + 0.02 * min(kw_norm, 3)
            else:
                score = 0.40 + 0.03 * min(kw_orig, 3) + 0.02 * min(kw_norm, 3) + 0.10 * min(amt_hits, 2)

            # Vehicle type boost (check both diacritics and normalized)
            if vehicle and vehicle in VEHICLE_MARKERS:
                disp_lower = doc.get("text_for_display", "").lower()
                disp_norm = QueryExpander.normalize(disp_lower)
                markers = VEHICLE_MARKERS[vehicle]
                markers_norm = [QueryExpander.normalize(m) for m in markers]
                if any(m in disp_lower for m in markers) or any(m in disp_norm for m in markers_norm):
                    score += 0.20

            score = min(1.0, score)

            key = (doc.get("chunk_id", ""), (doc.get("text_for_display") or "")[:200])
            if key not in seen:
                seen.add(key)
                candidates.append({
                    "rank": 0, "id": doc_id, "score": score,
                    "source": doc.get("source", "N/A"), "page": doc.get("page", "N/A"),
                    "text": doc.get("text_for_display") or doc.get("text", ""),
                    "legal_refs": doc.get("metadata", {}).get("legal_refs", {}),
                })

        if candidates:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(candidates[:k], 1):
                item["rank"] = i
            return candidates[:k]
        return []

    def _fusion_search(self, query: str, k: int, alpha: float) -> List[Dict]:
        """FAISS + BM25 fusion with query expansion (fallback). Single batch embedding call."""
        queries = QueryExpander.expand(query)

        # FAISS: batch all query variants in one embedding call
        faiss_scores: Dict[int, float] = {}
        all_candidates = set()
        for results in self.faiss.batch_search(queries, k=k * 3):
            for doc in results:
                did = doc["id"]
                all_candidates.add(did)
                faiss_scores[did] = max(faiss_scores.get(did, 0), doc["score"])

        # BM25: only score FAISS candidates (not entire corpus)
        bm25_scores: Dict[int, float] = {}
        for q in queries:
            for i in all_candidates:
                s = self.bm25.score(q, i)
                if s > 0:
                    bm25_scores[i] = max(bm25_scores.get(i, 0), s)

        # Normalize + combine
        fmax = max(faiss_scores.values()) if faiss_scores else 1.0
        bmax = max(bm25_scores.values()) if bm25_scores else 1.0
        faiss_scores = {did: v / fmax for did, v in faiss_scores.items()} if fmax > 0 else {}
        bm25_scores = {did: v / bmax for did, v in bm25_scores.items()} if bmax > 0 else {}

        all_ids = set(faiss_scores) | set(bm25_scores)
        combined = [(did, alpha * faiss_scores.get(did, 0) + (1 - alpha) * bm25_scores.get(did, 0))
                    for did in all_ids]
        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (did, score) in enumerate(combined[:k], 1):
            if did < len(self.corpus):
                doc = self.corpus[did]
                results.append({
                    "rank": rank, "id": int(did),
                    "score": round(min(score, 1.0), 4),
                    "source": doc.get("source", "N/A"), "page": doc.get("page", "N/A"),
                    "text": doc.get("text_for_display") or doc.get("text", ""),
                    "legal_refs": doc.get("metadata", {}).get("legal_refs", {}),
                })
        return results
