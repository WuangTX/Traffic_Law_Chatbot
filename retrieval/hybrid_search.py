"""
Hybrid Search: Direct Keyword + BM25 + FAISS + Query Expansion.
Priority layers for Vietnamese legal document retrieval.
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

from .query_intent import get_dynamic_k, diversify_results, INTENT_LIST


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
        # Alcohol / nồng độ cồn
        "uong ruou": ["nồng độ cồn", "có cồn", "miligam", "khí thở"],
        "nong do con": ["uống rượu", "uống bia", "có cồn", "miligam", "vi phạm nồng độ cồn"],
        "co con": ["nồng độ cồn", "uống rượu"],
        "say ruou": ["nồng độ cồn", "uống rượu bia", "vi phạm nồng độ cồn"],
        "ruou bia": ["nồng độ cồn", "uống rượu", "chất kích thích"],
        # Red light / đèn tín hiệu
        "vuot den do": ["đèn tín hiệu", "chấp hành hiệu lệnh", "tín hiệu giao thông"],
        "den do": ["tín hiệu giao thông", "hiệu lệnh đèn", "chấp hành hiệu lệnh"],
        # Speed / tốc độ
        "toc do": ["chạy quá tốc độ", "vượt tốc độ quy định"],
        "chay qua": ["tốc độ", "vượt tốc độ", "quá tốc độ cho phép"],
        # Fines / phạt
        "phat bao nhieu": ["mức phạt", "hình phạt", "phạt tiền", "xử phạt", "mức xử phạt"],
        "bi phat": ["xử phạt", "mức phạt", "hình phạt"],
        "phat nguoi": ["xử phạt qua camera", "phạt tự động", "phạt qua hình ảnh"],
        # Overtaking / vượt
        "cam vuot": ["vượt xe", "cấm vượt", "đoạn đường cấm vượt"],
        "vuot xe": ["cấm vượt", "vượt không đúng quy định", "vượt sai"],
        # Lane / làn đường
        "sai lan": ["làn đường", "chuyển làn", "sai làn đường", "lấn tuyến"],
        "lan duong": ["làn đường", "chuyển làn", "sai làn", "phần đường"],
        "chuyen lan": ["làn đường", "chuyển làn không đúng", "tín hiệu chuyển làn"],
        "lan tuyen": ["làn đường", "sai làn", "lấn tuyến", "phần đường"],
        # Helmet / mũ bảo hiểm
        "mu bao hiem": ["mũ bảo hiểm", "không đội mũ", "đội mũ không cài quai"],
        "khong doi mu": ["mũ bảo hiểm", "đội mũ bảo hiểm", "không đội mũ bảo hiểm"],
        # Parking / đỗ xe
        "do xe": ["dừng xe", "đỗ xe", "đậu xe", "dừng đỗ"],
        "dung xe": ["đỗ xe", "dừng đỗ", "dừng xe sai quy định"],
        # License / giấy phép
        "bang lai": ["giấy phép lái xe", "bằng lái xe", "tước bằng lái"],
        "giay phep lai": ["bằng lái", "tước bằng", "giấy phép lái xe"],
        "tuoc bang": ["tước giấy phép lái xe", "tạm giữ bằng", "tước bằng lái"],
        # Accident / tai nạn
        "tai nan": ["va chạm", "gây tai nạn", "tai nạn giao thông"],
        # Wrong way / ngược chiều
        "nguoc chieu": ["đi ngược chiều", "đường một chiều", "cấm đi ngược chiều"],
        "mot chieu": ["đường một chiều", "ngược chiều", "đi ngược chiều"],
        # Registration / giấy tờ
        "giay to": ["đăng ký xe", "giấy đăng ký", "bảo hiểm", "đăng kiểm"],
        "dang kiem": ["đăng ký xe", "giấy tờ xe", "tem kiểm định"],
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
        """Generate query variants with legal synonyms (max 10)."""
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
        return variants[:10]


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

    def search(self, query: str, k: int | None = None, alpha: float = 0.35) -> Tuple[List[Dict], str, int]:
        """
        Hybrid search with 3 strategies + adaptive retrieval depth.

        If k is None (default), uses dynamic k based on query intent:
          definition → k=1   comparison → k=5
          list_query  → k=15  detail     → k=3

        Returns (results, intent_label, dynamic_k_used).
        """
        # Detect intent and compute dynamic k (unless caller overrides)
        intent, dynamic_k = get_dynamic_k(query)
        if k is None:
            k = dynamic_k
        print(f"[Hybrid] Intent={intent}, k={k}")

        results: List[Dict] = []

        # Strategy 0: Definition question detection
        results = self._definition_search(query, max(k, 3))
        if results:
            results = results[:k]
            return (results, intent, k)

        # Strategy 1: Direct keyword search
        results = self._keyword_search(query, max(k, 5))
        if results:
            results = results[:k]
            return (results, intent, k)

        # Strategy 2: FAISS + BM25 fusion with query expansion
        # Fetch more candidates for list queries, then diversify
        fetch_k = max(k * 2, 10) if intent == INTENT_LIST else max(k, 5)
        results = self._fusion_search(query, fetch_k, alpha)

        # Diversity post-processing for list queries
        if intent == INTENT_LIST and len(results) > 1:
            results = diversify_results(results, k)
        else:
            results = results[:k]

        return (results, intent, k)

    def _definition_search(self, query: str, k: int) -> List[Dict]:
        """Detect 'là gì' / 'định nghĩa' questions and search ALL chunks for the term."""
        # Extract key term from patterns like "X là gì", "định nghĩa X"
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

        if not term or len(term) < 3:
            return []

        # Search ALL chunks for the term — score by definition quality
        candidates = []
        seen = set()
        for doc_id, doc in enumerate(self.corpus):
            text = (doc.get("text_for_display") or doc.get("text", "")).lower()
            raw_text = (doc.get("text") or "").lower()

            # Term must appear somewhere in the chunk
            if term not in text and term not in raw_text:
                continue

            chunk_id = doc.get("chunk_id", "")
            key = chunk_id
            if key in seen:
                continue
            seen.add(key)

            # Score: definition-like context gets higher score
            is_art2 = "ART2_" in chunk_id or "giải thích" in chunk_id.lower()
            # Check if term appears in "X là Y" pattern (strong definition signal)
            has_def_pattern = bool(re.search(
                re.escape(term) + r"\s+(?:là|được\s+hiểu\s+là|được\s+xác\s+định\s+là|gồm|bao\s+gồm)",
                text
            ))

            if is_art2 and has_def_pattern:
                score = 0.98
            elif has_def_pattern:
                score = 0.96
            elif is_art2:
                score = 0.88
            else:
                score = 0.72

            candidates.append({
                "rank": 0, "id": doc_id, "score": score, "strategy": "definition",
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
                    "rank": 0, "id": doc_id, "score": score, "strategy": "keyword",
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
        """FAISS + BM25 fusion with query expansion (fallback). Single batch embedding call.

        Score design:
        - FAISS raw scores (1/(1+L2)) are already in [0,1] but heavily compressed
          (typical: 0.02-0.15). We scale by a FIXED factor (not max-normalize) to
          avoid inflating bad matches. Genuine good matches get 0.5-0.9; noise stays low.
        - BM25 is unbounded → normalized by per-query max.
        - Final: alpha * scaled_faiss + (1-alpha) * norm_bm25
        """
        FAISS_SCALE = 5.0  # raw 0.15 → 0.75; raw 0.02 → 0.10

        queries = QueryExpander.expand(query)

        # FAISS: batch all query variants in one embedding call
        faiss_raw: Dict[int, float] = {}
        all_candidates = set()
        for results in self.faiss.batch_search(queries, k=k * 3):
            for doc in results:
                did = doc["id"]
                all_candidates.add(did)
                faiss_raw[did] = max(faiss_raw.get(did, 0), doc["score"])

        # BM25: only score FAISS candidates (not entire corpus)
        bm25_scores: Dict[int, float] = {}
        for q in queries:
            for i in all_candidates:
                s = self.bm25.score(q, i)
                if s > 0:
                    bm25_scores[i] = max(bm25_scores.get(i, 0), s)

        # Scale FAISS (fixed multiplier, not max-normalize — preserves absolute quality)
        faiss_scaled = {did: min(v * FAISS_SCALE, 1.0) for did, v in faiss_raw.items()}

        # Normalize BM25 by per-query max (BM25 is unbounded)
        bmax = max(bm25_scores.values()) if bm25_scores else 1.0
        bm25_norm = {did: v / bmax for did, v in bm25_scores.items()} if bmax > 0 else {}

        # Quality penalty: if the best raw FAISS match is weak, penalize ALL fusion scores.
        # raw FAISS scores (1/(1+L2)) typically: 0.10+ = relevant, 0.05-0.10 = fuzzy, <0.05 = noise.
        # Without this, BM25 alone can drive a bad match to 1.0 when both are max-normalized.
        raw_faiss_best = max(faiss_raw.values()) if faiss_raw else 0
        quality_factor = min(1.0, raw_faiss_best / 0.10)  # raw 0.10 → factor=1.0; raw 0.02 → factor=0.2
        print(f"[Fusion] raw_faiss_best={raw_faiss_best:.4f}, quality_factor={quality_factor:.3f}, k={k}")

        # Combine with quality penalty
        all_ids = set(faiss_scaled) | set(bm25_norm)
        combined = [(did, quality_factor * (alpha * faiss_scaled.get(did, 0) + (1 - alpha) * bm25_norm.get(did, 0)))
                    for did in all_ids]
        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (did, score) in enumerate(combined[:k], 1):
            if did < len(self.corpus):
                doc = self.corpus[did]
                results.append({
                    "rank": rank, "id": int(did),
                    "score": round(min(score, 1.0), 4), "strategy": "fusion",
                    "source": doc.get("source", "N/A"), "page": doc.get("page", "N/A"),
                    "text": doc.get("text_for_display") or doc.get("text", ""),
                    "legal_refs": doc.get("metadata", {}).get("legal_refs", {}),
                })
        return results
