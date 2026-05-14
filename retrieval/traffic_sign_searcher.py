"""
Traffic Sign Search: FAISS + keyword search for Vietnamese traffic signs.
"""
from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SIGNS_JSON_PATH = PROJECT_ROOT / "data" / "raw" / "traffic_signs_complete.json"
SIGNS_INDEX_PATH = PROJECT_ROOT / "vector_store" / "traffic_signs_index.bin"
SIGNS_CORPUS_PATH = PROJECT_ROOT / "vector_store" / "traffic_signs_corpus.json"
IMAGES_DIR = PROJECT_ROOT / "data" / "images" / "traffic_signs_images"

# Sign category display names
CATEGORY_NAMES = {
    "P": "Biển cấm",
    "W": "Biển nguy hiểm",
    "R": "Biển hiệu lệnh",
    "I": "Biển chỉ dẫn",
    "IE": "Biển chỉ dẫn trên đường cao tốc",
    "S": "Biển phụ",
    "DP": "Biển phụ",
    "Vạch": "Vạch kẻ đường",
}

# Vietnamese stopwords that dilute search signals
_VI_STOPWORDS = {
    "là", "gì", "nào", "sao", "cho", "của", "và", "các", "có",
    "được", "những", "này", "biển", "báo", "trong", "khi", "với", "về",
    "một", "hay", "đó", "đây", "bị", "ra", "vào", "để", "ở", "tại",
    "như", "thế", "phải", "từ", "đến", "nếu", "thì", "mà",
}


def _extract_category(name: str) -> str:
    """Extract category code from sign name: 'Biển P.101' -> 'P'"""
    m = re.match(r"Biển\s+([A-Za-z]+)", name)
    return m.group(1) if m else ""


def _search_text(item: dict) -> str:
    """Full searchable text for a sign."""
    parts = [item.get("name", ""), item.get("description", "")]
    category = _extract_category(item.get("name", ""))
    if category in CATEGORY_NAMES:
        parts.append(CATEGORY_NAMES[category])
    return " ".join(parts)


def _tokenize(text: str) -> set[str]:
    """Tokenize and filter stopwords."""
    words = set(re.findall(r"\w+", text.lower()))
    return words - _VI_STOPWORDS


class TrafficSignSearcher:
    """Search traffic signs by keyword or semantic similarity."""

    def __init__(self):
        self._signs: List[Dict] = []
        self._index = None
        self._load()

    def _load(self):
        """Load sign corpus and FAISS index from disk."""
        if not SIGNS_CORPUS_PATH.exists():
            raise FileNotFoundError(
                f"Traffic sign corpus not found: {SIGNS_CORPUS_PATH}\n"
                "Run: python -m retrieval.traffic_sign_searcher --build"
            )
        with open(SIGNS_CORPUS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._signs = data["signs"]

        if SIGNS_INDEX_PATH.exists():
            import faiss
            self._index = faiss.read_index(str(SIGNS_INDEX_PATH))
            print(f"Traffic signs ready: {len(self._signs)} signs, {self._index.ntotal} vectors")
        else:
            print(f"Traffic signs loaded (no FAISS index): {len(self._signs)} signs")

    @staticmethod
    def build():
        """Build FAISS index from traffic_signs_complete.json. Run once offline."""
        import faiss
        from pipeline.embedding import get_embeddings

        print("Building traffic sign search index...")

        with open(SIGNS_JSON_PATH, "r", encoding="utf-8") as f:
            raw_signs = json.load(f)

        print(f"  Loaded {len(raw_signs)} signs from JSON")

        # Prepare search texts and image paths
        texts = []
        signs_out = []
        for item in raw_signs:
            texts.append(_search_text(item))

            # Determine local image path
            local_path = item.get("local_path", "")
            # Normalize path: traffic_signs_images\X.svg -> data/images/traffic_signs_images/X.svg
            if local_path:
                filename = Path(local_path).name
                img_rel = "data/images/traffic_signs_images/" + filename
                img_exists = (PROJECT_ROOT / img_rel).exists()
            else:
                filename = ""
                img_rel = ""
                img_exists = False

            signs_out.append({
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "category": _extract_category(item.get("name", "")),
                "category_name": CATEGORY_NAMES.get(_extract_category(item.get("name", "")), ""),
                "image_path": img_rel,
                "image_exists": img_exists,
            })

        # Generate embeddings
        print(f"  Generating embeddings for {len(texts)} signs...")
        embeddings = get_embeddings(texts, batch_size=64)

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))

        # Save
        faiss.write_index(index, str(SIGNS_INDEX_PATH))
        with open(SIGNS_CORPUS_PATH, "w", encoding="utf-8") as f:
            json.dump({"signs": signs_out}, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Index saved: {SIGNS_INDEX_PATH} ({index.ntotal} vectors)")
        print(f"  ✓ Corpus saved: {SIGNS_CORPUS_PATH} ({len(signs_out)} signs)")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search traffic signs by keyword + FAISS fusion."""
        q_lower = query.lower()
        q_words = _tokenize(query)

        # Keyword match
        keyword_results = []
        for i, sign in enumerate(self._signs):
            name_lower = sign["name"].lower()
            desc_lower = sign["description"].lower()
            score = 0.0

            # Exact sign code match (e.g. "P.101", "Biển P.101", "p126", "W201a")
            code_match = re.search(r"[A-Za-z]+\.?\d+[a-z]?", query, re.IGNORECASE)
            if code_match:
                sign_code = code_match.group(0).upper().replace(".", "")
                name_clean = name_lower.replace("biển ", "").strip().replace(".", "")
                if sign_code.lower() == name_clean:
                    score = max(score, 0.98)
                elif sign_code.lower() in name_clean:
                    score = max(score, 0.95)
                elif name_clean.startswith(sign_code.lower()):
                    score = max(score, 0.90)

            # Phrase match: full description as substring
            if len(desc_lower) >= 4 and desc_lower in q_lower:
                score = max(score, 0.92)

            # Keyword overlap — prioritize query coverage (what user typed matters most)
            desc_words = _tokenize(desc_lower)
            if desc_words and q_words:
                overlap = len(q_words & desc_words)
                desc_coverage = overlap / max(len(desc_words), 1)
                query_coverage = overlap / max(len(q_words), 1)
                combined = (desc_coverage * 0.3 + query_coverage * 0.7)
                score = max(score, min(0.94, combined * 0.92))

            # Category match
            cat = sign.get("category", "")
            cat_name = sign.get("category_name", "").lower()
            if cat.lower() in q_lower or cat_name in q_lower:
                score = max(score, 0.40)

            if score > 0.18:
                keyword_results.append((score, i))

        keyword_results.sort(key=lambda x: x[0], reverse=True)

        # FAISS semantic search
        faiss_results = []
        if self._index is not None:
            try:
                from pipeline.embedding import get_embedding
                q_emb = get_embedding(query).reshape(1, -1).astype("float32")
                distances, indices = self._index.search(q_emb, k * 3)
                for j, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self._signs):
                        sim_score = round(1 / (1 + float(distances[0][j])), 4)
                        faiss_results.append((sim_score, int(idx)))
            except Exception as e:
                print(f"[Sign FAISS] Error: {e}")

        # Fusion: keyword first, then FAISS (dedup by name)
        seen = set()
        merged = []
        for score, idx in keyword_results + faiss_results:
            name = self._signs[idx]["name"]
            if name not in seen:
                seen.add(name)
                merged.append((score, idx))

        merged.sort(key=lambda x: x[0], reverse=True)
        top = merged[:k]

        # Format results
        results = []
        for rank, (score, idx) in enumerate(top, 1):
            sign = self._signs[idx]
            results.append({
                "rank": rank,
                "score": round(score, 4),
                "name": sign["name"],
                "description": sign["description"],
                "category": sign.get("category", ""),
                "category_name": sign.get("category_name", ""),
                "image_path": sign.get("image_path", ""),
                "image_exists": sign.get("image_exists", False),
            })
        return results

    def get_sign_by_name(self, name: str) -> Optional[Dict]:
        """Get a single sign by exact name (for image serving)."""
        for sign in self._signs:
            if sign["name"] == name:
                return sign
        return None


# CLI: build index or quick test
if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        TrafficSignSearcher.build()
    else:
        searcher = TrafficSignSearcher()
        for q in ["biển cấm đi ngược chiều", "P.101", "biển báo đường ưu tiên", "biển dừng lại"]:
            print(f"\n  Query: {q}")
            for r in searcher.search(q, k=3):
                print(f"    {r['name']} ({r['description']}) score={r['score']}")
