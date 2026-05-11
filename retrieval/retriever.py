"""
Retriever: FAISS vector search + corpus management.
"""

import faiss
import json
from pathlib import Path
import numpy as np
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pipeline.embedding import get_embeddings

VECTOR_STORE_DIR = project_root / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index_new.bin"
CORPUS_PATH = VECTOR_STORE_DIR / "corpus_data_new.json"


class Retriever:
    """Loads FAISS index + corpus, performs vector similarity search."""

    def __init__(self):
        self.index = self._load_faiss_index()
        self.corpus = self._load_corpus()
        if self.index and self.corpus:
            print(f"Retriever ready: {self.index.ntotal} vectors, {len(self.corpus)} documents")
        else:
            print("ERROR: Retriever failed to initialize. Check vector_store/ files.")

    def _load_faiss_index(self):
        if not FAISS_INDEX_PATH.exists():
            print(f"FAISS index not found: {FAISS_INDEX_PATH}")
            return None
        try:
            return faiss.read_index(str(FAISS_INDEX_PATH))
        except Exception as e:
            print(f"Error reading FAISS index: {e}")
            return None

    def _load_corpus(self):
        if not CORPUS_PATH.exists():
            print(f"Corpus not found: {CORPUS_PATH}")
            return None
        try:
            with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                return data
        except Exception as e:
            print(f"Error reading corpus: {e}")
            return None

    def _formatted_result(self, idx: int, distance: float, rank: int) -> dict:
        """Build a result dict from index + distance."""
        doc = self.corpus[idx]
        return {
            "rank": rank,
            "id": int(idx),
            "score": round(1 / (1 + float(distance)), 4),
            "source": doc.get("source", "N/A"),
            "page": doc.get("page", "N/A"),
            "text": doc.get("text_for_display") or doc.get("text", ""),
            "legal_refs": doc.get("metadata", {}).get("legal_refs", {}),
        }

    def search(self, query: str, k: int = 3) -> list[dict]:
        """FAISS vector similarity search for a single query."""
        results = self.batch_search([query], k=k)
        return results[0] if results else []

    def batch_search(self, queries: list[str], k: int = 3) -> list[list[dict]]:
        """FAISS search for multiple queries with a single embedding call."""
        if not self.index or not self.corpus:
            return [[] for _ in queries]

        try:
            embeddings = get_embeddings(queries)
            if embeddings.shape[0] == 0:
                return [[] for _ in queries]
        except Exception as e:
            print(f"Embedding error: {e}")
            return [[] for _ in queries]

        distances_all, indices_all = self.index.search(embeddings.astype("float32"), k)

        all_results = []
        for i in range(len(queries)):
            results = []
            for j, idx in enumerate(indices_all[i]):
                if 0 <= idx < len(self.corpus):
                    results.append(self._formatted_result(idx, distances_all[i][j], j + 1))
            all_results.append(results)
        return all_results


if __name__ == '__main__':
    retriever = Retriever()
    if retriever.index and retriever.corpus:
        for q in [
            "vượt đèn đỏ bị phạt bao nhiêu tiền?",
            "nồng độ cồn cho phép khi lái xe máy là bao nhiêu?",
        ]:
            for res in retriever.search(q, k=2):
                print(f"  Score={res['score']:.4f} | {res['source']}")
                print(f"  {res['text'][:200]}")
