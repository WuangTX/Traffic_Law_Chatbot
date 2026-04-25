"""
Module: retriever.py
Author: GitHub Copilot
Date: March 29, 2026
Description: 
    Thành phần "Trình truy xuất" (Retriever) của hệ thống RAG.
    Chịu trách nhiệm tải cơ sở dữ liệu vector (FAISS index) và kho văn bản (corpus),
    sau đó thực hiện tìm kiếm các văn bản liên quan nhất cho một câu hỏi đầu vào.
"""

import faiss
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass

# Thêm đường dẫn gốc của dự án vào sys.path để import các module khác
import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import các hàm cần thiết từ các module đã tạo
from rag.embedding import get_embeddings

# --- Cấu hình các đường dẫn ---
VECTOR_STORE_DIR = project_root / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
CORPUS_PATH = VECTOR_STORE_DIR / "corpus_data.json"
INDEX_MANIFEST_PATH = VECTOR_STORE_DIR / "index_manifest.json"


@dataclass
class IndexShard:
    source: str
    slug: str
    index: faiss.Index
    corpus_ids: list[int]


class Retriever:
    """
    Lớp Retriever quản lý việc tải và tìm kiếm trong cơ sở dữ liệu vector.
    """
    def __init__(self):
        """
        Khởi tạo Retriever, tải index và corpus vào bộ nhớ.
        """
        print("🚀 Khởi tạo Retriever...")
        self.index = self._load_faiss_index()
        self.shards = self._load_index_shards()
        self.corpus = self._load_corpus()
        if (self.index or self.shards) and self.corpus:
            if self.shards:
                total_vectors = sum(shard.index.ntotal for shard in self.shards)
                print(
                    f"✓ Đã tải thành công {len(self.shards)} shard index ({total_vectors} vectors) "
                    f"và corpus ({len(self.corpus)} văn bản)."
                )
            else:
                print(f"✓ Đã tải thành công FAISS index ({self.index.ntotal} vectors) và corpus ({len(self.corpus)} văn bản).")
            print("✓ Retriever đã sẵn sàng!")
        else:
            print("✗ Lỗi: Không thể khởi tạo Retriever. Vui lòng kiểm tra lại các file trong 'vector_store'.")

    def _load_index_shards(self) -> list[IndexShard]:
        """Load per-document FAISS shards if manifest is available."""
        if not INDEX_MANIFEST_PATH.exists():
            return []

        try:
            with open(INDEX_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            print(f"Cảnh báo: Không đọc được manifest shard index: {exc}")
            return []

        shards: list[IndexShard] = []
        for item in manifest.get("shards", []):
            try:
                index_path = Path(item.get("index_path", ""))
                if not index_path.is_absolute():
                    index_path = project_root / index_path
                if not index_path.exists():
                    continue

                shard_index = faiss.read_index(str(index_path))
                corpus_ids = [int(x) for x in item.get("corpus_ids", [])]
                shards.append(
                    IndexShard(
                        source=str(item.get("source", "N/A")),
                        slug=str(item.get("slug", "unknown")),
                        index=shard_index,
                        corpus_ids=corpus_ids,
                    )
                )
            except Exception as exc:
                print(f"Cảnh báo: Bỏ qua shard index lỗi '{item}': {exc}")

        return shards

    def _search_legacy(self, query_embedding: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Legacy search against single index. Returns list of (global_idx, distance)."""
        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        pairs: list[tuple[int, float]] = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            pairs.append((int(idx), float(distances[0][i])))
        return pairs

    def _search_shards(self, query_embedding: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Search all shards and merge nearest neighbors globally."""
        candidates: list[tuple[int, float]] = []
        per_shard_k = max(1, min(k * 3, 20))

        for shard in self.shards:
            local_k = min(per_shard_k, shard.index.ntotal)
            if local_k <= 0:
                continue

            distances, indices = shard.index.search(query_embedding.astype("float32"), local_k)
            for i, local_idx in enumerate(indices[0]):
                if local_idx < 0:
                    continue
                if local_idx >= len(shard.corpus_ids):
                    continue
                global_idx = shard.corpus_ids[int(local_idx)]
                candidates.append((int(global_idx), float(distances[0][i])))

        # Merge and keep best distance per global doc.
        best_by_id: dict[int, float] = {}
        for global_idx, dist in candidates:
            if global_idx not in best_by_id or dist < best_by_id[global_idx]:
                best_by_id[global_idx] = dist

        merged = sorted(best_by_id.items(), key=lambda item: item[1])
        return merged[:k]

    def _load_faiss_index(self):
        """
        Tải FAISS index từ file.
        """
        if not FAISS_INDEX_PATH.exists():
            print(f"Lỗi: Không tìm thấy file FAISS index tại: {FAISS_INDEX_PATH}")
            return None
        try:
            return faiss.read_index(str(FAISS_INDEX_PATH))
        except Exception as e:
            print(f"Lỗi khi đọc file FAISS index: {e}")
            return None

    def _load_corpus(self):
        """
        Tải kho văn bản (corpus) từ file JSON.
        """
        if not CORPUS_PATH.exists():
            print(f"Lỗi: Không tìm thấy file corpus tại: {CORPUS_PATH}")
            return None
        try:
            with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Lỗi khi đọc file corpus: {e}")
            return None

    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        Thực hiện tìm kiếm các văn bản liên quan nhất cho một câu hỏi.

        Args:
            query (str): Câu hỏi của người dùng.
            k (int): Số lượng kết quả liên quan nhất cần trả về. Mặc định là 3.

        Returns:
            list[dict]: Danh sách các dictionary, mỗi dictionary chứa thông tin
                        về một văn bản liên quan (bao gồm cả điểm số tương đồng).
                        Trả về danh sách rỗng nếu có lỗi.
        """
        if (not self.index and not self.shards) or not self.corpus:
            print("Lỗi: Retriever chưa được khởi tạo thành công.")
            return []

        print(f"\n🔎 Đang tìm kiếm cho câu hỏi: '{query}' với k={k}")

        # 1. Mã hóa câu hỏi thành vector
        try:
            query_embedding = get_embeddings([query])
            if query_embedding.shape[0] == 0:
                print("Lỗi: Không thể tạo embedding cho câu hỏi.")
                return []
        except Exception as e:
            print(f"Lỗi trong quá trình embedding câu hỏi: {e}")
            return []

        # 2. Dùng FAISS để tìm kiếm k vector gần nhất
        if self.shards:
            nearest = self._search_shards(query_embedding, k)
        else:
            nearest = self._search_legacy(query_embedding, k)

        # 3. Lấy kết quả và trả về
        results = []
        print(f"✓ Tìm thấy {len(nearest)} kết quả liên quan:")
        for i, (idx, distance) in enumerate(nearest):
            if idx < len(self.corpus):
                # Lấy thông tin từ corpus
                retrieved_doc = self.corpus[idx]
                
                # Tính điểm tương đồng (similarity score) từ khoảng cách (distance)
                # Khoảng cách càng nhỏ, độ tương đồng càng cao.
                # Một cách đơn giản để chuyển đổi: score = 1 / (1 + distance)
                similarity_score = 1 / (1 + distance)

                result_item = {
                    "rank": i + 1,
                    "id": int(idx),
                    "score": float(f"{similarity_score:.4f}"),
                    "source": retrieved_doc.get("source", "N/A"),
                    "page": retrieved_doc.get("page", "N/A"),
                    "text": retrieved_doc.get("text_display") or retrieved_doc.get("text_original", ""),
                    "text_processed": retrieved_doc.get("text_processed", ""),
                    "legal_refs": retrieved_doc.get("legal_refs", {}),
                }
                results.append(result_item)
                print(f"  - Rank {result_item['rank']}: Score={result_item['score']:.4f}, Nguồn: {result_item['source']} (Trang {result_item['page']})")
            else:
                print(f"  - Cảnh báo: Index {idx} không hợp lệ, vượt quá kích thước corpus.")

        return results

# --- Main block để chạy thử nghiệm ---
if __name__ == '__main__':
    # Khởi tạo retriever
    retriever = Retriever()

    # Nếu khởi tạo thành công, thực hiện tìm kiếm thử
    if (retriever.index or retriever.shards) and retriever.corpus:
        
        # Các câu hỏi ví dụ
        test_queries = [
            "vượt đèn đỏ bị phạt bao nhiêu tiền?",
            "nồng độ cồn cho phép khi lái xe máy là bao nhiêu?",
            "quy định về tốc độ tối đa trong khu dân cư",
            "xe máy có được đi vào đường cao tốc không?"
        ]

        # Thực hiện tìm kiếm cho từng câu hỏi
        for q in test_queries:
            search_results = retriever.search(q, k=3) # Tìm 3 kết quả liên quan nhất
            print("-" * 50)
            # In chi tiết kết quả
            for res in search_results:
                print(f"Rank {res['rank']} (Score: {res['score']})")
                print(f"Nguồn: {res['source']} - Trang: {res['page']}")
                print(f"Nội dung: {res['text'][:300]}...") # In 300 ký tự đầu
                print()
            print("=" * 50 + "\n")
