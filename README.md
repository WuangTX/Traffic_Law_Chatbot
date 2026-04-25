# 📚 TABLE OF CONTENTS - Documentation Index

## 🎯 Start Here

**New to this project?** Start with these files in this order:

1. **[QUICK_START.md](QUICK_START.md)** ⭐ START HERE
   - What is this project?
   - How data flows through the pipeline
   - Where files are saved
   - How to search the database

2. **[PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md)** 
   - Complete technical documentation
   - 5-step data processing flow
   - File structure and priorities
   - Data flow diagrams

3. **[CLEANUP_REPORT.md](CLEANUP_REPORT.md)**
   - What files were deleted (20 files)
   - Why they were removed
   - Storage reduction (83% ↓)
   - Final structure verification

---

## 📁 Key Data Files

### Source Documents (data/raw/)
```
luat_duong_bo_35-2024.json    [111 articles]
nd168_xu_phat_2024.json        [301 articles]
```

### Processed Chunks (data/processed/)
```
luật_35_chunks.json            [111 chunks with metadata]
nd168_chunks.json              [301 chunks with metadata]
```

### Vector Database (vector_store/)
```
faiss_index_new.bin            [FAISS index, 412 vectors]
corpus_data_new.json           [Chunk data + metadata]
index_manifest_new.json        [Build information]
```

---

## ⚙️ Pipeline Modules (rag/)

| File | Purpose | Status |
|------|---------|--------|
| `parser.py` | Parse documents into chunks | ✅ Complete |
| `metadata_extractor.py` | Extract penalties, vehicle types, etc. | ✅ Complete |
| `pipeline.py` | Orchestrate end-to-end processing | ✅ Complete |
| `build_vector_db_new.py` | Build FAISS index from chunks | ✅ Complete |
| `embedding.py` | PhoBERT encoding (lazy-loaded) | ✅ Complete |
| `retriever.py` | Query results (needs update for hybrid search) | ⏳ TODO |

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| Total Chunks | 412 |
| Luật 35 Chunks | 111 |
| ND168 Chunks | 301 |
| Embedding Dimension | 768 (PhoBERT) |
| FAISS Index Type | IndexFlatL2 |
| Storage (Total) | ~10 MB |
| Metadata Coverage | 20% vehicle type, 5% penalties |

---

## 🔄 Data Flow

```
Raw JSON (data/raw/)
    ↓
Parser (rag/parser.py)
    ↓
Chunking + Metadata (rag/pipeline.py)
    ↓
JSON Output (data/processed/)
    ↓
PhoBERT Embedding (rag/embedding.py)
    ↓
FAISS Indexing (rag/build_vector_db_new.py)
    ↓
Vector Store (vector_store/)
    ↓
Retrieval (rag/retriever.py + Bước 5 hybrid search)
```

---

## 🚀 What's Next (Bước 5)

### To-Do:
1. Install BM25: `pip install rank_bm25`
2. Create `rag/hybrid_retriever.py`
3. Implement:
   - BM25 keyword search
   - Metadata filtering
   - Result fusion/reranking
4. Update `retriever.py` and `api.py`
5. Run end-to-end tests

### Test Query Examples:
- "vượt đèn đỏ ô tó phạt bao nhiêu"
- "xe máy vi phạm giao thông"
- "Điều 6 trừ điểm"
- "phạt tiền giao thông đường bộ"

---

## 📝 Metadata Fields (In Each Chunk)

```json
{
  "id": "CHI_ART6_K1",
  "text_for_embedding": "[full context]",
  "text_original": "[original text]",
  "metadata": {
    "law_source": "Nghị định 168/2024/NĐ-CP",
    "article_number": 6,
    "article_title": "Xử phạt...",
    "vehicle_type": "ô tó",
    "penalty_range": [400000, 600000],
    "points_deducted": null,
    "references": [],
    "effective_date": "2025-01-01"
  }
}
```

---

## 🧹 Cleanup Summary

**Deleted 20 files** to streamline project:
- 11 legacy preprocessed data files
- 4 deprecated modules  
- 4 old index files
- 1 debug script

**Result**: 83% storage reduction (58 MB → 10 MB)

---

## 💡 Key Concepts

### Context Hoisting
Full hierarchical context preserved in embedding text:
```
"Chương I: I - Điều 6: [title] - Khoản 1: [content]"
```

### Vietnamese Number Format
- Input: "400.000 đồng" (. as thousand separator)
- Output: 400000 (in VND)

### PhoBERT Max Length
- 256 tokens max → text auto-truncates if longer
- L2 distance metric for similarity search

### Metadata Filtering
- Use `vehicle_type == 'ô tó'` to filter ô tó violations
- Use `penalty_range` to filter by penalty amount
- Reduces search space dramatically

---

## 🔗 File Dependencies

```
pipeline.py
├── requires: parser.py
├── requires: metadata_extractor.py
└── outputs: data/processed/*.json

build_vector_db_new.py
├── requires: data/processed/*.json
├── requires: embedding.py
├── requires: faiss library
└── outputs: vector_store/*.bin + .json

retriever.py (UPDATE NEEDED)
├── requires: vector_store/*.bin
├── requires: embedding.py
└── future: rank_bm25 for BM25 search
```

---

## ✅ Status Board

| Component | Status | Last Updated |
|-----------|--------|--------------|
| Parser | ✅ Complete | Apr 25, 2026 |
| Metadata Extractor | ✅ Complete | Apr 25, 2026 |
| Pipeline | ✅ Complete | Apr 25, 2026 |
| Vector DB | ✅ Complete | Apr 25, 2026 |
| Vector Search | ✅ Working | Apr 25, 2026 |
| Hybrid Search | ⏳ To-Do | - |
| API Integration | ⏳ To-Do | - |

---

## 📞 Quick Reference

**Want to...**

- **Understand the full pipeline?** → Read `PIPELINE_SUMMARY.md`
- **Learn how to use the code?** → Read `QUICK_START.md`
- **See what was cleaned up?** → Read `CLEANUP_REPORT.md`
- **Regenerate all chunks?** → Run `python rag/pipeline.py`
- **Rebuild vector DB?** → Run `python rag/build_vector_db_new.py`
- **Test search?** → Run `python test_retriever.py`

---

**Project Progress**: 80% Complete (Bước 1-4 ✅ | Bước 5 ⏳)  
**Last Updated**: April 25, 2026  
**Maintenance**: Low - Clean, modular architecture
