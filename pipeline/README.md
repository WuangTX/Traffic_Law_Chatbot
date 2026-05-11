# 📖 Pipeline Package README

## 🎯 Purpose

This package is responsible for the entire **document processing pipeline**. It takes raw legal documents in JSON format and transforms them into a structured, enriched, and chunked format ready for vector embedding and retrieval.

---

## 🚀 How to Run the Pipeline

To execute the full document processing pipeline, run the `pipeline_executor.py` module from the project's root directory.

**Command:**
```bash
# Make sure your virtual environment is activated
python -m pipeline.pipeline_executor
```

---

## ⚙️ What It Does (Step-by-Step)

When executed, the pipeline performs the following sequence of operations:

1.  **Load Raw Documents**:
    - Reads the source JSON files from `data/raw/`.
    - Specifically: `luat_duong_bo_35-2024.json` and `nd168_xu_phat_2024.json`.

2.  **Parse Hierarchical Structure (`parser.py`)**:
    - Analyzes the raw text to understand its legal structure.
    - Builds an in-memory representation of Chapters, Articles, Clauses, and Points.

3.  **Generate Chunks with Context (`chunker.py`)**:
    - Splits the parsed articles into smaller, more focused "chunks" (usually at the Clause or Point level).
    - Applies **Context Hoisting**: Prepends parent titles (e.g., "Chương X: ... - Điều Y: ...") to each chunk's text to provide essential context for the embedding model.

4.  **Enrich with Metadata (`metadata_extractor.py`)**:
    - Scans the text of each chunk to extract valuable metadata using regular expressions.
    - Extracted data includes:
        - Vehicle types (`ô tô`, `xe máy`)
        - Penalty ranges (`phạt tiền từ ... đến ...`)
        - Points deducted (`trừ ... điểm giấy phép lái xe`)
        - License suspension periods
        - References to other articles/laws

5.  **Save Processed Chunks**:
    - Writes the final list of enriched chunks to new JSON files in the `data/processed/` directory.
    - Output files:
        - `luật_35_chunks.json`
        - `nd168_chunks.json`

---

## 📦 Key Modules

-   **`parser.py`**: Handles the initial parsing of raw JSON into a structured `Article` format.
-   **`chunker.py`**: Contains the logic for splitting articles into smaller chunks using the context hoisting strategy.
-   **`metadata_extractor.py`**: Provides tools to extract specific legal details (penalties, vehicle types) from text.
-   **`embedding.py`**: A utility for generating text embeddings using the PhoBERT model (used by the `retrieval` package, not directly in this pipeline's execution flow).
-   **`pipeline_executor.py`**: The main orchestrator. It imports and calls the other modules in the correct order to run the end-to-end process.

---

## 📊 Output

The primary output of this pipeline is the set of JSON files in `data/processed/`. These files serve as the direct input for the next major step in the RAG system: building the vector database (handled by the `retrieval` package).
