"""
Module: test_retriever.py
Purpose: Test basic vector search functionality
"""

import json
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    import faiss
except ImportError:
    print("❌ Missing faiss-cpu")
    sys.exit(1)

from rag.embedding import get_embeddings


class SimpleRetriever:
    """Test retriever using vector similarity."""
    
    def __init__(self, index_path: str, corpus_path: str):
        """Load FAISS index and corpus."""
        self.index = faiss.read_index(index_path)
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.corpus = data['data']
        
        print(f"✓ Loaded index with {self.index.ntotal} vectors")
        print(f"✓ Loaded {len(self.corpus)} corpus entries")
    
    def search(self, query: str, top_k: int = 5) -> list:
        """Search for top-k similar chunks."""
        # Generate embedding for query
        query_embedding = get_embeddings([query], batch_size=1)
        if query_embedding is None or len(query_embedding) == 0:
            print("❌ Failed to generate query embedding")
            return []
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), 
            top_k
        )
        
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.corpus):
                entry = self.corpus[int(idx)]
                results.append({
                    'rank': rank + 1,
                    'distance': float(dist),
                    'score': 1 / (1 + float(dist)),  # Similarity score (0-1)
                    'chunk_id': entry['chunk_id'],
                    'source': entry['source'],
                    'article': entry['metadata'].get('article_number'),
                    'vehicle_type': entry['metadata'].get('vehicle_type'),
                    'penalty': entry['metadata'].get('penalty_range'),
                    'text_preview': entry['text'][:150],
                })
        
        return results


def main():
    """Test retriever with sample queries."""
    print("\n" + "="*80)
    print("TESTING VECTOR RETRIEVER")
    print("="*80)
    
    retriever = SimpleRetriever(
        'vector_store/faiss_index_new.bin',
        'vector_store/corpus_data_new.json'
    )
    
    test_queries = [
        "vượt đèn đỏ ô tô phạt bao nhiêu",
        "xe máy vi phạm giao thông",
        "phạt tiền giao thông",
        "Điều 6 ô tô",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 80)
        
        results = retriever.search(query, top_k=3)
        
        if not results:
            print("❌ No results found")
            continue
        
        for result in results:
            print(f"\n[#{result['rank']}] Score: {result['score']:.3f}")
            print(f"  Source: {result['source']}")
            print(f"  Article: Điều {result['article']}")
            if result['vehicle_type']:
                print(f"  Vehicle: {result['vehicle_type']}")
            if result['penalty']:
                print(f"  Penalty: {result['penalty'][0]:,} - {result['penalty'][1]:,} VND")
            print(f"  Text: {result['text_preview']}...")


if __name__ == '__main__':
    main()
