"""
Module: pipeline_executor.py
Purpose: Integrated pipeline to process legal documents end-to-end:
1. Parse hierarchical structure (Chapter > Article > Clause > Point)
2. Extract metadata (vehicle_type, penalties, references, etc.)
3. Generate chunks with Context Hoisting
4. Output flat JSON structure for RAG
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import local modules
from .parser import parse_law_document, Article
from .chunker import articles_to_chunks_with_context
from .metadata_extractor import MetadataExtractor


class DocumentProcessingPipeline:
    """End-to-end pipeline for processing legal documents."""
    
    def __init__(self, law_source: str):
        """
        Initialize pipeline.
        
        Args:
            law_source: Source law/regulation identifier (e.g., "Luật 35/2024", "ND168/2024")
        """
        self.law_source = law_source
        self.articles: List[Article] = []
        self.chunks_raw: List[Dict[str, Any]] = []
        self.chunks_final: List[Dict[str, Any]] = []
    
    def load_document(self, file_path: str) -> bool:
        """Load raw JSON document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            self.articles = parse_law_document(raw_data)
            print(f"✓ Loaded {len(self.articles)} articles from {file_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading document: {e}")
            return False
    
    def generate_chunks_with_context(self) -> bool:
        """Generate chunks with Context Hoisting."""
        try:
            self.chunks_raw = articles_to_chunks_with_context(self.articles)
            print(f"✓ Generated {len(self.chunks_raw)} raw chunks with context hoisting")
            return True
        except Exception as e:
            print(f"✗ Error generating chunks: {e}")
            return False
    
    def enrich_with_metadata(self) -> bool:
        """
        Enrich each chunk with metadata (vehicle_type, penalties, etc.).
        This produces the final chunk structure for RAG.
        """
        try:
            self.chunks_final = []
            
            for chunk in self.chunks_raw:
                # Extract metadata from the chunk text
                metadata = MetadataExtractor.extract_metadata(
                    text=chunk['text_original'],
                    article_title=chunk.get('article_title', ''),
                    article_number=chunk.get('article_number', 0),
                    law_source=self.law_source
                )
                
                # Build final chunk with flat structure
                final_chunk = {
                    'id': chunk['id'],
                    'text_for_embedding': chunk['text_for_embedding'],  # Use hoisted context for embedding
                    'text_original': chunk['text_original'],  # Keep original for reference
                    'metadata': {
                        'law_source': self.law_source,
                        'article_title': chunk.get('article_title', ''),
                        'article_number': chunk.get('article_number'),
                        'clause_number': chunk.get('clause_number'),
                        'point_id': chunk.get('point_id'),
                        'chapter_number': chunk.get('chapter_number'),
                        'chapter_title': chunk.get('chapter_title', ''),
                        
                        # Extracted metadata
                        'vehicle_type': metadata['vehicle_type'],
                        'penalty_range': metadata['penalty_range'],
                        'points_deducted': metadata['points_deducted'],
                        'license_suspension': metadata['license_suspension'],
                        'references': metadata['references'],
                        'effective_date': metadata['effective_date'] or '2025-01-01',  # Default for 2024 laws
                    }
                }
                
                self.chunks_final.append(final_chunk)
            
            print(f"✓ Enriched {len(self.chunks_final)} chunks with metadata")
            return True
        except Exception as e:
            print(f"✗ Error enriching metadata: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_output(self, output_path: str) -> bool:
        """Save final chunks to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks_final, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved {len(self.chunks_final)} chunks to {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error saving output: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {
            'total_articles': len(self.articles),
            'total_chunks': len(self.chunks_final),
            'vehicle_types': {},
            'penalty_distribution': {},
            'with_points_deduction': 0,
            'with_license_suspension': 0,
            'with_references': 0,
        }
        
        for chunk in self.chunks_final:
            meta = chunk.get('metadata', {})
            
            # Vehicle type distribution
            if meta.get('vehicle_type'):
                vtype = meta['vehicle_type']
                stats['vehicle_types'][vtype] = stats['vehicle_types'].get(vtype, 0) + 1
            
            # Count chunks with metadata
            if meta.get('points_deducted'):
                stats['with_points_deduction'] += 1
            if meta.get('license_suspension'):
                stats['with_license_suspension'] += 1
            if meta.get('references'):
                stats['with_references'] += 1
            
            # Penalty distribution
            if meta.get('penalty_range'):
                key = f"{meta['penalty_range'][0]//1000}k-{meta['penalty_range'][1]//1000}k"
                stats['penalty_distribution'][key] = stats['penalty_distribution'].get(key, 0) + 1
        
        return stats
    
    def print_sample_chunks(self, num_samples: int = 3):
        """Print sample chunks for inspection."""
        print(f"\n{'='*80}")
        print(f"SAMPLE CHUNKS (showing {num_samples} of {len(self.chunks_final)})")
        print(f"{'='*80}")
        
        for idx, chunk in enumerate(self.chunks_final[:num_samples], 1):
            print(f"\n[Sample {idx}] ID: {chunk['id']}")
            print(f"Text for embedding: {chunk['text_for_embedding'][:100]}...")
            print(f"Metadata:")
            for key, value in chunk['metadata'].items():
                if value is not None:
                    print(f"  - {key}: {value}")


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("DOCUMENT PROCESSING PIPELINE - LEGAL TEXT TO RAG-READY CHUNKS")
    print("="*80 + "\n")
    
    # Process Luật 35/2024
    print("PROCESSING: Luật 35/2024/QH15 (Luật Đường bộ)")
    print("-" * 80)
    
    pipeline_law35 = DocumentProcessingPipeline("Luật 35/2024/QH15")
    
    if not pipeline_law35.load_document('data/raw/luat_duong_bo_35-2024.json'):
        sys.exit(1)
    
    if not pipeline_law35.generate_chunks_with_context():
        sys.exit(1)
    
    if not pipeline_law35.enrich_with_metadata():
        sys.exit(1)
    
    if not pipeline_law35.save_output('data/processed/luật_35_chunks.json'):
        sys.exit(1)
    
    # Print statistics and samples
    stats = pipeline_law35.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Vehicle types: {stats['vehicle_types']}")
    print(f"  Chunks with points deduction: {stats['with_points_deduction']}")
    print(f"  Chunks with license suspension: {stats['with_license_suspension']}")
    print(f"  Chunks with references: {stats['with_references']}")
    
    pipeline_law35.print_sample_chunks(3)
    
    # Process Nghị định 168/2024 if file exists
    nd168_path = Path('data/raw/nd168_xu_phat_2024.json')
    if nd168_path.exists():
        print("\n" + "="*80)
        print("PROCESSING: Nghị định 168/2024/NĐ-CP (Xử phạt giao thông)")
        print("-" * 80)
        
        pipeline_nd168 = DocumentProcessingPipeline("Nghị định 168/2024/NĐ-CP")
        
        if pipeline_nd168.load_document(str(nd168_path)):
            if pipeline_nd168.generate_chunks_with_context():
                if pipeline_nd168.enrich_with_metadata():
                    if pipeline_nd168.save_output('data/processed/nd168_chunks.json'):
                        stats = pipeline_nd168.get_statistics()
                        print(f"\nStatistics:")
                        print(f"  Total articles: {stats['total_articles']}")
                        print(f"  Total chunks: {stats['total_chunks']}")
                        print(f"  Vehicle types: {stats['vehicle_types']}")
                        print(f"  Chunks with points deduction: {stats['with_points_deduction']}")
                        print(f"  Chunks with references: {stats['with_references']}")
                        
                        pipeline_nd168.print_sample_chunks(3)
    else:
        print(f"\n⚠ Nghị định 168 file not found at {nd168_path}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE ✓")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
