"""
Pipeline package: Data processing and preparation for RAG.
Includes parsing, chunking, metadata extraction, and embedding generation.
"""

from .parser import parse_law_document, Article, Clause, Point
from .chunker import articles_to_chunks_with_context

__all__ = [
    'parse_law_document',
    'articles_to_chunks_with_context',
    'Article',
    'Clause',
    'Point',
]
