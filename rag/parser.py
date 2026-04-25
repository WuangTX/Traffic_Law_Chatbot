"""
Module: parser.py
Purpose: Parse legal documents into hierarchical structure (Chapter > Article > Clause > Point)
with Context Hoisting for RAG.
"""

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class Point:
    """Represents a single point (điểm) in a clause."""
    point_id: str  # e.g., "a", "b", "c"
    text: str
    point_number: int = 0


@dataclass
class Clause:
    """Represents a clause (khoản) with optional points."""
    clause_number: int
    text: str
    points: List[Point]
    
    def has_points(self) -> bool:
        return len(self.points) > 0


@dataclass
class Article:
    """Represents a legal article (điều) with clauses."""
    article_number: int
    article_title: str
    clauses: List[Clause]
    chapter_number: int
    chapter_title: str


def _extract_clauses_and_points(article_text: str) -> List[Clause]:
    """
    Parse article text to extract clauses (khoản) and points (điểm).
    
    Expected pattern:
    Khoản 1. [text]
      a) [point text]
      b) [point text]
    Khoản 2. [text]
    """
    clauses: List[Clause] = []
    
    # Split by "Khoản" pattern
    khoản_pattern = r'Khoản\s+(\d+)'
    khoản_matches = list(re.finditer(khoản_pattern, article_text))
    
    if not khoản_matches:
        # No explicit clauses - treat entire article as single clause
        clauses.append(Clause(
            clause_number=1,
            text=article_text.strip(),
            points=[]
        ))
        return clauses
    
    for idx, match in enumerate(khoản_matches):
        clause_num = int(match.group(1))
        
        # Extract text from current Khoản to next Khoản (or end)
        start_pos = match.start()
        next_start = khoản_matches[idx + 1].start() if idx + 1 < len(khoản_matches) else len(article_text)
        clause_section = article_text[start_pos:next_start].strip()
        
        # Remove "Khoản X" prefix
        clause_text = re.sub(r'Khoản\s+\d+\.?\s*', '', clause_section, count=1).strip()
        
        # Try to extract points (a), (b), (c), etc.
        points = _extract_points(clause_text)
        
        # If points found, remove them from main text
        if points:
            for point in points:
                clause_text = clause_text.replace(f"{point.point_id})", "").strip()
        
        clauses.append(Clause(
            clause_number=clause_num,
            text=clause_text.strip(),
            points=points
        ))
    
    return clauses


def _extract_points(text: str) -> List[Point]:
    """Extract points (a, b, c, etc.) from clause text."""
    points: List[Point] = []
    
    # Pattern: a) [text], b) [text], etc.
    điểm_pattern = r'([a-zđ])\)\s*(.+?)(?=(?:[a-zđ]\))|$)'
    matches = list(re.finditer(điểm_pattern, text, re.DOTALL | re.IGNORECASE))
    
    for idx, match in enumerate(matches):
        point_id = match.group(1)
        point_text = match.group(2).strip()
        
        # Clean up - remove trailing semicolons or commas
        point_text = re.sub(r'[;,]\s*$', '', point_text)
        
        points.append(Point(
            point_id=point_id,
            text=point_text,
            point_number=idx
        ))
    
    return points


def parse_law_document(raw_data: Dict[str, Any]) -> List[Article]:
    """
    Parse a complete law document into Article objects with full hierarchy.
    
    Args:
        raw_data: Dictionary with 'chapters' key containing list of chapters
        
    Returns:
        List of Article objects with parsed clauses and points
    """
    articles: List[Article] = []
    
    chapters = raw_data.get('chapters', [])
    for chapter in chapters:
        chapter_num = chapter.get('chapter_number', 0)
        chapter_title = chapter.get('chapter_title', 'N/A')
        
        # Handle both structures:
        # 1. Luật 35: chapters -> articles (direct)
        # 2. ND168: chapters -> sections -> articles (nested)
        chapter_articles = chapter.get('articles', [])
        
        # If no direct articles, try to get from sections
        if not chapter_articles:
            sections = chapter.get('sections', [])
            for section in sections:
                chapter_articles.extend(section.get('articles', []))
        
        for art_data in chapter_articles:
            article_num = art_data.get('article_number', 0)
            article_title = art_data.get('article_title', 'N/A')
            article_content = art_data.get('content', '')
            
            # Parse clauses from article content
            clauses = _extract_clauses_and_points(article_content)
            
            article = Article(
                article_number=article_num,
                article_title=article_title,
                clauses=clauses,
                chapter_number=chapter_num,
                chapter_title=chapter_title
            )
            articles.append(article)
    
    return articles


def articles_to_chunks_with_context(articles: List[Article], 
                                   min_context: bool = True) -> List[Dict[str, Any]]:
    """
    Convert parsed articles into chunks with Context Hoisting.
    Each chunk includes full hierarchy context in text prefix.
    
    Args:
        articles: List of Article objects
        min_context: If True, include context even for single-clause articles
        
    Returns:
        List of dictionaries with structure: {
            'id': 'CH1_ART5_K2_Da',
            'text_original': 'Khoản 2, Điểm a: ...',
            'text_context_hoisted': 'Chương I... Điều 5 (title)... Khoản 2, Điểm a:...',
            'chapter_number': 1,
            'article_number': 5,
            'clause_number': 2,
            'point_id': 'a' or None,
            'chapter_title': '...',
            'article_title': '...',
        }
    """
    chunks: List[Dict[str, Any]] = []
    
    for article in articles:
        for clause in article.clauses:
            if clause.has_points():
                # Create a chunk for each point
                for point in clause.points:
                    chunk_id = (
                        f"CH{article.chapter_number}_"
                        f"ART{article.article_number}_"
                        f"K{clause.clause_number}_"
                        f"D{point.point_id}"
                    )
                    
                    # Build hoisted context text
                    context_parts = [
                        f"Chương {article.chapter_number}: {article.chapter_title}",
                        f"Điều {article.article_number}: {article.article_title}",
                        f"Khoản {clause.clause_number}",
                        f"Điểm {point.point_id}: {point.text}"
                    ]
                    
                    chunks.append({
                        'id': chunk_id,
                        'text_original': point.text,
                        'text_context_hoisted': ' - '.join(context_parts),
                        'chapter_number': article.chapter_number,
                        'article_number': article.article_number,
                        'clause_number': clause.clause_number,
                        'point_id': point.point_id,
                        'chapter_title': article.chapter_title,
                        'article_title': article.article_title,
                        'clause_text': clause.text,
                    })
            else:
                # No points - create chunk for entire clause
                chunk_id = (
                    f"CH{article.chapter_number}_"
                    f"ART{article.article_number}_"
                    f"K{clause.clause_number}"
                )
                
                context_parts = [
                    f"Chương {article.chapter_number}: {article.chapter_title}",
                    f"Điều {article.article_number}: {article.article_title}",
                    f"Khoản {clause.clause_number}: {clause.text}"
                ]
                
                chunks.append({
                    'id': chunk_id,
                    'text_original': clause.text,
                    'text_context_hoisted': ' - '.join(context_parts),
                    'chapter_number': article.chapter_number,
                    'article_number': article.article_number,
                    'clause_number': clause.clause_number,
                    'point_id': None,
                    'chapter_title': article.chapter_title,
                    'article_title': article.article_title,
                    'clause_text': clause.text,
                })
    
    return chunks


def main():
    """Test parser on sample data."""
    # Load sample data
    law_path = Path(__file__).parent.parent / "data" / "raw" / "luat_duong_bo_35-2024.json"
    with open(law_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print("=" * 80)
    print("PARSING TEST: Luật 35/2024/QH15")
    print("=" * 80)
    
    # Parse
    articles = parse_law_document(raw_data)
    print(f"✓ Parsed {len(articles)} articles from {len(raw_data.get('chapters', []))} chapters")
    
    # Show first 3 articles
    for article in articles[:3]:
        print(f"\nChương {article.chapter_number}: {article.chapter_title}")
        print(f"  Điều {article.article_number}: {article.article_title}")
        for clause in article.clauses:
            print(f"    Khoản {clause.clause_number}: {clause.text[:80]}...")
            if clause.points:
                for point in clause.points:
                    print(f"      Điểm {point.point_id}: {point.text[:60]}...")
    
    # Convert to chunks
    chunks = articles_to_chunks_with_context(articles)
    print(f"\n✓ Generated {len(chunks)} chunks with context hoisting")
    print("\nSample chunk:")
    if chunks:
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
