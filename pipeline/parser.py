"""
Module: parser.py
Purpose: Parse legal documents into hierarchical structure (Chapter > Article > Clause > Point).
Responsibility: Only handles parsing and creating Article objects.
"""

import re
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


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
    This version is more robust to variations in formatting.
    """
    clauses: List[Clause] = []
    
    # Normalize text to handle different newline characters
    article_text = article_text.replace('\r\n', '\n').strip()

    # Split by clause numbers: "1. Text... 2. Text..."
    # First try newline-separated, then inline-separated
    prelim = re.split(r'\n(?=\d+\.\s)', '\n' + article_text)
    if len(prelim) <= 2:
        # Few splits — try inline splitting (numbers followed by dot + space with cap letter)
        clause_splits = re.split(r'(?:^|(?<=\s))(?=\d+\.\s)', article_text)
    else:
        clause_splits = prelim
    
    # The first split part might be empty or could be a preamble before the first clause.
    # We will process it if it's not empty.
    preamble = clause_splits[0].strip()
    
    # If there are no numbered clauses, treat the whole text as one clause
    if len(clause_splits) <= 1 and preamble:
        points = _extract_points(preamble)
        clauses.append(Clause(
            clause_number=1,
            text=preamble if not points else re.sub(r'([a-zđ])\).+', '', preamble, flags=re.DOTALL).strip(),
            points=points
        ))
        return clauses

    # Process each split part as a potential clause
    for part in clause_splits:
        part = part.strip()
        if not part:
            continue

        # The clause number is at the start of the part
        match = re.match(r'(\d+)\.\s*(.*)', part, re.DOTALL)
        if not match:
            # This part might be a continuation or malformed, skip for now
            continue
            
        clause_num = int(match.group(1))
        clause_content = match.group(2).strip()
        
        # Extract points from this clause's content
        points = _extract_points(clause_content)
        
        # The main text of the clause is the part before any points start
        clause_main_text = clause_content
        if points:
            # Find the start of the first point to trim the main text
            first_point_marker = f"{points[0].point_id})"
            first_point_pos = clause_content.find(first_point_marker)
            if first_point_pos != -1:
                clause_main_text = clause_content[:first_point_pos].strip()

        clauses.append(Clause(
            clause_number=clause_num,
            text=clause_main_text,
            points=points
        ))
        
    return clauses


def _extract_points(text: str) -> List[Point]:
    """Extract points (a, b, c, etc.) from clause text."""
    points: List[Point] = []
    
    # Pattern: looks for "a)", "b)", etc. at the beginning of a line.
    point_pattern = r'^\s*([a-zđ])\)\s*(.*)'
    
    # Split the text by lines to check each one
    lines = text.split('\n')
    
    current_point_id = None
    current_point_text = []
    point_counter = 0

    for line in lines:
        match = re.match(point_pattern, line)
        if match:
            # Found a new point, save the previous one if it exists
            if current_point_id is not None:
                points.append(Point(
                    point_id=current_point_id,
                    text=' '.join(current_point_text).strip(),
                    point_number=point_counter
                ))
                point_counter += 1

            # Start a new point
            current_point_id = match.group(1)
            current_point_text = [match.group(2).strip()]
        elif current_point_id is not None:
            # This line is a continuation of the current point
            current_point_text.append(line.strip())

    # Add the last point
    if current_point_id is not None:
        points.append(Point(
            point_id=current_point_id,
            text=' '.join(current_point_text).strip(),
            point_number=point_counter
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


if __name__ == '__main__':
    print("Parser module - see pipeline.py for complete example")
