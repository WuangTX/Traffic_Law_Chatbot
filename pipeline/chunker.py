"""
Chunker: Convert parsed articles to compact embedding-ready chunks.
Uses short context format to stay within PhoBERT's 256-token limit.
"""

from typing import Any, Dict, List
from .parser import Article


def _short_article_context(article: Article) -> str:
    """Compact article header. E.g. 'Dieu 6: Xu phat xe o to'."""
    title = article.article_title
    # Strip common verbose prefixes
    for prefix in ["Xử phạt, trừ điểm giấy phép lái xe của ", "Xử phạt, trừ điểm giấy phép lái của "]:
        title = title.replace(prefix, "")
    # Handle malformed titles (parser artifacts)
    if not title or title.startswith("của ") or len(title) < 5:
        title = article.chapter_title or ""
        for prefix in ["Xử phạt, trừ điểm giấy phép lái xe của ", "Xử phạt, trừ điểm giấy phép lái của "]:
            title = title.replace(prefix, "")
    return f"Dieu {article.article_number}: {title}"


def _short_clause_context(clause_text: str | None) -> str:
    """Extract just the penalty amount from clause text."""
    if not clause_text:
        return ""
    import re
    match = re.search(r'(Phạt tiền từ [^;]+)', clause_text)
    if match:
        return match.group(1)
    return clause_text[:120]  # Fallback: first 120 chars


def _truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate violation text to stay under PhoBERT's 256-token limit."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Cut at last complete sentence boundary
    cut = text.rfind(';', 0, max_chars)
    if cut == -1:
        cut = text.rfind('.', 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut + 1]


def articles_to_chunks_with_context(articles: List[Article]) -> List[Dict[str, Any]]:
    """
    Convert articles into compact embedding chunks.
    Format: 'Dieu X: short_title - Khoan Y: penalty - Diem Z): violation_text'
    This keeps most chunks under 256 tokens for PhoBERT.
    """
    chunks: List[Dict[str, Any]] = []

    for article in articles:
        art_ctx = _short_article_context(article)

        for clause in article.clauses:
            penalty_ctx = _short_clause_context(clause.text)
            clause_header = f"Khoan {clause.clause_number}"
            if penalty_ctx:
                clause_header += f": {penalty_ctx}"

            if clause.has_points():
                for point in clause.points:
                    chunk_id = (
                        f"CH{article.chapter_number}_"
                        f"ART{article.article_number}_"
                        f"K{clause.clause_number}_"
                        f"D{point.point_id}"
                    )
                    # Compact format: Article header | Clause penalty | Point text
                    text_for_embedding = f"{art_ctx} - {clause_header} - Diem {point.point_id}): {_truncate_text(point.text)}"

                    chunks.append({
                        'id': chunk_id,
                        'text_original': point.text,
                        'text_for_embedding': text_for_embedding,
                        'metadata': {
                            'law_source': 'N/A',
                            'chapter_number': article.chapter_number,
                            'article_number': article.article_number,
                            'clause_number': clause.clause_number,
                            'point_id': point.point_id,
                            'chapter_title': article.chapter_title,
                            'article_title': article.article_title,
                        }
                    })
            else:
                chunk_id = (
                    f"CH{article.chapter_number}_"
                    f"ART{article.article_number}_"
                    f"K{clause.clause_number}"
                )
                text_for_embedding = f"{art_ctx} - {clause_header}: {_truncate_text(clause.text or '')}"

                chunks.append({
                    'id': chunk_id,
                    'text_original': clause.text,
                    'text_for_embedding': text_for_embedding,
                    'metadata': {
                        'law_source': 'N/A',
                        'chapter_number': article.chapter_number,
                        'article_number': article.article_number,
                        'clause_number': clause.clause_number,
                        'point_id': None,
                        'chapter_title': article.chapter_title,
                        'article_title': article.article_title,
                    }
                })

    return chunks
