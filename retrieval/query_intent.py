"""
Query Intent Detection + Dynamic k sizing.

Detects user intent from Vietnamese traffic-law questions and returns
an appropriate retrieval depth (k), avoiding over-retrieval for exact
lookups while expanding coverage for list/enumeration queries.
"""

import re
from typing import Dict, List, Tuple

# ── Intent labels ──────────────────────────────────────────────────────────

INTENT_DEFINITION = "definition"   # "X là gì" — need 1 exact match
INTENT_DETAIL     = "detail"       # "phạt bao nhiêu" — need 1-3 relevant chunks
INTENT_COMPARISON = "comparison"   # "A khác gì B" — need 3-5 chunks from both sides
INTENT_LIST       = "list_query"   # "các biển cấm" — need many diverse results

# ── Trigger lists ──────────────────────────────────────────────────────────

DEFINITION_PATTERNS = [
    r"là\s+g[iì]\b",           # "là gì"
    r"là\s+sao\b",              # "là sao"
    r"là\s+như\s+thế\s+nào",   # "là như thế nào"
    r"định\s+ngh[ĩi]a",        # "định nghĩa"
    r"khái\s+niệm",            # "khái niệm"
    r"ngh[ĩi]a\s+là\s+g[iì]",  # "nghĩa là gì"
    r"hiểu\s+như\s+thế\s+nào", # "hiểu như thế nào"
]

LIST_TRIGGERS = [
    "những", "các", "bao gồm", "gồm", "liệt kê",
    "có những", "tất cả", "những loại", "các loại",
    "mấy loại", "bao nhiêu loại", "những trường hợp",
    "các trường hợp", "danh sách",
]

COMPARE_TRIGGERS = [
    "khác gì", "khác nhau", "so sánh", "phân biệt",
    "khác biệt", "giống nhau",
]

# ── Penalty indicators (strong signal for detail intent) ───────────────────

PENALTY_PATTERNS = [
    r"phạt\s+bao\s+nhiêu", r"mức\s+phạt", r"bị\s+phạt",
    r"xử\s+phạt", r"phạt\s+tiền", r"phạt\s+thế\s+nào",
    r"phạt\s+ra\s+sao",
]


def detect_intent(query: str) -> Tuple[str, int]:
    """
    Detect query intent and return (intent_label, recommended_k).

    Detection order matters — check narrowest intents first:
      1. definition  → k=1  (exact match needed)
      2. comparison  → k=5  (both sides)
      3. list_query  → k=15 (broad coverage)
      4. detail      → k=3  (default for penalty/specific questions)

    Returns
    -------
    (intent_label, k)  e.g. ("detail", 3)
    """
    q_lower = query.lower().strip()

    # 1. Definition: patterns like "X là gì", "định nghĩa X"
    for pat in DEFINITION_PATTERNS:
        if re.search(pat, q_lower):
            return (INTENT_DEFINITION, 1)

    # 2. Comparison: "A khác gì B", "so sánh A và B"
    for trigger in COMPARE_TRIGGERS:
        if trigger in q_lower:
            return (INTENT_COMPARISON, 5)

    # 3. List query: "các biển cấm", "những lỗi bị phạt"
    if _has_list_intent(q_lower):
        return (INTENT_LIST, 15)

    # 4. Default: detail/specific question
    return (INTENT_DETAIL, 3)


def _has_list_intent(q_lower: str) -> bool:
    """Check if query asks for enumeration/listing."""
    # Leading trigger words strongly indicate list intent
    for trigger in LIST_TRIGGERS:
        # Trigger at start of query or after punctuation
        if q_lower.startswith(trigger) or trigger in q_lower:
            return True
    return False


def get_dynamic_k(query: str) -> Tuple[str, int]:
    """
    Public API: returns (intent_label, k) for a query.

    Usage:
        intent, k = get_dynamic_k("vượt đèn đỏ phạt bao nhiêu")
        # → ("detail", 3)
    """
    return detect_intent(query)


# ── Diversity filtering for list queries ───────────────────────────────────

def diversify_results(
    results: List[Dict],
    k: int,
    dedup_by: str = "article",
) -> List[Dict]:
    """
    Post-process results to maximise diversity.

    Strategy:
      - Each unique article gets at most 1 slot in the top results
      - Results from different articles are preferred over multiple from same article
      - Falls back to original order if no more unique articles remain

    Parameters
    ----------
    results : list of result dicts (must have "text" field)
    k : target number of results
    dedup_by : grouping key — "article" extracts Điều X from chunk text

    Returns
    -------
    Diversified list of at most k results.
    """
    if len(results) <= 1:
        return results

    # Extract grouping key from each result
    def _extract_key(r: Dict) -> str:
        text = r.get("text", "")
        if dedup_by == "article":
            m = re.search(r"Điều\s+(\d+)", text)
            return f"art_{m.group(1)}" if m else text[:60]
        return text[:60]

    diverse: List[Dict] = []
    seen_keys: set = set()

    # First pass: take one from each unique key
    for r in results:
        key = _extract_key(r)
        if key not in seen_keys:
            seen_keys.add(key)
            diverse.append(r)
            if len(diverse) >= k:
                break

    # Second pass: fill remaining slots if not enough unique ones
    if len(diverse) < k:
        for r in results:
            if r not in diverse:
                diverse.append(r)
                if len(diverse) >= k:
                    break

    return diverse[:k]
