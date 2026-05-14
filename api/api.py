"""
Simple HTTP API for Traffic Law retrieval + Traffic Sign search.

Endpoints:
- GET  /health
- GET  /signs/image?name=Biển P.101
- POST /search
- POST /ask
- POST /signs/search
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Add project root + api dir to path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
for p in (str(PROJECT_ROOT), str(API_DIR)):
    if p not in sys.path:
        sys.path.append(p)

from retrieval.retriever import Retriever
from retrieval.hybrid_search import HybridSearcher
from retrieval.traffic_sign_searcher import TrafficSignSearcher, IMAGES_DIR
from llm import DeepSeekRAG  # noqa: E402 — same-package relative import

# Sign-related query patterns
# Dynamic thresholds per search strategy — each strategy has its own score distribution
THRESHOLDS = {
    "definition": 0.88,   # Scores 0.72-0.98, keep only strong definition matches
    "keyword":    0.78,   # Scores 0.25-0.98, keep good keyword + penalty matches
    "fusion":     0.72,   # Scores normalized 0-1, need higher bar for fusion
}
# Citations: only include results above this score (prevents irrelevant citations)
MIN_CITATION_SCORE = 0.75
MIN_SIGN_SCORE = 0.75     # Traffic signs: code 90-98%, phrase 92%, keyword 75-85%

SIGN_QUERY_PATTERNS = [
    # Direct sign code queries: "P.101", "biển P.101", "W.201", "p126" (no dot)
    r"biển\s*[a-zA-Z]+\.?\d+",
    r"\b[a-zA-Z]+\.?\d+[a-z]?\b",
    # "biển nào", "biển gì" — user is asking WHICH sign
    r"biển\s+(nào|g[iì])\b",
    # "là biển" — asking what sign something is
    r"\blà\s+biển\b",
    # Sign categories
    r"biển\s*báo", r"biển\s*cấm", r"biển\s*nguy\s*hiểm",
    r"biển\s*hiệu\s*lệnh", r"biển\s*chỉ\s*dẫn", r"biển\s*phụ",
    r"biển\s*báo\s*giao\s*thông",
    # Road markings
    r"vạch\s*kẻ\s*đường",
]

SIGN_SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn luật giao thông đường bộ Việt Nam. "
    "Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng.\n\n"
    "QUAN TRỌNG — Chỉ được dùng thông tin có trong ngữ cảnh được cung cấp bên dưới:\n"
    "- Tuyệt đối KHÔNG được bịa ra mã biển báo, tên biển báo, hay số hiệu không có trong ngữ cảnh.\n"
    "- KHÔNG suy đoán các biển báo liên quan nếu không được liệt kê trong ngữ cảnh.\n"
    "- Nếu không chắc về mối liên hệ giữa các biển báo, chỉ mô tả biển báo được cung cấp.\n"
    "- Khi nói về biển báo, luôn kèm mã biển chính xác từ ngữ cảnh.\n\n"
    "Dưới đây là thông tin về các biển báo giao thông liên quan đến câu hỏi. "
    "Hãy giải thích ý nghĩa, tác dụng và những điều cần lưu ý về các biển báo này."
)


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000


def json_response(handler: BaseHTTPRequestHandler, status: int, data: dict[str, Any]) -> None:
    """Send JSON response."""
    try:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type")
        handler.end_headers()
        handler.wfile.write(body)
    except Exception as e:
        print(f"[ERROR] Failed to send response: {e}")


def shorten_text(text: str, limit: int = 500) -> str:
    """Shorten text to limit characters."""
    text = " ".join(str(text).split())
    return text[:limit] + "..." if len(text) > limit else text


def search_traffic_law(hybrid_searcher: HybridSearcher, question: str, k: int | None = None) -> tuple[list[dict[str, Any]], str, int]:
    """Search traffic law documents using hybrid search with adaptive k.

    Returns (formatted_results, intent_label, k_used)."""
    results, intent, k_used = hybrid_searcher.search(question, k=k)
    formatted = []
    for r in results:
        score = round(r.get("score", 0), 4)
        strategy = r.get("strategy", "fusion")
        threshold = THRESHOLDS.get(strategy, THRESHOLDS["fusion"])
        if score < threshold:
            continue
        formatted.append({
            "id": r.get("id"),
            "rank": len(formatted) + 1,
            "score": score,
            "strategy": strategy,
            "source": r.get("source", "Unknown"),
            "page": r.get("page", "N/A"),
            "text": shorten_text(r.get("text", "")),
            "legal_refs": r.get("legal_refs", {}),
        })
    return formatted, intent, k_used


def extract_info(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract key info from text: fines, points, violations."""
    text_lower = text.lower()

    # Extract fines - match real legal text formats:
    # "Phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng"
    # "Phạt tiền từ 400.000 đồng đến 600.000 đồng"
    # "phạt ... đồng"
    fine_range = re.findall(
        r"phạt\s+tiền\s+(?:từ\s+)?([\d.,]+)\s*(?:triệu\s+)?đồng(?:\s+đến\s+([\d.,]+)\s*(?:triệu\s+)?đồng)?",
        text_lower,
    )

    # Also match "X.000.000 đồng đến Y.000.000 đồng" standalone
    amount_pattern = re.findall(
        r"([\d.,]+)\s*(?:triệu\s+)?đồng\s+đến\s+([\d.,]+)\s*(?:triệu\s+)?đồng",
        text_lower,
    )   

    fines = []
    for m in fine_range:
        min_val = m[0].replace(".", "")
        max_val = m[1].replace(".", "") if m[1] else ""
        if max_val:
            fines.append(f"{m[0]} - {m[1]} đồng")
        else:
            fines.append(f"{m[0]} đồng")

    if not fines and amount_pattern:
        for m in amount_pattern[:3]:
            fines.append(f"{m[0]} - {m[1]} đồng")

    # Extract blood alcohol levels (nồng độ cồn)
    alcohol = re.findall(
        r"(?:nồng\s+độ\s+cồn)\s*(?:.*?)(\d+)\s*miligam.*?(\d+)\s*mililít",
        text_lower,
    )

    # Extract points (trừ điểm)
    points = re.findall(r"trừ\s+(\d+)\s+điểm", text_lower)

    # Extract suspension (tước/tạm giữ bằng lái)
    suspension = re.findall(
        r"(?:tước|tạm\s+giữ|tịch\s+thu).*?(?:bằng|giấy\s+phép\s+lái).*?(\d+)\s*(tháng|năm|ngày)?",
        text_lower,
    )
    # Simpler form
    suspension2 = re.findall(
        r"tước\s+(?:quyền\s+sử\s+dụng\s+)?(?:giấy\s+phép\s+lái\s+xe|bằng\s+lái)[^.]*?(\d+)\s*(tháng|năm)?",
        text_lower,
    )

    # Check metadata for structured info
    meta_fines = None
    if metadata:
        penalty = metadata.get("penalty_range")
        if penalty and isinstance(penalty, list) and len(penalty) == 2:
            meta_fines = [f"{penalty[0]:,} - {penalty[1]:,} đồng"]

    return {
        "fines": meta_fines or (fines[:3] if fines else ["Không tìm thấy thông tin mức phạt"]),
        "alcohol_level": alcohol[:1] if alcohol else [],
        "points": list(set(points))[:3] if points else ["Không tìm thấy thông tin trừ điểm"],
        "suspension": list(set(m[0] + " " + (m[1] or "tháng") for m in (suspension or suspension2)))[:2]
        if (suspension or suspension2)
        else [],
        "violations": [shorten_text(text, 300)],
    }


def _is_sign_query(question: str) -> bool:
    """Check if a question is about traffic signs."""
    q = question.lower()
    return any(re.search(pat, q) for pat in SIGN_QUERY_PATTERNS)


def make_handler(retriever: Retriever, hybrid_searcher: HybridSearcher, sign_searcher: TrafficSignSearcher, llm: DeepSeekRAG | None):
    """Create HTTP request handler class."""

    class ApiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Handle GET requests."""
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path == "/health":
                json_response(self, 200, {"status": "ok"})

            elif parsed.path == "/signs/image":
                params = urllib.parse.parse_qs(parsed.query)
                name = params.get("name", [""])[0]
                self._serve_sign_image(name)

            else:
                self._serve_static(parsed.path)

        def _serve_sign_image(self, name: str) -> None:
            """Serve a traffic sign SVG image."""
            if not name:
                json_response(self, 400, {"error": "Missing 'name' parameter"})
                return

            sign = sign_searcher.get_sign_by_name(name)
            if not sign:
                json_response(self, 404, {"error": f"Sign not found: {name}"})
                return

            img_path = sign.get("image_path", "")
            if not img_path:
                json_response(self, 404, {"error": f"No image for sign: {name}"})
                return

            full_path = PROJECT_ROOT / img_path
            if not full_path.exists():
                json_response(self, 404, {"error": f"Image file not found: {full_path}"})
                return

            try:
                svg_data = full_path.read_bytes()
                mime_type, _ = mimetypes.guess_type(str(full_path))
                if not mime_type:
                    mime_type = "image/svg+xml"
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(svg_data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(svg_data)
            except Exception as e:
                print(f"[ERROR] Serving image: {e}")
                json_response(self, 500, {"error": str(e)})

        def _serve_static(self, path: str) -> None:
            """Serve a static file from the frontend directory."""
            # Normalize path: / → /index.html
            if path == "/" or path == "":
                path = "/index.html"

            # Remove leading slash and resolve relative to frontend dir
            rel = path.lstrip("/")
            # Prevent directory traversal
            if ".." in rel or rel.startswith("/"):
                json_response(self, 403, {"error": "Forbidden"})
                return

            file_path = FRONTEND_DIR / rel
            if not file_path.exists() or not file_path.is_file():
                # Fall back to index.html for SPA routing
                file_path = FRONTEND_DIR / "index.html"
                if not file_path.exists():
                    json_response(self, 404, {"error": "Not found"})
                    return

            try:
                body = file_path.read_bytes()
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if not mime_type:
                    mime_type = "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                print(f"[ERROR] Serving static: {e}")
                json_response(self, 500, {"error": str(e)})

        def do_POST(self) -> None:
            """Handle POST requests."""
            print(f"\n[POST] {self.path}")

            if self.path not in {"/search", "/ask", "/signs/search"}:
                json_response(self, 404, {"error": "Not found"})
                return

            # Parse JSON body
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(body)
                question = str(payload.get("question", "")).strip()
                k = payload.get("k", 3)

                if not question:
                    json_response(self, 400, {"error": "Missing 'question'"})
                    return

            except Exception as e:
                json_response(self, 400, {"error": f"Invalid JSON: {e}"})
                return

            # Process request
            try:
                if self.path == "/search":
                    self._handle_search(hybrid_searcher, question, k)

                elif self.path == "/ask":
                    self._handle_ask(hybrid_searcher, sign_searcher, llm, question, k)

                elif self.path == "/signs/search":
                    self._handle_sign_search(sign_searcher, question, k)

            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()
                json_response(self, 500, {"error": str(e)})

        def _handle_search(self, hybrid_searcher: HybridSearcher, question: str, k: int) -> None:
            print(f"[SEARCH] Question: {question}, k={k}")
            results, intent, k_used = search_traffic_law(hybrid_searcher, question, k=k)
            json_response(self, 200, {
                "ok": True,
                "question": question,
                "intent": intent,
                "k": k_used,
                "count": len(results),
                "results": results,
            })

        def _handle_ask(self, hybrid_searcher: HybridSearcher, sign_searcher: TrafficSignSearcher, llm: DeepSeekRAG | None, question: str, k: int) -> None:
            print(f"[ASK] Question: {question}, k={k}")

            answer = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            extracted: dict[str, Any] = {}
            signs: list[dict[str, Any]] = []
            citations: list[dict[str, Any]] = []
            intent = "detail"
            k_used = 3

            # Check if question is about traffic signs — handle separately
            if _is_sign_query(question):
                print(f"[ASK] Traffic sign query detected")
                all_signs = sign_searcher.search(question, k=5)
                # Filter by score then cap at 3 best matches
                signs = [s for s in all_signs if s.get("score", 0) >= MIN_SIGN_SCORE][:3]
                print(f"[ASK] Signs: {len(all_signs)} total, {len(signs)} pass threshold (>= {MIN_SIGN_SCORE})")

                if signs:
                    if llm:
                        sign_contexts = [
                            f"{s['name']}: {s['description']} (Loại: {s.get('category_name', '')}, độ khớp: {s['score']:.0%})"
                            for s in signs
                        ]
                        try:
                            sign_prompt = (
                                f"Thông tin biển báo giao thông (chỉ lấy kết quả có độ khớp cao):\n"
                                + "\n".join(sign_contexts)
                                + f"\n\nCâu hỏi: {question}\n\n"
                                "Hãy giải thích ý nghĩa của biển báo phù hợp nhất với câu hỏi. "
                                "Nếu có nhiều biển, hãy liệt kê ngắn gọn từng biển."
                            )
                            response = llm._client.chat.completions.create(
                                model=llm._model,
                                messages=[
                                    {"role": "system", "content": SIGN_SYSTEM_PROMPT},
                                    {"role": "user", "content": sign_prompt},
                                ],
                                temperature=0.3,
                                max_tokens=1024,
                            )
                            answer = response.choices[0].message.content.strip()
                        except Exception as e:
                            print(f"[Sign LLM ERROR] {e}")
                            answer = f"Các biển báo liên quan:\n" + "\n".join(
                                f"- {s['name']}: {s['description']}" for s in signs
                            )
                    else:
                        answer = f"Các biển báo liên quan:\n" + "\n".join(
                            f"- {s['name']}: {s['description']}" for s in signs
                        )
                else:
                    best = all_signs[0] if all_signs else None
                    if best:
                        answer = (
                            f"Không tìm thấy biển báo nào khớp chính xác. "
                            f"Gần nhất: {best['name']} (độ khớp {best['score']:.0%}). "
                            f"Thử nhập mã biển báo cụ thể (VD: {best['name']})."
                        )
                    else:
                        answer = "Không tìm thấy biển báo nào liên quan. Vui lòng nhập mã biển báo cụ thể."

            else:
                # Standard legal Q&A — NOT a sign query
                results, intent, k_used = search_traffic_law(hybrid_searcher, question, k=None)

                if results:
                    print(f"[ASK] Intent={intent}, k={k_used}, {len(results)} results pass thresholds")
                    contexts = [r.get("text", "") for r in results]
                    if llm:
                        answer = llm.generate_or_fallback(
                            question, contexts,
                            fallback_text=results[0].get("text", ""),
                        )
                        extracted = extract_info(
                            results[0].get("text", ""),
                            results[0].get("legal_refs", {}),
                        )
                    else:
                        top_result = results[0]
                        top_text = top_result.get("text", "")
                        top_meta = top_result.get("legal_refs", {})
                        print(f"[ASK] Top: score={top_result.get('score')}, source={top_result.get('source')}")
                        answer = shorten_text(top_text, 800)
                        extracted = extract_info(top_text, top_meta)

                    # Only show citations that individually meet quality threshold
                    citations = [r for r in results if r.get("score", 0) >= MIN_CITATION_SCORE]
                    # If no citations pass the quality bar, don't show any
                    if not citations:
                        print(f"[ASK] No citations pass quality threshold ({MIN_CITATION_SCORE})")
                else:
                    citations = []

            json_response(self, 200, {
                "ok": True,
                "question": question,
                "answer": answer,
                "extracted": extracted,
                "citations": citations,
                "signs": signs,
                "intent": intent,
                "k": k_used,
            })

        def _handle_sign_search(self, sign_searcher: TrafficSignSearcher, question: str, k: int) -> None:
            print(f"[SIGNS] Search: {question}, k={k}")
            all_signs = sign_searcher.search(question, k=max(k, 8))
            signs = [s for s in all_signs if s.get("score", 0) >= MIN_SIGN_SCORE][:k]
            json_response(self, 200, {
                "ok": True,
                "question": question,
                "count": len(signs),
                "signs": signs,
            })

        def log_message(self, format: str, *args: Any) -> None:
            """Suppress default logging."""
            pass

    return ApiHandler


def run_server(settings: Settings = Settings()) -> None:
    """Start API server."""
    global retriever, hybrid_searcher, sign_searcher, llm

    # Preload embedding model NOW (not on first query)
    from pipeline.embedding import preload_model
    preload_model()

    # Initialize retriever
    retriever = Retriever()
    if not retriever.index or not retriever.corpus:
        raise RuntimeError("Vector DB not ready. Run: python -m retrieval.build_vector_db_new")

    print("✓ Retriever initialized")

    # Initialize hybrid searcher
    hybrid_searcher = HybridSearcher(retriever, retriever.corpus)
    print("✓ Hybrid Search ready (BM25 + FAISS + Direct Keyword)")

    # Initialize traffic sign searcher
    try:
        sign_searcher = TrafficSignSearcher()
        print("✓ Traffic Sign Search ready")
    except Exception as e:
        print(f"⚠ Traffic Sign Search init failed: {e}")
        sign_searcher = None

    # Initialize LLM (optional — falls back to raw retrieval if no API key)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if api_key:
        try:
            llm = DeepSeekRAG(api_key)
            print("✓ DeepSeek V3 LLM ready")
        except Exception as e:
            print(f"⚠ DeepSeek init failed: {e}")
            llm = None
    else:
        print("⚠ DEEPSEEK_API_KEY not set — using raw retrieval mode")
        llm = None

    # Start server
    handler = make_handler(retriever, hybrid_searcher, sign_searcher, llm)
    server = ThreadingHTTPServer((settings.host, settings.port), handler)
    print(f"\nAPI + Frontend running at http://{settings.host}:{settings.port}")
    print(f"  Corpus: {retriever.index.ntotal} vectors")
    print("- GET  /health")
    print("- GET  /signs/image?name=...")
    print("- POST /search   - {'question': '...', 'k': 3}")
    print("- POST /ask      - {'question': '...', 'k': 3}")
    print("- POST /signs/search - {'question': '...', 'k': 5}")
    if llm:
        print("- Mode: RAG (DeepSeek V3)")
    else:
        print("- Mode: Retrieval-only (no LLM)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.server_close()
        print("API stopped")


retriever = None
hybrid_searcher = None
sign_searcher = None
llm = None

if __name__ == "__main__":
    run_server()
