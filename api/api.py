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
for p in (str(PROJECT_ROOT), str(API_DIR)):
    if p not in sys.path:
        sys.path.append(p)

from retrieval.retriever import Retriever
from retrieval.hybrid_search import HybridSearcher
from retrieval.traffic_sign_searcher import TrafficSignSearcher, IMAGES_DIR
from llm import DeepSeekRAG  # noqa: E402 — same-package relative import

# Sign-related query patterns
SIGN_QUERY_PATTERNS = [
    r"biển\s*báo", r"biển\s*cấm", r"biển\s*nguy\s*hiểm",
    r"biển\s*hiệu\s*lệnh", r"biển\s*chỉ\s*dẫn", r"biển\s*phụ",
    r"biển\s*[a-zA-Z]+\.?\d+", r"vạch\s*kẻ\s*đường",
    r"biển\s*báo\s*giao\s*thông",
]

SIGN_SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn luật giao thông đường bộ Việt Nam. "
    "Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng.\n\n"
    "Dưới đây là thông tin về các biển báo giao thông liên quan đến câu hỏi. "
    "Hãy giải thích ý nghĩa, tác dụng và những điều cần lưu ý về các biển báo này. "
    "Trích dẫn tên biển báo khi trả lời."
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


def search_traffic_law(hybrid_searcher: HybridSearcher, question: str, k: int = 3) -> list[dict[str, Any]]:
    """Search traffic law documents using hybrid search (BM25 + FAISS + keyword)."""
    results = hybrid_searcher.search(question, k=k)
    return [
        {
            "id": r.get("id"),
            "rank": i + 1,
            "score": round(r.get("score", 0), 4),
            "source": r.get("source", "Unknown"),
            "page": r.get("page", "N/A"),
            "text": shorten_text(r.get("text", "")),
            "legal_refs": r.get("legal_refs", {}),
        }
        for i, r in enumerate(results)
    ]


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
                json_response(self, 404, {"error": "Not found"})

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
            results = search_traffic_law(hybrid_searcher, question, k=k)
            json_response(self, 200, {
                "ok": True,
                "question": question,
                "count": len(results),
                "results": results,
            })

        def _handle_ask(self, hybrid_searcher: HybridSearcher, sign_searcher: TrafficSignSearcher, llm: DeepSeekRAG | None, question: str, k: int) -> None:
            print(f"[ASK] Question: {question}, k={k}")
            results = search_traffic_law(hybrid_searcher, question, k=min(k, 5))

            answer = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            extracted: dict[str, Any] = {}
            signs: list[dict[str, Any]] = []

            # Check if question is about traffic signs
            if _is_sign_query(question):
                print(f"[ASK] Traffic sign query detected")
                signs = sign_searcher.search(question, k=5)
                if signs and llm:
                    sign_contexts = [
                        f"{s['name']}: {s['description']} (Loại: {s.get('category_name', '')})"
                        for s in signs
                    ]
                    try:
                        sign_prompt = (
                            f"Thông tin biển báo giao thông:\n"
                            + "\n".join(sign_contexts)
                            + f"\n\nCâu hỏi: {question}\n\n"
                            "Hãy giải thích các biển báo trên, ý nghĩa và những điều cần lưu ý."
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
                elif signs:
                    answer = f"Các biển báo liên quan:\n" + "\n".join(
                        f"- {s['name']}: {s['description']}" for s in signs
                    )

            elif results:
                # Standard legal Q&A flow
                contexts = [r.get("text", "") for r in results]
                if llm:
                    top_text = results[0].get("text", "")
                    answer = llm.generate_or_fallback(
                        question, contexts, fallback_text=top_text
                    )
                    extracted = extract_info(
                        results[0].get("text", ""),
                        results[0].get("legal_refs", {}),
                    )
                else:
                    top_result = results[0]
                    top_text = top_result.get("text", "")
                    top_meta = top_result.get("legal_refs", {})
                    print(f"[ASK] Top result: score={top_result.get('score')}, source={top_result.get('source')}")
                    if top_result.get("score", 0) >= 0.35:
                        answer = shorten_text(top_text, 800)
                        extracted = extract_info(top_text, top_meta)

            json_response(self, 200, {
                "ok": True,
                "question": question,
                "answer": answer,
                "extracted": extracted,
                "citations": results[:k],
                "signs": signs,
            })

        def _handle_sign_search(self, sign_searcher: TrafficSignSearcher, question: str, k: int) -> None:
            print(f"[SIGNS] Search: {question}, k={k}")
            signs = sign_searcher.search(question, k=k)
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
    print(f"\nAPI running at http://{settings.host}:{settings.port}")
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
