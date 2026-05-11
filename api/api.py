"""
Simple HTTP API for Traffic Law retrieval.

Endpoints:
- GET  /health
- POST /search
- POST /ask
"""

from __future__ import annotations

import json
import os
import re
import sys
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
from llm import DeepSeekRAG  # noqa: E402 — same-package relative import


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


def make_handler(retriever: Retriever, hybrid_searcher: HybridSearcher, llm: DeepSeekRAG | None):
    """Create HTTP request handler class."""

    class ApiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Handle GET requests."""
            if self.path == "/health":
                json_response(self, 200, {"status": "ok"})
            else:
                json_response(self, 404, {"error": "Not found"})

        def do_POST(self) -> None:
            """Handle POST requests."""
            print(f"\n[POST] {self.path}")

            if self.path not in {"/search", "/ask"}:
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
                    print(f"[SEARCH] Question: {question}, k={k}")
                    results = search_traffic_law(hybrid_searcher, question, k=k)
                    json_response(self, 200, {
                        "ok": True,
                        "question": question,
                        "count": len(results),
                        "results": results,
                    })

                elif self.path == "/ask":
                    print(f"[ASK] Question: {question}, k={k}")
                    results = search_traffic_law(hybrid_searcher, question, k=min(k, 5))

                    answer = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
                    extracted: dict[str, Any] = {}

                    if results:
                        # Collect contexts for LLM (top N result texts)
                        contexts = [r.get("text", "") for r in results]

                        # Use LLM if available, otherwise fall back to raw text
                        if llm:
                            top_text = results[0].get("text", "")
                            answer = llm.generate_or_fallback(
                                question, contexts, fallback_text=top_text
                            )
                            # Still extract fines/points from top result
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
                    })

            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()
                json_response(self, 500, {"error": str(e)})

        def log_message(self, format: str, *args: Any) -> None:
            """Suppress default logging."""
            pass

    return ApiHandler


def run_server(settings: Settings = Settings()) -> None:
    """Start API server."""
    global retriever, hybrid_searcher, llm

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
    handler = make_handler(retriever, hybrid_searcher, llm)
    server = ThreadingHTTPServer((settings.host, settings.port), handler)
    print(f"\nAPI running at http://{settings.host}:{settings.port}")
    print(f"  Corpus: {retriever.index.ntotal} vectors")
    print("- GET  /health")
    print("- POST /search - {'question': '...', 'k': 3}")
    print("- POST /ask - {'question': '...', 'k': 3}")
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
llm = None

if __name__ == "__main__":
    run_server()
