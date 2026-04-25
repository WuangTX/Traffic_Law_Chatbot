"""
Simple HTTP API for Traffic Law retrieval.

Endpoints:
- GET  /health
- POST /search
- POST /ask
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Add project root to path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from rag.retriever import Retriever


@dataclass
class Settings:
	host: str = "127.0.0.1"
	port: int = 8000


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
	body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
	handler.send_response(status)
	handler.send_header("Content-Type", "application/json; charset=utf-8")
	handler.send_header("Content-Length", str(len(body)))
	handler.send_header("Access-Control-Allow-Origin", "*")
	handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	handler.send_header("Access-Control-Allow-Headers", "Content-Type")
	handler.end_headers()
	handler.wfile.write(body)


def _shorten_text(text: str, limit: int = 420) -> str:
	clean = " ".join(str(text).split())
	if len(clean) <= limit:
		return clean
	return clean[:limit].rstrip() + "..."


def _unique_keep_order(items: list[str]) -> list[str]:
	seen = set()
	output: list[str] = []
	for item in items:
		norm = item.strip().lower()
		if not norm or norm in seen:
			continue
		seen.add(norm)
		output.append(item.strip())
	return output


def _extract_keyword_windows(text: str, keywords: list[str], window: int = 140, max_items: int = 4) -> list[str]:
	if not text:
		return []

	lower = text.lower()
	windows: list[str] = []
	for keyword in keywords:
		start_idx = 0
		key = keyword.lower()
		while True:
			idx = lower.find(key, start_idx)
			if idx == -1:
				break
			left = max(0, idx - window)
			right = min(len(text), idx + len(keyword) + window)
			windows.append(_shorten_text(text[left:right], limit=260))
			start_idx = idx + len(keyword)

	return _unique_keep_order(windows)[:max_items]


def _extract_fine_ranges(text: str) -> list[str]:
	if not text:
		return []
	matches = re.findall(
		r"phạt\s+tiền\s+(?:từ\s+)?\d[\d\s\.]*\s*đồng\s*(?:đến|tới|-)\s*\d[\d\s\.]*\s*đồng",
		text.lower(),
	)
	return _unique_keep_order([" ".join(m.split()) for m in matches])


def _fine_has_million_value(fine_text: str) -> bool:
	if not fine_text:
		return False
	number_chunks = re.findall(r"\d[\d\.\s]*", fine_text)
	for chunk in number_chunks:
		digits = re.sub(r"\D", "", chunk)
		if digits and int(digits) >= 1_000_000:
			return True
	return False


def _extract_gplx_points(text: str) -> list[str]:
	if not text:
		return []
	matches = re.findall(r"trừ\s+điểm\s+giấy\s+phép\s+lái\s+xe\s+\d+\s*điểm", text.lower())
	return _unique_keep_order([" ".join(m.split()) for m in matches])


def _normalize_query_text(text: str) -> str:
	if not text:
		return ""
	norm = unicodedata.normalize("NFD", text.lower())
	norm = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")
	norm = norm.replace("đ", "d")
	norm = " ".join(norm.split())
	return norm


def _is_red_light_question(question: str) -> bool:
	q = _normalize_query_text(question)
	red_question_keywords = ["den do", "den tin hieu", "vuot den", "khong chap hanh hieu lenh"]
	return any(keyword in q for keyword in red_question_keywords)


def _vehicle_keywords_from_question(question: str) -> list[str]:
	q = _normalize_query_text(question)
	if "o to" in q or "oto" in q or "xe hoi" in q:
		return ["ô tô", "xe ô tô", "loại xe tương tự xe ô tô", "xe chở người bốn bánh gắn động cơ"]
	if "xe may" in q or "mo to" in q or "gan may" in q:
		return ["xe máy", "mô tô", "xe gắn máy", "loại xe tương tự xe mô tô"]
	return []


def _other_vehicle_keywords_from_question(question: str) -> list[str]:
	q = _normalize_query_text(question)
	if "o to" in q or "oto" in q or "xe hoi" in q:
		return ["xe máy", "mô tô", "xe gắn máy", "loại xe tương tự xe mô tô"]
	if "xe may" in q or "mo to" in q or "gan may" in q:
		return ["ô tô", "xe ô tô", "loại xe tương tự xe ô tô", "xe chở người bốn bánh gắn động cơ"]
	return []


def _question_intent_keywords(question: str) -> list[str]:
	q = question.lower()
	intents: list[str] = []

	if _is_red_light_question(q):
		intents.extend([
			"chấp hành hiệu lệnh đèn tín hiệu giao thông",
			"đèn tín hiệu giao thông",
			"vượt đèn đỏ",
		])

	if "nồng độ cồn" in q or "cồn" in q:
		intents.extend(["nồng độ cồn", "kiểm tra nồng độ cồn"])

	if "ngược chiều" in q:
		intents.extend(["đi ngược chiều", "đường một chiều", "đường cao tốc"])

	if "không bằng lái" in q or "không có bằng lái" in q or "không giấy phép lái xe" in q:
		intents.extend(["không có giấy phép lái xe", "giấy phép lái xe"])

	if "phạt" in q:
		intents.append("phạt tiền")

	return _unique_keep_order(intents)


def _target_article_for_vehicle(question: str) -> str | None:
	q = _normalize_query_text(question)
	if "o to" in q or "oto" in q or "xe hoi" in q:
		return "6"
	if "xe may" in q or "mo to" in q or "gan may" in q:
		return "7"
	return None


def _rerank_results_for_ask(question: str, results: list[dict[str, Any]], keep_k: int) -> list[dict[str, Any]]:
	if not results:
		return []

	red_keywords = ["đèn đỏ", "đèn tín hiệu", "vượt đèn", "chấp hành hiệu lệnh"]
	vehicle_keywords = _vehicle_keywords_from_question(question)
	other_vehicle_keywords = _other_vehicle_keywords_from_question(question)
	intent_keywords = _question_intent_keywords(question)
	red_question = _is_red_light_question(question)
	target_article = _target_article_for_vehicle(question)

	if red_question:
		specific_red_scope = [
			item
			for item in results
			if any(
				keyword in f"{item.get('text', '')} {item.get('text_processed', '')}".lower()
				for keyword in [
					"không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
					"chấp hành hiệu lệnh của đèn tín hiệu giao thông",
					"vượt đèn đỏ",
				]
			)
		]
		if specific_red_scope:
			results = specific_red_scope

		red_scope = [
			item
			for item in results
			if any(
				keyword in f"{item.get('text', '')} {item.get('text_processed', '')}".lower()
				for keyword in ["đèn đỏ", "đèn tín hiệu", "đèn tín hiệu giao thông", "vượt đèn", "chấp hành hiệu lệnh"]
			)
		]
		if red_scope:
			results = red_scope

	if vehicle_keywords:
		vehicle_scope = [
			item
			for item in results
			if any(keyword in f"{item.get('text', '')} {item.get('text_processed', '')}".lower() for keyword in vehicle_keywords)
		]
		if vehicle_scope:
			results = vehicle_scope

	def boost(item: dict[str, Any]) -> float:
		text = f"{item.get('text', '')} {item.get('text_processed', '')}".lower()
		legal_refs = item.get("legal_refs", {}) or {}
		citations = legal_refs.get("citations", []) or []
		articles = [str(x) for x in legal_refs.get("articles", [])]
		s = 0.0

		if red_question and "chấp hành hiệu lệnh đèn tín hiệu giao thông" in text:
			s += 4.0
		elif red_question and "đèn tín hiệu giao thông" in text:
			s += 2.5

		if any(k in text for k in red_keywords):
			s += 2.0

		if intent_keywords:
			s += sum(1.5 for k in intent_keywords if k in text)

		if vehicle_keywords and any(k in text for k in vehicle_keywords):
			s += 3.0
		if vehicle_keywords and not any(k in text for k in vehicle_keywords):
			s -= 2.0
		if other_vehicle_keywords and any(k in text for k in other_vehicle_keywords) and not any(k in text for k in vehicle_keywords):
			s -= 1.5

		if "phạt tiền" in text:
			s += 1.0
		if "trừ điểm giấy phép lái xe" in text:
			s += 1.0

		if citations:
			s += min(1.5, 0.2 * len(citations))

		if red_question and target_article:
			if target_article in articles:
				s += 8.0
			elif articles:
				s -= 3.0

		if red_question and _fine_has_million_value(text):
			s += 4.0

		q_norm = _normalize_query_text(question)
		if "o to" in q_norm or "oto" in q_norm or "xe hoi" in q_norm:
			if any(k in text for k in ["xe gắn máy", "xe máy", "mô tô", "loại xe tương tự xe gắn máy"]):
				s -= 6.0

		return s

	ranked = sorted(results, key=lambda item: (boost(item), float(item.get("score", 0.0))), reverse=True)
	return ranked[:keep_k]


def _keyword_candidates_from_corpus(retriever: Retriever, question: str, limit: int = 12) -> list[dict[str, Any]]:
	if not retriever.corpus:
		return []

	red_keywords = ["đèn tín hiệu giao thông", "đèn đỏ", "vượt đèn", "chấp hành hiệu lệnh"]
	vehicle_keywords = _vehicle_keywords_from_question(question)
	other_vehicle_keywords = _other_vehicle_keywords_from_question(question)
	intent_keywords = _question_intent_keywords(question)
	red_question = _is_red_light_question(question)

	candidates: list[dict[str, Any]] = []
	for idx, doc in enumerate(retriever.corpus):
		text = str(doc.get("text_display") or doc.get("text_original") or "")
		text_processed = str(doc.get("text_processed") or "")
		check_text = f"{text} {text_processed}".lower()

		hit_score = 0
		if red_question:
			hit_score += sum(2 for kw in red_keywords if kw in check_text)
		if intent_keywords:
			hit_score += sum(2 for kw in intent_keywords if kw in check_text)
		if vehicle_keywords:
			vehicle_hits = sum(2 for kw in vehicle_keywords if kw in check_text)
			hit_score += vehicle_hits
			if vehicle_hits == 0:
				hit_score -= 2
		if other_vehicle_keywords and vehicle_keywords and any(kw in check_text for kw in other_vehicle_keywords) and not any(kw in check_text for kw in vehicle_keywords):
			hit_score -= 1

		if "phạt tiền" in check_text:
			hit_score += 1
		if "trừ điểm giấy phép lái xe" in check_text:
			hit_score += 1

		if hit_score <= 0:
			continue

		candidates.append(
			{
				"rank": 0,
				"id": int(idx),
				"score": float(0.25 + 0.03 * hit_score),
				"source": doc.get("source", "N/A"),
				"page": doc.get("page", "N/A"),
				"text": text,
				"text_processed": text_processed,
				"legal_refs": doc.get("legal_refs", {}),
			}
		)

	candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
	return candidates[:limit]


def _build_ask_payload(question: str, results: list[dict[str, Any]]) -> dict[str, Any]:
	if not results:
		return {
			"answer": "Chưa tìm thấy đoạn luật phù hợp trong cơ sở dữ liệu.",
			"extracted": {
				"violation": [],
				"fine_ranges": [],
				"gplx_points": [],
			},
			"evidence": [],
		}

	red_keywords = [
		"đèn đỏ",
		"đèn tín hiệu giao thông",
		"chấp hành hiệu lệnh đèn tín hiệu giao thông",
		"chấp hành hiệu lệnh",
		"vượt đèn",
	]

	violation_snippets: list[str] = []
	fine_ranges: list[str] = []
	gplx_points: list[str] = []
	fine_ranges_red: list[str] = []
	gplx_points_red: list[str] = []
	evidence: list[dict[str, Any]] = []
 
	red_question = _is_red_light_question(question)
	vehicle_keywords = _vehicle_keywords_from_question(question)
	target_article = _target_article_for_vehicle(question)

	red_scoped_items: list[dict[str, Any]] = []
	for item in results:
		check_text = f"{item.get('text', '')} {item.get('text_processed', '')}".lower()
		if any(keyword in check_text for keyword in red_keywords):
			red_scoped_items.append(item)

	scoped_results = red_scoped_items if red_question and red_scoped_items else results

	# If this is a red-light question, prefer NĐ168 article-level matches by vehicle type.
	if red_question:
		legal_scoped: list[dict[str, Any]] = []
		for item in scoped_results:
			source = str(item.get("source", "")).lower()
			if "168/2024" not in source and "nghị định" not in source:
				continue
			legal_refs = item.get("legal_refs", {}) or {}
			articles = [str(x) for x in legal_refs.get("articles", [])]
			if target_article:
				if target_article in articles:
					legal_scoped.append(item)
			else:
				if any(a in {"6", "7"} for a in articles):
					legal_scoped.append(item)

		if legal_scoped:
			scoped_results = legal_scoped

	for item in scoped_results:
		text_display = str(item.get("text", ""))
		text_processed = str(item.get("text_processed", ""))
		text_for_extract = text_display if text_display else text_processed

		local_snippets = _extract_keyword_windows(text_for_extract, red_keywords, window=160, max_items=2)
		local_fines = _extract_fine_ranges(text_for_extract)
		local_points = _extract_gplx_points(text_for_extract)

		# For red-light queries, prioritize million-level fines to avoid generic low-penalty leakage.
		if red_question:
			million_fines = [f for f in local_fines if _fine_has_million_value(f) or "triệu" in f]
			if million_fines:
				local_fines = million_fines

		violation_snippets.extend(local_snippets)
		fine_ranges.extend(local_fines)
		gplx_points.extend(local_points)

		target_snippets = local_snippets
		if vehicle_keywords:
			vehicle_snippets = [
				snippet
				for snippet in local_snippets
				if any(keyword in snippet.lower() for keyword in vehicle_keywords)
			]
			if vehicle_snippets:
				target_snippets = vehicle_snippets

		for snippet in target_snippets:
			fine_ranges_red.extend(_extract_fine_ranges(snippet))
			gplx_points_red.extend(_extract_gplx_points(snippet))

		evidence.append(
			{
				"rank": item.get("rank"),
				"score": item.get("score"),
				"source": item.get("source"),
				"page": item.get("page"),
				"legal_refs": item.get("legal_refs", {}),
				"snippet": local_snippets[0] if local_snippets else _shorten_text(text_for_extract, limit=220),
			}
		)

	violation_snippets = _unique_keep_order(violation_snippets)[:3]
	fine_ranges = _unique_keep_order(fine_ranges)[:3]
	gplx_points = _unique_keep_order(gplx_points)[:3]
	fine_ranges_red = _unique_keep_order(fine_ranges_red)[:3]
	gplx_points_red = _unique_keep_order(gplx_points_red)[:3]

	if red_question:
		violation_snippets = _unique_keep_order(
			["Không chấp hành hiệu lệnh đèn tín hiệu giao thông (vượt đèn đỏ)."] + violation_snippets
		)[:3]
		if fine_ranges_red:
			fine_ranges = fine_ranges_red
		if gplx_points_red:
			gplx_points = gplx_points_red

	answer_parts: list[str] = []
	if violation_snippets:
		answer_parts.append(f"Hành vi liên quan: {violation_snippets[0]}")
	if fine_ranges:
		answer_parts.append(f"Mức phạt tham chiếu: {'; '.join(fine_ranges[:2])}.")
	if gplx_points:
		answer_parts.append(f"Mức trừ điểm GPLX: {', '.join(gplx_points[:2])}.")

	if red_question and not violation_snippets:
		violation_snippets = ["Không chấp hành hiệu lệnh đèn tín hiệu giao thông (vượt đèn đỏ)."]

	if not answer_parts:
		top = results[0]
		answer = (
			f"Dựa trên kết quả truy xuất, nội dung liên quan nhất cho câu hỏi '{question}' là: "
			f"{_shorten_text(top.get('text', ''), limit=380)}"
		)
	else:
		answer = " ".join(answer_parts)

	return {
		"answer": answer,
		"extracted": {
			"violation": violation_snippets,
			"fine_ranges": fine_ranges,
			"gplx_points": gplx_points,
		},
		"evidence": evidence[:3],
	}


def _build_answer(question: str, results: list[dict[str, Any]]) -> str:
	if not results:
		return "Chưa tìm thấy đoạn luật phù hợp trong cơ sở dữ liệu."

	top = results[0]
	base = _shorten_text(top.get("text", ""), limit=380)
	return (
		f"Dựa trên kết quả truy xuất, nội dung liên quan nhất cho câu hỏi '{question}' là: "
		f"{base}"
	)


def query_traffic_law(retriever: Retriever, question: str, k: int = 3) -> list[dict[str, Any]]:
	clean_question = question.strip()
	if not clean_question:
		return []
	return retriever.search(clean_question, k=k)


def _build_search_results(question: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
	search_items: list[dict[str, Any]] = []
	for item in results:
		text = str(item.get("text", ""))
		text_processed = str(item.get("text_processed", ""))
		search_items.append(
			{
				"rank": item.get("rank"),
				"id": item.get("id"),
				"score": item.get("score"),
				"source": item.get("source"),
				"page": item.get("page"),
				"legal_refs": item.get("legal_refs", {}),
				"snippet": _shorten_text(text or text_processed, limit=260),
				"text": _shorten_text(text, limit=500),
				"text_processed": _shorten_text(text_processed, limit=500),
			}
		)

	if _is_red_light_question(question):
		for idx, item in enumerate(search_items, start=1):
			item["rank"] = idx

	return search_items


def make_handler(retriever: Retriever) -> type[BaseHTTPRequestHandler]:
	class ApiHandler(BaseHTTPRequestHandler):
		def do_OPTIONS(self) -> None:
			_json_response(self, 200, {"ok": True})

		def do_GET(self) -> None:
			if self.path == "/":
				_json_response(
					self,
					200,
					{
						"ok": True,
						"service": "traffic-law-chatbot-api",
						"endpoints": {
							"health": {"method": "GET", "path": "/health"},
							"search": {"method": "POST", "path": "/search"},
							"ask": {"method": "POST", "path": "/ask"},
						},
						"example": {
							"method": "POST",
							"path": "/search",
							"body": {"question": "Mức phạt vượt đèn đỏ", "k": 3},
						},
					},
				)
				return

			if self.path == "/health":
				_json_response(
					self,
					200,
					{
						"ok": True,
						"service": "traffic-law-chatbot-api",
						"index_size": retriever.index.ntotal if retriever.index else 0,
					},
				)
				return

			_json_response(self, 404, {"ok": False, "error": "Not found"})

		def do_POST(self) -> None:
			if self.path not in {"/search", "/ask"}:
				_json_response(self, 404, {"ok": False, "error": "Not found"})
				return

			try:
				length = int(self.headers.get("Content-Length", "0"))
				raw_bytes = self.rfile.read(length) if length > 0 else b"{}"

				content_type = self.headers.get("Content-Type", "")
				charset = "utf-8"
				if "charset=" in content_type:
					charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip()

				try:
					raw = raw_bytes.decode(charset)
				except UnicodeDecodeError:
					# Fallback for common PowerShell request body encodings.
					try:
						raw = raw_bytes.decode("utf-8-sig")
					except UnicodeDecodeError:
						raw = raw_bytes.decode("utf-16")

				raw = raw.lstrip("\ufeff")
				payload = json.loads(raw)
				if not isinstance(payload, dict):
					raise ValueError("JSON body must be an object")
			except Exception:
				_json_response(self, 400, {"ok": False, "error": "Invalid JSON body"})
				return

			question = str(payload.get("question", "")).strip()
			k = int(payload.get("k", 3)) if str(payload.get("k", "")).strip() else 3
			k = max(1, min(k, 8))

			if not question:
				_json_response(self, 400, {"ok": False, "error": "Field 'question' is required"})
				return

			try:
				results = query_traffic_law(retriever, question=question, k=k)
				if self.path == "/ask":
					candidate_results = query_traffic_law(retriever, question=question, k=max(k, 20))
					keyword_results = _keyword_candidates_from_corpus(retriever, question=question, limit=60)
					merged: list[dict[str, Any]] = []
					seen_ids: set[int] = set()

					for item in candidate_results + keyword_results:
						item_id = int(item.get("id", -1))
						if item_id in seen_ids:
							continue
						seen_ids.add(item_id)
						merged.append(item)

					ask_pool = _rerank_results_for_ask(question, merged, keep_k=max(5, k))
					ask_payload = _build_ask_payload(question, ask_pool)
					ask_results = ask_pool[:k]
					citations = [
						{
							"rank": item.get("rank"),
							"score": item.get("score"),
							"source": item.get("source"),
							"page": item.get("page"),
							"legal_refs": item.get("legal_refs", {}),
						}
						for item in ask_results
					]
					_json_response(
						self,
						200,
						{
							"ok": True,
							"question": question,
							"k": k,
							"answer": ask_payload["answer"],
							"extracted": ask_payload["extracted"],
							"evidence": ask_payload["evidence"],
							"citations": citations,
						},
					)
					return

				candidate_results = query_traffic_law(retriever, question=question, k=max(k, 8))
				keyword_results = _keyword_candidates_from_corpus(retriever, question=question, limit=12)
				merged: list[dict[str, Any]] = []
				seen_ids: set[int] = set()

				for item in candidate_results + keyword_results:
					item_id = int(item.get("id", -1))
					if item_id in seen_ids:
						continue
					seen_ids.add(item_id)
					merged.append(item)

				ranked_results = _rerank_results_for_ask(question, merged, keep_k=k)
				search_payload = _build_search_results(question, ranked_results)

				_json_response(
					self,
					200,
					{
						"ok": True,
						"question": question,
						"k": k,
						"count": len(search_payload),
						"results": search_payload,
					},
				)
			except Exception as exc:
				_json_response(self, 500, {"ok": False, "error": f"Search failed: {exc}"})

		def log_message(self, _format: str, *args: Any) -> None:
			# Keep terminal output clean; retriever already logs search details.
			return

	return ApiHandler


def run_server(settings: Settings = Settings()) -> None:
	retriever = Retriever()
	if not retriever.index or not retriever.corpus:
		raise RuntimeError("Retriever chưa sẵn sàng. Hãy build vector DB trước.")

	handler_cls = make_handler(retriever)
	server = ThreadingHTTPServer((settings.host, settings.port), handler_cls)
	print(f"API đang chạy tại http://{settings.host}:{settings.port}")
	print("- Health check: GET /health")
	print("- Search: POST /search với JSON {'question': '...', 'k': 3}")
	print("- Ask: POST /ask với JSON {'question': '...', 'k': 3}")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print("\nĐã nhận Ctrl+C. Đang dừng API...")
	finally:
		server.server_close()
		print("API đã dừng.")


if __name__ == "__main__":
	run_server()
