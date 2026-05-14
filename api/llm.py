"""
DeepSeek V3 RAG wrapper: generates natural-language answers from retrieved legal chunks.
Uses OpenAI-compatible API (base_url = api.deepseek.com).
"""

from __future__ import annotations

from typing import List


SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn luật giao thông đường bộ Việt Nam. "
    "Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng.\n\n"
    "QUY TẮC:\n"
    "1. CHỈ trả lời dựa trên ngữ cảnh pháp lý được cung cấp bên dưới.\n"
    "2. Chỉ sử dụng thông tin trong phần CONTEXT được cung cấp.\nNếu thông tin không xuất hiện trong CONTEXT:- không được tự suy luận, không được sử dụng kiến thức huấn luyện sẵn, không được nhắc tới điều luật khác mà không có trong CONTEXT.\n  "
    "3. Không sử dụng kiến thức pháp luật bên ngoài hoặc dữ liệu đã huấn luyện trước đó.\n"
    "4. Nếu CONTEXT không đủ thông tin thì báo là Hệ thống chưa có dữ liệu phù hợp:\n"
    "5. Luôn trích dẫn số điều, khoản, điểm khi trả lời (vd: 'Theo Điều 6, Khoản 9, Điểm b...').\n"
    "6. Nếu ngữ cảnh không chứa thông tin để trả lời, hãy nói rõ điều đó.\n"
    "7. Tuyệt đối KHÔNG bịa đặt thông tin không có trong ngữ cảnh.\n"
    "8. Nếu câu hỏi về định nghĩa/khái niệm, giải thích dựa trên ngữ cảnh có sẵn."
)

USER_PROMPT_TEMPLATE = """Ngữ cảnh pháp lý tham khảo:
{contexts}

Câu hỏi: {question}

Hãy trả lời dựa trên ngữ cảnh trên."""


class DeepSeekRAG:
    """Thin wrapper around DeepSeek V3 for RAG-based legal Q&A."""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self._model = model

    def _build_contexts(self, contexts: List[str]) -> str:
        """Format retrieved chunks into numbered context block."""
        parts = []
        for i, ctx in enumerate(contexts, 1):
            clean = " ".join(str(ctx).split())
            parts.append(f"[{i}] {clean}")
        return "\n\n".join(parts)

    def generate(self, question: str, contexts: List[str]) -> str:
        """Generate answer from retrieved legal contexts."""
        ctx_block = self._build_contexts(contexts)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            contexts=ctx_block,
            question=question,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def generate_or_fallback(
        self, question: str, contexts: List[str], fallback_text: str = ""
    ) -> str:
        """Generate answer, falling back to raw text on error."""
        try:
            return self.generate(question, contexts)
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            if fallback_text:
                return (
                    f"(Không thể sinh câu trả lời tự động do lỗi: {e})\n\n"
                    f"Kết quả tra cứu thô:\n{fallback_text[:800]}"
                )
            return f"Xin lỗi, đã có lỗi xảy ra khi sinh câu trả lời: {e}"
