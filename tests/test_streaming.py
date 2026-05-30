import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "src"))

from streaming import format_sse, split_text_for_sse


def test_format_sse_uses_event_stream_shape():
    message = format_sse("delta", {"content": "Xin chào"})

    assert message.startswith("event: delta\n")
    assert message.endswith("\n\n")

    data_line = [line for line in message.splitlines() if line.startswith("data: ")][0]
    payload = json.loads(data_line.removeprefix("data: "))
    assert payload == {"content": "Xin chào"}


def test_split_text_for_sse_preserves_content():
    text = "Một câu trả lời pháp lý dài cần được chia thành nhiều phần nhỏ."
    chunks = list(split_text_for_sse(text, chunk_size=18))

    assert len(chunks) > 1
    assert "".join(chunks).strip() == text
