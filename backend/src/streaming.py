import json
from typing import Dict, Iterable


def format_sse(event: str, data: Dict) -> str:
    """Format one Server-Sent Events message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def split_text_for_sse(text: str, chunk_size: int = 80) -> Iterable[str]:
    """Split text into small readable chunks for safe simulated streaming."""
    if not text:
        return

    buffer = ""
    for part in text.split(" "):
        candidate = f"{buffer} {part}".strip()
        if len(candidate) >= chunk_size and buffer:
            yield buffer + " "
            buffer = part
        else:
            buffer = candidate

    if buffer:
        yield buffer
