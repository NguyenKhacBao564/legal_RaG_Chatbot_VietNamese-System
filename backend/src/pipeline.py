import json
import logging
import os
import time
from typing import Dict, List

from agent import ai_agent_handle
from brain import (
    detect_route,
    detect_user_intent,
    gen_doc_prompt,
    openai_chat_complete,
    vietnamese_llm_chat_complete,
)
from configs import DEFAULT_COLLECTION_NAME
from models import get_conversation_messages, update_chat_conversation
from query_rewriter import rewrite_query_to_multi_queries
from rerank import rerank_documents
from search import hybrid_search
from summarizer import summarize_text
from tavily_tool import tavily_search_legal
from vectorize import get_collection_stats

logger = logging.getLogger(__name__)

VALID_ROUTES = {"legal_rag", "agent_tools", "web_search", "general_chat"}

RAG_UNAVAILABLE_NOTICE = (
    "Lưu ý: Hiện tại hệ thống không truy xuất được ngữ cảnh từ kho dữ liệu RAG. "
    "Câu trả lời dưới đây được tạo dựa trên kiến thức chung của mô hình và cần "
    "được kiểm chứng lại với văn bản pháp luật hoặc chuyên gia pháp lý."
)

LLM_ERROR_MESSAGE = (
    "Xin lỗi, hệ thống AI đang gặp lỗi khi tạo câu trả lời. "
    "Vui lòng thử lại sau hoặc rút gọn câu hỏi."
)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _new_metrics() -> Dict:
    return {
        "route": "unknown",
        "route_detection_ms": 0,
        "retrieval_ms": 0,
        "rerank_ms": 0,
        "llm_generation_ms": 0,
        "total_request_ms": 0,
        "fallback_used": False,
        "fallback_reasons": [],
    }


def _mark_fallback(metrics: Dict, reason: str) -> None:
    metrics["fallback_used"] = True
    if reason not in metrics["fallback_reasons"]:
        metrics["fallback_reasons"].append(reason)


def _log_metrics(metrics: Dict) -> None:
    logger.info(
        "rag_pipeline_metrics=%s",
        json.dumps(metrics, ensure_ascii=False, sort_keys=True),
    )


def _safe_llm_complete(messages: List[Dict], metrics: Dict, max_tokens: int = 4096) -> str:
    started = time.perf_counter()
    try:
        content = vietnamese_llm_chat_complete(messages, max_tokens=max_tokens)
        if not str(content).strip():
            _mark_fallback(metrics, "llm_empty_response")
            return LLM_ERROR_MESSAGE
        return str(content).strip()
    except Exception as exc:
        _mark_fallback(metrics, "llm_provider_error")
        logger.exception("LLM provider failed: %s", exc)
        return LLM_ERROR_MESSAGE
    finally:
        metrics["llm_generation_ms"] += _elapsed_ms(started)


def _fallback_general_legal_answer(
    question: str, history: List[Dict], metrics: Dict, reason: str
) -> str:
    _mark_fallback(metrics, reason)
    recent_history = history[-4:] if history else []
    messages = (
        [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý tư vấn pháp luật Việt Nam. Nếu không có ngữ cảnh RAG, "
                    "hãy trả lời thận trọng, nói rõ đây chỉ là thông tin tham khảo và "
                    "khuyến nghị kiểm chứng với văn bản pháp luật chính thức."
                ),
            }
        ]
        + recent_history
        + [{"role": "user", "content": question}]
    )
    answer = _safe_llm_complete(messages, metrics)
    return f"{RAG_UNAVAILABLE_NOTICE}\n\n{answer}"


def _follow_up_question(history: List[Dict], question: str) -> str:
    return detect_user_intent(history, question)


def _retrieve_documents(queries: List[str], top_k: int = 4) -> List[Dict]:
    all_docs = []
    seen_contents = set()

    for query in queries:
        docs = hybrid_search(query, limit=top_k)
        for doc in docs:
            content_hash = hash(doc.get("content", ""))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                doc["retrieval_query"] = query
                all_docs.append(doc)

    return all_docs


def _handle_legal_rag(history: List[Dict], question: str, metrics: Dict) -> str:
    recent_history = history[-4:] if history else []

    stats = get_collection_stats(DEFAULT_COLLECTION_NAME)
    if not stats or stats.get("error"):
        logger.warning("Qdrant collection unavailable: %s", stats)
        return _fallback_general_legal_answer(
            question, history, metrics, "qdrant_unavailable"
        )

    doc_count = stats.get("points_count") or stats.get("vectors_count") or 0
    if not doc_count:
        logger.warning("Qdrant collection is empty: %s", stats)
        return _fallback_general_legal_answer(
            question, history, metrics, "qdrant_collection_empty"
        )

    retrieval_started = time.perf_counter()
    try:
        standalone_question = _follow_up_question(history, question)
        query_variations = rewrite_query_to_multi_queries(
            standalone_question, num_queries=3
        )
        if not query_variations:
            query_variations = [standalone_question]
        retrieved_docs = _retrieve_documents(query_variations, top_k=4)
    except Exception as exc:
        _mark_fallback(metrics, "retrieval_error")
        logger.exception("RAG retrieval failed: %s", exc)
        retrieved_docs = []
        standalone_question = question
    finally:
        metrics["retrieval_ms"] = _elapsed_ms(retrieval_started)

    if not retrieved_docs:
        logger.warning("RAG retrieval returned no documents")
        return _fallback_general_legal_answer(
            question, history, metrics, "retrieval_no_documents"
        )

    rerank_started = time.perf_counter()
    try:
        if not os.getenv("COHERE_API_KEY"):
            _mark_fallback(metrics, "rerank_api_key_missing")
        ranked_docs = rerank_documents(retrieved_docs, standalone_question, top_n=5)
        if not ranked_docs:
            _mark_fallback(metrics, "rerank_no_documents")
            ranked_docs = retrieved_docs[:5]
    except Exception as exc:
        _mark_fallback(metrics, "rerank_error")
        logger.exception("Rerank failed, using retrieved docs: %s", exc)
        ranked_docs = retrieved_docs[:5]
    finally:
        metrics["rerank_ms"] = _elapsed_ms(rerank_started)

    system_prompt = """Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là:
1. Trả lời câu hỏi dựa trên các tài liệu pháp luật được cung cấp
2. Trích dẫn chính xác các điều khoản, khoản, điểm từ văn bản pháp luật nếu có trong ngữ cảnh
3. Giải thích rõ ràng, dễ hiểu cho người không chuyên
4. Nếu thông tin không đủ trong tài liệu, hãy nói rõ điều đó
5. Luôn đưa ra câu trả lời có căn cứ pháp lý

QUAN TRỌNG: Chỉ sử dụng thông tin từ các tài liệu được cung cấp bên dưới."""

    doc_context = gen_doc_prompt(ranked_docs)
    messages = (
        [{"role": "system", "content": system_prompt}]
        + recent_history
        + [
            {
                "role": "user",
                "content": (
                    f"{doc_context}\n\nCâu hỏi: {question}\n\n"
                    "Hãy trả lời dựa trên các tài liệu pháp luật trên."
                ),
            }
        ]
    )

    return _safe_llm_complete(messages, metrics)


def _handle_agent_tools(history: List[Dict], question: str, metrics: Dict) -> str:
    started = time.perf_counter()
    try:
        standalone_question = _follow_up_question(history, question)
        response = ai_agent_handle(standalone_question)
        return str(response).strip() or LLM_ERROR_MESSAGE
    except Exception as exc:
        _mark_fallback(metrics, "agent_tools_error")
        logger.exception("Agent tools failed: %s", exc)
        return _fallback_general_legal_answer(
            question, history, metrics, "agent_tools_fallback"
        )
    finally:
        metrics["llm_generation_ms"] += _elapsed_ms(started)


def _handle_web_search(history: List[Dict], question: str, metrics: Dict) -> str:
    started = time.perf_counter()
    try:
        standalone_question = _follow_up_question(history, question)
        search_results = tavily_search_legal(standalone_question, max_results=5)
        if not search_results:
            return _fallback_general_legal_answer(
                question, history, metrics, "web_search_no_results"
            )

        system_prompt = (
            "Bạn là trợ lý AI giúp tìm kiếm thông tin pháp luật trên internet. "
            "Hãy tổng hợp và trả lời câu hỏi dựa trên kết quả tìm kiếm được cung cấp."
        )
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [
                {
                    "role": "user",
                    "content": (
                        f"Kết quả tìm kiếm:\n{search_results}\n\n"
                        f"Câu hỏi: {question}\n\n"
                        "Hãy tổng hợp thông tin và trả lời."
                    ),
                }
            ]
        )
        return openai_chat_complete(messages)
    except Exception as exc:
        _mark_fallback(metrics, "web_search_error")
        logger.exception("Web search route failed: %s", exc)
        return _fallback_general_legal_answer(
            question, history, metrics, "web_search_fallback"
        )
    finally:
        metrics["llm_generation_ms"] += _elapsed_ms(started)


def _handle_general_chat(history: List[Dict], question: str, metrics: Dict) -> str:
    system_prompt = """Bạn là trợ lý AI thân thiện của hệ thống tư vấn pháp luật Việt Nam.
Hãy trả lời lịch sự và hướng dẫn người dùng về các câu hỏi pháp luật bạn có thể giúp đỡ."""
    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )
    return _safe_llm_complete(messages, metrics, max_tokens=1024)


def _route_answer(history: List[Dict], question: str, metrics: Dict) -> str:
    route_started = time.perf_counter()
    try:
        route = detect_route(history, question)
        if route not in VALID_ROUTES:
            _mark_fallback(metrics, "invalid_route")
            route = "legal_rag"
    except Exception as exc:
        _mark_fallback(metrics, "route_detection_error")
        logger.exception("Route detection failed, using legal_rag: %s", exc)
        route = "legal_rag"
    finally:
        metrics["route_detection_ms"] = _elapsed_ms(route_started)

    metrics["route"] = route

    if route == "legal_rag":
        return _handle_legal_rag(history, question, metrics)
    if route == "agent_tools":
        return _handle_agent_tools(history, question, metrics)
    if route == "web_search":
        return _handle_web_search(history, question, metrics)
    return _handle_general_chat(history, question, metrics)


def _safe_summarize_for_storage(response: str) -> str:
    try:
        return summarize_text(response)
    except Exception as exc:
        logger.warning("Failed to summarize response for storage: %s", exc)
        return response


def handle_chat_message(bot_id: str, user_id: str, question: str) -> Dict[str, str]:
    """Handle one chatbot message with latency logging and safe fallbacks."""
    metrics = _new_metrics()
    request_started = time.perf_counter()
    history: List[Dict] = []

    logger.info("Start handle message bot_id=%s user_id=%s", bot_id, user_id)

    try:
        conversation_id = update_chat_conversation(bot_id, user_id, question, True)
        messages = get_conversation_messages(conversation_id)
        history = messages[:-1]
    except Exception as exc:
        _mark_fallback(metrics, "chat_history_unavailable")
        logger.exception("Could not load or save conversation history: %s", exc)

    response = _route_answer(history, question, metrics)

    try:
        summarized_response = _safe_summarize_for_storage(response)
        update_chat_conversation(bot_id, user_id, summarized_response, False)
    except Exception as exc:
        _mark_fallback(metrics, "chat_response_not_persisted")
        logger.exception("Could not save assistant response: %s", exc)

    metrics["total_request_ms"] = _elapsed_ms(request_started)
    _log_metrics(metrics)

    return {"role": "assistant", "content": response}
