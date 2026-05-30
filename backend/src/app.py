import logging
import os
import time
from typing import Dict, Optional

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from configs import DEFAULT_COLLECTION_NAME
from pipeline import handle_chat_message
from models import insert_document
from tasks import index_document_v2, llm_handle_message
from streaming import format_sse, split_text_for_sse
from utils import setup_logging
from vectorize import create_collection, get_collection_stats

# Constants
TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "420"))
FORCE_SYNC_CHAT = os.getenv("FORCE_SYNC_CHAT", "false").lower() == "true"
POLLING_INTERVAL = 0.5

setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI()


class CompleteRequest(BaseModel):
    bot_id: Optional[str] = "botLawyer"
    user_id: str
    user_message: str
    sync_request: Optional[bool] = False


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Vietnamese Legal Chatbot Backend"}


@app.get("/ready")
async def ready():
    dependencies = {
        "chat_provider": {
            "configured": bool(
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("VIETNAMESE_LLM_API_URL")
            )
        },
        "qdrant": {"collection": DEFAULT_COLLECTION_NAME, "status": "unknown"},
    }

    try:
        stats = get_collection_stats(DEFAULT_COLLECTION_NAME)
        if stats and not stats.get("error"):
            dependencies["qdrant"].update(
                {
                    "status": str(stats.get("status", "ok")),
                    "points_count": stats.get("points_count"),
                    "vectors_count": stats.get("vectors_count"),
                }
            )
        else:
            dependencies["qdrant"].update(
                {
                    "status": "unavailable",
                    "error": stats.get("error") if stats else "unknown",
                }
            )
    except Exception as exc:
        logger.warning("Readiness Qdrant check failed: %s", exc)
        dependencies["qdrant"].update({"status": "unavailable", "error": str(exc)})

    service_ready = dependencies["chat_provider"]["configured"]
    status = "ready" if service_ready else "degraded"
    return {
        "status": status,
        "service": "Vietnamese Legal Chatbot Backend",
        "dependencies": dependencies,
    }


@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    bot_id = data.bot_id
    user_id = data.user_id
    user_message = data.user_message
    logger.info(f"Complete chat from user {user_id} to {bot_id}: {user_message}")

    if not user_message or not user_id:
        raise HTTPException(
            status_code=400, detail="User id and user message are required"
        )

    if data.sync_request or FORCE_SYNC_CHAT:
        response = handle_chat_message(bot_id, user_id, user_message)
        return {"response": str(response)}
    else:
        task = llm_handle_message.delay(bot_id, user_id, user_message)
        return {"task_id": task.id}


@app.post("/chat/stream")
async def stream(data: CompleteRequest):
    bot_id = data.bot_id
    user_id = data.user_id
    user_message = data.user_message
    logger.info("Streaming chat from user %s to %s", user_id, bot_id)

    if not user_message or not user_id:
        raise HTTPException(
            status_code=400, detail="User id and user message are required"
        )

    async def event_generator():
        yield format_sse("start", {"message": "request_received"})
        try:
            response = await run_in_threadpool(
                handle_chat_message, bot_id, user_id, user_message
            )
            content = str(response.get("content", ""))
            yield format_sse("metadata", {"role": "assistant", "streaming": "simulated"})
            for chunk in split_text_for_sse(content):
                yield format_sse("delta", {"content": chunk})
            yield format_sse("done", {"content": content})
        except Exception as exc:
            logger.exception("Streaming chat failed: %s", exc)
            yield format_sse(
                "error",
                {
                    "message": (
                        "Xin lỗi, hệ thống đang gặp lỗi khi tạo câu trả lời. "
                        "Vui lòng thử lại sau."
                    )
                },
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/chat/complete/{task_id}")
async def get_response(task_id: str):
    start_time = time.time()
    while True:
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        logger.info(f"Task result: {task_result.result}")

        if task_status == "PENDING":
            if time.time() - start_time > TASK_TIMEOUT:
                return {
                    "task_id": task_id,
                    "task_status": task_result.status,
                    "task_result": task_result.result,
                    "error_message": "Service timeout, retry please",
                }
            else:
                time.sleep(POLLING_INTERVAL)  # sleep for 0.5 seconds before retrying
        else:
            result = {
                "task_id": task_id,
                "task_status": task_result.status,
                "task_result": task_result.result,
            }
            return result


@app.post("/collection/create")
async def create_vector_collection(data: Dict):
    collection_name = data.get("collection_name")
    create_status = create_collection(collection_name)
    logging.info(f"Create collection {collection_name} status: {create_status}")
    return {"status": create_status is not None}


@app.post("/document/create")
async def create_document(data: Dict):
    doc_id = data.get("id")
    question = data.get("question")
    content = data.get("content")
    create_status = insert_document(question, content)
    logging.info(f"Create document status: {create_status}")
    index_status = index_document_v2(doc_id, question, content)
    return {"status": create_status is not None, "index_status": index_status}


@app.post("/data/import")
async def import_qa_data_endpoint():
    from import_data import import_qa_data

    success = import_qa_data()
    return {"success": success}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        workers=int(os.getenv("WEB_CONCURRENCY", "1")),
        log_level="info",
    )
