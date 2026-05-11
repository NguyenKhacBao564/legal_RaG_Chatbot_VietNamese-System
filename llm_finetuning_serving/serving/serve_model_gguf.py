#!/usr/bin/env python3
"""
FastAPI server for local GGUF inference with llama.cpp.

This is intended for Apple Silicon / CPU local serving where loading the
Transformers FP16 model is too heavy. It serves an OpenAI-compatible
/v1/chat/completions endpoint used by the chatbot backend.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm: Llama | None = None
generation_lock = asyncio.Lock()
PROJECT_DIR = Path(__file__).resolve().parents[1]


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(256, ge=1, le=4096)
    stream: bool = False


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def _load_model() -> Llama:
    model_path = _env_path(
        "GGUF_MODEL_PATH",
        str(PROJECT_DIR / "models/gguf/Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    )
    lora_path_raw = os.getenv(
        "GGUF_LORA_PATH",
        str(PROJECT_DIR / "models/gguf/vietnamese-legal-lora.gguf"),
    )
    lora_path = Path(lora_path_raw).expanduser().resolve() if lora_path_raw else None

    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")
    if lora_path and not lora_path.exists():
        raise FileNotFoundError(f"GGUF LoRA adapter not found: {lora_path}")

    n_ctx = int(os.getenv("LLAMA_CPP_N_CTX", "2048"))
    n_batch = int(os.getenv("LLAMA_CPP_N_BATCH", "512"))
    n_threads = int(os.getenv("LLAMA_CPP_N_THREADS", str(os.cpu_count() or 4)))
    n_gpu_layers = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "-1"))
    chat_format = os.getenv("LLAMA_CPP_CHAT_FORMAT") or None

    logger.info("Loading GGUF model: %s", model_path)
    if lora_path:
        logger.info("Loading GGUF LoRA adapter: %s", lora_path)

    return Llama(
        model_path=str(model_path),
        lora_path=str(lora_path) if lora_path else None,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        chat_format=chat_format,
        verbose=os.getenv("LLAMA_CPP_VERBOSE", "false").lower() == "true",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    start = time.time()
    llm = _load_model()
    logger.info("GGUF model loaded in %.2fs", time.time() - start)
    yield
    logger.info("Shutting down GGUF model server")


app = FastAPI(
    title="Vietnamese Legal LLM GGUF API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _completion(request: ChatRequest) -> dict[str, Any]:
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    max_tokens_limit = int(os.getenv("LLAMA_CPP_MAX_TOKENS", "384"))
    max_tokens = min(request.max_tokens, max_tokens_limit)
    messages = [message.model_dump() for message in request.messages]

    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=False,
    )
    content = result["choices"][0]["message"].get("content", "").strip()
    if not content:
        raise HTTPException(status_code=502, detail="Model returned empty content")

    return {
        "id": result.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": result.get("created", int(time.time())),
        "model": request.model or "vietnamese-legal-llama-gguf",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": result["choices"][0].get("finish_reason", "stop"),
            }
        ],
        "usage": result.get("usage", {}),
    }


@app.get("/")
async def root():
    return {"message": "Vietnamese Legal LLM GGUF API", "model_loaded": llm is not None}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if llm is not None else "model_not_loaded",
        "model_loaded": llm is not None,
        "backend": "llama.cpp",
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this local server")
    async with generation_lock:
        start = time.time()
        response = await asyncio.to_thread(_completion, request)
        logger.info("Generated response in %.2fs", time.time() - start)
        return response


@app.get("/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "vietnamese-legal-llama-gguf",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "6000"))
    uvicorn.run("serve_model_gguf:app", host=host, port=port, reload=False, workers=1)
