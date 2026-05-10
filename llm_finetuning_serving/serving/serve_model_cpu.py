#!/usr/bin/env python3
"""
Local serving script for Vietnamese Legal LLM.
Supports CPU, Apple Silicon MPS, and Hugging Face PEFT/LoRA adapters.
"""

import os
import json
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env file will not be loaded.")
    print("Install with: pip install python-dotenv")

# Import DO Spaces manager
import sys
sys.path.append('..')
from do_spaces_manager import DOSpacesManager

# FastAPI and Pydantic
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# ML libraries for local inference
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftConfig, PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None
model_config = {}

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    stream: bool = Field(default=False, description="Enable streaming response")

class ChatResponse(BaseModel):
    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Response type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")

def get_device() -> str:
    """Select the best local device."""
    requested_device = os.environ.get("DEVICE", "auto").lower()
    if requested_device != "auto":
        return requested_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_torch_dtype(device: str):
    if device == "cpu":
        return torch.float32
    return torch.float16


def load_model_cpu():
    """Load model optimized for local inference."""
    global model, tokenizer, model_config
    
    try:
        model_path = os.environ.get("MODEL_PATH", "")
        model_name = os.environ.get(
            "MODEL_NAME",
            "NguyenBao564/vietnamese-legal-llama-3.1-8b",
        )
        adapter_model_name = os.environ.get("ADAPTER_MODEL_NAME") or model_name
        base_model_name = os.environ.get("BASE_MODEL_NAME")
        hf_token = os.environ.get("HF_TOKEN") or None
        device = get_device()
        torch_dtype = resolve_torch_dtype(device)

        model_source = adapter_model_name or model_path
        logger.info(f"Loading model source: {model_source}")
        logger.info(f"Selected device: {device}")
        
        try:
            peft_config = PeftConfig.from_pretrained(model_source, token=hf_token)
            base_model_name = base_model_name or peft_config.base_model_name_or_path
            if not base_model_name:
                base_model_name = "unsloth/Llama-3.1-8B-Instruct"

            logger.info(f"Detected PEFT adapter: {model_source}")
            logger.info(f"Loading base model: {base_model_name}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                token=hf_token,
                trust_remote_code=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=hf_token,
            )
            model = PeftModel.from_pretrained(
                base_model,
                model_source,
                token=hf_token,
            )
        except Exception as peft_error:
            logger.info(f"PEFT loading skipped/failed: {peft_error}")
            local_or_remote_model = model_path or model_name
            tokenizer = AutoTokenizer.from_pretrained(
                local_or_remote_model,
                token=hf_token,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                local_or_remote_model,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=hf_token,
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(device)
        model.eval()
        
        # Load model config
        config_path = Path(model_path) / "config.json" if model_path else None
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def format_conversation(messages: List[ChatMessage]) -> str:
    """Format conversation with the tokenizer chat template when available."""
    payload = [{"role": msg.role, "content": msg.content} for msg in messages]
    if tokenizer and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            payload,
            tokenize=False,
            add_generation_prompt=True,
        )

    system = "Bạn là trợ lý tư vấn pháp luật Việt Nam."
    turns = []
    for msg in payload:
        if msg["role"] == "system":
            system = msg["content"]
        elif msg["role"] == "user":
            turns.append(f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>")
        elif msg["role"] == "assistant":
            turns.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>")
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>\n"
        + "\n".join(turns)
        + "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def generate_response(messages: List[ChatMessage], temperature: float, max_tokens: int, top_p: float) -> str:
    """Generate response using CPU inference"""
    global model, tokenizer
    
    try:
        # Format conversation
        prompt = format_conversation(messages)
        
        # Tokenize input
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the new generated text
        generated_text = response[len(prompt):].strip()
        for token in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
            generated_text = generated_text.replace(token, "")
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Vietnamese Legal LLM API Server (local CPU/MPS version)...")
    load_model_cpu()
    yield
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Vietnamese Legal LLM API (CPU)",
    description="CPU-optimized API for Vietnamese Legal Language Model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Vietnamese Legal LLM API (CPU Version)", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(next(model.parameters()).device) if model is not None else "unknown",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    try:
        start_time = time.time()
        
        # Generate response
        generated_text = generate_response(
            request.messages, 
            request.temperature, 
            request.max_tokens,
            request.top_p
        )
        
        generation_time = time.time() - start_time
        
        # Count tokens (approximate)
        prompt_tokens = len(format_conversation(request.messages).split())
        completion_tokens = len(generated_text.split())
        total_tokens = prompt_tokens + completion_tokens
        
        # Create response
        response = ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model="vietnamese-legal-llama",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        
        logger.info(f"Generated response in {generation_time:.2f}s, tokens: {total_tokens}")
        return response
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 6000))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
