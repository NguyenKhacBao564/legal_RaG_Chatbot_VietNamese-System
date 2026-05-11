"""
FastAPI serving system for Vietnamese Legal LLM
Optimized for high-performance inference on GPU
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

# ML libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
try:
    from unsloth import FastLanguageModel
except (ImportError, NotImplementedError):
    print("Unsloth not available or not supported on this hardware, using standard transformers")

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
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p sampling")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    stream: bool = Field(False, description="Stream response")

class ChatResponse(BaseModel):
    id: str = Field(..., description="Response ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage")

async def load_model():
    """Load the fine-tuned model on startup"""
    global model, tokenizer, model_config
    
    model_path = os.getenv("MODEL_PATH", "/app/model")
    model_name = os.getenv("MODEL_NAME", "latest")
    hf_token = os.getenv("HF_TOKEN") or None
    load_in_4bit = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    local_files_only = os.getenv("HF_HUB_OFFLINE", "false").lower() in ("1", "true", "yes")
    base_model_name = os.getenv("BASE_MODEL_NAME")
    adapter_model_name = os.getenv("ADAPTER_MODEL_NAME")
    torch_dtype = torch.float16
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    model_source = adapter_model_name or (
        model_name if model_name not in ("", "latest") else model_path
    )
    logger.info(f"Loading model source: {model_source}")
    
    # Download model from Digital Ocean Spaces if not exists locally
    try:
        if not Path(model_path).exists() or not any(Path(model_path).iterdir()):
            logger.info("Model not found locally, downloading from Digital Ocean Spaces...")
            
            spaces_manager = DOSpacesManager()
            
            if model_name == "latest":
                # Find latest model
                models = spaces_manager.list_objects("models/")
                if models:
                    # Get latest model (sort by name/timestamp)
                    latest_model = sorted([m for m in models if "vietnamese-legal-llama" in m])[-1]
                    model_name = latest_model.split('/')[1]  # Extract model name
                    logger.info(f"Found latest model: {model_name}")
                else:
                    raise ValueError("No models found in Digital Ocean Spaces")
            
            # Download model
            spaces_prefix = f"models/{model_name}"
            if spaces_manager.download_directory(spaces_prefix, model_path):
                logger.info(f"✅ Model downloaded from DO Spaces: {model_name}")
            else:
                raise ValueError("Failed to download model from Digital Ocean Spaces")
                
    except Exception as e:
        logger.warning(f"Could not download from DO Spaces: {e}")
        logger.info("Using local model path...")
    
    try:
        try:
            peft_config = PeftConfig.from_pretrained(
                model_source,
                token=hf_token,
                local_files_only=local_files_only,
            )
            base_model_name = base_model_name or peft_config.base_model_name_or_path
            if not base_model_name:
                base_model_name = "unsloth/Llama-3.1-8B-Instruct"

            logger.info(f"Detected PEFT adapter: {model_source}")
            logger.info(f"Loading base model: {base_model_name}")

            model_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
                "token": hf_token,
                "local_files_only": local_files_only,
                "low_cpu_mem_usage": True,
            }
            if has_cuda:
                model_kwargs["device_map"] = "auto"
            else:
                logger.info("CUDA is not available; loading model without device_map auto/offload")

            if load_in_4bit and has_cuda:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif load_in_4bit:
                logger.info("Skipping runtime 4-bit quantization because CUDA is not available")

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **model_kwargs,
            )
            model = PeftModel.from_pretrained(
                base_model,
                model_source,
                token=hf_token,
                local_files_only=local_files_only,
            )
            if has_mps:
                logger.info("Moving PEFT model to MPS")
                model = model.to("mps")
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                token=hf_token,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            logger.info("PEFT adapter loaded with transformers")
        except Exception as peft_error:
            logger.info(f"PEFT loading skipped/failed: {peft_error}")
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_source,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=load_in_4bit,
                    token=hf_token,
                )
                FastLanguageModel.for_inference(model)
                logger.info("Model loaded with Unsloth")
            except Exception as unsloth_error:
                logger.info(f"Unsloth loading failed: {unsloth_error}")
                logger.info("Loading as a full transformers model...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    token=hf_token,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    token=hf_token,
                    local_files_only=local_files_only,
                    low_cpu_mem_usage=True,
                )
                if has_mps:
                    model = model.to("mps")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
            
        # Load model config if available
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
        
        # Load model info if available
        info_path = Path(model_path) / "model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
                logger.info(f"Model info: {model_info.get('model_name', 'unknown')}")
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Vietnamese Legal LLM API server...")
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down server...")

# Initialize FastAPI app
app = FastAPI(
    title="Vietnamese Legal LLM API",
    description="Fine-tuned Llama-3.1-8B for Vietnamese Legal Questions",
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

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Format messages into Llama-3.1 chat format"""
    
    system_token = "<|start_header_id|>system<|end_header_id|>"
    user_token = "<|start_header_id|>user<|end_header_id|>"
    assistant_token = "<|start_header_id|>assistant<|end_header_id|>"
    eos_token = "<|eot_id|>"
    
    # Default system message for legal domain
    default_system = (
        "Bạn là một chuyên gia tư vấn pháp luật Việt Nam với nhiều năm kinh nghiệm. "
        "Hãy trả lời các câu hỏi một cách chính xác, chi tiết và dễ hiểu. "
        "Luôn dẫn nguồn từ các văn bản pháp luật cụ thể khi có thể."
    )
    
    formatted_prompt = ""
    
    # Add system message
    system_message = None
    for msg in messages:
        if msg.role == "system":
            system_message = msg.content
            break
    
    if not system_message:
        system_message = default_system
    
    formatted_prompt += f"{system_token}\n\n{system_message}{eos_token}\n"
    
    # Add conversation history
    for msg in messages:
        if msg.role == "user":
            formatted_prompt += f"{user_token}\n\n{msg.content}{eos_token}\n"
        elif msg.role == "assistant":
            formatted_prompt += f"{assistant_token}\n\n{msg.content}{eos_token}\n"
    
    # Add assistant token for generation
    formatted_prompt += f"{assistant_token}\n\n"
    
    return formatted_prompt

def extract_response(generated_text: str, prompt: str) -> str:
    """Extract assistant response from generated text"""
    # Remove the prompt
    if prompt in generated_text:
        response = generated_text.replace(prompt, "").strip()
    else:
        response = generated_text
    
    # Remove end tokens
    end_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]
    for token in end_tokens:
        response = response.replace(token, "")
    
    return response.strip()

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text.split())  # Rough estimate

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    
    memory_usage = {}
    if gpu_available:
        memory_usage = {
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        }
    
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        gpu_available=gpu_available,
        memory_usage=memory_usage
    )

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """Chat completions endpoint compatible with OpenAI API"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format prompt
        prompt = format_chat_prompt(request.messages)
        
        # Generate response
        start_time = time.time()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response_text = extract_response(generated_text, prompt)
        
        generation_time = time.time() - start_time
        
        # Count tokens
        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(response_text)
        total_tokens = prompt_tokens + completion_tokens
        
        # Create response
        response = ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model="vietnamese-legal-llama",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        
        logger.info(f"Generated response in {generation_time:.2f}s, {completion_tokens} tokens")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions/stream")
async def chat_completions_stream(request: ChatRequest):
    """Streaming chat completions endpoint"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def generate_stream():
        try:
            # Format prompt
            prompt = format_chat_prompt(request.messages)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Create a streaming generator
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            with torch.no_grad():
                # Generate with streaming
                generation_kwargs = {
                    **inputs,
                    "max_new_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "do_sample": True,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "streamer": streamer,
                }
                
                # Note: This is a simplified streaming implementation
                # For production, you'd want to implement proper async streaming
                outputs = model.generate(**generation_kwargs)
                
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            response_text = extract_response(generated_text, prompt)
            
            # Simulate streaming by sending chunks
            words = response_text.split()
            for i, word in enumerate(words):
                chunk_data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "vietnamese-legal-llama",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": word + " "},
                            "finish_reason": None
                        }
                    ]
                }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            # Send final chunk
            final_chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "vietnamese-legal-llama",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/models")
async def list_models():
    """List available models endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "vietnamese-legal-llama",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vietnamese-legal-system"
            }
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vietnamese Legal LLM API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 6000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "serve_model:app",
        host=host,
        port=port,
        reload=False,
        workers=1  # Single worker for GPU usage
    )
