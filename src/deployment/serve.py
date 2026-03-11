#!/usr/bin/env python3
"""
FastAPI server for LLM inference.

Provides REST API endpoints for text generation.
"""

import time
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.deployment.inference import InferenceEngine
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM LoRA Inference API",
    description="REST API for fine-tuned LLM inference",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine (loaded on startup)
inference_engine: Optional[InferenceEngine] = None


# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt")
    max_new_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(50, ge=0, description="Top-k sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Whether to use sampling")


class ChatRequest(BaseModel):
    """Request model for chat completion."""
    instruction: str = Field(..., description="Instruction for the model")
    input: Optional[str] = Field("", description="Optional input context")
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class GenerationResponse(BaseModel):
    """Response model for generation."""
    generated_text: str
    input_length: int
    output_length: int
    generation_time: float
    tokens_per_second: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: float


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global inference_engine
    
    import os
    model_path = os.getenv("MODEL_PATH", "outputs/merged-model")
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        inference_engine = InferenceEngine(model_path)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        inference_engine = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if inference_engine else "unhealthy",
        "model_loaded": inference_engine is not None,
        "timestamp": time.time(),
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate text from prompt.
    
    Args:
        request: Generation request
        
    Returns:
        Generated text and metadata
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=GenerationResponse)
async def chat(request: ChatRequest):
    """
    Chat completion endpoint.
    
    Args:
        request: Chat request
        
    Returns:
        Generated response and metadata
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_engine.chat(
            instruction=request.instruction,
            input_text=request.input,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LLM LoRA Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/generate": "Text generation",
            "/chat": "Chat completion",
            "/docs": "API documentation",
        }
    }


def main():
    """Run the FastAPI server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM inference API server")
    parser.add_argument("--model-path", type=str, default="outputs/merged-model", help="Path to model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()
    
    # Set model path as environment variable
    import os
    os.environ["MODEL_PATH"] = args.model_path
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Workers: {args.workers}")
    logger.info("API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "src.deployment.serve:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
