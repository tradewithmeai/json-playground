"""
Simple OpenAI-compatible API server for local LLM models.
Loads a HuggingFace model and serves it via FastAPI.
"""

import argparse
import json
import time
import uuid
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import uvicorn


# Request/Response models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# Global model and tokenizer
model = None
tokenizer = None
model_name = "local-model"


def load_model(model_path: str, use_4bit: bool = True):
    """Load the model and tokenizer from the specified path."""
    global model, tokenizer, model_name

    print(f"Loading model from: {model_path}")
    print("This may take a few minutes...")

    # Detect available device
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using CUDA GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
    else:
        device = "cpu"
        print("Using CPU (this will be slower)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Load model
    if device == "cuda" and use_4bit:
        print("Loading with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    elif device == "cuda":
        print("Loading with full precision (float16)...")
        print("WARNING: 7B model needs ~14GB VRAM - may use CPU offloading")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("Loading on CPU (float32)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model = model.to(device)

    mode = "4-bit" if (device == "cuda" and use_4bit) else "full precision"
    model_name = model_path.split("\\")[-1].split("/")[-1]
    print(f"Model loaded successfully: {model_name} ({mode})")


def generate_response(messages: List[Message], temperature: float, max_tokens: int, top_p: float) -> tuple:
    """Generate a response from the model."""
    global model, tokenizer

    # Format messages for the model
    # Qwen uses ChatML format
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif msg.role == "user":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"

    # Add assistant prompt for generation
    prompt += "<|im_start|>assistant\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    prompt_tokens = input_ids.shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    # Decode response
    new_tokens = outputs[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up response
    response_text = response_text.strip()
    if response_text.endswith("<|im_end|>"):
        response_text = response_text[:-10].strip()

    completion_tokens = len(new_tokens)

    return response_text, prompt_tokens, completion_tokens


# Create FastAPI app
app = FastAPI(title="Local LLM Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "running", "model": model_name}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported")

    try:
        response_text, prompt_tokens, completion_tokens = generate_response(
            request.messages,
            request.temperature or 0.7,
            request.max_tokens or 512,
            request.top_p or 0.9
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Local LLM Server")
    parser.add_argument(
        "--model-path",
        type=str,
        default=r"C:\ai-models\text-generation-webui\models\Qwen2.5-7B-Instruct",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full precision (float16) instead of 4-bit quantization"
    )

    args = parser.parse_args()

    # Load the model
    load_model(args.model_path, use_4bit=not args.full)

    # Start the server
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
