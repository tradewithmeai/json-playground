"""
Model server that can run in background thread.
"""

import warnings
import threading
import time
import uuid
from typing import List, Optional

# Suppress flash attention warning on Windows
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

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


class ModelServer:
    """Manages loading a model and serving it via FastAPI."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "local-model"
        self.app = None
        self.server_thread = None
        self.server = None
        self._setup_app()

    def _setup_app(self):
        """Setup FastAPI app with routes."""
        self.app = FastAPI(title="Local LLM Server")

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/")
        def root():
            return {"status": "running", "model": self.model_name}

        @self.app.get("/v1/models")
        def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local"
                    }
                ]
            }

        @self.app.post("/v1/chat/completions")
        def chat_completions(request: ChatCompletionRequest):
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            if request.stream:
                raise HTTPException(status_code=400, detail="Streaming not supported")

            try:
                response_text, prompt_tokens, completion_tokens = self._generate(
                    request.messages,
                    request.temperature or 0.7,
                    request.max_tokens or 512,
                    request.top_p or 0.9
                )

                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    created=int(time.time()),
                    model=self.model_name,
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

    def load_model(self, model_path: str, use_4bit: bool = True, callback=None):
        """Load the model and tokenizer."""
        def log(msg):
            if callback:
                callback(msg)

        log(f"Model path: {model_path}")

        # Detect device
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log(f"GPU detected: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        else:
            device = "cpu"
            log("No GPU - using CPU (this will be slower)")

        # Load tokenizer
        log("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        log("Tokenizer loaded.")

        # Load model
        if device == "cuda" and use_4bit:
            if callback:
                callback("Loading with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        elif device == "cuda":
            if callback:
                callback("Loading with full precision (float16)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            if callback:
                callback("Loading on CPU (float32)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            self.model = self.model.to(device)

        mode = "4-bit" if (device == "cuda" and use_4bit) else "full"
        self.model_name = model_path.split("\\")[-1].split("/")[-1]

        if callback:
            callback(f"Model loaded: {self.model_name} ({mode})")

        return True

    def _generate(self, messages: List[Message], temperature: float, max_tokens: int, top_p: float) -> tuple:
        """Generate a response from the model."""
        # Format messages (ChatML format for Qwen)
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif msg.role == "user":
                prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif msg.role == "assistant":
                prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        prompt_tokens = input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            )

        # Decode
        new_tokens = outputs[0][input_ids.shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        response_text = response_text.strip()
        if response_text.endswith("<|im_end|>"):
            response_text = response_text[:-10].strip()

        return response_text, prompt_tokens, len(new_tokens)

    def start(self, host: str = "127.0.0.1", port: int = 5000):
        """Start the server in a background thread."""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config)

        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()

        return f"http://{host}:{port}"

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.should_exit = True
            if self.server_thread:
                self.server_thread.join(timeout=5)

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.server_thread is not None and self.server_thread.is_alive()
