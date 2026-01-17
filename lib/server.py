"""
Model server that can run in background thread.
"""

import warnings
import threading
import time
import uuid
import logging
from typing import List, Optional

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*flash-attention.*")
warnings.filterwarnings("ignore", message=".*CUDA extension not installed.*")
warnings.filterwarnings("ignore", message=".*Exllamav2 kernel.*")
warnings.filterwarnings("ignore", message=".*CUDA kernels for auto_gptq.*")
warnings.filterwarnings("ignore", message=".*offload_buffers.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress auto_gptq logging
logging.getLogger("auto_gptq").setLevel(logging.ERROR)

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
                import traceback
                error_details = traceback.format_exc()
                print(f"Generation error: {error_details}", flush=True)
                raise HTTPException(status_code=500, detail=str(e))

    def _is_gptq_model(self, model_path: str) -> bool:
        """Check if the model is a pre-quantized GPTQ model."""
        import os
        # Check for GPTQ in folder name or quantize_config.json
        folder_name = os.path.basename(model_path).lower()
        if "gptq" in folder_name or "awq" in folder_name:
            return True
        # Check for quantize_config.json (GPTQ marker file)
        config_path = os.path.join(model_path, "quantize_config.json")
        if os.path.exists(config_path):
            return True
        return False

    def load_model(self, model_path: str, use_4bit: bool = True, callback=None):
        """Load the model and tokenizer."""
        def log(msg):
            if callback:
                callback(msg)

        log(f"Model path: {model_path}")

        # Check if this is a pre-quantized GPTQ model
        is_gptq = self._is_gptq_model(model_path)
        if is_gptq:
            log("Detected pre-quantized GPTQ model")

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

        # Load model based on type
        if is_gptq:
            # GPTQ models - use auto_gptq directly
            log("Loading GPTQ model (pre-quantized 4-bit)...")
            try:
                # Suppress auto_gptq warnings
                import sys
                import io
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()

                from auto_gptq import AutoGPTQForCausalLM
                self.model = AutoGPTQForCausalLM.from_quantized(
                    model_path,
                    device_map="auto",
                    use_safetensors=True,
                    trust_remote_code=True
                )
                sys.stderr = old_stderr  # Restore stderr
                mode = "GPTQ-4bit"
            except Exception as e:
                sys.stderr = old_stderr  # Restore stderr
                log(f"AutoGPTQ failed: {e}")
                log("Falling back to float16 loading (model may be unsupported by GPTQ)...")
                # Fallback: load as regular float16 model, ignoring GPTQ config
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                # Remove quantization config to prevent GPTQ loading
                if hasattr(config, 'quantization_config'):
                    delattr(config, 'quantization_config')
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
                mode = "float16-fallback"
        elif device == "cuda" and use_4bit:
            # Runtime quantization with bitsandbytes
            log("Loading with bitsandbytes 4-bit quantization...")
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
            mode = "bnb-4bit"
        elif device == "cuda":
            log("Loading with full precision (float16)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            mode = "float16"
        else:
            log("Loading on CPU (float32)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            self.model = self.model.to(device)
            mode = "float32"

        self.model_name = model_path.split("\\")[-1].split("/")[-1]

        log(f"Model loaded: {self.model_name} ({mode})")

        return True

    def _generate(self, messages: List[Message], temperature: float, max_tokens: int, top_p: float) -> tuple:
        """Generate a response from the model."""
        # Convert messages to list of dicts for apply_chat_template
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Use tokenizer's built-in chat template (works for all models)
        prompt = self.tokenizer.apply_chat_template(
            messages_dict,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Get device - handle different model types
        try:
            device = self.model.device
        except AttributeError:
            # GPTQ models may not have .device, check first parameter
            device = next(self.model.parameters()).device

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        prompt_tokens = input_ids.shape[1]

        # Generate - use keyword args for compatibility with GPTQ models
        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the new tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        response_text = response_text.strip()

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
