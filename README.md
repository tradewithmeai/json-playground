# JSON Playground - LLM Testing Suite

Tools for testing and benchmarking local and remote LLM models via OpenAI-compatible APIs.

## Quick Start (Recommended)

The unified benchmark tool handles everything in one terminal:

```bash
python llm_bench.py
```

Interactive menu guides you through:
1. **Local Model** - Load a model and start chatting (no separate server needed)
2. **Remote Endpoint** - Connect to any OpenAI-compatible API
3. **Compare** - Benchmark local model vs cloud endpoint (loads ONE local model)

## Components

| File | Description |
|------|-------------|
| `llm_bench.py` | **Unified tool** - Interactive setup, auto server management |
| `model_server.py` | Standalone server (for advanced use) |
| `chat_client.py` | Standalone client (for advanced use) |
| `lib/` | Shared modules (server, client, utils) |
| `ui_prompt_playground.py` | Gradio web UI (experimental) |

## Standalone Usage

For more control, use the separate server and client:

### 1. Start the Model Server

```bash
# 4-bit quantization (default, fits in 8GB VRAM)
python model_server.py

# Full precision (needs ~14GB VRAM for 7B model)
python model_server.py --full

# Custom port
python model_server.py --port 5001
```

### 2. Chat with the Model

```bash
python chat_client.py
```

## Model Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | `C:\ai-models\text-generation-webui\models\Qwen2.5-7B-Instruct` | Path to model |
| `--host` | `127.0.0.1` | Host to bind |
| `--port` | `5000` | Port to bind |
| `--full` | off | Full precision (float16) instead of 4-bit |

## Chat Client Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | `http://127.0.0.1:5000/v1/chat/completions` | API URL |
| `--compare` | off | Compare two models side-by-side |
| `--url1` | port 5000 | First model URL (comparison mode) |
| `--url2` | port 5001 | Second model URL (comparison mode) |
| `--name1` | `Model-A` | Label for first model |
| `--name2` | `Model-B` | Label for second model |

### Chat Commands
- `quit` or `q` - Exit
- `new` - Clear conversation history

## Comparison Mode (Benchmarking)

Compare two model configurations side-by-side:

```bash
# Terminal 1: 4-bit model on port 5000
python model_server.py

# Terminal 2: Full precision on port 5001
python model_server.py --full --port 5001

# Terminal 3: Compare both
python chat_client.py --compare --name1 "4-bit" --name2 "Full"
```

Both requests run in parallel - shows response times and declares winner.

## Connecting to Other LLM Servers

The chat client works with any OpenAI-compatible API:

```bash
# Ollama
python chat_client.py --url http://127.0.0.1:11434/v1/chat/completions

# LM Studio
python chat_client.py --url http://127.0.0.1:1234/v1/chat/completions

# vLLM
python chat_client.py --url http://127.0.0.1:8000/v1/chat/completions

# Remote server
python chat_client.py --url https://your-server.com/v1/chat/completions
```

## Web UI (Experimental)

The Gradio-based playground UI has some stability issues but can be started with:

```bash
python ui_prompt_playground.py
```

Features:
- JSON editor with parameter controls
- Chat view with message history
- Multi-endpoint support (Local, RunPod, Custom)
- Public URL via localtunnel

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- `requests` - HTTP client
- `torch` - PyTorch for model loading
- `transformers` - HuggingFace model loading
- `fastapi`, `uvicorn` - API server
- `bitsandbytes` - 4-bit quantization
- `accelerate` - Model distribution
- `gradio` - Web UI (optional)

## API Endpoints

When model_server.py is running:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server status |
| `/v1/models` | GET | List loaded models |
| `/v1/chat/completions` | POST | Chat completion (OpenAI format) |

### Example Request

```bash
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```
