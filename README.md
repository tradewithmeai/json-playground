# OpenAI JSON Playground

A web-based JSON editor for creating and testing OpenAI-style chat completion requests. Built with Gradio.

## Features

- **JSON Editor Tab**: Compose requests with controls for temperature, top_p, max_tokens
- **Chat View Tab**: View conversation history and inspect individual messages
- **Multi-Endpoint Support**:
  - **Local**: Connect to local LLM servers (text-generation-webui, llama.cpp, etc.)
  - **RunPod**: Auto-connects to RunPod GPU instances via `~/.myai/pod.json`
  - **Custom**: Any OpenAI-compatible API endpoint
- **Message Inspector**: Navigate through chat history and see the exact JSON payload at each point
- **Public URL**: Automatically creates a public tunnel via localtunnel

## Installation

```bash
pip install gradio requests
npm install -g localtunnel  # Optional, for public URL
```

## Usage

```bash
python ui_prompt_playground.py
```

Opens at `http://127.0.0.1:7861` with a public URL printed to console.

## Endpoint Configuration

### Local (default)
Connects to `http://127.0.0.1:5000/v1/chat/completions`. Start your local model server first:

```bash
# For text-generation-webui
cd /path/to/text-generation-webui
start_windows.bat  # or ./start_linux.sh
```

### RunPod
Reads pod configuration from `~/.myai/pod.json`. Requires `VLLM_API_KEY` environment variable.

### Custom
Enter any OpenAI-compatible endpoint URL, model name, and API key.

## Project Structure

```
json-playground/
├── ui_prompt_playground.py  # Main application
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- gradio
- requests
- Node.js (for localtunnel, optional)
