"""
LLM Benchmark Tool - Unified entry point for model testing and benchmarking.

This script auto-manages its own virtual environment to avoid dependency conflicts.
On first run, it creates a venv and installs required packages.
"""

import os
import sys
import subprocess

# =============================================================================
# VENV BOOTSTRAP - Must run before any other imports
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, "venv")
VENV_PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")

# Required packages for the LLM benchmark tool
REQUIRED_PACKAGES = [
    "numpy<2",
    "torch==2.2.2+cu118",
    "transformers==4.44.2",
    "accelerate",
    "bitsandbytes==0.43.1",
    "auto-gptq==0.7.1",
    "optimum",
    "fastapi",
    "uvicorn",
    "requests",
    "pydantic",
    "protobuf",
    "sentencepiece",
]

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"


def is_in_venv():
    """Check if we're running inside our venv."""
    return sys.executable.lower() == VENV_PYTHON.lower()


def create_venv():
    """Create the virtual environment."""
    print(f"Creating virtual environment in {VENV_DIR}...")
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
    print("Virtual environment created.")


def install_packages():
    """Install required packages in the venv."""
    print("\nInstalling required packages (this may take several minutes on first run)...")

    # Upgrade pip first
    subprocess.run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"],
                   check=True, capture_output=True)

    # Install torch separately with index URL
    print("  Installing PyTorch with CUDA support...")
    subprocess.run([
        VENV_PYTHON, "-m", "pip", "install",
        "torch==2.2.2+cu118",
        "--index-url", TORCH_INDEX_URL
    ], check=True)

    # Install other packages
    other_packages = [p for p in REQUIRED_PACKAGES if not p.startswith("torch==")]
    print("  Installing other dependencies...")
    subprocess.run([VENV_PYTHON, "-m", "pip", "install"] + other_packages, check=True)

    print("All packages installed.\n")


def ensure_venv():
    """Ensure venv exists and has required packages, then relaunch if needed."""
    if is_in_venv():
        # Already in venv, continue with normal execution
        return

    # Not in venv - need to set up and relaunch
    print("=" * 60)
    print("LLM Benchmark Tool - Environment Setup")
    print("=" * 60)

    venv_exists = os.path.exists(VENV_PYTHON)

    if not venv_exists:
        create_venv()
        install_packages()

    # Relaunch script inside the venv
    print(f"Launching in virtual environment...\n")
    result = subprocess.run([VENV_PYTHON] + sys.argv)
    sys.exit(result.returncode)


# Run venv check before anything else
ensure_venv()

# =============================================================================
# MAIN APPLICATION - Only runs inside venv
# =============================================================================

import time
import atexit
from dataclasses import dataclass
from typing import Optional

from lib.server import ModelServer
from lib.client import run_chat, run_compare
from lib.utils import check_endpoint, wait_for_server, print_header, print_separator, get_runpod_endpoint

# Default paths
MODELS_DIR = r"C:\ai-models\text-generation-webui\models"
DEFAULT_PORT = 5000


@dataclass
class AppState:
    """Centralized application state."""
    server: Optional[ModelServer] = None
    local_url: Optional[str] = None
    local_port: Optional[int] = None
    model_name: Optional[str] = None
    cloud_url: Optional[str] = None
    cloud_name: Optional[str] = None
    active_mode: Optional[str] = None  # "local", "remote", "compare", "compare_cloud"

    def has_local_server(self) -> bool:
        return self.server is not None and self.server.is_running()

    def stop_local(self):
        if self.server and self.server.is_running():
            print("\nStopping local server...")
            self.server.stop()
        self.server = None
        self.local_url = None
        self.local_port = None
        self.model_name = None

    def cleanup(self):
        self.stop_local()
        self.cloud_url = None
        self.cloud_name = None
        self.active_mode = None

    def status_line(self) -> str:
        parts = []
        if self.has_local_server():
            parts.append(f"Local: {self.model_name} @ :{self.local_port}")
        if self.cloud_url:
            parts.append(f"Cloud: {self.cloud_name}")
        return " | ".join(parts) if parts else "No active connections"


def get_available_models():
    """Scan models directory and return list of available models."""
    if not os.path.exists(MODELS_DIR):
        return []

    models = []
    for name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, name)
        # Check if it's a directory with model files (safetensors or config.json)
        if os.path.isdir(model_path):
            files = os.listdir(model_path)
            has_model = any(f.endswith('.safetensors') or f == 'config.json' for f in files)
            if has_model:
                models.append(name)
    return sorted(models)

# Global application state
_state = AppState()


def cleanup():
    """Cleanup on exit."""
    global _state
    _state.cleanup()


atexit.register(cleanup)


def get_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default."""
    if default:
        display = f"{prompt} [{default}]: "
    else:
        display = f"{prompt}: "

    sys.stdout.flush()
    value = input(display).strip()
    return value if value else default


def mode_local():
    """Mode 1: Start local model server and chat."""
    global _state

    print_separator()
    print("LOCAL MODEL SETUP")
    print_separator()

    # List available models
    models = get_available_models()
    if not models:
        print("No models found in:", MODELS_DIR)
        return

    print("\nAvailable models:")
    for i, name in enumerate(models, 1):
        print(f"  [{i}] {name}")

    choice = get_input(f"\nSelect model (1-{len(models)})", "1")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            model_name = models[idx]
            model_path = os.path.join(MODELS_DIR, model_name)
        else:
            print("Invalid choice")
            return
    except ValueError:
        print("Invalid choice")
        return

    # Check if GPTQ model (already quantized)
    is_gptq = "gptq" in model_name.lower() or "awq" in model_name.lower()

    if is_gptq:
        print(f"\n{model_name} is pre-quantized (GPTQ 4-bit)")
    else:
        print(f"\n{model_name} will load in float16")

    use_4bit = False  # Disabled - bitsandbytes runtime quantization not reliable

    # Get port
    port = int(get_input("Port", str(DEFAULT_PORT)))

    print_separator()

    # Create and load model
    _state.server = ModelServer()

    def log(msg):
        print(f"  {msg}", flush=True)

    print("Loading model (this may take a few minutes)...", flush=True)
    print("", flush=True)
    try:
        _state.server.load_model(model_path, use_4bit=use_4bit, callback=log)
    except Exception as e:
        print(f"\nError loading model: {e}", flush=True)
        _state.server = None
        return

    # Start server
    print("\nStarting server...", flush=True)
    url = _state.server.start(port=port)
    api_url = f"{url}/v1/chat/completions"

    # Wait for server to be ready
    print("Waiting for server...", end=" ", flush=True)
    if wait_for_server(api_url, max_wait=30):
        print("OK")
    else:
        print("FAILED")
        _state.stop_local()
        return

    # Store state
    _state.local_url = api_url
    _state.local_port = port
    _state.model_name = model_name
    _state.active_mode = "local"

    print_separator()
    print(f"Server ready at {url}")
    print_separator()

    # Ask about JSON view
    json_view = get_input("Show JSON requests/responses? (y/n)", "n")
    show_json = json_view.lower() == "y"

    # Start chat
    run_chat(api_url, show_json=show_json)


def mode_remote():
    """Mode 2: Connect to remote endpoint."""
    print_separator()
    print("REMOTE ENDPOINT")
    print_separator()

    # Try to auto-detect running RunPod pod
    url = None
    model = "local"  # Default for local servers
    runpod_url, runpod_model, runpod_err = get_runpod_endpoint()

    if runpod_url:
        print(f"\nFound running pod: {runpod_model}")
        print(f"URL: {runpod_url}")
        use_detected = get_input("Use this endpoint? (y/n)", "y")
        if use_detected.lower() == "y":
            url = runpod_url
            model = runpod_model  # Use actual model name for vLLM
    else:
        print(f"\nNo running RunPod detected: {runpod_err}")

    # Fall back to manual entry if no auto-detected endpoint
    if not url:
        url = get_input("API URL", "http://127.0.0.1:5000/v1/chat/completions")
        # For manual entry, ask for model name (local servers ignore this, vLLM needs it)
        model = get_input("Model name", "local")

    # Get API key from environment
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        print("\nUsing API key from VLLM_API_KEY environment variable")
    else:
        print("\nNo VLLM_API_KEY found (requests will be unauthenticated)")

    print("\nTesting connection...", end=" ", flush=True)
    ok, msg = check_endpoint(url, api_key=api_key)
    if ok:
        print("OK")
    else:
        print(f"FAILED ({msg})")
        retry = get_input("Continue anyway? (y/n)", "n")
        if retry.lower() != "y":
            return

    print_separator()

    # Ask about JSON view
    json_view = get_input("Show JSON requests/responses? (y/n)", "n")
    show_json = json_view.lower() == "y"

    run_chat(url, show_json=show_json, api_key=api_key, model=model)


def mode_compare():
    """Mode 3: Compare local model vs cloud endpoint."""
    global _state

    print_separator()
    print("COMPARISON MODE: Local vs Cloud")
    print_separator()
    print("\nThis mode loads ONE local model and compares it against a cloud endpoint.")

    # First, set up the local model
    print("\n--- LOCAL MODEL (Endpoint A) ---")

    # List available models
    models = get_available_models()
    if not models:
        print("No models found in:", MODELS_DIR)
        return

    print("\nAvailable models:")
    for i, name in enumerate(models, 1):
        print(f"  [{i}] {name}")

    choice = get_input(f"\nSelect model (1-{len(models)})", "1")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            model_name = models[idx]
            model_path = os.path.join(MODELS_DIR, model_name)
        else:
            print("Invalid choice")
            return
    except ValueError:
        print("Invalid choice")
        return

    # Check if GPTQ model
    is_gptq = "gptq" in model_name.lower() or "awq" in model_name.lower()
    if is_gptq:
        print(f"\n{model_name} is pre-quantized (GPTQ 4-bit)")
    else:
        print(f"\n{model_name} will load in float16")

    use_4bit = False  # Disabled - bitsandbytes not reliable

    port = int(get_input("Local port", str(DEFAULT_PORT)))

    # Load and start local model
    print_separator()
    _state.server = ModelServer()

    def log(msg):
        print(f"  {msg}")

    print("Loading local model...")
    try:
        _state.server.load_model(model_path, use_4bit=use_4bit, callback=log)
    except Exception as e:
        print(f"\nError loading model: {e}")
        _state.server = None
        return

    print("\nStarting local server...")
    local_url = _state.server.start(port=port)
    local_api = f"{local_url}/v1/chat/completions"

    print("Waiting for server...", end=" ", flush=True)
    if wait_for_server(local_api, max_wait=30):
        print("OK")
    else:
        print("FAILED")
        _state.stop_local()
        return

    # Store local state
    _state.local_url = local_api
    _state.local_port = port
    _state.model_name = model_name

    local_name = _state.server.model_name + " (local)"

    # Now get the cloud endpoint
    print("\n--- CLOUD ENDPOINT (Endpoint B) ---")

    # Try to auto-detect running RunPod pod
    cloud_url = None
    cloud_name = None
    cloud_model = "local"  # Default model name for API requests
    runpod_url, runpod_model, runpod_err = get_runpod_endpoint()

    if runpod_url:
        print(f"\nFound running pod: {runpod_model}")
        print(f"URL: {runpod_url}")
        use_detected = get_input("Use this endpoint? (y/n)", "y")
        if use_detected.lower() == "y":
            cloud_url = runpod_url
            cloud_name = runpod_model + " (cloud)"
            cloud_model = runpod_model  # Use actual model name for vLLM
    else:
        print(f"\nNo running RunPod detected: {runpod_err}")

    # Fall back to manual entry if no auto-detected endpoint
    if not cloud_url:
        cloud_url = get_input("Cloud API URL", "https://your-server.com/v1/chat/completions")
        cloud_name = get_input("Cloud name", "Cloud")
        cloud_model = get_input("Model name", "local")

    # Get API key from environment
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        print("\nUsing API key from VLLM_API_KEY environment variable")
    else:
        print("\nNo VLLM_API_KEY found (requests will be unauthenticated)")

    print(f"\nTesting cloud connection...", end=" ", flush=True)
    ok, msg = check_endpoint(cloud_url, api_key=api_key)
    if ok:
        print("OK")
    else:
        print(f"FAILED ({msg})")
        retry = get_input("Continue anyway? (y/n)", "n")
        if retry.lower() != "y":
            _state.stop_local()  # Cleanup local server before returning
            return

    # Store cloud state
    _state.cloud_url = cloud_url
    _state.cloud_name = cloud_name
    _state.active_mode = "compare"

    print_separator()
    print(f"Ready to compare:")
    print(f"  A: {local_name} @ {local_api}")
    print(f"  B: {cloud_name} @ {cloud_url}")
    if api_key:
        print(f"  Auth: Bearer token for cloud endpoint")
    print_separator()

    # Ask about JSON view
    json_view = get_input("Show JSON requests/responses? (y/n)", "n")
    show_json = json_view.lower() == "y"

    run_compare(local_api, cloud_url, local_name, cloud_name, show_json=show_json,
                api_key_b=api_key, model_a="local", model_b=cloud_model)


def main_menu():
    """Display main menu based on current state."""
    print_header("LLM Benchmark Tool")

    # Show current status
    print(f"\nStatus: {_state.status_line()}")
    print()

    print("  [1] Local Model - Load and serve a local model")
    print("  [2] Remote Endpoint - Connect to existing API")
    print("  [3] Compare - Local model vs cloud endpoint")
    print("  [4] Compare Cloud - Compare two remote endpoints")

    if _state.has_local_server():
        print(f"  [S] Stop Server ({_state.model_name})")

    print("  [Q] Quit")
    print()

    return get_input("Choice", "1").upper()


def mode_compare_cloud():
    """Mode 4: Compare two cloud endpoints."""
    global _state

    print_separator()
    print("COMPARISON MODE: Cloud vs Cloud")
    print_separator()
    print("\nThis mode compares two remote endpoints (no local model needed).")

    # Get first cloud endpoint
    print("\n--- CLOUD ENDPOINT A ---")

    url_a = None
    model_a = "local"
    name_a = None
    api_key_a = None

    # Try auto-detect for first endpoint
    runpod_url, runpod_model, runpod_err = get_runpod_endpoint()
    if runpod_url:
        print(f"\nFound running pod: {runpod_model}")
        print(f"URL: {runpod_url}")
        use_detected = get_input("Use this for Endpoint A? (y/n)", "y")
        if use_detected.lower() == "y":
            url_a = runpod_url
            model_a = runpod_model
            name_a = runpod_model + " (A)"
    else:
        print(f"\nNo running RunPod detected: {runpod_err}")

    if not url_a:
        url_a = get_input("Endpoint A URL", "https://your-server.com/v1/chat/completions")
        model_a = get_input("Model name A", "local")
        name_a = get_input("Display name A", "Cloud-A")

    # Get API key for endpoint A
    api_key_a = os.environ.get("VLLM_API_KEY")
    if api_key_a:
        print("Using VLLM_API_KEY for Endpoint A")

    # Get second cloud endpoint
    print("\n--- CLOUD ENDPOINT B ---")

    url_b = get_input("Endpoint B URL", "https://other-server.com/v1/chat/completions")
    model_b = get_input("Model name B", "local")
    name_b = get_input("Display name B", "Cloud-B")

    # Ask about API key for B (could be different)
    use_same_key = get_input("Use same API key for Endpoint B? (y/n)", "y")
    if use_same_key.lower() == "y":
        api_key_b = api_key_a
    else:
        api_key_b = get_input("API key for Endpoint B (or blank)", "")
        if not api_key_b:
            api_key_b = None

    # Test connections
    print(f"\nTesting Endpoint A...", end=" ", flush=True)
    ok, msg = check_endpoint(url_a, api_key=api_key_a)
    if ok:
        print("OK")
    else:
        print(f"FAILED ({msg})")
        retry = get_input("Continue anyway? (y/n)", "n")
        if retry.lower() != "y":
            return

    print(f"Testing Endpoint B...", end=" ", flush=True)
    ok, msg = check_endpoint(url_b, api_key=api_key_b)
    if ok:
        print("OK")
    else:
        print(f"FAILED ({msg})")
        retry = get_input("Continue anyway? (y/n)", "n")
        if retry.lower() != "y":
            return

    # Store state
    _state.cloud_url = url_a  # Store first as primary cloud
    _state.cloud_name = name_a
    _state.active_mode = "compare_cloud"

    print_separator()
    print(f"Ready to compare:")
    print(f"  A: {name_a} @ {url_a}")
    print(f"  B: {name_b} @ {url_b}")
    print_separator()

    # Ask about JSON view
    json_view = get_input("Show JSON requests/responses? (y/n)", "n")
    show_json = json_view.lower() == "y"

    run_compare(url_a, url_b, name_a, name_b, show_json=show_json,
                api_key_a=api_key_a, api_key_b=api_key_b, model_a=model_a, model_b=model_b)


def main():
    """Main entry point."""
    global _state

    try:
        while True:
            choice = main_menu()

            if choice == "1":
                mode_local()
            elif choice == "2":
                mode_remote()
            elif choice == "3":
                mode_compare()
            elif choice == "4":
                mode_compare_cloud()
            elif choice == "S":
                if _state.has_local_server():
                    _state.stop_local()
                    print("Server stopped.")
                else:
                    print("No server running.")
                continue  # Don't ask to return to menu
            elif choice == "Q":
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice\n")
                continue

            # After exiting a mode, ask if user wants to continue
            print()
            again = get_input("Return to menu? (y/n)", "y")
            if again.lower() != "y":
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
