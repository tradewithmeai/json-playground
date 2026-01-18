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

from lib.server import ModelServer
from lib.client import run_chat, run_compare
from lib.utils import check_endpoint, wait_for_server, print_header, print_separator, get_runpod_endpoint

# Default paths
MODELS_DIR = r"C:\ai-models\text-generation-webui\models"
DEFAULT_PORT = 5000


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

# Global server instance for cleanup
_server = None


def cleanup():
    """Cleanup on exit."""
    global _server
    if _server and _server.is_running():
        print("\nShutting down server...")
        _server.stop()


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
    global _server

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
    _server = ModelServer()

    def log(msg):
        print(f"  {msg}", flush=True)

    print("Loading model (this may take a few minutes)...", flush=True)
    print("", flush=True)
    try:
        _server.load_model(model_path, use_4bit=use_4bit, callback=log)
    except Exception as e:
        print(f"\nError loading model: {e}", flush=True)
        return

    # Start server
    print("\nStarting server...", flush=True)
    url = _server.start(port=port)
    api_url = f"{url}/v1/chat/completions"

    # Wait for server to be ready
    print("Waiting for server...", end=" ", flush=True)
    if wait_for_server(api_url, max_wait=30):
        print("OK")
    else:
        print("FAILED")
        return

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

    url = get_input("API URL", "http://127.0.0.1:5000/v1/chat/completions")

    print("\nTesting connection...", end=" ", flush=True)
    ok, msg = check_endpoint(url)
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

    run_chat(url, show_json=show_json)


def mode_compare():
    """Mode 3: Compare local model vs cloud endpoint."""
    global _server

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
    _server = ModelServer()

    def log(msg):
        print(f"  {msg}")

    print("Loading local model...")
    try:
        _server.load_model(model_path, use_4bit=use_4bit, callback=log)
    except Exception as e:
        print(f"\nError loading model: {e}")
        return

    print("\nStarting local server...")
    local_url = _server.start(port=port)
    local_api = f"{local_url}/v1/chat/completions"

    print("Waiting for server...", end=" ", flush=True)
    if wait_for_server(local_api, max_wait=30):
        print("OK")
    else:
        print("FAILED")
        return

    local_name = _server.model_name + " (local)"

    # Now get the cloud endpoint
    print("\n--- CLOUD ENDPOINT (Endpoint B) ---")

    # Try to auto-detect running RunPod pod
    cloud_url = None
    cloud_name = None
    runpod_url, runpod_model, runpod_err = get_runpod_endpoint()

    if runpod_url:
        print(f"\nFound running pod: {runpod_model}")
        print(f"URL: {runpod_url}")
        use_detected = get_input("Use this endpoint? (y/n)", "y")
        if use_detected.lower() == "y":
            cloud_url = runpod_url
            cloud_name = runpod_model + " (cloud)"
    else:
        print(f"\nNo running RunPod detected: {runpod_err}")

    # Fall back to manual entry if no auto-detected endpoint
    if not cloud_url:
        cloud_url = get_input("Cloud API URL", "https://your-server.com/v1/chat/completions")
        cloud_name = get_input("Cloud name", "Cloud")

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
            return

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

    run_compare(local_api, cloud_url, local_name, cloud_name, show_json=show_json, api_key_b=api_key)


def main_menu():
    """Display main menu and get choice."""
    print_header("LLM Benchmark Tool")
    print()
    print("  [1] Local Model - Load and serve a local model")
    print("  [2] Remote Endpoint - Connect to existing API")
    print("  [3] Compare - Local model vs cloud endpoint")
    print("  [4] Exit")
    print()

    choice = get_input("Choice", "1")
    return choice


def main():
    """Main entry point."""
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
