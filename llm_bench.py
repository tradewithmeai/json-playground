"""
LLM Benchmark Tool - Unified entry point for model testing and benchmarking.
"""

import sys
import time
import atexit

from lib.server import ModelServer
from lib.client import run_chat, run_compare
from lib.utils import check_endpoint, wait_for_server, print_header, print_separator

# Default paths
DEFAULT_MODEL_PATH = r"C:\ai-models\text-generation-webui\models\Qwen2.5-7B-Instruct"
DEFAULT_PORT = 5000

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

    # Get model path
    model_path = get_input("Model path", DEFAULT_MODEL_PATH)

    # Get quantization
    print("\nQuantization:")
    print("  [1] 4-bit (recommended for <16GB VRAM)")
    print("  [2] Full precision (float16)")
    quant = get_input("Choice", "1")
    use_4bit = quant != "2"

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
    model_path = get_input("Model path", DEFAULT_MODEL_PATH)

    print("\nQuantization:")
    print("  [1] 4-bit (recommended)")
    print("  [2] Full precision")
    quant = get_input("Choice", "1")
    use_4bit = quant != "2"

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
    cloud_url = get_input("Cloud API URL", "https://your-server.com/v1/chat/completions")
    cloud_name = get_input("Cloud name", "Cloud")

    print(f"\nTesting cloud connection...", end=" ", flush=True)
    ok, msg = check_endpoint(cloud_url)
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
    print_separator()

    # Ask about JSON view
    json_view = get_input("Show JSON requests/responses? (y/n)", "n")
    show_json = json_view.lower() == "y"

    run_compare(local_api, cloud_url, local_name, cloud_name, show_json=show_json)


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
