"""
Utility functions for LLM Benchmark Tool.
"""

import json
import os
import requests
import time


def get_runpod_endpoint() -> tuple:
    """
    Read RunPod pod state from ~/.myai/pod.json and return endpoint info.

    Returns:
        (url, model_name, error) - URL and model name if found, or error message
    """
    state_file = os.path.expanduser("~/.myai/pod.json")

    if not os.path.exists(state_file):
        return None, None, "No pod state file found (~/.myai/pod.json)"

    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
    except json.JSONDecodeError as e:
        return None, None, f"Invalid JSON in pod state file: {e}"
    except Exception as e:
        return None, None, f"Error reading pod state: {e}"

    pod_id = state.get("pod_id")
    model = state.get("model", "Unknown Model")

    if not pod_id:
        return None, None, "No pod_id in state file (pod may not be running)"

    # vLLM uses port 8000
    url = f"https://{pod_id}-8000.proxy.runpod.net/v1/chat/completions"
    return url, model, None


def check_endpoint(url: str, timeout: float = 5.0, api_key: str = None) -> tuple:
    """
    Check if an endpoint is reachable.

    Args:
        url: API endpoint URL
        timeout: Request timeout in seconds
        api_key: Optional API key for Bearer token auth

    Returns (success: bool, message: str)
    """
    # Use /v1/models endpoint for health check (works with vLLM, OpenAI-compatible servers)
    health_url = url.replace("/v1/chat/completions", "/v1/models")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(health_url, timeout=timeout, headers=headers)
        if response.status_code == 200:
            return True, "OK"
        elif response.status_code == 401:
            return False, "Unauthorized (check API key)"
        elif response.status_code == 403:
            return False, "Forbidden (invalid API key)"
        else:
            return False, f"Status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def wait_for_server(url: str, max_wait: int = 120, interval: float = 2.0) -> bool:
    """
    Wait for a server to become ready.
    Returns True if server is ready, False if timeout.
    """
    health_url = url.replace("/v1/chat/completions", "/v1/models")
    start = time.time()

    while time.time() - start < max_wait:
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(interval)

    return False


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def clear_screen():
    """Clear terminal screen (cross-platform)."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str, width: int = 50):
    """Print a formatted header."""
    print("=" * width)
    print(title.center(width))
    print("=" * width)


def print_separator(char: str = "-", width: int = 50):
    """Print a separator line."""
    print(char * width)
