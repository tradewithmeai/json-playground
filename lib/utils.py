"""
Utility functions for LLM Benchmark Tool.
"""

import requests
import time


def check_endpoint(url: str, timeout: float = 5.0) -> tuple:
    """
    Check if an endpoint is reachable.
    Returns (success: bool, message: str)
    """
    # Convert chat completions URL to base URL for health check
    base_url = url.replace("/v1/chat/completions", "")

    try:
        response = requests.get(base_url, timeout=timeout)
        if response.status_code == 200:
            return True, "OK"
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
    base_url = url.replace("/v1/chat/completions", "")
    start = time.time()

    while time.time() - start < max_wait:
        try:
            response = requests.get(base_url, timeout=2)
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
