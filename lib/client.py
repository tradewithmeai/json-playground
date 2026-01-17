"""
Chat client logic for LLM Benchmark Tool.
"""

import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor

MAX_HISTORY = 20  # Keep last 20 messages


def send_message(url: str, user_input: str, history: list, show_json: bool = False) -> tuple:
    """
    Send message to API and return response with timing.
    Returns (reply, elapsed_time, error, request_json, response_json)
    """
    messages = history + [{"role": "user", "content": user_input}]

    payload = {
        "model": "local",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7
    }

    start = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=300)
        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            # Get error details from response
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", str(error_data))
            except:
                error_msg = response.text[:500] if response.text else f"HTTP {response.status_code}"
            return None, elapsed, f"Server error: {error_msg}", payload, None

        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        return reply, elapsed, None, payload, data
    except requests.exceptions.ConnectionError:
        return None, 0, "Connection refused", payload, None
    except requests.exceptions.Timeout:
        return None, 0, "Timeout", payload, None
    except Exception as e:
        return None, 0, str(e), payload, None


def trim_history(history: list) -> list:
    """Keep only the last MAX_HISTORY messages."""
    if len(history) > MAX_HISTORY:
        return history[-MAX_HISTORY:]
    return history


def print_json(label: str, data: dict):
    """Pretty print JSON with label."""
    print(f"\n{'─'*20} {label} {'─'*20}")
    print(json.dumps(data, indent=2))
    print(f"{'─'*50}")


def run_chat(url: str, name: str = "Assistant", show_json: bool = False):
    """
    Run interactive chat session.
    Returns when user types 'quit'.
    """
    print(f"\nChat ready. Connected to: {url}")
    print(f"Memory: last {MAX_HISTORY} messages")
    if show_json:
        print("JSON view: ON (showing request/response)")
    print("Commands: 'quit' to exit, 'new' to clear history, 'json' to toggle JSON view\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if user_input.lower() == 'new':
            history = []
            print("[Conversation cleared]\n")
            continue

        if user_input.lower() == 'json':
            show_json = not show_json
            print(f"[JSON view: {'ON' if show_json else 'OFF'}]\n")
            continue

        reply, elapsed, error, req_json, res_json = send_message(url, user_input, history, show_json)

        if show_json and req_json:
            print_json("REQUEST", req_json)

        if error:
            print(f"Error: {error}\n")
            continue

        if show_json and res_json:
            print_json("RESPONSE", res_json)

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        history = trim_history(history)

        print(f"\n{name}: {reply}")
        print(f"[{elapsed:.2f}s | {len(history)} msgs]\n")


def run_compare(url1: str, url2: str, name1: str = "Model-A", name2: str = "Model-B", show_json: bool = False):
    """
    Run comparison chat session with two models.
    """
    print(f"\nComparison Mode")
    print(f"  {name1}: {url1}")
    print(f"  {name2}: {url2}")
    print(f"Memory: last {MAX_HISTORY} messages (shared)")
    if show_json:
        print("JSON view: ON")
    print("Commands: 'quit' to exit, 'new' to clear history, 'json' to toggle JSON view\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if user_input.lower() == 'new':
            history = []
            print("[Conversation cleared]\n")
            continue

        if user_input.lower() == 'json':
            show_json = not show_json
            print(f"[JSON view: {'ON' if show_json else 'OFF'}]\n")
            continue

        # Run both requests in parallel
        results = {}

        def query_model(name, url):
            results[name] = send_message(url, user_input, history, show_json)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(query_model, name1, url1)
            executor.submit(query_model, name2, url2)

        reply1, time1, err1, req1, res1 = results.get(name1, (None, 0, "No response", None, None))
        reply2, time2, err2, req2, res2 = results.get(name2, (None, 0, "No response", None, None))

        # Show JSON if enabled (just show one request since they're the same)
        if show_json and req1:
            print_json("REQUEST (sent to both)", req1)

        # Display comparison
        print(f"\n{'─'*60}")

        print(f"[{name1}] ({time1:.2f}s)")
        if err1:
            print(f"  Error: {err1}")
        else:
            if show_json and res1:
                print_json(f"{name1} RESPONSE", res1)
            print(f"  {reply1}")

        print(f"\n{'─'*30}")

        print(f"[{name2}] ({time2:.2f}s)")
        if err2:
            print(f"  Error: {err2}")
        else:
            if show_json and res2:
                print_json(f"{name2} RESPONSE", res2)
            print(f"  {reply2}")

        print(f"\n{'─'*60}")
        if not err1 and not err2:
            diff = abs(time1 - time2)
            faster = name1 if time1 < time2 else name2
            print(f"Winner: {faster} (faster by {diff:.2f}s)")
        print(f"{'─'*60}\n")

        # Update history with first successful response
        history.append({"role": "user", "content": user_input})
        if reply1:
            history.append({"role": "assistant", "content": reply1})
        elif reply2:
            history.append({"role": "assistant", "content": reply2})
        history = trim_history(history)
