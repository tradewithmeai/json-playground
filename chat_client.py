"""
Simple terminal chat client for benchmarking the local LLM model server.
Supports single model or comparison mode (two models side by side).
"""

import argparse
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor

MAX_HISTORY = 20  # Keep last 20 messages (10 exchanges)


def chat_single(url: str, user_input: str, history: list) -> tuple:
    """Send message to a single API and return response with timing."""
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
        response.raise_for_status()
        elapsed = time.perf_counter() - start

        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        return reply, elapsed, None
    except requests.exceptions.ConnectionError:
        return None, 0, "Connection refused"
    except requests.exceptions.Timeout:
        return None, 0, "Timeout"
    except Exception as e:
        return None, 0, str(e)


def trim_history(history: list) -> list:
    """Keep only the last MAX_HISTORY messages."""
    if len(history) > MAX_HISTORY:
        return history[-MAX_HISTORY:]
    return history


def run_single_mode(url: str):
    """Run in single model mode."""
    print(f"\nSimple Chat Client")
    print(f"Connected to: {url}")
    print(f"Memory: last {MAX_HISTORY} messages")
    print("Commands: 'quit' to exit, 'new' for fresh conversation\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'new':
            history = []
            print("[Conversation cleared]\n")
            continue

        reply, elapsed, error = chat_single(url, user_input, history)

        if error:
            print(f"Error: {error}\n")
            continue

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        history = trim_history(history)

        print(f"\nAssistant: {reply}")
        print(f"[{elapsed:.2f}s | {len(history)} msgs in memory]\n")


def run_compare_mode(url1: str, url2: str, name1: str, name2: str):
    """Run in comparison mode with two models."""
    print(f"\n{'='*60}")
    print(f"COMPARISON MODE")
    print(f"{'='*60}")
    print(f"Model A ({name1}): {url1}")
    print(f"Model B ({name2}): {url2}")
    print(f"Memory: last {MAX_HISTORY} messages (shared)")
    print("Commands: 'quit' to exit, 'new' for fresh conversation")
    print(f"{'='*60}\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'new':
            history = []
            print("[Conversation cleared]\n")
            continue

        # Run both requests in parallel
        results = {}

        def query_model(name, url):
            results[name] = chat_single(url, user_input, history)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(query_model, name1, url1)
            executor.submit(query_model, name2, url2)

        reply1, time1, err1 = results.get(name1, (None, 0, "No response"))
        reply2, time2, err2 = results.get(name2, (None, 0, "No response"))

        # Display comparison
        print(f"\n{'─'*60}")

        # Model A
        print(f"[{name1}] ({time1:.2f}s)")
        if err1:
            print(f"  Error: {err1}")
        else:
            print(f"  {reply1}")

        print(f"\n{'─'*30}")

        # Model B
        print(f"[{name2}] ({time2:.2f}s)")
        if err2:
            print(f"  Error: {err2}")
        else:
            print(f"  {reply2}")

        # Summary
        print(f"\n{'─'*60}")
        if not err1 and not err2:
            diff = abs(time1 - time2)
            faster = name1 if time1 < time2 else name2
            print(f"Winner: {faster} (faster by {diff:.2f}s)")
        print(f"{'─'*60}\n")

        # Update history (use first successful response)
        history.append({"role": "user", "content": user_input})
        if reply1:
            history.append({"role": "assistant", "content": reply1})
        elif reply2:
            history.append({"role": "assistant", "content": reply2})
        history = trim_history(history)


def main():
    parser = argparse.ArgumentParser(description="Chat client for LLM benchmarking")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:5000/v1/chat/completions",
        help="API URL for single mode"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable comparison mode (two models)"
    )
    parser.add_argument(
        "--url1",
        type=str,
        default="http://127.0.0.1:5000/v1/chat/completions",
        help="First model URL (comparison mode)"
    )
    parser.add_argument(
        "--url2",
        type=str,
        default="http://127.0.0.1:5001/v1/chat/completions",
        help="Second model URL (comparison mode)"
    )
    parser.add_argument(
        "--name1",
        type=str,
        default="Model-A",
        help="Name for first model"
    )
    parser.add_argument(
        "--name2",
        type=str,
        default="Model-B",
        help="Name for second model"
    )

    args = parser.parse_args()

    if args.compare:
        run_compare_mode(args.url1, args.url2, args.name1, args.name2)
    else:
        run_single_mode(args.url)


if __name__ == "__main__":
    main()
