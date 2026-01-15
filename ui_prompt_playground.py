# ui_prompt_playground.py
# Gradio UI for local OpenAI-compatible endpoint with chat history and JSON inspection

import json
import os
import requests
import subprocess
import threading
from pathlib import Path
import gradio as gr

# Endpoint configurations
ENDPOINTS = {
    "Local": {
        "url": "http://127.0.0.1:5000/v1/chat/completions",
        "model": "Qwen2.5-7B-Instruct",
        "api_key": None,
    },
    "RunPod": {
        "url": None,  # Loaded dynamically from ~/.myai/pod.json
        "model": None,
        "api_key": os.getenv("VLLM_API_KEY"),
    },
    "Custom": {
        "url": None,  # User-provided
        "model": None,
        "api_key": None,
    },
}


def get_runpod_config():
    """Load RunPod endpoint from ~/.myai/pod.json if available."""
    state_file = Path.home() / ".myai" / "pod.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
                pod_id = state.get("pod_id")
                model = state.get("model", "mistralai/Mistral-7B-Instruct-v0.1")
                if pod_id:
                    return {
                        "url": f"https://{pod_id}-8000.proxy.runpod.net/v1/chat/completions",
                        "model": model,
                    }
        except (json.JSONDecodeError, KeyError):
            pass
    return {"url": None, "model": None}


def get_endpoint_info(endpoint_choice, custom_url, custom_api_key, custom_model):
    """Get the URL, model, and API key for the selected endpoint."""
    if endpoint_choice == "Local":
        return (
            ENDPOINTS["Local"]["url"],
            ENDPOINTS["Local"]["model"],
            None
        )
    elif endpoint_choice == "RunPod":
        runpod = get_runpod_config()
        return (
            runpod["url"],
            runpod["model"],
            ENDPOINTS["RunPod"]["api_key"]
        )
    elif endpoint_choice == "Custom":
        return (
            custom_url.strip() if custom_url else None,
            custom_model.strip() if custom_model else "gpt-3.5-turbo",
            custom_api_key.strip() if custom_api_key else None
        )
    return (None, None, None)

# Custom CSS for professional styling
CUSTOM_CSS = """
.chat-panel {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
}
.nav-btn {
    min-width: 80px !important;
}
.selected-indicator {
    background-color: #e3f2fd !important;
}
.header-text {
    color: #1976d2;
    margin-bottom: 0.5rem;
}
.clear-btn {
    background-color: #ffebee !important;
    color: #c62828 !important;
}
.send-btn {
    background-color: #1976d2 !important;
    color: white !important;
}
"""


def format_chat_display(history):
    """Convert chat_history to Gradio Chatbot format."""
    if not history:
        return []

    chatbot_messages = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            # System messages shown as a note
            chatbot_messages.append((None, f"[System] {content}"))
        elif role == "user":
            chatbot_messages.append((content, None))
        elif role == "assistant":
            chatbot_messages.append((None, content))

    return chatbot_messages


def build_payload(history, focus_idx, temperature, max_tokens, top_p, model="unknown"):
    """Build JSON payload from history up to focus_index."""
    if not history or focus_idx < 0:
        return {"model": model, "messages": [], "temperature": float(temperature),
                "top_p": float(top_p), "max_tokens": int(max_tokens)}

    messages = history[:focus_idx + 1]
    return {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }


def get_nav_choices(history):
    """Generate dropdown choices for message navigation."""
    if not history:
        return ["(empty)"]

    choices = []
    for i, msg in enumerate(history):
        role = msg["role"]
        preview = msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
        preview = preview.replace("\n", " ")
        choices.append(f"[{i}] {role}: {preview}")
    return choices


def send_message(history, focus_idx, system_text, user_text, temperature, max_tokens, top_p,
                 endpoint_choice, custom_url, custom_api_key, custom_model):
    """Send message to API and update chat history."""
    # Get endpoint configuration
    api_url, model, api_key = get_endpoint_info(endpoint_choice, custom_url, custom_api_key, custom_model)

    if not api_url:
        error_msg = f"No endpoint configured for {endpoint_choice}"
        if endpoint_choice == "RunPod":
            error_msg += ". Make sure you have an active pod (check ~/.myai/pod.json)"
        return (history, focus_idx, format_chat_display(history), get_nav_choices(history),
                "", json.dumps({"error": error_msg}, indent=2), "", system_text)

    if not user_text.strip():
        # No message to send
        payload = build_payload(history, focus_idx, temperature, max_tokens, top_p, model)
        return (history, focus_idx, format_chat_display(history), get_nav_choices(history),
                json.dumps(payload, indent=2), "", "", system_text)

    # Branch: truncate history to focus point if we're not at the end
    if history and focus_idx >= 0 and focus_idx < len(history) - 1:
        history = history[:focus_idx + 1]

    # Handle system message
    if history is None:
        history = []

    system_text_stripped = system_text.strip()

    # Update or add system message
    if system_text_stripped:
        if history and history[0]["role"] == "system":
            history[0]["content"] = system_text_stripped
        else:
            history.insert(0, {"role": "system", "content": system_text_stripped})
    else:
        # Remove system message if it exists and text is empty
        if history and history[0]["role"] == "system":
            history.pop(0)

    # Append user message
    history.append({"role": "user", "content": user_text.strip()})

    # Build payload with full history
    payload = build_payload(history, len(history) - 1, temperature, max_tokens, top_p, model)

    # Build headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        r = requests.post(api_url, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"]

        # Append assistant response
        history.append({"role": "assistant", "content": reply})
        new_focus = len(history) - 1

        # Get current system text from history
        current_system = ""
        if history and history[0]["role"] == "system":
            current_system = history[0]["content"]

        return (history, new_focus, format_chat_display(history), get_nav_choices(history),
                json.dumps(payload, indent=2), json.dumps(data, indent=2), reply, current_system)

    except Exception as e:
        # Still append user message but show error
        new_focus = len(history) - 1

        current_system = ""
        if history and history[0]["role"] == "system":
            current_system = history[0]["content"]

        return (history, new_focus, format_chat_display(history), get_nav_choices(history),
                json.dumps(payload, indent=2), json.dumps({"error": str(e)}, indent=2), "", current_system)


def navigate_prev(history, focus_idx, temperature, max_tokens, top_p):
    """Navigate to previous message."""
    if not history or focus_idx <= 0:
        focus_idx = 0 if history else -1
    else:
        focus_idx -= 1

    return load_message_at_index(history, focus_idx, temperature, max_tokens, top_p)


def navigate_next(history, focus_idx, temperature, max_tokens, top_p):
    """Navigate to next message."""
    if not history:
        focus_idx = -1
    elif focus_idx >= len(history) - 1:
        focus_idx = len(history) - 1
    else:
        focus_idx += 1

    return load_message_at_index(history, focus_idx, temperature, max_tokens, top_p)


def navigate_to_selection(history, selection, temperature, max_tokens, top_p):
    """Navigate to selected message from dropdown."""
    if not history or not selection or selection == "(empty)":
        return load_message_at_index(history, -1, temperature, max_tokens, top_p)

    # Extract index from selection string "[0] role: preview"
    try:
        idx = int(selection.split("]")[0].replace("[", ""))
        return load_message_at_index(history, idx, temperature, max_tokens, top_p)
    except (ValueError, IndexError):
        return load_message_at_index(history, len(history) - 1 if history else -1, temperature, max_tokens, top_p)


def load_message_at_index(history, focus_idx, temperature, max_tokens, top_p):
    """Load message content into editor fields."""
    if not history or focus_idx < 0 or focus_idx >= len(history):
        payload = build_payload(history, focus_idx, temperature, max_tokens, top_p)
        return (focus_idx, "", "user", json.dumps(payload, indent=2),
                f"Position: - / {len(history) if history else 0}")

    msg = history[focus_idx]
    role = msg["role"]
    content = msg["content"]

    payload = build_payload(history, focus_idx, temperature, max_tokens, top_p)
    position_text = f"Position: {focus_idx + 1} / {len(history)}"

    # For system messages, load into system field; for others, load into user field
    if role == "system":
        return (focus_idx, content, "user", json.dumps(payload, indent=2), position_text)
    else:
        return (focus_idx, content, role, json.dumps(payload, indent=2), position_text)


def clear_history():
    """Clear all chat history and reset state."""
    return ([], -1, [], ["(empty)"], "", "", "", "", "user", "Position: - / 0")


def go_to_end(history, temperature, max_tokens, top_p):
    """Jump to the end of chat history."""
    if not history:
        return load_message_at_index([], -1, temperature, max_tokens, top_p)
    return load_message_at_index(history, len(history) - 1, temperature, max_tokens, top_p)


# Build UI
with gr.Blocks(title="Local OpenAI JSON Playground") as demo:
    # State
    chat_history = gr.State([])
    focus_index = gr.State(-1)

    # Header
    gr.Markdown("## OpenAI JSON Playground", elem_classes=["header-text"])

    with gr.Tabs():
        # Tab 1: JSON Editor (default)
        with gr.Tab("JSON Editor"):
            # Endpoint configuration row
            with gr.Row():
                endpoint_choice = gr.Dropdown(
                    ["Local", "RunPod", "Custom"],
                    value="Local",
                    label="Endpoint",
                    scale=1
                )
                custom_url = gr.Textbox(
                    label="Custom URL",
                    placeholder="https://api.example.com/v1/chat/completions",
                    visible=False,
                    scale=2
                )
                custom_model = gr.Textbox(
                    label="Model Name",
                    placeholder="gpt-3.5-turbo",
                    visible=False,
                    scale=1
                )
                custom_api_key = gr.Textbox(
                    label="API Key",
                    placeholder="sk-...",
                    type="password",
                    visible=False,
                    scale=1
                )

            with gr.Row():
                with gr.Column(scale=1):
                    # Controls
                    role = gr.Dropdown(
                        ["user", "assistant"],
                        value="user",
                        label="Role"
                    )
                    temperature = gr.Dropdown(
                        [0.0, 0.2, 0.4, 0.7, 1.0],
                        value=0.7,
                        label="Temperature"
                    )
                    top_p = gr.Dropdown(
                        [0.7, 0.9, 1.0],
                        value=1.0,
                        label="Top P"
                    )
                    max_tokens = gr.Dropdown(
                        [64, 128, 256, 512, 1024],
                        value=128,
                        label="Max Tokens"
                    )

                with gr.Column(scale=2):
                    # Input fields
                    system_text = gr.Textbox(
                        lines=2,
                        label="System Prompt",
                        placeholder="Optional system prompt (persists across messages)..."
                    )
                    user_text = gr.Textbox(
                        lines=4,
                        label="Message Content",
                        placeholder="Type your message here..."
                    )
                    send_btn = gr.Button("Send", elem_classes=["send-btn"])

            gr.Markdown("---")

            # JSON views
            with gr.Row():
                with gr.Column():
                    payload_view = gr.Code(
                        label="Request JSON",
                        language="json",
                        lines=12
                    )
                with gr.Column():
                    response_view = gr.Code(
                        label="Response JSON",
                        language="json",
                        lines=12
                    )

            assistant_text = gr.Textbox(
                lines=3,
                label="Last Assistant Response",
                interactive=False
            )

        # Tab 2: Chat View
        with gr.Tab("Chat View"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=450,
                        elem_classes=["chat-panel"]
                    )
                    with gr.Row():
                        clear_btn = gr.Button("Clear History", elem_classes=["clear-btn"], size="sm")
                        end_btn = gr.Button("Go to End", size="sm")

                with gr.Column(scale=1):
                    gr.Markdown("### Message Inspector")
                    with gr.Row():
                        prev_btn = gr.Button("< Prev", elem_classes=["nav-btn"], size="sm")
                        next_btn = gr.Button("Next >", elem_classes=["nav-btn"], size="sm")

                    nav_dropdown = gr.Dropdown(
                        choices=["(empty)"],
                        value="(empty)",
                        label="Select Message"
                    )
                    position_display = gr.Markdown("Position: - / 0")

                    gr.Markdown("---")
                    gr.Markdown("**Selected Message JSON:**")
                    inspector_json = gr.Code(
                        label="Payload up to selected message",
                        language="json",
                        lines=15
                    )

    # Toggle custom fields visibility based on endpoint selection
    def toggle_custom_fields(choice):
        is_custom = choice == "Custom"
        return (
            gr.update(visible=is_custom),  # custom_url
            gr.update(visible=is_custom),  # custom_model
            gr.update(visible=is_custom),  # custom_api_key
        )

    endpoint_choice.change(
        fn=toggle_custom_fields,
        inputs=[endpoint_choice],
        outputs=[custom_url, custom_model, custom_api_key]
    )

    # Event handlers
    send_btn.click(
        fn=send_message,
        inputs=[chat_history, focus_index, system_text, user_text, temperature, max_tokens, top_p,
                endpoint_choice, custom_url, custom_api_key, custom_model],
        outputs=[chat_history, focus_index, chatbot, nav_dropdown, payload_view, response_view, assistant_text, system_text]
    ).then(
        fn=lambda: "",
        outputs=[user_text]
    )

    # Navigation (Chat View tab) - updates inspector_json
    prev_btn.click(
        fn=navigate_prev,
        inputs=[chat_history, focus_index, temperature, max_tokens, top_p],
        outputs=[focus_index, user_text, role, inspector_json, position_display]
    )

    next_btn.click(
        fn=navigate_next,
        inputs=[chat_history, focus_index, temperature, max_tokens, top_p],
        outputs=[focus_index, user_text, role, inspector_json, position_display]
    )

    nav_dropdown.change(
        fn=navigate_to_selection,
        inputs=[chat_history, nav_dropdown, temperature, max_tokens, top_p],
        outputs=[focus_index, user_text, role, inspector_json, position_display]
    )

    end_btn.click(
        fn=go_to_end,
        inputs=[chat_history, temperature, max_tokens, top_p],
        outputs=[focus_index, user_text, role, inspector_json, position_display]
    )

    clear_btn.click(
        fn=clear_history,
        outputs=[chat_history, focus_index, chatbot, nav_dropdown, payload_view, response_view, assistant_text, user_text, role, position_display]
    ).then(
        fn=lambda: "",
        outputs=[inspector_json]
    )


def start_localtunnel(port):
    """Start localtunnel in background and print the URL."""
    def run_tunnel():
        process = subprocess.Popen(
            ["npx", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True
        )
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"[localtunnel] {line}")

    thread = threading.Thread(target=run_tunnel, daemon=True)
    thread.start()


if __name__ == "__main__":
    port = 7861

    print(f"\n* Local URL: http://127.0.0.1:{port}")
    print("* Starting localtunnel... (public URL will appear below)\n")

    # Start localtunnel in background
    start_localtunnel(port)

    # Launch Gradio
    demo.launch(server_name="127.0.0.1", server_port=port, css=CUSTOM_CSS)
