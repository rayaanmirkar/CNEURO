#!/usr/bin/env python3
"""
cneuro.py — Production-Grade Full-Screen CLI Agent powered by Ollama.
Dependency-free implementation using Python standard library only.
"""

import argparse
import ast
import json
import os
import py_compile
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Global Configurations & Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 25
NUDGE_LIMIT = 1  # how many times to ask the model to actually call write_file
                 # before we extract its last code block and write it ourselves
DEFAULT_MODEL = "qwen2.5-coder:14b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
LOG_PATH = os.environ.get("AGENT_LOG", "cneuro_log.jsonl")
LOG_LOCK = threading.Lock()

SYSTEM_PROMPT = (
    "You are CNEURO, an advanced AI software engineer and computational neuroscience assistant.\n"
    "When asked to create, edit, read files, or run commands, execute the appropriate tool calls directly.\n"
    "DO NOT output raw JSON blocks, code snippets for the user to copy/paste, or instructions asking the user to manually make files when a tool can do it for them.\n"
    "Respond directly in text only for conversation, explanations, or general advice."
)

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read full text content of a local file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with specified content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory contents at target path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the local environment.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Terminal Styling & UI Components
# ---------------------------------------------------------------------------

class Style:
    enabled = True

    @staticmethod
    def wrap(code, text):
        return f"\033[{code}m{text}\033[0m" if Style.enabled else str(text)

    @staticmethod
    def dim(text): return Style.wrap("90", text)
    @staticmethod
    def bold(text): return Style.wrap("1", text)
    @staticmethod
    def cyan(text): return Style.wrap("36", text)
    @staticmethod
    def green(text): return Style.wrap("32", text)
    @staticmethod
    def yellow(text): return Style.wrap("33", text)
    @staticmethod
    def red(text): return Style.wrap("31", text)
    @staticmethod
    def magenta(text): return Style.wrap("35", text)


def clear_screen():
    if sys.stdout.isatty():
        print("\033[H\033[2J", end="")


def term_width():
    return shutil.get_terminal_size(fallback=(80, 24)).columns


BIG_BANNER = [
    "  ██████╗███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ",
    " ██╔════╝████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗",
    " ██║     ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║",
    " ██║     ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║",
    " ╚██████╗██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝",
    "  ╚═════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ "
]

SLASH_HELP = """
  Available commands:
    /help              show this help
    /clear             clear conversation history
    /model [name]      show or switch the model
    /host [url]        show or switch the Ollama host
    /auto [on|off]     show or toggle auto mode (skip write/run confirmations)
    /log [n]           show the last n log entries (default 10)
    /quit, /exit       exit cneuro
"""


def print_cneuro_header(model_name):
    cwd = os.getcwd()
    print()
    for line in BIG_BANNER:
        print(Style.magenta(line))
    print()
    print(f" {Style.cyan('CNEURO CLI')} {Style.dim('v2.5.0')} | {Style.dim(model_name)}")
    print(f" {Style.dim('Working Dir:')} {Style.dim(cwd)}")
    print()


def print_status_footer():
    w = term_width()
    footer_info = "* local · /cneuro"
    # Pad the PLAIN text first, then color it — coloring first would make the
    # invisible ANSI escape codes count toward rjust's width and misalign it.
    print(f"\n{Style.dim(footer_info.rjust(w))}")


def print_assistant_box(content):
    if not content or not content.strip():
        return
    print(Style.magenta("\n┌──[ Cneuro ]"))
    lines = content.strip().split("\n")
    print(Style.magenta("└─> ") + lines[0])
    for line in lines[1:]:
        print(f"    {line}")
    print()


def print_action(fn, args):
    if fn == "write_file":
        print(f"  {Style.magenta('*')} {Style.bold('Write')} {args.get('path')}")
    elif fn == "read_file":
        print(f"  {Style.magenta('*')} {Style.bold('Read')} {args.get('path')}")
    elif fn == "run_command":
        print(f"  {Style.magenta('*')} {Style.bold('Bash')} {args.get('command')}")
    elif fn == "list_dir":
        print(f"  {Style.magenta('*')} {Style.bold('List')} {args.get('path', '.')}")
    else:
        print(f"  {Style.magenta('*')} {Style.bold('Tool')} {fn}")


def print_result(result):
    if result.get("ok"):
        msg = result.get("message") or "Done"
        print(f"  {Style.green('[OK]')} {Style.dim(msg)}")
    else:
        err = result.get("error") or "Failed"
        print(f"  {Style.red('[ERR]')} {Style.dim(err)}")


# ---------------------------------------------------------------------------
# Activity Spinner
# ---------------------------------------------------------------------------

class Spinner:
    FRAMES = ["-", "\\", "|", "/"]

    def __init__(self, label="Thinking"):
        self.label = label
        self._stop = threading.Event()
        self._t0 = time.time()
        self._thread = None
        self._active = Style.enabled and sys.stdout.isatty()

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            frame = self.FRAMES[i % len(self.FRAMES)]
            line = f"  {Style.magenta(frame)} {Style.dim(f'{self.label}... ({elapsed:0.1f}s)')}"
            print(line, end="\r", flush=True)
            i += 1
            time.sleep(0.08)

    def start(self):
        if self._active:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    def stop(self):
        elapsed = time.time() - self._t0
        if self._active and self._thread:
            self._stop.set()
            self._thread.join(timeout=0.3)
            print(" " * min(term_width(), 60), end="\r", flush=True)
        return elapsed


# ---------------------------------------------------------------------------
# Logging & Utilities
# ---------------------------------------------------------------------------

def log_event(event_type, data):
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "type": event_type, "data": data}
    try:
        with LOG_LOCK:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def tail_log(n=10):
    if not os.path.exists(LOG_PATH):
        print(Style.dim("  (no log file yet)\n"))
        return
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    for line in lines:
        try:
            entry = json.loads(line)
            summary = json.dumps(entry["data"])
            if len(summary) > 100:
                summary = summary[:100] + "..."
            print(Style.dim(f"  [{entry['ts']}] {entry['type']}: {summary}"))
        except Exception:
            print(Style.dim("  " + line.strip()))
    print()


def verify_python_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        ast.parse(content)
        py_compile.compile(path, doraise=True)
        return True, None
    except Exception as e:
        return False, str(e)


def find_code_blocks(content):
    """Fenced code blocks in a text response, e.g. ```python ... ```."""
    return re.findall(r"```(?:\w+)?\n([\s\S]*?)```", content)


def guess_filename(content, task):
    """Best-effort filename when the model described a file but never named
    it via a tool call. Prefers an explicit .py mention in the text or task,
    falls back to a slug of the task."""
    m = re.search(r"([\w\-]+\.py)\b", content) or re.search(r"([\w\-]+\.py)\b", task)
    if m:
        return m.group(1)
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")[:40] or "script"
    return f"{slug}.py"


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def tool_read_file(args, auto):
    path = args.get("path")
    if not path:
        return {"ok": False, "error": "Missing 'path' argument"}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return {"ok": True, "content": f.read()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_write_file(args, auto):
    path = args.get("path")
    content = args.get("content", "")
    if not path:
        return {"ok": False, "error": "Missing 'path' argument"}

    if not auto and sys.stdin.isatty():
        print(f"  {Style.yellow('?')} Create file {Style.bold(path)}? [y/N] ", end="", flush=True)
        try:
            resp = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            resp = "n"
        if resp != "y":
            return {"ok": False, "error": "User declined file creation"}

    try:
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    res = {"ok": True, "message": f"Wrote {path}"}
    if path.endswith(".py"):
        ok, err = verify_python_file(path)
        if not ok:
            res["ok"] = False
            res["error"] = f"Syntax verification failed: {err}"
    return res


def tool_list_dir(args, auto):
    path = args.get("path", ".")
    try:
        entries = os.listdir(path)
        return {"ok": True, "entries": sorted(entries)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_run_command(args, auto):
    command = args.get("command")
    if not command:
        return {"ok": False, "error": "Missing 'command' argument"}

    if not auto and sys.stdin.isatty():
        print(f"  {Style.yellow('?')} Run command {Style.bold(command)}? [y/N] ", end="", flush=True)
        try:
            resp = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            resp = "n"
        if resp != "y":
            return {"ok": False, "error": "User declined command execution"}

    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        return {
            "ok": res.returncode == 0,
            "stdout": res.stdout[-4000:],
            "stderr": res.stderr[-4000:],
            "returncode": res.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Command execution timed out after 60s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


TOOL_IMPL = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_dir": tool_list_dir,
    "run_command": tool_run_command,
}


def extract_fallback_tool_call(content):
    """Catches the case where the model emits a complete, well-formed tool
    call as JSON text instead of using the native tool_calls field."""
    if not content or "{" not in content:
        return None

    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content)
    candidates = matches if matches else [content]

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("tool")
                args = obj.get("arguments") or obj.get("args") or obj.get("parameters")
                if name in TOOL_IMPL and isinstance(args, dict):
                    return name, args
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------------------------
# Inference Engine Interface
# ---------------------------------------------------------------------------

def trim_context(messages, max_history=12):
    if len(messages) <= max_history + 1:
        return messages
    system_msg = messages[0]
    recent_history = messages[-max_history:]
    return [system_msg] + recent_history


def call_ollama(host, model, messages):
    payload = {
        "model": model,
        "messages": trim_context(messages),
        "tools": TOOL_SCHEMA,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "top_p": 0.85,
            "num_ctx": 4096,
        },
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )

    content_parts, tool_calls = [], []
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message", {})
                if isinstance(msg, dict):
                    if msg.get("content"):
                        content_parts.append(msg["content"])
                    if msg.get("tool_calls"):
                        tool_calls.extend(msg["tool_calls"])
                if chunk.get("done", False):
                    break
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed to Ollama at '{host}': {e.reason}")
    except Exception as e:
        raise RuntimeError(f"Ollama API Error: {e}")

    message = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"message": message}


# ---------------------------------------------------------------------------
# Core Agent Loop
# ---------------------------------------------------------------------------

def run_task(host, model, auto, task, messages=None):
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": task})
    log_event("task", {"task": task})

    nudges_used = 0

    for step in range(MAX_STEPS):
        spinner = Spinner("Thinking")
        spinner.start()
        try:
            response = call_ollama(host, model, messages)
        except RuntimeError as e:
            spinner.stop()
            print(f"  {Style.red('[ERR]')} {e}\n")
            return messages
        except KeyboardInterrupt:
            spinner.stop()
            print(f"\n  {Style.yellow('[!]')} Task interrupted by user.\n")
            return messages
        spinner.stop()

        message = response.get("message", {})
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            content = message.get("content", "")
            fallback = extract_fallback_tool_call(content)
            if fallback:
                fn_name, fn_args = fallback
                tool_calls = [{"function": {"name": fn_name, "arguments": fn_args}}]
                message = {"role": "assistant", "content": "", "tool_calls": tool_calls}

        if not tool_calls:
            content = message.get("content", "")
            code_blocks = find_code_blocks(content)

            if code_blocks and nudges_used < NUDGE_LIMIT:
                nudges_used += 1
                messages.append(message)
                messages.append({
                    "role": "user",
                    "content": (
                        "Don't just show code in your response — actually create the file "
                        "yourself right now using the write_file tool, with that exact "
                        "content. Do not respond with text only."
                    ),
                })
                log_event("nudge", {"attempt": nudges_used})
                continue

            if code_blocks:
                code = code_blocks[-1]
                filename = guess_filename(content, task)
                print(Style.yellow(
                    f"  [!] model described code instead of writing it — saving it "
                    f"directly as {filename}"
                ))
                result = tool_write_file({"path": filename, "content": code}, auto)
                print_action("write_file", {"path": filename})
                print_result(result)
                log_event("auto_recovered_write", {"path": filename, "result": result})

            print_assistant_box(content)
            log_event("final_answer", {"content": content})
            messages.append(message)
            return messages

        messages.append(message)
        for call in tool_calls:
            fn = call.get("function", {}).get("name")
            raw_args = call.get("function", {}).get("arguments", {})
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)

            print_action(fn, args)

            impl = TOOL_IMPL.get(fn)
            result = impl(args, auto) if impl else {"ok": False, "error": f"Unknown tool: {fn}"}

            print_result(result)

            log_event("tool_call", {"tool": fn, "args": args, "result": result})
            messages.append({"role": "tool", "content": json.dumps(result)})

    print(f"  {Style.yellow('[!]')} Reached maximum step limit ({MAX_STEPS}). Stopping turn.\n")
    return messages


# ---------------------------------------------------------------------------
# Slash Commands
# ---------------------------------------------------------------------------

def handle_slash_command(task, args):
    """Returns 'quit' if the command should end the session, otherwise None.
    Mutates `args` in place for /model, /host, /auto so the running session
    picks up the change on the next task. `messages` is the current chat
    history list; return value replaces it via the caller (needed for /clear)."""
    parts = task[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit"):
        return "quit"
    elif cmd == "help":
        print(SLASH_HELP)
    elif cmd == "clear":
        return "clear"
    elif cmd == "model":
        if rest:
            args.model = rest
            print(Style.dim(f"  model set to {args.model}\n"))
        else:
            print(Style.dim(f"  current model: {args.model}\n"))
    elif cmd == "host":
        if rest:
            args.host = rest
            print(Style.dim(f"  host set to {args.host}\n"))
        else:
            print(Style.dim(f"  current host: {args.host}\n"))
    elif cmd == "auto":
        if rest.lower() in ("on", "off"):
            args.auto = rest.lower() == "on"
        print(Style.dim(f"  auto mode is {'ON' if args.auto else 'OFF'}\n"))
    elif cmd == "log":
        n = int(rest) if rest.isdigit() else 10
        tail_log(n)
    else:
        print(Style.yellow(f"  unknown command /{cmd} — try /help\n"))
    return None


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(prog="cneuro", description="CNEURO CLI Agent")
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST))
    p.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    p.add_argument("--auto", action="store_true", default=os.environ.get("AGENT_AUTO", "0") == "1")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--task", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    Style.enabled = not args.no_color and sys.stdout.isatty()

    clear_screen()
    print_cneuro_header(args.model)

    if args.task:
        run_task(args.host, args.model, args.auto, args.task)
        print_status_footer()
        return

    messages = None
    while True:
        print(Style.cyan("┌──[ You ]"))
        try:
            task = input(Style.cyan("└─> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if task.lower() in ("quit", "exit"):
            print()
            break
        if not task:
            continue
        if task.startswith("/"):
            outcome = handle_slash_command(task, args)
            if outcome == "quit":
                print()
                break
            if outcome == "clear":
                messages = None
                print(Style.dim("  conversation cleared\n"))
            continue

        messages = run_task(args.host, args.model, args.auto, task, messages)

    print_status_footer()


if __name__ == "__main__":
    main()