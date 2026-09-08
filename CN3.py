#!/usr/bin/env python3
"""
CNEURO 3.2
A dependency-free local coding/science agent for Ollama.
Python standard library only.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path as FSPath
from typing import Any, Callable


# ============================================================================
# Configuration
# ============================================================================
APP_NAME = "CNEURO"
VERSION = "3.2.0"
DEFAULT_MODEL = "qwen2.5-coder:14b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"

MAX_STEPS = 30
MAX_HISTORY_MESSAGES = 24
MAX_TOOL_OUTPUT = 6000
COMMAND_TIMEOUT = 60
LOG_PATH = os.environ.get("AGENT_LOG", "cneuro_log.jsonl")

SYSTEM_PROMPT = """
You are CNEURO 3.2, a local coding and computational-science agent.

You work directly in the user's current project workspace.

CORE BEHAVIOR:
- When the user asks you to create or modify files, use the file tools.
- When the user asks you to run or test something, use run_command.
- Inspect existing files before changing them when useful.
- Prefer small, targeted edits over rewriting an unchanged file.
- After changing code, verify it when practical.
- If a command fails, inspect the failure and fix the underlying problem.
- Do not repeatedly perform a successful action.
- Once the user's requested work is complete, stop calling tools and give a concise final answer.
- Report actual command output; never invent output.
- For scientific code, check units, dimensions, numerical scales, and obvious methodological mistakes.
- Do not change scientific parameters merely to force an interesting result.
- For scientific Python, run scientific_check before treating a simulation as scientifically valid.
- A process exit code of 0 means only that the program executed successfully; it does NOT prove scientific correctness.
- If scientific_check reports an error, fix the code before reporting success.
- Do not tell the user to manually copy code when you can perform the requested file operation.
- Do not emit fake tool-call JSON. Use the provided tools.
- If native tool calling is unavailable, you may emit exactly one JSON tool-call object:
  {"name":"tool_name","arguments":{...}}
- Never put source code in a JSON tool-call response unless it is the value of a tool argument.

SCIENTIFIC VALIDATION:
- Preserve the user's requested duration, timestep, stimulus, and model parameters unless they are scientifically inconsistent.
- For LIF models, distinguish volts, amperes, ohms, seconds, and Hz.
- A firing rate computed as spike_count / duration is already Hz when duration is in seconds.
- Do not multiply that rate by 1000 unless the denominator is explicitly in milliseconds.
- For a current-driven LIF, a standard form is:
    dVdt = (V_rest - V + R * I) / tau
  or an algebraically equivalent expression.
- Never accept an expression that subtracts current (A) directly from voltage (V), such as (current - V).
- If a scientific check fails after execution, inspect, repair, re-verify, and rerun.

AVAILABLE TOOLS:
read_file, write_file, edit_file, list_dir, search_files, run_command,
verify_python, scientific_check

COMPLETION RULE:
If the requested file has been created/updated, syntax verification passes,
the requested command/test succeeds, and scientific_check has no errors,
do not recreate the file. Use the actual result to answer the user.
"""


# ============================================================================
# Terminal UI
# ============================================================================
class Style:
    enabled = True

    @staticmethod
    def wrap(code: str, text: Any) -> str:
        return f"\033[{code}m{text}\033[0m" if Style.enabled else str(text)

    @staticmethod
    def dim(text: Any) -> str:
        return Style.wrap("90", text)

    @staticmethod
    def bold(text: Any) -> str:
        return Style.wrap("1", text)

    @staticmethod
    def cyan(text: Any) -> str:
        return Style.wrap("36", text)

    @staticmethod
    def green(text: Any) -> str:
        return Style.wrap("32", text)

    @staticmethod
    def yellow(text: Any) -> str:
        return Style.wrap("33", text)

    @staticmethod
    def red(text: Any) -> str:
        return Style.wrap("31", text)

    @staticmethod
    def magenta(text: Any) -> str:
        return Style.wrap("35", text)


def term_width() -> int:
    return shutil.get_terminal_size((90, 24)).columns


def clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[H\033[2J", end="")


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} chars truncated]"


def print_banner(model: str, workspace: FSPath) -> None:
    banner = [
        "   ██████╗███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ",
        "  ██╔════╝████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗",
        "  ██║     ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║",
        "  ██║     ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║",
        "  ╚██████╗██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝",
        "   ╚═════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝",
    ]
    print()
    for line in banner:
        print(Style.magenta(line))
    print()
    print(
        f" {Style.bold(APP_NAME)} {Style.dim(f'v{VERSION}')}  "
        f"{Style.dim('•')}  {Style.cyan(model)}"
    )
    print(f" {Style.dim('workspace')}  {workspace}")
    print(
        f" {Style.dim('commands')}   /help  /clear  /model  /auto  /status  /quit"
    )
    print()


def print_user_prompt() -> str:
    print(Style.cyan("╭─ You"))
    return input(Style.cyan("╰─> ")).strip()


def print_assistant(text: str) -> None:
    if not text.strip():
        return
    print()
    print(Style.magenta("╭─ CNEURO"))
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        prefix = "╰─> " if i == 0 else "    "
        print(Style.magenta(prefix) + line)
    print()


class Spinner:
    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, label: str = "Thinking"):
        self.label = label
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.active = Style.enabled and sys.stdout.isatty()
        self.started = time.monotonic()

    def _run(self) -> None:
        i = 0
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - self.started
            print(
                f"  {Style.magenta(self.FRAMES[i % len(self.FRAMES)])} "
                f"{self.label} {elapsed:.1f}s",
                end="\r",
                flush=True,
            )
            i += 1
            time.sleep(0.08)

    def start(self) -> None:
        if self.active:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self) -> float:
        elapsed = time.monotonic() - self.started
        if self.active and self.thread:
            self.stop_event.set()
            self.thread.join(timeout=0.3)
            print(" " * min(term_width(), 80), end="\r", flush=True)
        return elapsed


# ============================================================================
# Logging
# ============================================================================
_LOG_LOCK = threading.Lock()


def log_event(kind: str, data: dict[str, Any]) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": kind,
        "data": data,
    }
    try:
        with _LOG_LOCK:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ============================================================================
# Workspace
# ============================================================================
class WorkspaceError(Exception):
    pass


class Workspace:
    def __init__(self, root: str):
        self.root = FSPath(root).expanduser().resolve()

    def resolve(self, user_path: str = ".") -> FSPath:
        raw = FSPath(user_path)
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise WorkspaceError(f"Path escapes workspace: {user_path!r}")
        return resolved

    def display(self, path: FSPath) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


# ============================================================================
# Verification helpers
# ============================================================================
def verify_python_source(source: str, filename: str = "<string>") -> tuple[bool, str]:
    try:
        tree = ast.parse(source, filename=filename)
        compile(tree, filename, "exec")
        return True, "Python syntax and compilation check passed."
    except Exception as exc:
        return False, str(exc)


def verify_python_path(path: FSPath) -> tuple[bool, str]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, str(exc)
    return verify_python_source(source, str(path))


def make_diff(old: str, new: str, path: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"{path} (before)",
        tofile=f"{path} (after)",
        lineterm="",
    )
    return "\n".join(diff)


# ============================================================================
# Scientific validation
# ============================================================================
def _literal_assignments(tree: ast.AST) -> dict[str, float]:
    values: dict[str, float] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue

        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue

        value_node = node.value
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except Exception:
            continue

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[targets[0].id] = float(value)
    return values


def _constant_value(node: ast.AST | None, values: dict[str, float]) -> float | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        return values.get(node.id)
    return None


def _name_lower(node: ast.AST | None) -> str:
    return node.id.lower() if isinstance(node, ast.Name) else ""


def _is_name(node: ast.AST | None, *names: str) -> bool:
    return isinstance(node, ast.Name) and node.id.lower() in {n.lower() for n in names}


def _lif_equation_checks(
    tree: ast.AST,
    source: str,
    values: dict[str, float],
) -> tuple[list[str], list[dict[str, str]], bool]:
    checks: list[str] = []
    findings: list[dict[str, str]] = []
    equation_seen = False
    lower = source.lower()

    # Look at AST BinOp trees rather than only raw strings.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        targets = node.targets
        if not any(isinstance(t, ast.Name) and t.id.lower() in {"dvdt", "dv"} for t in targets):
            continue

        equation_seen = True
        expr = node.value

        # Detect the invalid pattern: current - V, V - current, etc.
        invalid_mixed_subtraction = False

        for sub in ast.walk(expr):
            if not isinstance(sub, ast.BinOp) or not isinstance(sub.op, ast.Sub):
                continue

            left_name = _name_lower(sub.left)
            right_name = _name_lower(sub.right)

            current_like = {"current", "i", "input_current", "injected_current"}
            voltage_like = {
                "v",
                "voltage",
                "v_rest",
                "vreset",
                "v_reset",
                "v_threshold",
                "vth",
                "threshold",
            }

            if (
                (left_name in current_like and right_name in voltage_like)
                or (left_name in voltage_like and right_name in current_like)
            ):
                invalid_mixed_subtraction = True

        if invalid_mixed_subtraction:
            findings.append(
                {
                    "level": "error",
                    "message": (
                        "LIF dimensional error: the membrane equation subtracts "
                        "current (A) directly from voltage (V). Use a resistance "
                        "term, e.g. (V_rest - V + R * current) / tau."
                    ),
                }
            )
            continue

        normalized = re.sub(r"\s+", "", ast.unparse(expr) if hasattr(ast, "unparse") else source)
        has_current_drive = bool(
            re.search(r"\bR\*current\b", normalized, re.I)
            or re.search(r"\bcurrent\*R\b", normalized, re.I)
            or re.search(r"\bR\*I\b", normalized)
            or re.search(r"\bI\*R\b", normalized)
        )
        has_voltage_leak = bool(
            re.search(r"V_rest-V", normalized, re.I)
            or re.search(r"-V\+V_rest", normalized, re.I)
            or re.search(r"\(-V\+V_rest", normalized, re.I)
        )
        has_tau = bool(re.search(r"/tau\b", normalized, re.I))

        if has_current_drive and has_voltage_leak and has_tau:
            checks.append("LIF membrane update has the expected current-driven form.")
        elif "dVdt" in source or "dvdt" in lower:
            findings.append(
                {
                    "level": "warning",
                    "message": (
                        "A dV/dt assignment was found, but its exact LIF form "
                        "could not be confidently recognized. Check that voltage "
                        "leak and current drive have consistent units."
                    ),
                }
            )

    return checks, findings, equation_seen


def _extract_spike_count(stdout: str) -> int | None:
    patterns = [
        r"total\s+spike\s+count\s*:\s*(\d+)",
        r"spike\s+count\s*:\s*(\d+)",
        r"spikes?\s*:\s*(\d+)",
        r"n[_\s-]*spikes?\s*:\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout, re.I)
        if match:
            return int(match.group(1))
    return None


def _extract_firing_rate(stdout: str) -> float | None:
    match = re.search(
        r"(?:average\s+)?firing\s+rate\s*:\s*([-+]?\d+(?:\.\d+)?)\s*hz",
        stdout,
        re.I,
    )
    return float(match.group(1)) if match else None


def scientific_check_python(path: FSPath, stdout: str = "") -> dict[str, Any]:
    """
    Conservative, explainable scientific checks for numerical neuroscience.

    This is not a theorem prover. It deliberately reports warnings when a
    pattern is ambiguous and errors only for high-confidence mistakes.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except Exception as exc:
        return {"ok": False, "level": "error", "error": f"Could not analyze Python source: {exc}"}

    findings: list[dict[str, str]] = []
    checks: list[str] = []
    values = _literal_assignments(tree)
    lower = source.lower()

    # ---- Firing-rate units -------------------------------------------------
    firing_rate_assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and "firing_rate" in target.id.lower()
            for target in node.targets
        )
    ]

    for node in firing_rate_assignments:
        expr = node.value
        normalized = re.sub(
            r"\s+",
            "",
            ast.unparse(expr) if hasattr(ast, "unparse") else "",
        )

        has_spike_count = bool(re.search(r"\bspike[_]?count\b", normalized, re.I))
        divides_duration = bool(re.search(r"/duration\b", normalized, re.I))
        multiplies_1000 = bool(re.search(r"\*1000(?:\.0+)?\b", normalized))

        if has_spike_count and divides_duration:
            if multiplies_1000:
                findings.append(
                    {
                        "level": "error",
                        "message": (
                            "Firing-rate unit error: spike_count / duration is "
                            "already Hz when duration is in seconds. Multiplying "
                            "by 1000 makes the reported rate 1000× too large."
                        ),
                    }
                )
            else:
                checks.append(
                    "Firing rate uses spike_count / duration (Hz when duration is seconds)."
                )

    # Catch the same mistake in a print expression if no assignment exists.
    for line in source.splitlines():
        compact = re.sub(r"\s+", "", line)
        if (
            "firingrate" in compact.lower()
            and "spike_count/duration" in compact.lower()
            and "*1000" in compact
        ):
            findings.append(
                {
                    "level": "error",
                    "message": (
                        "Firing-rate unit error: spike_count / duration is already "
                        "Hz for a duration expressed in seconds; remove * 1000."
                    ),
                }
            )

    # ---- LIF detection -----------------------------------------------------
    lif_names = {
        "tau",
        "v_rest",
        "v_threshold",
        "r",
        "current",
    }
    lif_detected = lif_names.issubset(values) and (
        "leaky integrate" in lower
        or "lif" in lower
        or "dvdt" in lower
        or "d v / dt" in lower
    )

    lif_checks, lif_findings, equation_seen = _lif_equation_checks(
        tree, source, values
    )
    checks.extend(lif_checks)
    findings.extend(lif_findings)

    if lif_detected:
        tau = values["tau"]
        v_rest = values["v_rest"]
        threshold = values["v_threshold"]
        resistance = values["r"]
        current = values["current"]

        if tau <= 0:
            findings.append(
                {"level": "error", "message": "LIF tau must be positive."}
            )
        if resistance <= 0:
            findings.append(
                {"level": "error", "message": "LIF resistance R must be positive."}
            )

        v_inf = v_rest + resistance * current

        checks.append(
            f"LIF steady-state voltage estimate: {v_inf:.6g} V "
            f"({v_inf * 1000:.3f} mV)."
        )
        checks.append(
            f"LIF threshold: {threshold:.6g} V "
            f"({threshold * 1000:.3f} mV)."
        )

        if v_inf < threshold:
            checks.append(
                "Steady state is below threshold for the supplied constant current."
            )
        else:
            checks.append(
                "Steady state reaches/exceeds threshold for the supplied constant current."
            )

        spike_count = _extract_spike_count(stdout)
        if spike_count is not None:
            if v_inf < threshold and spike_count > 0:
                findings.append(
                    {
                        "level": "error",
                        "message": (
                            f"Observed {spike_count} spikes even though the estimated "
                            f"steady state ({v_inf * 1000:.2f} mV) is below threshold "
                            f"({threshold * 1000:.2f} mV). This strongly suggests an "
                            "incorrect membrane equation, stimulus, or threshold logic."
                        ),
                    }
                )
            elif v_inf < threshold and spike_count == 0:
                checks.append(
                    "Observed 0 spikes is consistent with the steady state remaining below threshold."
                )
            elif v_inf >= threshold and spike_count == 0:
                findings.append(
                    {
                        "level": "warning",
                        "message": (
                            f"Steady state ({v_inf * 1000:.2f} mV) reaches/exceeds "
                            f"threshold ({threshold * 1000:.2f} mV), but the run "
                            "reported 0 spikes. Check timestep, threshold comparison, "
                            "stimulus duration, and reset logic."
                        ),
                    }
                )

        reported_rate = _extract_firing_rate(stdout)
        if spike_count is not None and reported_rate is not None:
            duration = values.get("duration")
            if duration and duration > 0:
                expected_rate = spike_count / duration
                if abs(reported_rate - expected_rate) > max(
                    0.01, abs(expected_rate) * 0.01
                ):
                    findings.append(
                        {
                            "level": "error",
                            "message": (
                                f"Reported firing rate {reported_rate:g} Hz does not "
                                f"match spike_count / duration = {expected_rate:g} Hz."
                            ),
                        }
                    )
                else:
                    checks.append(
                        f"Observed firing rate matches spike_count / duration ({expected_rate:g} Hz)."
                    )

        if not equation_seen:
            findings.append(
                {
                    "level": "warning",
                    "message": (
                        "LIF parameters were detected, but no recognizable dV/dt "
                        "assignment was found. Scientific validation is incomplete."
                    ),
                }
            )

    level = (
        "error"
        if any(f["level"] == "error" for f in findings)
        else "warning"
        if findings
        else "ok"
    )

    return {
        "ok": level != "error",
        "level": level,
        "checks": checks,
        "findings": findings,
    }


# ============================================================================
# Tool layer
# ============================================================================
@dataclass
class ToolContext:
    workspace: Workspace
    auto: bool
    interactive: bool


def ask_confirmation(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        answer = input(
            f"  {Style.yellow('?')} {question} {Style.dim('[y/N]')} "
        ).strip().lower()
        return answer in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def _validate_python_candidate(path: FSPath, content: str) -> tuple[bool, str]:
    if path.suffix.lower() != ".py":
        return True, ""
    return verify_python_source(content, str(path))


def tool_read_file(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path_arg = str(args.get("path", "")).strip()
    if not path_arg:
        return {"ok": False, "error": "Missing path."}
    try:
        path = ctx.workspace.resolve(path_arg)
        if not path.is_file():
            return {"ok": False, "error": f"Not a file: {path_arg}"}
        content = path.read_text(encoding="utf-8", errors="replace")
        return {
            "ok": True,
            "path": ctx.workspace.display(path),
            "content": truncate(content, 16000),
        }
    except (WorkspaceError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def tool_write_file(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path_arg = str(args.get("path", "")).strip()
    content = args.get("content")

    if not path_arg:
        return {"ok": False, "error": "Missing path."}
    if not isinstance(content, str):
        return {"ok": False, "error": "Content must be a string."}

    try:
        path = ctx.workspace.resolve(path_arg)
        existed = path.exists()
        old = (
            path.read_text(encoding="utf-8", errors="replace")
            if existed and path.is_file()
            else ""
        )

        if existed and old == content:
            return {
                "ok": True,
                "changed": False,
                "path": ctx.workspace.display(path),
                "message": "File already contains the requested content; no write performed.",
            }

        valid, error = _validate_python_candidate(path, content)
        if not valid:
            return {
                "ok": False,
                "error": f"Python verification failed; write rejected: {error}",
                "path": ctx.workspace.display(path),
            }

        if not ctx.auto and ctx.interactive:
            action = "overwrite" if existed else "create"
            if not ask_confirmation(
                f"{action.capitalize()} {ctx.workspace.display(path)}?"
            ):
                return {"ok": False, "error": "User declined file write."}

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        result: dict[str, Any] = {
            "ok": True,
            "changed": True,
            "path": ctx.workspace.display(path),
            "message": (
                f"{'Updated' if existed else 'Created'} "
                f"{ctx.workspace.display(path)}."
            ),
        }

        if path.suffix.lower() == ".py":
            result["verified"] = True

        diff = make_diff(old, content, ctx.workspace.display(path))
        if diff:
            result["diff"] = truncate(diff, 10000)
        return result

    except (WorkspaceError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def tool_edit_file(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path_arg = str(args.get("path", "")).strip()
    old = args.get("old")
    new = args.get("new")

    if not path_arg:
        return {"ok": False, "error": "Missing path."}
    if not isinstance(old, str) or not isinstance(new, str):
        return {"ok": False, "error": "old and new must both be strings."}

    try:
        path = ctx.workspace.resolve(path_arg)
        if not path.is_file():
            return {"ok": False, "error": f"File does not exist: {path_arg}"}

        content = path.read_text(encoding="utf-8", errors="replace")
        count = content.count(old)

        if count == 0:
            return {"ok": False, "error": "Exact old text was not found."}
        if count > 1:
            return {
                "ok": False,
                "error": f"Old text occurs {count} times. Provide a more specific block.",
            }

        updated = content.replace(old, new, 1)

        valid, error = _validate_python_candidate(path, updated)
        if not valid:
            return {
                "ok": False,
                "error": f"Python verification failed; edit rejected: {error}",
            }

        if not ctx.auto and ctx.interactive:
            diff = make_diff(content, updated, ctx.workspace.display(path))
            print()
            print(Style.dim(truncate(diff, 8000)))
            if not ask_confirmation(
                f"Apply edit to {ctx.workspace.display(path)}?"
            ):
                return {"ok": False, "error": "User declined file edit."}

        path.write_text(updated, encoding="utf-8")

        result: dict[str, Any] = {
            "ok": True,
            "changed": True,
            "path": ctx.workspace.display(path),
            "message": f"Edited {ctx.workspace.display(path)}.",
            "diff": truncate(
                make_diff(content, updated, ctx.workspace.display(path)),
                10000,
            ),
        }

        if path.suffix.lower() == ".py":
            result["verified"] = True

        return result

    except (WorkspaceError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def tool_list_dir(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path_arg = str(args.get("path", ".")).strip() or "."
    try:
        path = ctx.workspace.resolve(path_arg)
        if not path.is_dir():
            return {"ok": False, "error": f"Not a directory: {path_arg}"}

        entries = []
        for entry in sorted(
            path.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        ):
            entries.append(
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                }
            )

        return {
            "ok": True,
            "path": ctx.workspace.display(path),
            "entries": entries[:500],
        }
    except (WorkspaceError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def tool_search_files(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    path_arg = str(args.get("path", ".")).strip() or "."

    if not query:
        return {"ok": False, "error": "Missing search query."}

    ignored_dirs = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
    }

    try:
        root = ctx.workspace.resolve(path_arg)
        if not root.exists():
            return {"ok": False, "error": f"Path does not exist: {path_arg}"}

        matches = []
        files = [root] if root.is_file() else root.rglob("*")

        for path in files:
            if len(matches) >= 100:
                break
            if not path.is_file():
                continue
            if any(part in ignored_dirs for part in path.parts):
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for line_no, line in enumerate(text.splitlines(), 1):
                if query.lower() in line.lower():
                    matches.append(
                        {
                            "path": ctx.workspace.display(path),
                            "line": line_no,
                            "text": line[:500],
                        }
                    )
                    if len(matches) >= 100:
                        break

        return {"ok": True, "matches": matches}

    except (WorkspaceError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def tool_scientific_check(
    ctx: ToolContext, args: dict[str, Any]
) -> dict[str, Any]:
    path_arg = str(args.get("path", "")).strip()
    stdout = str(args.get("stdout", "") or "")

    if not path_arg:
        return {"ok": False, "error": "Missing path."}

    try:
        path = ctx.workspace.resolve(path_arg)
        if not path.is_file():
            return {"ok": False, "error": f"Not a file: {path_arg}"}
        if path.suffix.lower() != ".py":
            return {
                "ok": False,
                "error": "scientific_check currently supports Python files.",
            }
        return scientific_check_python(path, stdout)
    except (WorkspaceError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def _find_python_script(command: str) -> str | None:
    """
    Extract the first Python script from common invocations:
      python lif_test.py
      python3 ./lif_test.py --arg 1
      py lif_test.py
      python.exe "lif_test.py"
    """
    pattern = re.compile(
        r"""(?ix)
        (?:^|[\s;&|()])
        (?:(?:python(?:\d+(?:\.\d+)?)?)|py)(?:\.exe)?
        [\s]+
        (?:"([^"]+\.py)"|'([^']+\.py)'|([^\s;&|()]+\.py))
        """
    )
    match = pattern.search(command)
    if not match:
        return None
    return next((g for g in match.groups() if g), None)


def tool_run_command(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    command = str(args.get("command", "")).strip()

    if not command:
        return {"ok": False, "error": "Missing command."}

    if not ctx.auto and ctx.interactive:
        if not ask_confirmation(f"Run command: {command}?"):
            return {"ok": False, "error": "User declined command execution."}

    try:
        started = time.monotonic()
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.workspace.root),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
        elapsed = time.monotonic() - started

        result: dict[str, Any] = {
            "ok": proc.returncode == 0,
            "command": command,
            "returncode": proc.returncode,
            "duration_s": round(elapsed, 3),
            "stdout": truncate(proc.stdout),
            "stderr": truncate(proc.stderr),
        }

        script_name = _find_python_script(command)
        if proc.returncode == 0 and script_name:
            try:
                script = ctx.workspace.resolve(script_name)
                if script.is_file():
                    science = scientific_check_python(script, proc.stdout)
                    result["scientific_check"] = science

                    # Important: execution success and scientific success are
                    # separate. A scientific error makes the overall tool
                    # result unsuccessful so Qwen is prompted to repair it.
                    if science.get("level") == "error":
                        result["ok"] = False
                        result["error"] = (
                            "Command exited successfully, but scientific_check "
                            "found errors. Inspect and repair the code."
                        )
            except (WorkspaceError, OSError) as exc:
                result["scientific_check_error"] = str(exc)

        return result

    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "command": command,
            "error": f"Command timed out after {COMMAND_TIMEOUT}s.",
        }
    except OSError as exc:
        return {"ok": False, "command": command, "error": str(exc)}


def tool_verify_python(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    path_arg = str(args.get("path", "")).strip()

    if not path_arg:
        return {"ok": False, "error": "Missing path."}

    try:
        path = ctx.workspace.resolve(path_arg)
        if not path.is_file():
            return {"ok": False, "error": f"Not a file: {path_arg}"}

        valid, message = verify_python_path(path)
        return {
            "ok": valid,
            "path": ctx.workspace.display(path),
            "message": message if valid else None,
            "error": None if valid else message,
        }
    except (WorkspaceError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


TOOL_FUNCTIONS: dict[
    str, Callable[[ToolContext, dict[str, Any]], dict[str, Any]]
] = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_dir": tool_list_dir,
    "search_files": tool_search_files,
    "run_command": tool_run_command,
    "verify_python": tool_verify_python,
    "scientific_check": tool_scientific_check,
}


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file in the workspace.",
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
            "description": "Create or replace a workspace file. Python files are syntax-checked before writing.",
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
            "name": "edit_file",
            "description": "Make one exact text replacement in an existing file. Python edits are syntax-checked before writing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories in the workspace.",
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
            "name": "search_files",
            "description": "Search text across workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the workspace. For Python scripts, "
                "CNEURO automatically runs scientific validation after execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_python",
            "description": "Parse and compile a Python file to catch syntax errors.",
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
            "name": "scientific_check",
            "description": (
                "Perform conservative scientific sanity checks on a Python "
                "simulation. Checks firing-rate units, LIF dimensional "
                "consistency, recognizable LIF equations, steady-state behavior, "
                "and observed spike/rate output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "stdout": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
]


# ============================================================================
# Ollama
# ============================================================================
class OllamaError(RuntimeError):
    pass


def _message_group(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role == "system":
        return "system"
    if role == "user":
        return "user"
    if role == "tool":
        return "tool"
    if role == "assistant" and message.get("tool_calls"):
        return "assistant_tool"
    return "assistant"


def trim_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Trim by conversation groups rather than raw messages.

    This avoids cutting an assistant tool-call message away from its
    corresponding tool results.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages

    system = messages[:1]
    body = messages[1:]

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for message in body:
        group = _message_group(message)

        if group == "assistant_tool":
            if current:
                groups.append(current)
                current = []
            current.append(message)
        elif group == "tool":
            current.append(message)
        else:
            if current:
                groups.append(current)
                current = []
            current.append(message)

    if current:
        groups.append(current)

    selected: list[dict[str, Any]] = []
    count = 1

    for group in reversed(groups):
        if count + len(group) > MAX_HISTORY_MESSAGES:
            break
        selected[0:0] = group
        count += len(group)

    return system + selected


def call_ollama(
    host: str,
    model: str,
    messages: list[dict[str, Any]],
    num_ctx: int,
    temperature: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": trim_messages(messages),
        "tools": TOOL_SCHEMA,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": num_ctx,
        },
    }

    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        },
        method="POST",
    )

    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    done_reason = None

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message") or {}

                if isinstance(msg, dict):
                    piece = msg.get("content")
                    if piece:
                        content_parts.append(str(piece))

                    calls = msg.get("tool_calls")
                    if isinstance(calls, list):
                        tool_calls.extend(calls)

                if chunk.get("done"):
                    done_reason = chunk.get("done_reason")
                    break

    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(
            f"Ollama returned HTTP {exc.code}: {truncate(detail, 1000)}"
        )
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {host}: {exc.reason}"
        )
    except TimeoutError:
        raise OllamaError("Ollama request timed out.")
    except OSError as exc:
        raise OllamaError(f"Ollama connection error: {exc}")

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }

    if tool_calls:
        message["tool_calls"] = tool_calls

    if done_reason:
        message["_done_reason"] = done_reason

    return {"message": message}


# ============================================================================
# Text-tool fallback
# ============================================================================
def extract_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Recover explicit JSON tool calls emitted as ordinary assistant text."""
    if not content:
        return []

    candidates: list[str] = []

    for match in re.finditer(
        r"```(?:json)?\s*(.*?)```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        candidates.append(match.group(1).strip())

    candidates.append(content.strip())

    decoder = json.JSONDecoder()

    for start, char in enumerate(content):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(content[start:])
            candidates.append(json.dumps(obj, ensure_ascii=False))
        except json.JSONDecodeError:
            continue

    calls: list[dict[str, Any]] = []
    seen: set[str] = set()

    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        items = parsed if isinstance(parsed, list) else [parsed]

        for item in items:
            if not isinstance(item, dict):
                continue

            name = item.get("name") or item.get("tool")
            raw_args = (
                item.get("arguments")
                if "arguments" in item
                else item.get("args", item.get("parameters", {}))
            )

            if name not in TOOL_FUNCTIONS:
                continue

            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    continue

            if not isinstance(raw_args, dict):
                continue

            call = {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": raw_args,
                },
            }

            signature = json.dumps(
                call,
                sort_keys=True,
                ensure_ascii=False,
            )

            if signature not in seen:
                seen.add(signature)
                calls.append(call)

    return calls


# ============================================================================
# Agent
# ============================================================================
@dataclass
class AgentState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    completed_actions: set[str] = field(default_factory=set)
    denied_actions: set[str] = field(default_factory=set)
    recent_calls: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        host: str,
        model: str,
        workspace: Workspace,
        auto: bool,
        num_ctx: int = 8192,
        temperature: float = 0.1,
    ):
        self.host = host
        self.model = model
        self.workspace = workspace
        self.auto = auto
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.state = AgentState(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}]
        )

    def _tool_signature(self, name: str, args: dict[str, Any]) -> str:
        return json.dumps(
            {"name": name, "args": args},
            sort_keys=True,
            ensure_ascii=False,
        )

    def _print_tool(self, name: str, args: dict[str, Any]) -> None:
        labels = {
            "read_file": "Read",
            "write_file": "Write",
            "edit_file": "Edit",
            "list_dir": "List",
            "search_files": "Search",
            "run_command": "Run",
            "verify_python": "Verify",
            "scientific_check": "Science check",
        }

        label = labels.get(name, name)
        target = (
            args.get("path")
            or args.get("command")
            or args.get("query")
            or "."
        )
        print(
            f"  {Style.magenta('◆')} "
            f"{Style.bold(label)} {Style.dim(str(target))}"
        )

    def _print_result(self, name: str, result: dict[str, Any]) -> None:
        if result.get("ok"):
            print(
                f"    {Style.green('✓')} "
                f"{Style.dim(self._result_summary(name, result))}"
            )
        else:
            error = result.get("error") or "operation failed"
            print(f"    {Style.red('✗')} {Style.dim(str(error))}")

        if name == "run_command":
            stdout = result.get("stdout", "").strip()
            stderr = result.get("stderr", "").strip()

            if stdout:
                print(Style.dim("      stdout:"))
                for line in truncate(stdout, 4000).splitlines():
                    print(f"        {line}")

            if stderr:
                print(Style.yellow("      stderr:"))
                for line in truncate(stderr, 4000).splitlines():
                    print(f"        {line}")

            science = result.get("scientific_check")

            if isinstance(science, dict):
                level = science.get("level", "ok")
                symbol = (
                    Style.green("✓")
                    if level == "ok"
                    else Style.yellow("⚠")
                    if level == "warning"
                    else Style.red("✗")
                )

                print(
                    f"      {symbol} "
                    f"{Style.bold('science:')} {level}"
                )

                for item in science.get("checks", []):
                    print(f"        {Style.dim('•')} {item}")

                for finding in science.get("findings", []):
                    marker = (
                        Style.red("!")
                        if finding.get("level") == "error"
                        else Style.yellow("!")
                    )
                    print(
                        f"        {marker} "
                        f"{finding.get('message', 'Scientific check finding')}"
                    )

        if name in {"write_file", "edit_file"} and result.get("diff"):
            diff = str(result["diff"])
            print(Style.dim("      diff:"))

            for line in truncate(diff, 5000).splitlines():
                if line.startswith("+++") or line.startswith("---"):
                    print(Style.dim(f"        {line}"))
                elif line.startswith("+"):
                    print(Style.green(f"        {line}"))
                elif line.startswith("-"):
                    print(Style.red(f"        {line}"))
                else:
                    print(Style.dim(f"        {line}"))

    @staticmethod
    def _result_summary(name: str, result: dict[str, Any]) -> str:
        if name == "run_command":
            return (
                f"exit {result.get('returncode', '?')} • "
                f"{result.get('duration_s', '?')}s"
            )
        if name == "read_file":
            return f"read {result.get('path', 'file')}"
        if name == "write_file":
            return result.get("message", "write complete")
        if name == "edit_file":
            return result.get("message", "edit complete")
        if name == "verify_python":
            return result.get("message") or "verification passed"
        if name == "list_dir":
            return f"{len(result.get('entries', []))} entries"
        if name == "search_files":
            return f"{len(result.get('matches', []))} matches"
        if name == "scientific_check":
            return f"{result.get('level', 'ok')} • scientific sanity check"
        return "complete"

    def _parse_args(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw

        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid tool arguments JSON: {exc}")

            if not isinstance(parsed, dict):
                raise ValueError("Tool arguments must be a JSON object.")

            return parsed

        raise ValueError("Tool arguments have an unsupported type.")

    def _is_repeated(self, signature: str) -> bool:
        self.state.recent_calls.append(signature)
        self.state.recent_calls = self.state.recent_calls[-6:]

        return (
            len(self.state.recent_calls) >= 3
            and self.state.recent_calls[-1]
            == self.state.recent_calls[-2]
            == self.state.recent_calls[-3]
        )

    def _dedupe_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()

        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "").strip()

            if not name:
                continue

            try:
                args = self._parse_args(
                    function.get("arguments", {})
                )
                signature = self._tool_signature(name, args)
            except ValueError:
                signature = json.dumps(
                    {
                        "name": name,
                        "arguments": function.get("arguments", {}),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )

            if signature in seen:
                print(
                    f"    {Style.yellow('↷')} "
                    f"{Style.dim(f'duplicate {name} call removed')}"
                )
                continue

            seen.add(signature)
            unique.append(call)

        return unique

    def _should_global_dedupe(self, name: str) -> bool:
        # Reads/checks can legitimately be repeated after a file changes.
        return name in {
            "write_file",
            "edit_file",
            "run_command",
        }

    def run(self, task: str) -> str:
        self.state.messages.append(
            {"role": "user", "content": task}
        )
        log_event("task", {"task": task})

        for step in range(1, MAX_STEPS + 1):
            spinner = Spinner("Thinking")
            spinner.start()

            try:
                response = call_ollama(
                    self.host,
                    self.model,
                    self.state.messages,
                    self.num_ctx,
                    self.temperature,
                )
            except KeyboardInterrupt:
                spinner.stop()
                print(f"\n  {Style.yellow('!')} Interrupted.")
                return ""
            except OllamaError as exc:
                spinner.stop()
                print(f"  {Style.red('✗')} {exc}\n")
                return ""

            spinner.stop()

            message = response.get("message") or {}
            content = str(message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") or []

            fallback_used = False

            if not tool_calls and content:
                tool_calls = extract_text_tool_calls(content)
                fallback_used = bool(tool_calls)

            if tool_calls:
                tool_calls = self._dedupe_tool_calls(tool_calls)

            if not tool_calls:
                self.state.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
                log_event(
                    "final_answer",
                    {
                        "content": content,
                        "steps": step,
                    },
                )
                return content

            if fallback_used:
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                }

            self.state.messages.append(message)

            ctx = ToolContext(
                workspace=self.workspace,
                auto=self.auto,
                interactive=sys.stdin.isatty(),
            )

            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name")

                if name not in TOOL_FUNCTIONS:
                    result = {
                        "ok": False,
                        "error": f"Unknown tool: {name}",
                    }
                    name = str(name)
                    args = {}
                else:
                    try:
                        args = self._parse_args(
                            function.get("arguments", {})
                        )
                    except ValueError as exc:
                        args = {}
                        result = {
                            "ok": False,
                            "error": str(exc),
                        }
                    else:
                        signature = self._tool_signature(name, args)

                        if (
                            self._should_global_dedupe(name)
                            and signature in self.state.completed_actions
                        ):
                            result = {
                                "ok": True,
                                "skipped": True,
                                "message": (
                                    "Identical action already succeeded earlier "
                                    "in this task; skipped duplicate execution."
                                ),
                            }
                            print(
                                f"    {Style.yellow('↷')} "
                                f"{Style.dim('duplicate skipped')}"
                            )

                        elif signature in self.state.denied_actions:
                            result = {
                                "ok": False,
                                "skipped": True,
                                "error": (
                                    "This exact action was already declined by "
                                    "the user. Do not retry it."
                                ),
                            }
                            print(
                                f"    {Style.yellow('↷')} "
                                f"{Style.dim('previously declined; retry blocked')}"
                            )

                        elif self._is_repeated(signature):
                            result = {
                                "ok": False,
                                "error": (
                                    "Repeated identical tool call detected. "
                                    "Choose a different action or finish the task."
                                ),
                            }
                            print(
                                f"    {Style.yellow('↷')} "
                                f"{Style.dim('repeated call blocked')}"
                            )

                        else:
                            self._print_tool(name, args)

                            try:
                                result = TOOL_FUNCTIONS[name](ctx, args)
                            except Exception as exc:
                                result = {
                                    "ok": False,
                                    "error": (
                                        f"Tool crashed: "
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                }

                            self._print_result(name, result)

                            log_event(
                                "tool_call",
                                {
                                    "tool": name,
                                    "args": args,
                                    "result": result,
                                },
                            )

                            if (
                                result.get("ok")
                                and not result.get("skipped")
                                and self._should_global_dedupe(name)
                            ):
                                self.state.completed_actions.add(signature)

                            if (
                                not result.get("ok")
                                and "declined"
                                in str(result.get("error", "")).lower()
                            ):
                                self.state.denied_actions.add(signature)

                self.state.messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            result,
                            ensure_ascii=False,
                        ),
                    }
                )

        print(
            f"  {Style.yellow('!')} "
            f"Agent reached the {MAX_STEPS}-step limit."
        )
        return ""


# ============================================================================
# Slash commands
# ============================================================================
HELP = """
  /help                 Show commands
  /clear                Clear conversation
  /model [name]         Show or switch model
  /host [url]           Show or switch Ollama host
  /auto [on|off]        Show or toggle confirmations
  /status               Show CNEURO status
  /log [n]              Show recent log entries
  /quit                 Exit
  Ctrl-C                Interrupt the current prompt/task
"""


def tail_log(n: int = 10) -> None:
    if not os.path.exists(LOG_PATH):
        print(Style.dim("  No log yet.\n"))
        return

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
    except OSError as exc:
        print(Style.red(f"  Could not read log: {exc}\n"))
        return

    for line in lines:
        try:
            entry = json.loads(line)
            summary = json.dumps(
                entry.get("data", {}),
                ensure_ascii=False,
            )
            print(
                Style.dim(
                    f"  [{entry.get('ts', '?')}] "
                    f"{entry.get('type', '?')}: "
                    f"{truncate(summary, 140)}"
                )
            )
        except json.JSONDecodeError:
            print(Style.dim("  " + line.strip()))

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cneuro",
        description="CNEURO 3.2 — local Ollama coding/science agent",
    )

    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=os.environ.get("AGENT_AUTO", "0") == "1",
        help="Skip confirmations for file writes and commands",
    )
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--task", default=None)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.1)

    args = parser.parse_args()

    Style.enabled = not args.no_color and sys.stdout.isatty()

    try:
        workspace = Workspace(args.workspace)
    except Exception as exc:
        print(f"{Style.red('Error:')} {exc}")
        raise SystemExit(1)

    clear_screen()
    print_banner(args.model, workspace.root)

    agent = Agent(
        host=args.host,
        model=args.model,
        workspace=workspace,
        auto=args.auto,
        num_ctx=args.num_ctx,
        temperature=args.temperature,
    )

    if args.task:
        answer = agent.run(args.task)
        if answer:
            print_assistant(answer)
        return

    while True:
        try:
            task = print_user_prompt()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not task:
            continue

        if task.startswith("/"):
            parts = task[1:].split(maxsplit=1)
            command = parts[0].lower() if parts else ""
            value = parts[1].strip() if len(parts) > 1 else ""

            if command in {"quit", "exit"}:
                print()
                break

            if command == "help":
                print(HELP)
                continue

            if command == "clear":
                agent.state = AgentState(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT}
                    ]
                )
                clear_screen()
                print_banner(args.model, workspace.root)
                print(Style.dim("  Conversation cleared.\n"))
                continue

            if command == "model":
                if value:
                    args.model = value
                    agent.model = value
                    print(Style.dim(f"  Model → {value}\n"))
                else:
                    print(Style.dim(f"  Model: {agent.model}\n"))
                continue

            if command == "host":
                if value:
                    args.host = value
                    agent.host = value
                    print(Style.dim(f"  Host → {value}\n"))
                else:
                    print(Style.dim(f"  Host: {agent.host}\n"))
                continue

            if command == "auto":
                if value.lower() == "on":
                    agent.auto = True
                elif value.lower() == "off":
                    agent.auto = False

                print(
                    Style.dim(
                        f"  Auto mode: {'ON' if agent.auto else 'OFF'}\n"
                    )
                )
                continue

            if command == "status":
                print()
                print(f"  {Style.bold('CNEURO')} v{VERSION}")
                print(f"  {Style.dim('Model:')}     {agent.model}")
                print(f"  {Style.dim('Host:')}      {agent.host}")
                print(f"  {Style.dim('Workspace:')} {workspace.root}")
                print(
                    f"  {Style.dim('Auto:')}      "
                    f"{'ON' if agent.auto else 'OFF'}"
                )
                print(f"  {Style.dim('Context:')}   {agent.num_ctx}")
                print(f"  {Style.dim('Temperature:')} {agent.temperature}")
                print()
                continue

            if command == "log":
                n = int(value) if value.isdigit() else 10
                tail_log(max(1, min(n, 100)))
                continue

            print(
                Style.yellow(
                    f"  Unknown command /{command}. Try /help.\n"
                )
            )
            continue

        answer = agent.run(task)
        if answer:
            print_assistant(answer)


if __name__ == "__main__":
    main()
