from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    timestamp: str | None
    cwd: str | None
    source: str | None
    originator: str | None
    cli_version: str | None
    model_provider: str | None


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    phase: str | None


@dataclass(frozen=True)
class ExecCommandCall:
    call_id: str | None
    cmd: str


@dataclass(frozen=True)
class ExecCommandOutput:
    call_id: str
    raw: str
    exit_code: int | None


@dataclass(frozen=True)
class ApplyPatchCall:
    patch_text: str


@dataclass(frozen=True)
class PatchFileChange:
    operation: str  # Add|Update|Delete
    path: str
    additions: int
    deletions: int


@dataclass(frozen=True)
class ParsedSession:
    session_meta: SessionMeta | None
    messages: List[Message]
    exec_calls: List[ExecCommandCall]
    exec_outputs: Dict[str, ExecCommandOutput]
    patches: List[ApplyPatchCall]
    patch_files: List[PatchFileChange]
    task_completions: List[Tuple[str | None, str | None, str | None]]  # (turn_id, timestamp, last_agent_message)


def _safe_json_loads(line: str) -> Dict[str, Any] | None:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _extract_text_from_content(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts).strip()


_EXIT_CODE_RE = re.compile(r"Process exited with code (?P<code>-?\d+)")


def _parse_exit_code(raw_output: str) -> int | None:
    m = _EXIT_CODE_RE.search(raw_output)
    if not m:
        return None
    try:
        return int(m.group("code"))
    except Exception:
        return None


_PATCH_FILE_RE = re.compile(r"^\\*\\*\\*\\s+(Add|Update|Delete) File:\\s+(?P<path>.+?)\\s*$")


def _parse_patch_file_changes(patch_text: str) -> List[PatchFileChange]:
    changes: List[PatchFileChange] = []
    current: Tuple[str, str] | None = None
    add = 0
    delete = 0

    def flush() -> None:
        nonlocal current, add, delete
        if current is None:
            return
        op, path = current
        changes.append(PatchFileChange(operation=op, path=path, additions=add, deletions=delete))
        current = None
        add = 0
        delete = 0

    for raw_line in patch_text.splitlines():
        m = _PATCH_FILE_RE.match(raw_line)
        if m:
            flush()
            op = m.group(1)
            path = m.group("path")
            current = (op, path)
            continue

        if current is None:
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            add += 1
        elif raw_line.startswith("-") and not raw_line.startswith("---"):
            delete += 1

    flush()
    return changes


def iter_session_events(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _safe_json_loads(line)
            if obj is None:
                continue
            yield obj


def parse_session(path: Path) -> ParsedSession:
    meta: SessionMeta | None = None
    messages: List[Message] = []
    exec_calls: List[ExecCommandCall] = []
    exec_outputs: Dict[str, ExecCommandOutput] = {}
    patches: List[ApplyPatchCall] = []
    patch_files: List[PatchFileChange] = []
    task_completions: List[Tuple[str | None, str | None, str | None]] = []

    for ev in iter_session_events(path):
        ev_type = ev.get("type")
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else None

        if ev_type == "session_meta" and payload:
            sid = payload.get("id")
            if isinstance(sid, str):
                meta = SessionMeta(
                    session_id=sid,
                    timestamp=payload.get("timestamp") if isinstance(payload.get("timestamp"), str) else None,
                    cwd=payload.get("cwd") if isinstance(payload.get("cwd"), str) else None,
                    source=payload.get("source") if isinstance(payload.get("source"), str) else None,
                    originator=payload.get("originator") if isinstance(payload.get("originator"), str) else None,
                    cli_version=payload.get("cli_version") if isinstance(payload.get("cli_version"), str) else None,
                    model_provider=payload.get("model_provider")
                    if isinstance(payload.get("model_provider"), str)
                    else None,
                )
            continue

        if ev_type == "event_msg" and payload:
            if payload.get("type") == "task_complete":
                turn_id = payload.get("turn_id") if isinstance(payload.get("turn_id"), str) else None
                last_msg = (
                    payload.get("last_agent_message")
                    if isinstance(payload.get("last_agent_message"), str)
                    else None
                )
                ts = ev.get("timestamp") if isinstance(ev.get("timestamp"), str) else None
                task_completions.append((turn_id, ts, last_msg))
            continue

        if ev_type != "response_item" or not payload:
            continue

        ptype = payload.get("type")

        if ptype == "message":
            role = payload.get("role")
            if not isinstance(role, str):
                continue
            if role not in {"user", "assistant"}:
                continue
            content = payload.get("content")
            text = _extract_text_from_content(content)
            if not text:
                continue
            phase = payload.get("phase") if isinstance(payload.get("phase"), str) else None
            messages.append(Message(role=role, text=text, phase=phase))
            continue

        if ptype == "function_call":
            name = payload.get("name")
            if name != "exec_command":
                continue
            call_id = payload.get("call_id") if isinstance(payload.get("call_id"), str) else None
            raw_args = payload.get("arguments")
            cmd = None
            if isinstance(raw_args, str):
                try:
                    args_obj = json.loads(raw_args)
                    cmd = args_obj.get("cmd")
                except Exception:
                    cmd = None
            if isinstance(cmd, str) and cmd.strip():
                exec_calls.append(ExecCommandCall(call_id=call_id, cmd=cmd.strip()))
            continue

        if ptype == "function_call_output":
            call_id = payload.get("call_id")
            out = payload.get("output")
            if not (isinstance(call_id, str) and isinstance(out, str)):
                continue
            exec_outputs[call_id] = ExecCommandOutput(
                call_id=call_id,
                raw=out,
                exit_code=_parse_exit_code(out),
            )
            continue

        if ptype == "custom_tool_call":
            name = payload.get("name")
            if name != "apply_patch":
                continue
            patch_text = payload.get("input")
            if not isinstance(patch_text, str) or not patch_text.strip():
                continue
            patches.append(ApplyPatchCall(patch_text=patch_text))
            patch_files.extend(_parse_patch_file_changes(patch_text))
            continue

    return ParsedSession(
        session_meta=meta,
        messages=messages,
        exec_calls=exec_calls,
        exec_outputs=exec_outputs,
        patches=patches,
        patch_files=patch_files,
        task_completions=task_completions,
    )

