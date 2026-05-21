from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .parse_session import ParsedSession, PatchFileChange, parse_session
from .rollup import write_daily_rollup


@dataclass(frozen=True)
class SummarizeResult:
    session_file: Path
    date: str
    session_id: str
    written_markdown: str
    written_json: str


_SESSIONS_DATE_RE = re.compile(r"/sessions/(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})/")


def _derive_date_from_path(session_file: Path) -> str | None:
    m = _SESSIONS_DATE_RE.search(str(session_file))
    if not m:
        return None
    return f"{m.group('y')}-{m.group('m')}-{m.group('d')}"


def _first_user_prompt(messages: List[Tuple[str, str]]) -> str | None:
    for role, text in messages:
        if role != "user":
            continue
        t = text.strip()
        if not t:
            continue
        # Many Codex VSCode prompts include a structured block; prefer the explicit request section.
        req = _extract_codex_request(t)
        if req:
            return req
        if t.startswith("# AGENTS.md instructions"):
            continue
        if "<INSTRUCTIONS>" in t and "## Skills" in t:
            continue
        if t.startswith("# Context from my IDE setup"):
            continue
        return t
    return None


def _extract_codex_request(text: str) -> str | None:
    lines = [ln.rstrip() for ln in text.splitlines()]
    markers = {
        "## My request for Codex:",
        "# My request for Codex:",
        "My request for Codex:",
        "## My request for Codex",
        "# My request for Codex",
    }
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.strip() in markers:
            start_idx = i + 1
            break
        if "My request for Codex:" in ln:
            # Handle marker + inline request on the same line.
            after = ln.split("My request for Codex:", 1)[1].strip()
            if after:
                return after
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    for j in range(start_idx, len(lines)):
        candidate = lines[j].strip()
        if not candidate:
            continue
        # Skip obvious boilerplate headers.
        if candidate.startswith("## ") or candidate.startswith("# "):
            continue
        return candidate
    return None


def _short_title(text: str) -> str:
    first = text.strip().splitlines()[0].strip()
    # Clean common VSCode/Codex boilerplate that sometimes leaks into the first line.
    first = re.sub(r"\s*#\s*Terminal\s+\d+.*$", "", first).strip()
    first = re.sub(r"\s{2,}", " ", first).strip()
    if len(first) > 90:
        return first[:87] + "..."
    return first


def _key_actions(parsed: ParsedSession) -> List[str]:
    actions: List[str] = []

    if parsed.patch_files:
        changed = len({p.path for p in parsed.patch_files})
        actions.append(f"Edited {changed} file(s) via apply_patch.")

    failing_cmds = 0
    for call in parsed.exec_calls:
        if call.call_id and call.call_id in parsed.exec_outputs:
            out = parsed.exec_outputs[call.call_id]
            if out.exit_code not in (None, 0):
                failing_cmds += 1
    if failing_cmds:
        actions.append(f"Ran shell commands with {failing_cmds} non-zero exit(s).")
    elif parsed.exec_calls:
        actions.append(f"Ran {len(parsed.exec_calls)} shell command(s).")

    if parsed.task_completions:
        actions.append(f"Completed {len(parsed.task_completions)} Codex task(s).")

    return actions


def _format_patch_files(patch_files: List[PatchFileChange]) -> List[str]:
    by_path: Dict[str, PatchFileChange] = {}
    # If the same file appears multiple times, aggregate additions/deletions and keep last operation.
    for ch in patch_files:
        prev = by_path.get(ch.path)
        if prev is None:
            by_path[ch.path] = ch
        else:
            by_path[ch.path] = PatchFileChange(
                operation=ch.operation,
                path=ch.path,
                additions=prev.additions + ch.additions,
                deletions=prev.deletions + ch.deletions,
            )

    lines: List[str] = []
    for path, ch in sorted(by_path.items(), key=lambda kv: kv[0]):
        lines.append(f"- `{path}` ({ch.operation}) +{ch.additions} -{ch.deletions}")
    return lines


def _format_commands(parsed: ParsedSession) -> List[str]:
    lines: List[str] = []
    for call in parsed.exec_calls[:40]:
        code = None
        if call.call_id and call.call_id in parsed.exec_outputs:
            code = parsed.exec_outputs[call.call_id].exit_code
        suffix = "" if code is None else f" (exit {code})"
        lines.append(f"- `{call.cmd}`{suffix}")
    return lines


def _extract_chat_highlights(parsed: ParsedSession) -> Tuple[List[str], List[str]]:
    users: List[str] = []
    assistants: List[str] = []
    for m in parsed.messages:
        if m.role == "user":
            t = m.text.strip()
            if t.startswith("# AGENTS.md instructions"):
                continue
            if "<INSTRUCTIONS>" in t and "## Skills" in t:
                continue
            users.append(t)
        elif m.role == "assistant":
            # Prefer final answers, but keep something even if phase is missing.
            assistants.append(m.text.strip())

    # Prefer final_answer messages when present.
    final_answers = [m.text.strip() for m in parsed.messages if m.role == "assistant" and m.phase == "final_answer"]
    if final_answers:
        assistants = final_answers

    def clip(s: str, n: int = 500) -> str:
        s = s.strip()
        return s if len(s) <= n else (s[: n - 3] + "...")

    return [clip(x) for x in users[:6]], [clip(x) for x in assistants[:6]]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summarize_session_file(
    *,
    session_file: Path,
    out_dir: Path,
    sessions_dir: Path,
    update_rollup: bool = True,
) -> SummarizeResult:
    parsed = parse_session(session_file)
    session_id = parsed.session_meta.session_id if parsed.session_meta else session_file.stem
    date_str = _derive_date_from_path(session_file) or "unknown-date"

    day_dir = out_dir / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    md_path = day_dir / f"session-{session_id}.md"
    js_path = day_dir / f"session-{session_id}.json"

    user_prompts, assistant_outcomes = _extract_chat_highlights(parsed)
    first_prompt = _first_user_prompt([(m.role, m.text) for m in parsed.messages]) or f"Session {session_id}"
    title = _short_title(first_prompt)

    key_actions = _key_actions(parsed)

    lines: List[str] = []
    lines.append(f"# Codex Worklog — {date_str}")
    lines.append("")
    lines.append(f"**Title:** {title}")
    lines.append("")
    lines.append("## Session")
    lines.append(f"- Session ID: `{session_id}`")
    lines.append(f"- Session file: `{session_file}`")
    if parsed.session_meta and parsed.session_meta.cwd:
        lines.append(f"- CWD: `{parsed.session_meta.cwd}`")
    if parsed.session_meta and parsed.session_meta.source:
        lines.append(f"- Source: `{parsed.session_meta.source}`")
    if parsed.session_meta and parsed.session_meta.cli_version:
        lines.append(f"- CLI: `{parsed.session_meta.cli_version}`")
    lines.append("")

    lines.append("## Summary")
    if key_actions:
        for act in key_actions:
            lines.append(f"- {act}")
    else:
        lines.append("- (No actions detected; session may be mostly conversational.)")
    lines.append("")

    if parsed.patch_files:
        lines.append("## Changes")
        lines.extend(_format_patch_files(parsed.patch_files))
        lines.append("")

    if parsed.exec_calls:
        lines.append("## Commands")
        lines.extend(_format_commands(parsed))
        lines.append("")

    if user_prompts:
        lines.append("## Chat Highlights")
        lines.append("### User")
        for i, t in enumerate(user_prompts, 1):
            lines.append(f"{i}. {t}")
        lines.append("")

    if assistant_outcomes:
        lines.append("### Assistant")
        for i, t in enumerate(assistant_outcomes, 1):
            lines.append(f"{i}. {t}")
        lines.append("")

    md_text = "\n".join(lines).rstrip() + "\n"

    # Only write if changed
    previous = md_path.read_text(encoding="utf-8") if md_path.exists() else None
    if previous != md_text:
        md_path.write_text(md_text, encoding="utf-8")

    summary_obj = {
        "date": date_str,
        "session_id": session_id,
        "session_file": str(session_file),
        "markdown_path": str(md_path),
        "json_path": str(js_path),
        "title": title,
        "key_actions": key_actions,
        "files_changed": [
            {"path": ch.path, "operation": ch.operation, "additions": ch.additions, "deletions": ch.deletions}
            for ch in parsed.patch_files
        ],
        "commands": [
            {
                "cmd": c.cmd,
                "exit_code": (parsed.exec_outputs.get(c.call_id).exit_code if c.call_id and c.call_id in parsed.exec_outputs else None),
            }
            for c in parsed.exec_calls
        ],
        "task_completions": [
            {"turn_id": tid, "timestamp": ts, "last_agent_message": msg}
            for (tid, ts, msg) in parsed.task_completions
        ],
        "content_hash": _hash_text(md_text),
    }
    js_path.write_text(json.dumps(summary_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if update_rollup:
        write_daily_rollup(out_dir=out_dir, date_str=date_str)

    return SummarizeResult(
        session_file=session_file,
        date=date_str,
        session_id=session_id,
        written_markdown=str(md_path),
        written_json=str(js_path),
    )


def _iter_session_files(sessions_dir: Path) -> List[Path]:
    files: List[Path] = []
    for root, _dirs, filenames in os.walk(sessions_dir):
        for fn in filenames:
            if not fn.endswith(".jsonl"):
                continue
            files.append(Path(root) / fn)
    return files


def summarize_latest_session(*, sessions_dir: Path, out_dir: Path) -> SummarizeResult:
    files = _iter_session_files(sessions_dir)
    if not files:
        raise SystemExit(f"No session files found under {sessions_dir}")
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return summarize_session_file(session_file=latest, out_dir=out_dir, sessions_dir=sessions_dir, update_rollup=True)
