from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .summarize import summarize_session_file


@dataclass
class WatchedFileState:
    mtime_ns: int
    size: int
    last_written_hash: str | None


def _state_path(repo_root: Path) -> Path:
    d = repo_root / ".state"
    d.mkdir(parents=True, exist_ok=True)
    return d / "watch_state.json"


def _load_state(repo_root: Path) -> Dict[str, WatchedFileState]:
    sp = _state_path(repo_root)
    if not sp.exists():
        return {}
    try:
        raw = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, WatchedFileState] = {}
    files = raw.get("files")
    if not isinstance(files, dict):
        return {}
    for path, v in files.items():
        if not isinstance(path, str) or not isinstance(v, dict):
            continue
        try:
            out[path] = WatchedFileState(
                mtime_ns=int(v.get("mtime_ns", 0)),
                size=int(v.get("size", 0)),
                last_written_hash=v.get("last_written_hash") if isinstance(v.get("last_written_hash"), str) else None,
            )
        except Exception:
            continue
    return out


def _save_state(repo_root: Path, state: Dict[str, WatchedFileState]) -> None:
    sp = _state_path(repo_root)
    payload = {
        "files": {
            path: {"mtime_ns": st.mtime_ns, "size": st.size, "last_written_hash": st.last_written_hash}
            for path, st in state.items()
        }
    }
    sp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_session_files(sessions_dir: Path) -> list[Path]:
    out: list[Path] = []
    for root, _dirs, filenames in __import__("os").walk(sessions_dir):
        for fn in filenames:
            if fn.endswith(".jsonl"):
                out.append(Path(root) / fn)
    return out


def sync_once(
    *,
    sessions_dir: Path,
    out_dir: Path,
    repo_root: Path,
    stable_seconds: float,
    all_files: bool = False,
) -> str:
    """
    One-shot sync: summarize any new/changed session files since last run.

    Returns a short human-readable status string.
    """
    state = _load_state(repo_root)
    now = time.time()
    updated: List[str] = []
    pending_unstable = 0
    min_seconds_until_stable: float | None = None
    changed_state = False

    for path in sorted(_iter_session_files(sessions_dir)):
        try:
            st = path.stat()
        except FileNotFoundError:
            continue

        key = str(path)
        prev = state.get(key)
        is_new_or_changed = all_files or prev is None or prev.mtime_ns != st.st_mtime_ns or prev.size != st.st_size
        if not is_new_or_changed:
            continue

        if stable_seconds > 0 and (now - st.st_mtime) < stable_seconds:
            pending_unstable += 1
            remaining = stable_seconds - (now - st.st_mtime)
            if min_seconds_until_stable is None or remaining < min_seconds_until_stable:
                min_seconds_until_stable = remaining
            continue

        try:
            res = summarize_session_file(
                session_file=path,
                out_dir=out_dir,
                sessions_dir=sessions_dir,
                update_rollup=True,
            )
            updated.append(res.written_markdown)
        except Exception as exc:
            updated.append(f"ERROR: {path} ({exc})")

        state[key] = WatchedFileState(mtime_ns=st.st_mtime_ns, size=st.st_size, last_written_hash=None)
        changed_state = True

    if changed_state:
        _save_state(repo_root, state)

    if not updated:
        if pending_unstable:
            secs = 0 if min_seconds_until_stable is None else max(0, int(round(min_seconds_until_stable)))
            return f"[codex-worklog] sync: waiting for {pending_unstable} session(s) to stabilize (~{secs}s)"
        return "[codex-worklog] sync: no changes"

    lines = ["[codex-worklog] sync: updated summaries:"]
    lines.extend([f"- {p}" for p in updated])
    return "\n".join(lines)


def watch_sessions(
    *,
    sessions_dir: Path,
    out_dir: Path,
    repo_root: Path,
    poll_seconds: float,
    stable_seconds: float,
) -> None:
    state = _load_state(repo_root)
    print(f"[codex-worklog] Watching {sessions_dir} (poll={poll_seconds}s, stable={stable_seconds}s)")
    print(f"[codex-worklog] Writing summaries to {out_dir}")

    while True:
        # Reuse the same logic as sync, but keep state in-memory for efficiency.
        now = time.time()
        changed_state = False

        for path in sorted(_iter_session_files(sessions_dir)):
            try:
                st = path.stat()
            except FileNotFoundError:
                continue

            key = str(path)
            prev = state.get(key)
            is_new_or_changed = prev is None or prev.mtime_ns != st.st_mtime_ns or prev.size != st.st_size
            if not is_new_or_changed:
                continue

            if stable_seconds > 0 and (now - st.st_mtime) < stable_seconds:
                continue

            try:
                res = summarize_session_file(
                    session_file=path,
                    out_dir=out_dir,
                    sessions_dir=sessions_dir,
                    update_rollup=True,
                )
                print(f"[codex-worklog] Updated {res.written_markdown}")
            except Exception as exc:
                print(f"[codex-worklog] ERROR summarizing {path}: {exc}")

            state[key] = WatchedFileState(mtime_ns=st.st_mtime_ns, size=st.st_size, last_written_hash=None)
            changed_state = True

        if changed_state:
            _save_state(repo_root, state)

        time.sleep(poll_seconds)
