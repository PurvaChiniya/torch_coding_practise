from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SessionSummaryIndex:
    session_id: str
    date: str
    markdown_path: str
    json_path: str
    title: str
    key_actions: List[str]


def _load_session_summary_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_daily_rollup(out_dir: Path, date_str: str | None) -> str:
    if date_str is None:
        # Default: pick latest date folder if present; else today's date.
        # (Avoid importing datetime/timezone; output is best-effort.)
        date_dirs = sorted([p for p in out_dir.glob("*") if p.is_dir()])
        if date_dirs:
            date_str = date_dirs[-1].name
        else:
            from datetime import date

            date_str = date.today().isoformat()

    day_dir = out_dir / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    daily_md = day_dir / "daily.md"

    entries: List[SessionSummaryIndex] = []
    for js in sorted(day_dir.glob("session-*.json")):
        obj = _load_session_summary_json(js)
        if not obj:
            continue
        entries.append(
            SessionSummaryIndex(
                session_id=str(obj.get("session_id") or ""),
                date=str(obj.get("date") or date_str),
                markdown_path=str(obj.get("markdown_path") or ""),
                json_path=str(obj.get("json_path") or ""),
                title=str(obj.get("title") or js.stem),
                key_actions=list(obj.get("key_actions") or []),
            )
        )

    lines: List[str] = []
    lines.append(f"# Codex Daily Worklog — {date_str}")
    lines.append("")
    if not entries:
        lines.append("_No session summaries found yet for this day._")
        lines.append("")
        daily_md.write_text("\n".join(lines), encoding="utf-8")
        return str(daily_md)

    lines.append("## Sessions")
    for e in entries:
        md_name = Path(e.markdown_path).name if e.markdown_path else f"session-{e.session_id}.md"
        lines.append(f"- [{e.title}]({md_name})")
        for act in e.key_actions[:6]:
            lines.append(f"  - {act}")
    lines.append("")

    daily_md.write_text("\n".join(lines), encoding="utf-8")
    return str(daily_md)

