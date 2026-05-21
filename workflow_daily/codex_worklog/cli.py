from __future__ import annotations

import argparse
from pathlib import Path

from .rollup import write_daily_rollup
from .summarize import summarize_latest_session, summarize_session_file
from .watch import sync_once, watch_sessions


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codex-worklog",
        description="Generate daily worklogs from Codex session JSONL files.",
    )
    p.add_argument(
        "--repo-root",
        default=".",
        help="Repo root used as default output location (default: .).",
    )
    p.add_argument(
        "--sessions-dir",
        default=str(Path.home() / ".codex" / "sessions"),
        help="Codex sessions directory (default: ~/.codex/sessions).",
    )
    p.add_argument(
        "--out-dir",
        default="summaries",
        help="Output directory under --repo-root (default: summaries).",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("summarize", help="Summarize a specific session JSONL file.")
    s1.add_argument("session_file", help="Path to a Codex session .jsonl file.")
    s1.add_argument(
        "--no-rollup",
        action="store_true",
        help="Do not update the daily rollup after writing the session summary.",
    )

    sub.add_parser("summarize-latest", help="Summarize the most recently modified session file.")

    w = sub.add_parser("watch", help="Watch for new/updated sessions and write summaries.")
    w.add_argument("--poll-seconds", type=float, default=10.0)
    w.add_argument("--stable-seconds", type=float, default=30.0)

    y = sub.add_parser(
        "sync",
        help="One-shot update: write summaries for any new/changed sessions since last sync.",
    )
    y.add_argument("--stable-seconds", type=float, default=30.0)
    y.add_argument(
        "--all",
        action="store_true",
        help="Summarize all sessions (ignores prior sync state).",
    )

    r = sub.add_parser("rollup", help="Generate/refresh the daily rollup for a date.")
    r.add_argument("--date", required=False, help="Date like YYYY-MM-DD (default: today).")

    return p


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    sessions_dir = Path(args.sessions_dir).expanduser().resolve()
    out_dir = (repo_root / args.out_dir).resolve()

    if args.cmd == "summarize":
        session_file = Path(args.session_file).expanduser().resolve()
        result = summarize_session_file(
            session_file=session_file,
            out_dir=out_dir,
            sessions_dir=sessions_dir,
            update_rollup=not args.no_rollup,
        )
        print(result.written_markdown)
        return 0

    if args.cmd == "summarize-latest":
        result = summarize_latest_session(sessions_dir=sessions_dir, out_dir=out_dir)
        print(result.written_markdown)
        return 0

    if args.cmd == "watch":
        watch_sessions(
            sessions_dir=sessions_dir,
            out_dir=out_dir,
            repo_root=repo_root,
            poll_seconds=float(args.poll_seconds),
            stable_seconds=float(args.stable_seconds),
        )
        return 0

    if args.cmd == "sync":
        updated = sync_once(
            sessions_dir=sessions_dir,
            out_dir=out_dir,
            repo_root=repo_root,
            stable_seconds=float(args.stable_seconds),
            all_files=bool(args.all),
        )
        print(updated)
        return 0

    if args.cmd == "rollup":
        written = write_daily_rollup(out_dir=out_dir, date_str=args.date)
        print(written)
        return 0

    p.error(f"Unknown cmd: {args.cmd}")
    return 2
