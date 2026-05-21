# workflow_daily

Automates technical daily worklogs from your Codex chats.

## What it does

- Reads Codex session logs from `~/.codex/sessions/**/*.jsonl`
- Writes per-session Markdown summaries into `summaries/YYYY-MM-DD/`
- Keeps an automatically-updated `summaries/YYYY-MM-DD/daily.md` rollup
- Optional watcher mode to regenerate summaries when a chat finishes

## Quick start

Generate a summary for the most recent Codex session:

```bash
./bin/codex-worklog summarize-latest
```

One-shot “update anything new/changed” (recommended for manual use):

```bash
./bin/codex-worklog sync
```

Run the watcher (recommended: start once in the morning; it updates as sessions finish):

```bash
./bin/codex-worklog watch
```

## Automation (macOS)

If you want the watcher to run automatically on login:

```bash
./skills/codex-worklog/scripts/install_launchd_watch.sh
```

If `workflow_daily` lives under `~/Desktop`, macOS may block the launchd job with “Operation not permitted”; move the repo somewhere else (for example `~/workspace/workflow_daily`) and re-run the installer.

To uninstall:

```bash
./skills/codex-worklog/scripts/uninstall_launchd_watch.sh
```

Generate/refresh the rollup for a specific date:

```bash
./bin/codex-worklog rollup --date 2026-03-04
```

## Output layout

```
summaries/
  2026-03-04/
    session-<session_id>.md
    session-<session_id>.json
    daily.md
```

## Notes

- This repo intentionally uses only the Python standard library (no dependencies).
- Session timestamps are stored in UTC by Codex; folder dates are derived from the session path when possible.
