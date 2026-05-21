---
name: codex-worklog
description: >
  Generate a technical worklog (Markdown + JSON) from Codex session logs stored under
  ~/.codex/sessions/*.jsonl, and keep a per-day rollup (daily.md). Use when you want an
  end-of-chat or end-of-day summary of what was done in Codex (commands run, files edited
  via apply_patch, key chat highlights). Also use to set up an optional background
  watcher/launchd job that writes summaries automatically.
---

# Codex Worklog

## Overview

Turn Codex session JSONL logs into per-session and per-day technical worklogs inside this repo.

## Workflow

### 1) Summarize the most recent Codex session

From the repo root, run:

```bash
./bin/codex-worklog summarize-latest
```

This writes:

`summaries/YYYY-MM-DD/session-<session_id>.md` and `summaries/YYYY-MM-DD/session-<session_id>.json`, and refreshes `summaries/YYYY-MM-DD/daily.md`.

### 2) Summarize a specific session file

```bash
./bin/codex-worklog summarize /path/to/rollout-....jsonl
```

### 3) Run the watcher (end-of-chat automation)

Start a polling watcher that regenerates summaries when session files stop changing:

```bash
./bin/codex-worklog watch
```

If you want it to start automatically on macOS login, use:

```bash
./skills/codex-worklog/scripts/install_launchd_watch.sh
```

To remove it later:

```bash
./skills/codex-worklog/scripts/uninstall_launchd_watch.sh
```
