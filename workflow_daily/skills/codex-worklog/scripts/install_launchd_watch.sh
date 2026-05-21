#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=""
FORCE=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="1"
fi
if [[ "${1:-}" == "--force" ]]; then
  FORCE="1"
fi
if [[ "${2:-}" == "--force" ]]; then
  FORCE="1"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LABEL="com.pchiniya.codex-worklog.watch"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${LABEL}.plist"

OUT_LOG="${REPO_ROOT}/.state/launchd.watch.out.log"
ERR_LOG="${REPO_ROOT}/.state/launchd.watch.err.log"

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python3 not found on PATH; required to run the watcher."
  exit 1
fi

mkdir -p "${PLIST_DIR}"
mkdir -p "${REPO_ROOT}/.state"

if [[ -z "${FORCE}" && "${REPO_ROOT}" == "${HOME}/Desktop/"* ]]; then
  echo "ERROR: Repo is under ~/Desktop, and launchd often cannot execute scripts there (macOS privacy/TCC)."
  echo "Move this repo out of Desktop (e.g. ~/workspace/workflow_daily) and re-run this installer,"
  echo "or run the watcher manually: ./bin/codex-worklog watch"
  echo ""
  echo "If you already installed and see 'Operation not permitted' in:"
  echo "  ${REPO_ROOT}/.state/launchd.watch.err.log"
  echo "this is the same issue."
  echo ""
  echo "Override (not recommended): re-run with --force"
  exit 2
fi

if [[ -n "${DRY_RUN}" ]]; then
  echo "Would install: ${PLIST_PATH}"
  echo "Would run watcher in: ${REPO_ROOT}"
  exit 0
fi

cat > "${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>${LABEL}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>60</integer>
    <key>WorkingDirectory</key><string>${REPO_ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>PYTHONUNBUFFERED</key><string>1</string>
    </dict>
    <key>ProgramArguments</key>
    <array>
      <string>${PYTHON_BIN}</string>
      <string>-m</string>
      <string>codex_worklog</string>
      <string>--repo-root</string>
      <string>${REPO_ROOT}</string>
      <string>watch</string>
    </array>
    <key>StandardOutPath</key><string>${OUT_LOG}</string>
    <key>StandardErrorPath</key><string>${ERR_LOG}</string>
  </dict>
</plist>
PLIST

launchctl unload "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl load "${PLIST_PATH}"

echo "Installed and loaded: ${PLIST_PATH}"
echo "Logs: ${OUT_LOG}"
echo "      ${ERR_LOG}"
