#!/usr/bin/env bash
set -euo pipefail

LABEL="com.pchiniya.codex-worklog.watch"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

if [[ -f "${PLIST_PATH}" ]]; then
  launchctl unload "${PLIST_PATH}" >/dev/null 2>&1 || true
  rm -f "${PLIST_PATH}"
  echo "Uninstalled: ${PLIST_PATH}"
else
  echo "Not installed: ${PLIST_PATH}"
fi

