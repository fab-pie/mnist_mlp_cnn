#!/usr/bin/env bash
set -euo pipefail
# Simple wrapper: run a command, tee output to run_last.log
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run_last.log"

if [ "$#" -lt 1 ]; then
  echo "Usage: run_and_log.sh <command...>"
  exit 1
fi

cmd=("$@")
echo "Running: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]:-0}
exit $rc
