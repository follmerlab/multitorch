#!/usr/bin/env bash
# Monitor the v0 Fe XAS fit sweep running on this host.
#
# Usage (on exxa, or anywhere the sweep is running):
#   bash bench/monitor_v0_sweep.sh           # one-shot snapshot
#   bash bench/monitor_v0_sweep.sh --watch   # auto-refresh every 30 s
#
# Reads from /home/afollmer/code/multiplets/fits/results/v0_sweep.log
# (override with $V0_SWEEP_LOG env var).
#
# Output:
#   - sweep PID + elapsed time (or "no process running")
#   - which spectrum is in progress
#   - per-spectrum walls + losses for completed spectra
#   - ETA (rough, based on completed-spectrum average)
#   - last 5 log lines for context

set -euo pipefail

LOG="${V0_SWEEP_LOG:-/home/afollmer/code/multiplets/fits/results/v0_sweep.log}"
RESULTS_DIR="$(dirname "$LOG")"
WATCH=0
[[ "${1:-}" == "--watch" ]] && WATCH=1

show_status() {
    clear 2>/dev/null || true
    echo "═══ v0 Fe XAS sweep — $(date '+%H:%M:%S') ═══"
    echo

    if [[ ! -f "$LOG" ]]; then
        echo "No log at $LOG (sweep not started?)"
        return
    fi

    PID=$(pgrep -f 'fit_and_plot_all.py' || true)
    if [[ -n "$PID" ]]; then
        STARTED=$(ps -o lstart= -p "$PID" 2>/dev/null | xargs)
        ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | xargs)
        echo "▶ Running (PID $PID, started $STARTED, elapsed $ELAPSED)"
    else
        echo "■ Not running"
        if grep -q "TOTAL" "$LOG" 2>/dev/null; then
            echo "  → completed (found TOTAL in log)"
        elif grep -q "Traceback\|Error" "$LOG" 2>/dev/null; then
            echo "  ✗ exited with error"
        else
            echo "  ? exited without TOTAL marker — check tail of log"
        fi
    fi
    echo

    # Total spectra count
    TOTAL=$(ls "$RESULTS_DIR"/../../test-data/Fe*.txt 2>/dev/null | wc -l | tr -d ' ')
    [[ -z "$TOTAL" ]] && TOTAL="?"

    # Completed spectra (lines containing "wall=" in log)
    COMPLETED=$(grep -c "wall=" "$LOG" 2>/dev/null || echo 0)
    echo "Progress: $COMPLETED / $TOTAL spectra fit"

    # Currently fitting (last "===" line w/o a corresponding "wall=" after)
    LAST_HEAD=$(grep -n "^===" "$LOG" 2>/dev/null | tail -1 | cut -d: -f1)
    if [[ -n "$LAST_HEAD" ]]; then
        TAIL_LINES=$(tail -n +"$LAST_HEAD" "$LOG" | head -5)
        if echo "$TAIL_LINES" | grep -q "wall="; then
            : # most recent spectrum done
        else
            CURRENT=$(echo "$TAIL_LINES" | head -1 | sed 's/^=== \(.*\) ===$/\1/')
            echo "Currently fitting: $CURRENT"
        fi
    fi
    echo

    if [[ $COMPLETED -gt 0 ]]; then
        echo "Completed spectra:"
        grep -B1 "wall=" "$LOG" | grep -E "^===|wall=" | \
            paste -d ' ' - - | \
            sed 's/^=== \(.*\) ===  /  \1: /;s/  loss=/ loss=/;s/  rmse=/ rmse=/'
        echo

        # ETA
        if [[ $COMPLETED -lt ${TOTAL:-99} ]] && [[ -n "$PID" ]]; then
            AVG_SEC=$(grep "wall=" "$LOG" | awk -F'wall=' '{print $2}' | \
                awk '{sum += $1; count++} END {if (count > 0) printf "%.1f", sum/count}')
            REMAINING=$((${TOTAL:-0} - COMPLETED))
            if [[ -n "$AVG_SEC" ]] && [[ $REMAINING -gt 0 ]]; then
                ETA_SEC=$(awk "BEGIN {printf \"%.0f\", $AVG_SEC * $REMAINING}")
                ETA_MIN=$((ETA_SEC / 60))
                echo "ETA: ~${ETA_MIN} min ($REMAINING spectra × ${AVG_SEC}s avg)"
                echo
            fi
        fi
    fi

    echo "Last 5 log lines:"
    tail -5 "$LOG" | sed 's/^/  /'
    echo

    # Result files
    if ls "$RESULTS_DIR"/v0_summary.* >/dev/null 2>&1; then
        echo "Results landed:"
        ls -la "$RESULTS_DIR"/v0_summary.* "$RESULTS_DIR"/v0_runtime.* 2>/dev/null | \
            awk '{print "  "$NF" ("$5" bytes)"}'
    fi
}

if [[ $WATCH -eq 1 ]]; then
    while true; do
        show_status
        echo
        echo "── Refreshing in 30s — Ctrl-C to stop ──"
        sleep 30
    done
else
    show_status
fi
