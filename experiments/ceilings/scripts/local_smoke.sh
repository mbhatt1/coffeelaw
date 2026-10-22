#!/usr/bin/env bash
# Throttled small-model smoke run of the whole pipeline. Meant to run inside the capped
# container started by scripts/watchdog.py; every model call goes to the host's Ollama server.
#
# Throttling: one team run at a time (no --concurrency), short reasoning budgets, Ollama busy at
# most CEILINGS_OLLAMA_DUTY of the time, a cool-down pause between phases, and a small n.
set -uo pipefail

MODEL="${MODEL:-ollama/gemma3:1b}"
ROOT="${ROOT:-data/ollama-smoke}"
COOLDOWN="${COOLDOWN:-60}"
export CEILINGS_REASONING_TOKENS="${CEILINGS_REASONING_TOKENS:-0,48,96,160,256}"
export CEILINGS_OLLAMA_DUTY="${CEILINGS_OLLAMA_DUTY:-0.5}"
export CEILINGS_OLLAMA_NUM_THREAD="${CEILINGS_OLLAMA_NUM_THREAD:-4}"
export CEILINGS_OLLAMA_KEEP_ALIVE="${CEILINGS_OLLAMA_KEEP_ALIVE:-60s}"

mkdir -p "$ROOT"
phase() { echo; echo "=== $(date +%H:%M:%S) $*"; }
cool() { echo "--- cooling down ${COOLDOWN}s"; sleep "$COOLDOWN"; }
c() { ceilings --root "$ROOT" "$@"; echo "(exit $?)"; }

phase "1/6 difficulty search (quick grid, n=6 per setting)"
c calibrate --model "$MODEL" --grid quick --n 6
cool
phase "2/6 calibration gate on the chosen settings (no cost matching)"
c prepare --model "$MODEL" --n-main 8 --n-pilot 6 --delta medium --no-match
cool
phase "3/6 main grid: baseline, E, K, M, A1, P2 at medium delta"
c run --model "$MODEL" --delta medium --arms baseline,E,K,M,A1,P2
cool
phase "4/6 sweeps: tool calls (k=3, exact probe) and message bits (n=4)"
c sweep --model "$MODEL" --family info --k 3 --T 0,1,3 --flip-rate 0.0 --n 4
c sweep --model "$MODEL" --family comm --n-bits 4 --L 0,1,2,4 --n 4
phase "5/6 checks, analysis, audit"
c check --model "$MODEL" --out "$ROOT/reports"
c analyze --model "$MODEL" --out "$ROOT/reports" --n-boot 1000
c audit --model "$MODEL" --out "$ROOT/reports"
phase "6/6 done"
