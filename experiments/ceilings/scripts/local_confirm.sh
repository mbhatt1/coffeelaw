#!/usr/bin/env bash
# Throttled confirmation run: every arm and both sweeps at fixed easy difficulties, to check that
# the harness behaves as designed with a real model (message format followed, probabilities
# consistent, every run checkable). Run it inside the capped container started by
# scripts/watchdog.py. It is not a calibration: verdicts here carry no scientific weight.
set -uo pipefail

MODEL="${MODEL:-ollama/gemma3:1b}"
ROOT="${ROOT:-data/ollama-confirm}"
N="${N:-6}"
COOLDOWN="${COOLDOWN:-60}"
# Reasoning budgets need roughly 20+ tokens per chain step; `high` (320) covers d=4 comfortably.
export CEILINGS_REASONING_TOKENS="${CEILINGS_REASONING_TOKENS:-0,128,320,512,768}"
export CEILINGS_OLLAMA_DUTY="${CEILINGS_OLLAMA_DUTY:-0.5}"
export CEILINGS_OLLAMA_NUM_THREAD="${CEILINGS_OLLAMA_NUM_THREAD:-4}"
export CEILINGS_OLLAMA_KEEP_ALIVE="${CEILINGS_OLLAMA_KEEP_ALIVE:-60s}"

mkdir -p "$ROOT"
phase() { echo; echo "=== $(date +%H:%M:%S) $*"; }
cool() { echo "--- cooling down ${COOLDOWN}s"; sleep "$COOLDOWN"; }
c() { ceilings --root "$ROOT" "$@"; echo "(exit $?)"; }

phase "1/4 calibration gate at fixed difficulties (no cost matching)"
c prepare --model "$MODEL" --family info --difficulty k=3,p=0.0 --n-main "$N" --n-pilot "$N" --delta medium --no-match
c prepare --model "$MODEL" --family reason --difficulty d=4 --n-main "$N" --n-pilot "$N" --delta medium --no-match
c prepare --model "$MODEL" --family comm_parity,comm_ip --difficulty n=4 --n-main "$N" --n-pilot "$N" --delta medium --no-match
cool
phase "2/4 main grid: all 9 arms at medium delta"
c run --model "$MODEL" --delta medium
cool
phase "3/4 sweeps: tool calls (k=3, exact) and message bits (n=4)"
c sweep --model "$MODEL" --family info --k 3 --T 0,3 --flip-rate 0.0 --n 4
c sweep --model "$MODEL" --family comm --n-bits 4 --L 0,1,4 --n 4
phase "4/4 checks, analysis, audit"
c check --model "$MODEL" --out "$ROOT/reports"
c analyze --model "$MODEL" --out "$ROOT/reports" --n-boot 1000 --min-n 0 --min-models 1
c audit --model "$MODEL" --out "$ROOT/reports"
phase "done"
