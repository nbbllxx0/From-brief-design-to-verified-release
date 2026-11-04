#!/usr/bin/env bash
# X2 representation and schema-feedback arms (2026-11-01): 150 core Codex sessions, then 70 optional
# (GP repeat 2 on the 20-brief repeat subset; Claude Code GPC at effort high). Subscription CLIs only, no GPU.
# Finished runs are skipped, so the batch can be restarted. Log: results/X2/provenance_batch.log
cd "$(dirname "$0")/.." || exit 1
PY=~/anaconda3/envs/agentto/python.exe
LOG=results/X2/provenance_batch.log
SUB20="b01 b03 b04 b06 b07 b08 b09 b10 b19 b22 b25 b31 b39 b44 b45 b46 b47 b48 b49 b50"
echo "start $(date -Iseconds)" >> "$LOG"
for c in TS GP GPC; do
  $PY -B scripts/x2_run.py --cond "$c" >> "$LOG" 2>&1
  echo "done $c $(date -Iseconds)" >> "$LOG"
done
$PY -B scripts/x2_run.py --cond GP --rep 2 --briefs $SUB20 >> "$LOG" 2>&1
echo "done GP r2 $(date -Iseconds)" >> "$LOG"
$PY -B scripts/x2_run.py --cond GPC --provider claude --effort high >> "$LOG" 2>&1
echo "done claude GPC $(date -Iseconds)" >> "$LOG"
$PY -B scripts/x2_provenance.py >> "$LOG" 2>&1
echo "end $(date -Iseconds)" >> "$LOG"
