#!/usr/bin/env bash
# Generalized recovery monitor: the fleet's launch-ssh reliably hangs the local orchestrator
# (a gcloud + backgrounded-remote quirk), but the search runs detached on the VMs and is
# unaffected. This decoupled monitor polls each shard VM to completion, pulls its shard JSON
# + run.log, tears the VM down, then does the full merge. Pair it with graf_search_fleet.sh
# (whose own poll/teardown may hang).
#
# Usage: graf_search_recover.sh <vm_prefix> <nshards> <results_dir> <space> [zone]
#   e.g. graf_search_recover.sh exp-20260615-093426-graf-single 8 <rdir> single
set -uo pipefail
export CLOUDSDK_CORE_PROJECT=cr-memmingpark
PREFIX="$1"; NSH="$2"; RDIR="$3"; SPACE="$4"; ZONE="${5:-europe-west1-b}"
SA=claude-experiments@cr-memmingpark.iam.gserviceaccount.com
common(){ echo --zone="$ZONE" --impersonate-service-account="$SA"; }
log(){ echo "[recover $(date +%H:%M:%S)] $*"; }
mkdir -p "$RDIR/logs" "$RDIR/results"
for s in $(seq 0 $((NSH - 1))); do
  VM="$PREFIX-s$s"
  # skip a shard already pulled (e.g. an orchestrator that self-resolved)
  if [ -f "$RDIR/results/search_${SPACE}_shard$s.json" ]; then log "$VM: shard json already present, skip"; continue; fi
  log "monitoring $VM..."
  while true; do
    up=$(gcloud compute instances describe "$VM" $(common) --format='value(status)' 2>/dev/null | tr -dc 'A-Z')
    if [ -z "$up" ]; then log "$VM gone (torn down elsewhere); skip"; break; fi
    done=$(gcloud compute ssh "$VM" $(common) --command='grep -c "shard done" ~/experiment/run.log 2>/dev/null || true' 2>/dev/null | tr -dc '0-9')
    alive=$(gcloud compute ssh "$VM" $(common) --command='pgrep -f "[e]xperiments.v1_graf.search" >/dev/null && echo 1 || echo 0' 2>/dev/null | tr -dc '0-9')
    gcloud compute scp $(common) "$VM":~/experiment/run.log "$RDIR/logs/$VM.run.log" >/dev/null 2>&1
    prog=$(grep -c "  ->  " "$RDIR/logs/$VM.run.log" 2>/dev/null || true)
    log "$VM: ${prog:-0} configs done (alive=${alive:-?} done_marker=${done:-0})"
    { [ "${done:-0}" != "0" ] || [ "${alive:-0}" = "0" ]; } && break
    sleep 60
  done
  if [ -n "${up:-}" ]; then
    log "pulling shard $s + tearing down $VM..."
    gcloud compute scp $(common) "$VM":~/experiment/experiments/v1_graf/results/search_${SPACE}_shard$s.json "$RDIR/results/" >/dev/null 2>&1
    gcloud compute scp $(common) "$VM":~/experiment/run.log "$RDIR/logs/$VM.run.log" >/dev/null 2>&1
    gcloud compute instances delete "$VM" $(common) -q >/dev/null 2>&1
    log "$VM DONE"
  fi
done
log "all shards handled; merging..."
PYTHONPATH=. uv run --project . python -m experiments.v1_graf.search --space "$SPACE" --merge --results-dir "$RDIR/results"
log "FULL MERGE DONE -> $RDIR/results/best_${SPACE}.json"
