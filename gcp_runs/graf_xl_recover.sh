#!/usr/bin/env bash
# Recovery monitor for the xl fleet shards whose local orchestrator hung on the launch-ssh
# (s4-s7). Each VM is alive and the search is running detached; this polls each to
# completion (shard-done marker or python exit), pulls the shard JSON + run.log, tears the
# VM down, then does the full 8-shard merge.
set -uo pipefail
export CLOUDSDK_CORE_PROJECT=cr-memmingpark
SA=claude-experiments@cr-memmingpark.iam.gserviceaccount.com
ZONE=europe-west1-b
RDIR="gcp_runs/graf-search-single-xl-20260614-164113"
PREFIX="exp-20260614-164113-graf-single"
common(){ echo --zone="$ZONE" --impersonate-service-account="$SA"; }
log(){ echo "[recover $(date +%H:%M:%S)] $*"; }
for s in 4 5 6 7; do
  VM="$PREFIX-s$s"
  log "monitoring $VM..."
  while true; do
    done=$(gcloud compute ssh "$VM" $(common) --command='grep -c "shard done" ~/experiment/run.log 2>/dev/null || true' 2>/dev/null | tr -dc '0-9')
    alive=$(gcloud compute ssh "$VM" $(common) --command='pgrep -f "[e]xperiments.v1_graf.search" >/dev/null && echo 1 || echo 0' 2>/dev/null | tr -dc '0-9')
    gcloud compute scp $(common) "$VM":~/experiment/run.log "$RDIR/logs/$VM.run.log" >/dev/null 2>&1
    prog=$(grep -c "  ->  " "$RDIR/logs/$VM.run.log" 2>/dev/null || true)
    log "$VM: ${prog:-0}/6 done (alive=${alive:-?} done_marker=${done:-0})"
    { [ "${done:-0}" != "0" ] || [ "${alive:-0}" = "0" ]; } && break
    sleep 60
  done
  log "pulling shard $s + tearing down $VM..."
  gcloud compute scp $(common) "$VM":~/experiment/experiments/v1_graf/results/search_single_shard$s.json "$RDIR/results/" >/dev/null 2>&1
  gcloud compute scp $(common) "$VM":~/experiment/run.log "$RDIR/logs/$VM.run.log" >/dev/null 2>&1
  gcloud compute instances delete "$VM" $(common) -q >/dev/null 2>&1
  log "$VM DONE"
done
log "all 4 recovered; full 8-shard merge..."
PYTHONPATH=. uv run --project . python -m experiments.v1_graf.search --space single --merge --results-dir "$RDIR/results"
log "FULL MERGE DONE -> $RDIR/results/best_single.json"
