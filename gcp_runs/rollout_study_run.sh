#!/usr/bin/env bash
# One rollout_study direction, full lifecycle on a fresh GCE VM (scp mode).
# Stages the pinned code archive /tmp/vjf-rollout-code.tar.gz + the gitignored 77MB array_5.mat,
# runs experiments.v1_graf.rollout_study for one grating DIRECTION over SEEDS (env-driven, with
# per-seed checkpointing), polls to completion, pulls the (checkpoint or final) results JSON +
# table + figure, then tears the VM down. Safe to background several in parallel.
#
# Usage: rollout_study_run.sh <vm> <dir_deg> <seeds_csv> <results_dir> [zone]
set -uo pipefail
VM="$1"; DIR="$2"; SEEDS="$3"; RDIR="$4"; ZONE="${5:-europe-west1-b}"
PROJECT=cr-memmingpark
SA=claude-experiments@cr-memmingpark.iam.gserviceaccount.com
CODE=/tmp/vjf-rollout-code.tar.gz
DATA="$(pwd)/experiments/v1_graf/data/raw/array_5.mat"
TAG="d${DIR}"
RESJSON="rollout_study_results_${TAG}.json"
RESTEX="rollout_study_table_${TAG}.tex"
RESFIG="forecast_vs_horizon_${TAG}.png"
mkdir -p "$RDIR/logs" "$RDIR/results"
LOG="$RDIR/logs/$VM.setup.log"
log(){ echo "[$VM $(date +%H:%M:%S)] $*"; }
common(){ echo --project="$PROJECT" --zone="$ZONE" --impersonate-service-account="$SA"; }

# 1. create (retry in Madrid on capacity error)
log "creating VM ($ZONE, e2-standard-4)..."
create(){ gcloud compute instances create "$VM" $(common) --machine-type=e2-standard-4 \
    --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=20GB \
    --boot-disk-type=pd-balanced --service-account="$SA" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --labels=purpose=experiment,created-by=claude,exp=rollout-study; }
if ! create >>"$LOG" 2>&1; then
  log "create failed in $ZONE; retrying europe-southwest1-b"; ZONE=europe-southwest1-b
  create >>"$LOG" 2>&1 || { log "CREATE FAILED (see $LOG)"; exit 1; }
fi

# 2. wait for ssh (<=150s)
log "waiting for ssh..."
ok=0
for i in $(seq 1 30); do
  if gcloud compute ssh "$VM" $(common) --command='echo ready' >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
[ "$ok" = 0 ] && { log "SSH NEVER CAME UP; tearing down"; gcloud compute instances delete "$VM" $(common) -q >/dev/null 2>&1; exit 1; }

# 3. stage code + data
log "staging code + data (scp)..."
gcloud compute scp $(common) "$CODE" "$VM":~/code.tar.gz   >>"$LOG" 2>&1
gcloud compute scp $(common) "$DATA" "$VM":~/array_5.mat   >>"$LOG" 2>&1
gcloud compute ssh $(common) "$VM" --command='set -e; tar xzf ~/code.tar.gz; mkdir -p ~/experiment/experiments/v1_graf/data/raw; mv ~/array_5.mat ~/experiment/experiments/v1_graf/data/raw/; echo STAGED' >>"$LOG" 2>&1

# 4. install (uv venv + editable core deps + the two dev deps the study needs)
log "installing uv + deps (torch wheel is large; this takes a few min)..."
gcloud compute ssh $(common) "$VM" --command='set -e; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH="$HOME/.local/bin:$PATH"; cd ~/experiment; uv venv .venv >/dev/null 2>&1; uv pip install --python .venv/bin/python -e . >/dev/null 2>&1; uv pip install --python .venv/bin/python scikit-learn matplotlib >/dev/null 2>&1; echo INSTALL_OK' >>"$LOG" 2>&1
grep -q INSTALL_OK "$LOG" || { log "INSTALL FAILED (see $LOG); tearing down"; gcloud compute instances delete "$VM" $(common) -q >/dev/null 2>&1; exit 1; }

# 5. launch detached (setsid: clean session so the ssh channel closes; -n: no local stdin)
log "launching: rollout_study DIR=$DIR SEEDS=$SEEDS"
gcloud compute ssh $(common) --ssh-flag=-n "$VM" --command="cd ~/experiment && export PATH=\"\$HOME/.local/bin:\$PATH\" && source .venv/bin/activate && setsid env PYTHONPATH=. ROLLOUT_DIR=$DIR ROLLOUT_SEEDS=$SEEDS python -m experiments.v1_graf.rollout_study >run.log 2>&1 </dev/null & sleep 2; echo LAUNCHED" >>"$LOG" 2>&1

# 6. poll to completion; pull the checkpoint JSON each cycle so partial progress always survives
log "polling..."
while true; do
  sleep 60
  alive=$(gcloud compute ssh "$VM" $(common) --command='pgrep -f "[e]xperiments.v1_graf.rollout_study" >/dev/null && echo 1 || echo 0' 2>/dev/null | tr -dc '0-9')
  gcloud compute scp $(common) "$VM":~/experiment/run.log "$RDIR/logs/$VM.run.log" >/dev/null 2>&1
  gcloud compute scp $(common) "$VM":~/experiment/experiments/v1_graf/"$RESJSON" "$RDIR/results/" >/dev/null 2>&1 || true
  seeds_done=$(grep -c "checkpoint" "$RDIR/logs/$VM.run.log" 2>/dev/null || true)
  full=$(grep -c "$RESFIG" "$RDIR/logs/$VM.run.log" 2>/dev/null || true)
  log "progress: ${seeds_done:-0} seeds checkpointed (alive=${alive:-?} full_done=${full:-0})"
  [ "${full:-0}" != "0" ] && break
  [ "${alive:-0}" = "0" ] && { log "process gone (seeds=${seeds_done:-0}) -- inspect run.log"; break; }
done

# 7. retrieve final artifacts (json already pulled each cycle; table + figure if full-complete)
log "retrieving artifacts..."
gcloud compute scp $(common) "$VM":~/experiment/experiments/v1_graf/"$RESJSON" "$RDIR/results/" >>"$LOG" 2>&1 || true
gcloud compute scp $(common) "$VM":~/experiment/experiments/v1_graf/report_m1/"$RESTEX" "$RDIR/results/" >>"$LOG" 2>&1 || true
gcloud compute scp $(common) "$VM":~/experiment/experiments/v1_graf/report_m1/figs/"$RESFIG" "$RDIR/results/" >>"$LOG" 2>&1 || true
gcloud compute scp $(common) "$VM":~/experiment/run.log "$RDIR/logs/$VM.run.log" >/dev/null 2>&1

# 8. teardown (unconditional)
log "tearing down VM..."
gcloud compute instances delete "$VM" $(common) -q >/dev/null 2>&1
log "DONE ($ZONE)"
