#!/usr/bin/env bash
# One Graf sVJF search shard, full lifecycle on a fresh GCE VM.
# scp mode from the pinned code archive /tmp/vjf-graf-code.tar.gz (commit a8906f8) +
# the gitignored 77MB array_5.mat -- so no VM-side git auth and no data in git.
#
# Usage: graf_search_run.sh <vm> <space> <shard> <nshards> <quick:0|1> <results_dir> [zone] [grid]
# Runs the search detached on the VM, polls to completion, pulls the shard JSON + logs,
# then tears the VM down. Safe to background 8 of these in parallel. grid = base|xl.
set -uo pipefail
VM="$1"; SPACE="$2"; SHARD="$3"; NSH="$4"; QUICK="$5"; RDIR="$6"; ZONE="${7:-europe-west1-b}"; GRID="${8:-base}"
PROJECT=cr-memmingpark
SA=claude-experiments@cr-memmingpark.iam.gserviceaccount.com
CODE=/tmp/vjf-graf-code.tar.gz
DATA="$(pwd)/experiments/v1_graf/data/raw/array_5.mat"
mkdir -p "$RDIR/logs" "$RDIR/results"
LOG="$RDIR/logs/$VM.setup.log"
QFLAG=""; [ "$QUICK" = "1" ] && QFLAG="--quick"
log(){ echo "[$VM $(date +%H:%M:%S)] $*"; }
common(){ echo --project="$PROJECT" --zone="$ZONE" --impersonate-service-account="$SA"; }

# 1. create (retry in Madrid on capacity error)
log "creating VM ($ZONE, e2-standard-2)..."
create(){ gcloud compute instances create "$VM" $(common) --machine-type=e2-standard-2 \
    --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=20GB \
    --boot-disk-type=pd-balanced --service-account="$SA" \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --labels=purpose=experiment,created-by=claude,exp=graf-search; }
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

# 4. install (uv venv + editable core deps + the two dev deps the search needs)
log "installing uv + deps (torch wheel is large; this takes a few min)..."
gcloud compute ssh $(common) "$VM" --command='set -e; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH="$HOME/.local/bin:$PATH"; cd ~/experiment; uv venv .venv >/dev/null 2>&1; uv pip install --python .venv/bin/python -e . >/dev/null 2>&1; uv pip install --python .venv/bin/python scikit-learn matplotlib >/dev/null 2>&1; echo INSTALL_OK' >>"$LOG" 2>&1
grep -q INSTALL_OK "$LOG" || { log "INSTALL FAILED (see $LOG); tearing down"; gcloud compute instances delete "$VM" $(common) -q >/dev/null 2>&1; exit 1; }

# 5. launch detached
log "launching: search --space $SPACE --shard $SHARD --n-shards $NSH --grid $GRID $QFLAG"
gcloud compute ssh $(common) "$VM" --command="cd ~/experiment && export PATH=\"\$HOME/.local/bin:\$PATH\" && source .venv/bin/activate && nohup env PYTHONPATH=. python -m experiments.v1_graf.search --space $SPACE --shard $SHARD --n-shards $NSH --grid $GRID $QFLAG >run.log 2>&1 </dev/null & disown; sleep 2; echo LAUNCHED" >>"$LOG" 2>&1

# 6. poll to completion ('[e]xperiments...' bracket-trick so pgrep never matches its own shell)
RESJSON="search_${SPACE}_shard${SHARD}.json"
log "polling..."
while true; do
  sleep 30
  done=$(gcloud compute ssh "$VM" $(common) --command='grep -c "shard done" ~/experiment/run.log 2>/dev/null || true' 2>/dev/null | tr -dc '0-9')
  alive=$(gcloud compute ssh "$VM" $(common) --command='pgrep -f "[e]xperiments.v1_graf.search" >/dev/null && echo 1 || echo 0' 2>/dev/null | tr -dc '0-9')
  gcloud compute scp $(common) "$VM":~/experiment/run.log "$RDIR/logs/$VM.run.log" >/dev/null 2>&1
  prog=$(grep -c "  ->  " "$RDIR/logs/$VM.run.log" 2>/dev/null || true)
  log "progress: ${prog:-0} configs done (alive=${alive:-?} done_marker=${done:-0})"
  [ "${done:-0}" != "0" ] && break
  [ "${alive:-0}" = "0" ] && { log "process gone (done_marker=${done:-0}) -- inspect run.log"; break; }
done

# 7. retrieve (exact filename, no remote globbing)
log "retrieving $RESJSON + run.log..."
gcloud compute scp $(common) "$VM":~/experiment/experiments/v1_graf/results/"$RESJSON" "$RDIR/results/" >>"$LOG" 2>&1
gcloud compute scp $(common) "$VM":~/experiment/run.log "$RDIR/logs/$VM.run.log" >/dev/null 2>&1

# 8. teardown (unconditional)
log "tearing down VM..."
gcloud compute instances delete "$VM" $(common) -q >/dev/null 2>&1
log "DONE ($ZONE)"
