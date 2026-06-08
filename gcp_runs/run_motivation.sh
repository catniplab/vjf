#!/usr/bin/env bash
# /gcp_run: expanded E6 motivator on a fresh e2-standard-4. Four original-VJF variants
# (C init {random,oracle} x optimizer {Adam,SGD}) + sVJF, 5 seeds, filtered + one-step R^2
# over the stream. Self-contained (vjf + numpy + torch; no neurofisherSNR).
set -uo pipefail
PROJECT=cr-memmingpark; ZONE=europe-west1-b
SA=claude-experiments@cr-memmingpark.iam.gserviceaccount.com
IMP=--impersonate-service-account=$SA
REPO=/Users/memming/Dropbox/_projects/vjf
VM=exp-$(date +%Y%m%d-%H%M%S)-vjf-motiv
LOG=$REPO/gcp_runs/$VM.log
DEST=$REPO/gcp_runs/$VM
mkdir -p "$DEST"
echo "VM=$VM" | tee "$LOG"

gcloud compute instances create "$VM" --project=$PROJECT --zone=$ZONE $IMP \
  --machine-type=e2-standard-4 --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=20GB --boot-disk-type=pd-balanced --service-account=$SA \
  --scopes=https://www.googleapis.com/auth/cloud-platform --metadata=enable-oslogin=TRUE \
  --labels=purpose=experiment,created-by=claude,mode=git,exp=motiv >>"$LOG" 2>&1 || { echo "create failed" | tee -a "$LOG"; exit 1; }

SSH() { gcloud compute ssh "$VM" --project=$PROJECT --zone=$ZONE $IMP --command="$1" 2>>"$LOG"; }

echo "waiting for ssh..." | tee -a "$LOG"
for i in $(seq 1 40); do SSH 'echo ready' 2>/dev/null | grep -q ready && break; sleep 5; done

echo "installing (clone branch + uv + numpy/torch)..." | tee -a "$LOG"
SSH 'sudo apt-get -qq update >/dev/null 2>&1; sudo apt-get -qq install -y git >/dev/null 2>&1; \
     git clone --branch exp/better-experiments --single-branch https://github.com/catniplab/vjf.git ~/vjf >/dev/null 2>&1; \
     curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; \
     export PATH="$HOME/.local/bin:$PATH"; cd ~/vjf && uv venv .venv >/dev/null 2>&1 && \
     uv pip install --python .venv/bin/python -e . >/dev/null 2>&1 && \
     uv pip install --python .venv/bin/python numpy torch >/dev/null 2>&1 && echo INSTALL_OK' >>"$LOG" 2>&1

# scp the (uncommitted) experiment script onto the cloned repo
gcloud compute scp --project=$PROJECT --zone=$ZONE $IMP \
  "$REPO/experiments/lc_poisson_stream/motivation_compare.py" \
  "$VM:~/vjf/experiments/lc_poisson_stream/motivation_compare.py" >>"$LOG" 2>&1

echo "=== RUN motivation_compare (5 variants x 5 seeds) ===" | tee -a "$LOG"
SSH 'cd ~/vjf && source .venv/bin/activate && python experiments/lc_poisson_stream/motivation_compare.py' 2>&1 | tee -a "$LOG"

echo "pulling motivation_compare.json..." | tee -a "$LOG"
gcloud compute scp --project=$PROJECT --zone=$ZONE $IMP \
  "$VM:~/vjf/motivation_compare.json" "$DEST/motivation_compare.json" >>"$LOG" 2>&1

echo "=== tearing down $VM ===" | tee -a "$LOG"
gcloud compute instances delete "$VM" --project=$PROJECT --zone=$ZONE $IMP --quiet >>"$LOG" 2>&1
echo "done; VM deleted; data at $DEST/motivation_compare.json" | tee -a "$LOG"
