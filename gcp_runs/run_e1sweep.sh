#!/usr/bin/env bash
# /gcp_run: fill in the readout-stability (E1) experiment at the MISSING SNR regimes
# (-3, 3, 6 dB; n=15,50,150), 3 seeds x 4 arms, so the figure can show horizon vs SNR.
# E1 uses lc_data.calibrate_poisson -> needs neurofishersnr (a PyPI pkg the prior VMs had).
set -uo pipefail
PROJECT=cr-memmingpark; ZONE=europe-west1-b
SA=claude-experiments@cr-memmingpark.iam.gserviceaccount.com
IMP=--impersonate-service-account=$SA
REPO=/Users/memming/Dropbox/_projects/vjf
VM=exp-$(date +%Y%m%d-%H%M%S)-vjf-e1sweep
LOG=$REPO/gcp_runs/$VM.log
DEST=$REPO/gcp_runs/$VM/results
mkdir -p "$DEST"
echo "VM=$VM" | tee "$LOG"

gcloud compute instances create "$VM" --project=$PROJECT --zone=$ZONE $IMP \
  --machine-type=e2-standard-4 --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=20GB --boot-disk-type=pd-balanced --service-account=$SA \
  --scopes=https://www.googleapis.com/auth/cloud-platform --metadata=enable-oslogin=TRUE \
  --labels=purpose=experiment,created-by=claude,mode=git,exp=e1sweep >>"$LOG" 2>&1 || { echo "create failed" | tee -a "$LOG"; exit 1; }

SSH() { gcloud compute ssh "$VM" --project=$PROJECT --zone=$ZONE $IMP --command="$1" 2>>"$LOG"; }

echo "waiting for ssh..." | tee -a "$LOG"
for i in $(seq 1 40); do SSH 'echo ready' 2>/dev/null | grep -q ready && break; sleep 5; done

echo "installing (clone + uv + neurofisherSNR from git)..." | tee -a "$LOG"
NFS=git+https://github.com/catniplab/neurofisherSNR.git@96e83f68025db204a5dc6fde1a74403a045e61fb
SSH "sudo apt-get -qq update >/dev/null 2>&1; sudo apt-get -qq install -y git >/dev/null 2>&1; \
     git clone --branch exp/better-experiments --single-branch https://github.com/catniplab/vjf.git ~/vjf >/dev/null 2>&1; \
     curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; \
     export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/vjf && uv venv .venv >/dev/null 2>&1 && \
     uv pip install --python .venv/bin/python -e . >/dev/null 2>&1 && \
     uv pip install --python .venv/bin/python numpy torch scipy matplotlib '$NFS' >/dev/null 2>&1" >>"$LOG" 2>&1

# hard gate: abort (and tear down) if the calibrator did not import, instead of wasting the run
READY=$(SSH 'cd ~/vjf && .venv/bin/python -c "import neurofisherSNR" 2>&1 && echo NFS_READY')
if ! printf '%s' "$READY" | grep -q NFS_READY; then
  echo "ABORT: neurofisherSNR import failed -> $READY" | tee -a "$LOG"
  gcloud compute instances delete "$VM" --project=$PROJECT --zone=$ZONE $IMP --quiet >>"$LOG" 2>&1
  exit 1
fi
echo "NFS_READY ok" | tee -a "$LOG"

echo "=== RUN E1 sweep (-3/3/6 dB x 3 seeds x 4 arms) ===" | tee -a "$LOG"
SSH 'cd ~/vjf/experiments/lc_poisson_stream && source ~/vjf/.venv/bin/activate && \
  for sp in "-3 15" "3 50" "6 150"; do set -- $sp; snr=$1; n=$2; \
    for seed in 20260605 20260606 20260607; do \
      for arm in proj_oracle online_base frozen_pca freeze_after; do \
        echo ">>> $arm snr=$snr n=$n seed=$seed $(date -u +%H:%M:%S)"; \
        python exp_e1.py --arm $arm --seed $seed --n-neurons $n --snr-db $snr 2>&1 | tail -1; \
      done; done; done; echo "=== E1SWEEP done ==="' 2>&1 | tee -a "$LOG"

echo "pulling results_e1..." | tee -a "$LOG"
gcloud compute scp --recurse --project=$PROJECT --zone=$ZONE $IMP \
  "$VM:~/vjf/experiments/lc_poisson_stream/results_e1" "$DEST" >>"$LOG" 2>&1

echo "=== tearing down $VM ===" | tee -a "$LOG"
gcloud compute instances delete "$VM" --project=$PROJECT --zone=$ZONE $IMP --quiet >>"$LOG" 2>&1
echo "done; VM deleted; results at $DEST/results_e1" | tee -a "$LOG"
