#!/usr/bin/env bash
# Thin wrapper around `gcloud compute ssh/scp` for the current experiment VM,
# with SA impersonation, project, and zone baked in -- so you don't retype
# VM=... and --impersonate-service-account=... every time.
#
# VM name resolution order: $VJF_VM, else the first `VM=` line in /tmp/vjf_vm.txt.
#
# Usage:
#   gcp_runs/vm.sh ssh '<remote command>'      # run a command on the VM
#   gcp_runs/vm.sh log                          # tail ~/experiment/run.log
#   gcp_runs/vm.sh pull '<remote glob>' '<local dir>'   # scp --recurse from the VM
#   gcp_runs/vm.sh '<remote command>'           # bare form == ssh
#
# Examples:
#   gcp_runs/vm.sh ssh 'cd ~/experiment && git rev-parse --short HEAD'
#   gcp_runs/vm.sh pull '~/experiment/experiments/lc_poisson_stream/results/*' ./results
set -uo pipefail

VM="${VJF_VM:-$(sed -n 's/^VM=//p' /tmp/vjf_vm.txt 2>/dev/null | head -1)}"
PROJECT="${VJF_PROJECT:-cr-memmingpark}"
ZONE="${VJF_ZONE:-europe-west1-b}"
SA="claude-experiments@cr-memmingpark.iam.gserviceaccount.com"
COMMON=(--project="$PROJECT" --zone="$ZONE" --impersonate-service-account="$SA")

[ -z "$VM" ] && { echo "vm.sh: no VM (set \$VJF_VM or put 'VM=<name>' in /tmp/vjf_vm.txt)" >&2; exit 1; }

# gcloud prints SA-impersonation warnings to stderr on every call; drop the noise.
_filter() { grep -vE "service account|^WARNING:" || true; }

sub="${1:-log}"; shift || true
case "$sub" in
  ssh)  gcloud compute ssh "$VM" "${COMMON[@]}" --command="$*" 2>&1 | _filter ;;
  pull) gcloud compute scp --recurse "${COMMON[@]}" "$VM:$1" "$2" 2>&1 | _filter ;;
  log)  gcloud compute ssh "$VM" "${COMMON[@]}" --command='tail -n 25 ~/experiment/run.log' 2>&1 | _filter ;;
  *)    gcloud compute ssh "$VM" "${COMMON[@]}" --command="$sub $*" 2>&1 | _filter ;;
esac
