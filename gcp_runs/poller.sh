#!/usr/bin/env bash
# Watch the two in-flight experiment VMs. When a VM's orchestrator process
# (run_main.sh / run_extra.sh) is gone, the run is done: pull its results
# locally, then delete the instance. Exit once both are torn down.
# macOS bash-3.2 safe: no associative arrays; self-match avoided with the
# grep '[r]un_...' trick.
set -uo pipefail
PROJECT=cr-memmingpark
ZONE=europe-west1-b
SA=claude-experiments@cr-memmingpark.iam.gserviceaccount.com
REPO=/Users/memming/Dropbox/_projects/vjf
LCP='~/experiment/experiments/lc_poisson_stream'

vm_exists() { gcloud compute instances describe "$1" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; }
vm_ssh()    { gcloud compute ssh "$1" --project="$PROJECT" --zone="$ZONE" --impersonate-service-account="$SA" --command="$2" 2>/dev/null; }
vm_scp()    { gcloud compute scp --recurse --project="$PROJECT" --zone="$ZONE" --impersonate-service-account="$SA" "$1:$2" "$3" 2>/dev/null; }
vm_del()    { gcloud compute instances delete "$1" --project="$PROJECT" --zone="$ZONE" --quiet 2>/dev/null; }

MAIN=exp-20260607-094255-vjf-main
EXTRA=exp-20260607-100741-vjf-extra

handle() {  # $1=vm $2=orchestrator-script $3...=remote paths to pull (last arg is local dir)
  vm="$1"; orch="$2"; shift 2
  vm_exists "$vm" || return 0
  # grep -c '[X]...' returns 0 when the orchestrator has exited (no self-match)
  pat="[${orch:0:1}]${orch:1}"
  running="$(vm_ssh "$vm" "ps -e -o args 2>/dev/null | grep -c '$pat'")"
  running="$(printf '%s' "$running" | tr -dc '0-9')"
  [ "${running:-1}" != "0" ] && { echo "$(date '+%H:%M:%S') $vm still running"; return 0; }
  ldir="$REPO/gcp_runs/$vm/results"
  mkdir -p "$ldir"
  for rp in "$@"; do vm_scp "$vm" "$rp" "$ldir"; done
  vm_del "$vm"
  echo "$(date '+%H:%M:%S') $vm DONE -> pulled to $ldir, instance deleted"
}

echo "$(date '+%H:%M:%S') poller start"
while vm_exists "$MAIN" || vm_exists "$EXTRA"; do
  handle "$MAIN"  run_main.sh  "$LCP/results"
  handle "$EXTRA" run_extra.sh "$LCP/results_e1" "$LCP/results/x_timing"
  vm_exists "$MAIN" || vm_exists "$EXTRA" || break
  sleep 180
done
echo "$(date '+%H:%M:%S') poller done: both VMs torn down"
