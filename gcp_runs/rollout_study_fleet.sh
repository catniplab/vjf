#!/usr/bin/env bash
# Fan out rollout_study across one VM per grating DIRECTION (full lifecycle each via
# rollout_study_run.sh: create/stage/install/run/pull/teardown). Firms up the report's
# near-marginality + tests generality. Results pulled to <results_dir>/results; merge locally.
#
# Usage: rollout_study_fleet.sh <results_dir> [seeds_csv] [dirs...]
set -uo pipefail
RDIR="${1:?need results dir}"; SEEDS="${2:-20260612,20260613,20260614,20260615,20260616}"
shift || true; shift || true
DIRS=("$@"); [ ${#DIRS[@]} -eq 0 ] && DIRS=(225 45 105 295)
export CLOUDSDK_CORE_PROJECT=cr-memmingpark
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$RDIR/logs" "$RDIR/results"
echo "fleet: dirs ${DIRS[*]} x seeds {$SEEDS} -> $RDIR"
pids=()
for d in "${DIRS[@]}"; do
  VM="exp-$TS-rollout-d$d"
  bash gcp_runs/rollout_study_run.sh "$VM" "$d" "$SEEDS" "$RDIR" europe-west1-b \
    > "$RDIR/logs/fleet-d$d.out" 2>&1 &
  pids+=($!)
  echo "  launched dir $d -> $VM (pid $!)"
  sleep 15   # stagger VM creation so the API/quotas are not hammered
done
echo "waiting for ${#pids[@]} direction VMs (each creates+installs+runs ~5 seeds+tears-down)..."
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "  dir-VM $i lifecycle FAILED (see $RDIR/logs)"; fail=$((fail + 1)); fi
done
echo "all direction VMs finished ($fail failed). pulled files:"
ls -la "$RDIR/results/"
echo "fleet done -> $RDIR/results/ (merge with rollout_study_merge.py)"
