#!/usr/bin/env bash
# Fan out the Graf sVJF search across N VMs (one shard each), wait for all, then merge.
# Each shard runs the full lifecycle via graf_search_run.sh (create/stage/install/run/
# pull/teardown). Shard JSONs are pulled into <results_dir>/results and merged locally.
#
# Usage: graf_search_fleet.sh <space> <nshards> <results_dir> [quick:0|1] [grid:base|xl]
set -uo pipefail
SPACE="${1:-single}"; NSH="${2:-8}"; RDIR="${3:?need results dir}"; QUICK="${4:-0}"; GRID="${5:-base}"
export CLOUDSDK_CORE_PROJECT=cr-memmingpark
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$RDIR/logs" "$RDIR/results"
echo "fleet: $SPACE x $NSH shards (quick=$QUICK grid=$GRID) -> $RDIR"
pids=()
for s in $(seq 0 $((NSH - 1))); do
  VM="exp-$TS-graf-$SPACE-s$s"
  bash gcp_runs/graf_search_run.sh "$VM" "$SPACE" "$s" "$NSH" "$QUICK" "$RDIR" europe-west1-b "$GRID" \
    > "$RDIR/logs/fleet-s$s.out" 2>&1 &
  pids+=($!)
  echo "  launched shard $s -> $VM (pid $!)"
  sleep 10   # stagger VM creation so the API/quotas are not hammered
done
echo "waiting for ${#pids[@]} shards (each ~creates+installs+runs+tears-down)..."
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "  shard $i lifecycle FAILED (see $RDIR/logs/fleet-s$i.out)"; fail=$((fail + 1)); fi
done
echo "all shards finished ($fail failed lifecycles). pulled shard files:"
ls -la "$RDIR/results/"
echo "=== merging (gate + rank) ==="
PYTHONPATH=. uv run --project . python -m experiments.v1_graf.search --space "$SPACE" --merge --results-dir "$RDIR/results"
echo "fleet done -> $RDIR/results/best_${SPACE}.json"
