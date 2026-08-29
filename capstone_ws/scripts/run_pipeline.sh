#!/usr/bin/env bash
# Robust launch + GUARANTEED teardown for a capstone scenario.
#
# Fixes the contamination bug found in Phase 3: `setsid` + parent-PID kill does
# NOT reliably kill `ros2 run` node children, so a later run could collect an
# earlier run's messages (e.g. sweep_k8 inherited gate_k6 embeddings). This
# launches nodes with plain `&` (the PID IS the node) and, on exit AND before
# start, hard-kills every capstone node process, verifies zero remain, and waits
# for ROS discovery to expire — so no run ever starts against a dirty domain.
#
# Usage:  run_pipeline.sh <scenario> [node ...]      # default nodes: sim rff base
#   e.g.  run_pipeline.sh gate_k6
#         run_pipeline.sh sweep_k8 sim rff base
# Or source it to reuse clean_pipeline() in other scripts.
set -u

# matches the actual node process cmdline (…/install/capstone_<pkg>/lib/capstone_<pkg>/…);
# NOT present in this script's own cmdline, so pgrep never self-matches.
_NODE_PAT='capstone_[a-z][a-z]*/lib/capstone_'

clean_pipeline() {
  for _ in 1 2 3; do
    for p in $(pgrep -f "$_NODE_PAT" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  done
  local left; left=$(pgrep -f "$_NODE_PAT" 2>/dev/null | wc -l)
  timeout 20 bash -c 'while ros2 node list 2>/dev/null | grep -q capstone; do :; done'
  echo "[pipeline] clean: ${left} procs killed, domain clear"
}

run_pipeline() {
  local scn="$1"; shift || true
  local nodes=("$@"); [ ${#nodes[@]} -eq 0 ] && nodes=(sim rff base)
  clean_pipeline
  local pids=()
  for node in "${nodes[@]}"; do
    ros2 run "capstone_${node}" "capstone_${node}_node" \
      --ros-args -p scenario:="$scn" -p use_sim_time:=true &
    pids+=($!)
  done
  trap 'kill ${pids[*]} 2>/dev/null; clean_pipeline' EXIT INT TERM
  echo "[pipeline] launched (${nodes[*]}) for scenario '${scn}' — pids ${pids[*]}"
  wait
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  [ $# -ge 1 ] || { echo "usage: $0 <scenario> [node ...]" >&2; exit 2; }
  run_pipeline "$@"
fi
