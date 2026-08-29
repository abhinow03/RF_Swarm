#!/usr/bin/env bash
# Narrate before/after demo: /formation comes alive ~35s in (once its window fills),
# and narrate must switch from the degenerate abstention to the real formation ctx.
# narrate required_gib=999 => the bucket-A (LLM) path cleanly report-and-halts,
# GPU-safe (never competes with the other user), while still proving the transition.
WS=/home/pw26_akp_01/RF_Swarm/capstone_ws
SP=$(mktemp -d /tmp/rf_swarm_narrate_demo.XXXXXX)
HELPER=$WS/src/capstone_narrate/scripts/narration_collector.py
source $WS/activate.sh >/dev/null 2>&1
source $WS/scripts/run_pipeline.sh
clean_pipeline
pids=()
for node in sim rff base track formation; do
  ros2 run "capstone_${node}" "capstone_${node}_node" \
    --ros-args -p scenario:=gate_k6 -p use_sim_time:=true > "$SP/nd_${node}.log" 2>&1 &
  pids+=($!)
done
ros2 run capstone_narrate capstone_narrate_node \
  --ros-args -p use_sim_time:=true -p required_gib:=999.0 > "$SP/nd_narrate.log" 2>&1 &
pids+=($!)
echo "[demo] launched pids ${pids[*]}; collecting /narration timeline for 95s"
python3 "$HELPER" 95
kill ${pids[*]} 2>/dev/null
clean_pipeline

echo; echo "===== narrate node log: degenerate -> formation transition ====="
grep -nE 'ctx_degenerate|ctx_formation|formation ctx|loading narration|model unavailable|narration \[' "$SP/nd_narrate.log" | head -40
