#!/usr/bin/env bash
# Drive the 5 K=6 scenarios through the FULL pipeline (sim rff base track formation),
# collecting real-EKF-fed /formation predictions per scenario -> confusion matrix.
# NOTE: no `set -u` — sourcing ROS setup.bash references unset vars and aborts under it.
WS=/home/pw26_akp_01/RF_Swarm/capstone_ws
SP=$(mktemp -d /tmp/rf_swarm_confusion_gate.XXXXXX)
HELPER=$WS/src/capstone_formation/scripts/gate_confusion_live.py
RESULTS=$SP/confusion_live.jsonl
source $WS/activate.sh >/dev/null 2>&1
source $WS/scripts/run_pipeline.sh   # for clean_pipeline()
: > "$RESULTS"

# scenario -> true formation label
declare -A TRUE=( [gate_k6_vshape]=v_shape [gate_k6]=encirclement [gate_k6_column]=column \
                  [gate_k6_diamond]=diamond [gate_k6_dispersed]=dispersed )
DUR=${DUR:-85}

for scn in gate_k6_vshape gate_k6 gate_k6_column gate_k6_diamond gate_k6_dispersed; do
  echo "==================== $scn (true=${TRUE[$scn]}) ===================="
  clean_pipeline
  pids=()
  for node in sim rff base track formation; do
    ros2 run "capstone_${node}" "capstone_${node}_node" \
      --ros-args -p scenario:="$scn" -p use_sim_time:=true > "$SP/pl_${scn}.log" 2>&1 &
    pids+=($!)
  done
  echo "[gate] launched $scn pids ${pids[*]}; collecting /formation for ${DUR}s"
  python3 "$HELPER" "${TRUE[$scn]}" "$DUR" "$RESULTS"
  kill ${pids[*]} 2>/dev/null
  clean_pipeline
done

echo; echo "===== REAL-EKF-FED CONFUSION (raw records) ====="
cat "$RESULTS"
