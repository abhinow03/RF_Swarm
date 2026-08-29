#!/usr/bin/env bash
# Run capstone_rff (REAL-IQ -> RFFP router + frozen BLE encoder) under the CUDA env
# so the startup pre-encode runs on the GPU (the encoder is ~30 ms/window on CPU ->
# too slow; ~sub-ms on GPU). After pre-encoding the per-unit embedding banks the node
# frees the GPU and serves from numpy banks, so it only touches the GPU briefly at
# startup (before capstone_narrate loads its LLM).
#
# The project .venv holds CPU-only torch (for STGT); the CUDA torch + sklearn + scipy
# needed by RFFP live in ~/aadhya/uav-llm/.venv (py3.12.3), which also reaches rclpy +
# capstone_msgs + capstone_rff + rffp via the sourced overlays / PYTHONPATH.
#
# Usage: run the pipeline WITHOUT rff, then this:
#   bash scripts/run_pipeline.sh gate_k6 sim base track formation predict bridge
#   bash scripts/run_rff_real.sh [scenario]
CUDA_PY=/home/pw26_akp_01/aadhya/uav-llm/.venv/bin/python3
SCN="${1:-gate_k6}"
source /opt/ros/jazzy/setup.bash
source /home/pw26_akp_01/RF_Swarm/capstone_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export PYTHONPATH="${RFFP_ROOT:-/home/pw26_akp_01/Desktop/RFFP_DEPLOY}${PYTHONPATH:+:$PYTHONPATH}"
echo "[rff-real] python=$CUDA_PY  scenario=$SCN  (GPU pre-encode via RFFP router+encoder)"
exec "$CUDA_PY" -c "from capstone_rff.rff_node import main; main()" \
  --ros-args -p scenario:="$SCN" -p use_sim_time:=true
