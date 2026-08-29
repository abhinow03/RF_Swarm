#!/usr/bin/env bash
# Run capstone_narrate under the LLM ENVIRONMENT so the bucket-A path can load the
# 4-bit Qwen2.5-7B + LoRA adapter on the GPU and emit REAL LLM-generated narration.
#
# Why a separate launcher (not a pipeline change): the project .venv holds CPU-only
# torch (for the STGT classifier). The LLM needs CUDA torch + transformers +
# bitsandbytes + peft, which live in ~/aadhya/uav-llm/.venv (py3.12.3, matches jazzy).
# That .venv can also reach rclpy + capstone_msgs via the sourced overlays, so we run
# ONLY narrate under it — every other node keeps running under the normal env. No
# modification to the working pipeline; just a different python for one node.
#
# Usage:  run the pipeline WITHOUT narrate, then this:
#   bash scripts/run_pipeline.sh gate_k6 sim rff base track formation predict bridge
#   bash scripts/run_narrate_llm.sh
LLM_PY=/home/pw26_akp_01/aadhya/uav-llm/.venv/bin/python3
source /opt/ros/jazzy/setup.bash
source /home/pw26_akp_01/RF_Swarm/capstone_ws/install/setup.bash
export ROS_DOMAIN_ID=42
# vendored, self-contained peer code (pinned 69ac2b6) — no ~/aadhya/uav-llm at runtime
export PYTHONPATH=/home/pw26_akp_01/RF_Swarm/vendor/src:/home/pw26_akp_01/RF_Swarm/vendor/scripts:/home/pw26_akp_01/RF_Swarm/vendor/config${PYTHONPATH:+:$PYTHONPATH}
echo "[narrate-llm] python=$LLM_PY  domain=$ROS_DOMAIN_ID"
exec "$LLM_PY" -c "from capstone_narrate.narrate_node import main; main()" \
  --ros-args -p use_sim_time:=true -p required_gib:=15.0
