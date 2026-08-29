#!/usr/bin/env bash
# RF_Swarm project activation — Option E (extend system ROS 2 Jazzy).
# Deliberately NOT sourced from ~/.bashrc; source this by hand:
#     source RF_Swarm/capstone_ws/activate.sh
#
# Order matters:
#   1. system ROS 2 Jazzy  -> puts rclpy + ros2 CLI on PYTHONPATH/PATH
#   2. project venv         -> our extra Python deps, sees system+ROS pkgs
#      (venv created with --system-site-packages, so ROS's PYTHONPATH is honoured)
#   3. ROS_DOMAIN_ID=42     -> isolate our DDS traffic from the default-domain(0)
#      rf_rosbridge_test overlay on this shared machine
#   4. this workspace's overlay (only once it has been built)

# 1. system ROS 2 Jazzy (read-only base install)
source /opt/ros/jazzy/setup.bash

# 2. project venv (isolated, inside RF_Swarm only)
_RFSWARM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${_RFSWARM_ROOT}/.venv/bin/activate"

# 2b. expose venv-only pip packages (scikit-learn/scipy/joblib) to nodes launched
#     via the system-python shebang that `ros2 run` installs (it ignores the venv).
#     Appended so system/ROS packages keep priority (venv has no numpy -> no shadow).
_RFSWARM_VENV_SP="$(echo "${VIRTUAL_ENV}"/lib/python*/site-packages)"
if [ -d "${_RFSWARM_VENV_SP}" ]; then
  export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${_RFSWARM_VENV_SP}"
fi

# 3. dedicated DDS domain for this project (rf_rosbridge_test uses default 0)
export ROS_DOMAIN_ID=42

# 4. this workspace's overlay, if built
if [ -f "${_RFSWARM_ROOT}/capstone_ws/install/setup.bash" ]; then
  source "${_RFSWARM_ROOT}/capstone_ws/install/setup.bash"
fi

# 5. VENDORED peer code (self-contained, Part-2 packaging). The swarm_intent
#    package + demo_common + demo_config were copied from pinned commit
#    69ac2b69015b7d8e37b388acb3443f5073c0eae3 into RF_Swarm/vendor/ (see
#    DEPENDENCY_MANIFEST.md). capstone_sim/formation import swarm_intent.{formations,
#    stgt}; capstone_narrate imports swarm_intent.{coverage,context_spec} + (lazy GPU
#    path) demo_common/demo_config. Now resolved IN-REPO — no ~/aadhya/uav-llm at runtime.
_RFSWARM_VENDOR="${_RFSWARM_ROOT}/vendor"
if [ -d "${_RFSWARM_VENDOR}/src" ]; then
  export PYTHONPATH="${_RFSWARM_VENDOR}/src:${_RFSWARM_VENDOR}/scripts:${_RFSWARM_VENDOR}/config${PYTHONPATH:+:${PYTHONPATH}}"
fi

# 5b. RFFP deploy package (protocol router + frozen tier encoders). capstone_rff runs
#     REAL BLE IQ through it -> /embeddings (replaces the old pool replay). Needs torch
#     + sklearn (already in .venv). router.pkl is produced once by fit_router.py.
_RFFP_ROOT="${RFFP_ROOT:-${HOME}/Desktop/RFFP_DEPLOY}"
if [ -d "${_RFFP_ROOT}/rffp" ]; then
  export PYTHONPATH="${_RFFP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# 5c. CAPSTONE/DL_model — real WiSig (wifi) / DRFF (drone_ocusync) IQ loaders
#     (gen_rff.data.loaders), the same repo the router itself was fit from. Only
#     needed when a scenario's rf.protocol is wifi/drone_ocusync; the ble path
#     never imports gen_rff.
_DLMODEL_ROOT="${DLMODEL_ROOT:-${HOME}/CAPSTONE/DL_model}"
if [ -d "${_DLMODEL_ROOT}/gen_rff" ]; then
  export PYTHONPATH="${_DLMODEL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "[RF_Swarm] ROS_DISTRO=${ROS_DISTRO}  ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  venv=${VIRTUAL_ENV##*/}"
