#!/usr/bin/env bash
# rosbridge over wss:// for the WebXR / Quest client.
#
# WebXR requires the page be served over https, and a secure page may only open a
# SECURE websocket (wss://) — a plain ws:// is silently refused by the headset. The
# existing rf_rosbridge_test setup only ever ran plain ws:// on localhost, so wss +
# a TLS cert is genuinely new (see phase6). This reuses the PROVEN rosbridge overlay
# build and only turns on its ssl option with a self-signed cert.
#
# Reuses:
#   * the rosbridge_suite overlay already built at ~/rf_rosbridge_test/rosbridge_ws
#   * an mkcert-signed cert in capstone_ws/certs (SAN covers mars-4090 / localhost /
#     LAN 192.168.31.141 / tailscale 100.107.251.96)
# Runs on ROS_DOMAIN_ID=42 (our pipeline's domain) and port 9443 so it never
# collides with a default-domain / port-9090 rosbridge the viz demo might run.
#
# QUEST TRUST (mkcert, zero per-device accept step at demo time):
#   The leaf cert is signed by a project-contained mkcert CA at
#   capstone_ws/certs/mkcert-ca/rootCA.pem. We deliberately did NOT run
#   `mkcert -install` — nothing in this machine's system/user trust store is
#   modified (shared-machine safe). Instead, sideload rootCA.pem onto the Quest
#   ONCE (Settings > Security > install CA cert, or transfer the .pem and trust it),
#   after which wss:// connects with NO warning. See scripts/install_quest_ca.md.
set -e
WS=/home/pw26_akp_01/RF_Swarm/capstone_ws
CERT=$WS/certs/rosbridge_mkcert_cert.pem
KEY=$WS/certs/rosbridge_mkcert_key.pem
PORT=${PORT:-9443}

source /opt/ros/jazzy/setup.bash
source /home/pw26_akp_01/rf_rosbridge_test/rosbridge_ws/install/setup.bash   # proven rosbridge overlay
source "$WS/install/setup.bash"                                              # our capstone_msgs types
export ROS_DOMAIN_ID=42

echo "[rosbridge] wss://0.0.0.0:${PORT}  domain=${ROS_DOMAIN_ID}  cert=${CERT}"
exec ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
  ssl:=true certfile:="$CERT" keyfile:="$KEY" port:="$PORT"
