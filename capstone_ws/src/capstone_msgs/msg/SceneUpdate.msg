# THE ONLY TOPIC THE HEADSET SUBSCRIBES TO.
# QoS: RELIABLE + TRANSIENT_LOCAL (depth 1) so late-joining AR clients get
# state immediately instead of waiting for the next 0.1 Hz publication.
#
# NOTE: /scene_update ON THE WIRE is std_msgs/String (JSON), per the Option C
# reconciliation with the AR client — see reconciliation_answers.md. This typed
# definition is retained for schema documentation and capstone_eval use; it is
# NOT published as-is in Phase 0. Do not assume `ros2 interface show
# capstone_msgs/SceneUpdate` describes what is actually on /scene_update.
std_msgs/Header header

uint8 schema_version                    # bump on ANY change to this package

capstone_msgs/TrackState[]      tracks
capstone_msgs/PathPrediction[]  predictions
capstone_msgs/FormationEstimate formation
capstone_msgs/Narration         narration
capstone_msgs/ReceiverState[]   receivers

uint16  k_estimated
float64 pipeline_latency_ms             # observation timestamp -> this publish
string  scenario_id
