# TRUTH ONLY. Never consumed by the estimation chain.
std_msgs/Header header

uint16 emitter_id                       # ground-truth index 0..K-1
string hardware_uid                     # which real unit's embeddings are bound
                                        # e.g. "esp32_lab_03", "drff_airframe_7"

geometry_msgs/Point   position          # ENU metres
geometry_msgs/Vector3 velocity          # ENU m/s

float64 cfo_hz                          # true carrier frequency offset
float64 tx_power_dbm
