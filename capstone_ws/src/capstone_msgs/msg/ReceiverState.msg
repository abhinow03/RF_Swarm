# State of one swarm receiver. Published on /truth/receivers (exact) and
# /receivers (noise-corrupted, what the estimator actually sees).
std_msgs/Header header

uint8 rx_id

geometry_msgs/Point   position          # ENU metres, base-station origin
geometry_msgs/Vector3 velocity          # ENU m/s
float64[9]            position_covariance   # row-major 3x3, m^2

float64 clock_bias                      # seconds, vs base-station reference
float64 clock_drift                     # s/s
float64 clock_bias_sigma                # seconds
