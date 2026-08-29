std_msgs/Header header

string global_id                        # "d1".."dK" — from FingerprintSet

geometry_msgs/Point   position          # ENU metres
geometry_msgs/Vector3 velocity          # ENU m/s
float64[36]           covariance        # row-major 6x6, [x y z vx vy vz]

float64 gdop
uint8   n_receivers_used
uint8   status

uint8 STATUS_INITIALIZING=0
uint8 STATUS_TRACKING=1
uint8 STATUS_COASTING=2                 # emitter silent, propagating
uint8 STATUS_LOST=3
