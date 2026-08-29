# Output of the base station. ~0.1 Hz (N=120 accumulation).
std_msgs/Header header

uint16    k_estimated
float64[] eigenvalues                   # for eigengap audit
float64   eigengap_value
uint32    n_accumulated

capstone_msgs/FingerprintAssignment[] assignments
string    encoder_sha256
