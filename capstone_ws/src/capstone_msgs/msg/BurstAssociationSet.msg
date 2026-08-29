# The base station's CURRENT full per-burst association map (blind). Republished
# each base tick as the complete map over all accumulated bursts, so it is
# idempotent and a late-joining / re-labeling-tolerant consumer (the tracker) can
# just adopt the latest snapshot. Published latched on /associations.
std_msgs/Header header
capstone_msgs/BurstAssociation[] associations
uint16 k_estimated                      # mirrors FingerprintSet.k_estimated at publish time
string encoder_sha256
