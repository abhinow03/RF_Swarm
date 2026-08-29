# One burst's BLIND association (fingerprint-derived, NOT truth). The base station
# clusters embeddings; this is which discovered emitter (global_id) a given burst
# was assigned to, plus how much to trust it. Distinct from BurstAssociationTruth
# (which carries the real emitter id and is for scoring only).
uint32  burst_seq                       # join key to RfObservationSet.burst_seq
string  global_id                       # "d1".."dK" (base first-seen cluster order)
float64 confidence                      # [0,1]; 0 => withheld (under-accumulated / not separable)
