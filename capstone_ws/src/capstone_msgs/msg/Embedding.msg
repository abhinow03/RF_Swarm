std_msgs/Header header

uint32 burst_seq
uint8  rx_id

float32[] vector                        # length == dim
uint16    dim                           # 512

string encoder_sha256                   # frozen checkpoint provenance
string source                           # "replay:drff" | "replay:esp32_lab" | "live"
