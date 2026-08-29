# One burst as measured at one receiver. NO truth fields.
std_msgs/Header header

uint8  rx_id
uint32 burst_seq                        # join key across receivers

builtin_interfaces/Time toa             # time of arrival, RECEIVER clock
float64 toa_sigma                       # seconds

float64 rssi_dbm
float64 rssi_sigma                      # dB

float64 fdoa_hz                         # measured freq offset from nominal
float64 fdoa_sigma                      # Hz   (NOTE: CFO + Doppler, unseparated)

float64 aoa_azimuth                     # rad, ENU. NaN if unavailable
float64 aoa_elevation                   # rad
float64 aoa_sigma                       # rad
bool    aoa_valid                       # FALSE on single-antenna receivers.
                                        # Set true only under array-upgrade flag.

float64 snr_db
float64 nominal_freq_hz
