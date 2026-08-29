# capstone_msgs — FROZEN INTERFACE SPEC v1

**Status:** FROZEN as of Phase 0. Any change requires bumping `schema_version` in
`SceneUpdate.msg` and re-recording golden bags.

Design rules baked into this spec:

1. **Ground truth never enters a production message.** Truth lives on `/truth/*`
   topics consumed only by `capstone_eval`. No node in the estimation chain may
   subscribe to `/truth/*` unless its `passthrough_truth` parameter is `true`.
2. **`burst_seq` is the join key.** Observations, embeddings, and truth
   associations across all receivers for one emission share one `burst_seq`.
3. **One message to the headset.** The AR client subscribes to exactly one topic
   (`/scene_update`) so `roslibjs` stays trivial.
4. **Every node exposes `passthrough_truth`** (bool param) and publishes
   `NodeHealth`.

---

## Package layout

```
capstone_ws/src/
  capstone_msgs/          # THIS PACKAGE — freeze first
  capstone_scenarios/     # YAML scenario configs (no code)
  capstone_sim/           # scripted poses + observable generator
  capstone_rff/           # thin wrapper -> encoder module
  capstone_base/          # pooling, eigengap, global ID assignment
  capstone_track/         # closed-form TDoA init + fused EKF
  capstone_predict/       # CV baseline (permanent) + ML model slot
  capstone_narrate/       # template narrator (permanent) + LLM slot
  capstone_bridge/        # frame alignment + rosbridge launch
  capstone_eval/          # metrics -> CSV. Only package allowed to read truth.
```

---

## Message definitions

### `msg/ReceiverState.msg`
```
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
```

### `msg/ReceiverStateArray.msg`
```
std_msgs/Header header
capstone_msgs/ReceiverState[] receivers
```

### `msg/EmitterTruth.msg`
```
# TRUTH ONLY. Never consumed by the estimation chain.
std_msgs/Header header

uint16 emitter_id                       # ground-truth index 0..K-1
string hardware_uid                     # which real unit's embeddings are bound
                                        # e.g. "esp32_lab_03", "drff_airframe_7"

geometry_msgs/Point   position          # ENU metres
geometry_msgs/Vector3 velocity          # ENU m/s

float64 cfo_hz                          # true carrier frequency offset
float64 tx_power_dbm
```

### `msg/TruthScene.msg`
```
# TRUTH ONLY.
std_msgs/Header header
capstone_msgs/EmitterTruth[] emitters
uint16 k_true
string formation_label                  # ground-truth formation class
string scenario_id                      # matches capstone_scenarios YAML name
```

### `msg/RfObservation.msg`
```
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
```

### `msg/RfObservationSet.msg`
```
std_msgs/Header header
uint32 burst_seq
capstone_msgs/RfObservation[] observations
```

### `msg/BurstAssociationTruth.msg`
```
# TRUTH ONLY. Lets capstone_eval score association without leaking into RfObservation.
std_msgs/Header header
uint32 burst_seq
uint16 truth_emitter_id
```

### `msg/Embedding.msg`
```
std_msgs/Header header

uint32 burst_seq
uint8  rx_id

float32[] vector                        # length == dim
uint16    dim                           # 512

string encoder_sha256                   # frozen checkpoint provenance
string source                           # "replay:drff" | "replay:esp32_lab" | "live"
```

### `msg/FingerprintAssignment.msg`
```
uint32  burst_seq
string  global_id                       # "d1".."dK"
float64 confidence                      # [0,1]
```

### `msg/FingerprintSet.msg`
```
# Output of the base station. ~0.1 Hz (N=120 accumulation).
std_msgs/Header header

uint16    k_estimated
float64[] eigenvalues                   # for eigengap audit
float64   eigengap_value
uint32    n_accumulated

capstone_msgs/FingerprintAssignment[] assignments
string    encoder_sha256
```

### `msg/TrackState.msg`
```
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
```

### `msg/TrackArray.msg`
```
std_msgs/Header header
capstone_msgs/TrackState[] tracks
```

### `msg/PathPrediction.msg`
```
std_msgs/Header header

string global_id

geometry_msgs/Point[]     waypoints
builtin_interfaces/Time[] waypoint_times    # same length as waypoints
float64[]                 position_sigma    # metres, 1-sigma radius per waypoint

float64 horizon_sec
string  model_id                        # "cv_baseline" | "<ml model tag>"
```

### `msg/FormationEstimate.msg`
```
std_msgs/Header header

string    formation_class               # one of the 8 classes
float64   confidence
float64[] class_probabilities           # length 8, ordered per FORMATION_CLASSES
string[]  member_ids                    # global_ids participating
bool      from_truth                    # true when fed parametric truth (debug)
```

### `msg/Narration.msg`
```
std_msgs/Header header

string text
string template_id                      # grounding skeleton used
string source                           # "template" | "llm"
uint8  severity

string[] referenced_ids
builtin_interfaces/Time trigger_time

uint8 SEVERITY_INFO=0
uint8 SEVERITY_NOTICE=1
uint8 SEVERITY_ALERT=2
```

### `msg/NodeHealth.msg`
```
std_msgs/Header header

string  node_name
bool    passthrough_truth               # fault-isolation switch state
float64 publish_rate_hz
uint32  messages_dropped
string  status_text
```

### `msg/SceneUpdate.msg`
```
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
```

---

## Topic table

| Topic | Type | Rate | QoS | Publisher |
|---|---|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | 50 Hz | BEST_EFFORT | `capstone_sim` |
| `/truth/scene` | `TruthScene` | 50 Hz | BEST_EFFORT | `capstone_sim` |
| `/truth/receivers` | `ReceiverStateArray` | 50 Hz | BEST_EFFORT | `capstone_sim` |
| `/truth/association` | `BurstAssociationTruth` | ~10 Hz | RELIABLE | `capstone_sim` |
| `/receivers` | `ReceiverStateArray` | 50 Hz | BEST_EFFORT | `capstone_sim` |
| `/rf_observations` | `RfObservationSet` | ~10 Hz | RELIABLE | `capstone_sim` |
| `/embeddings` | `Embedding` | ~10 Hz | RELIABLE | `capstone_rff` |
| `/fingerprints` | `FingerprintSet` | ~0.1 Hz | RELIABLE, TRANSIENT_LOCAL | `capstone_base` |
| `/tracks` | `TrackArray` | 10 Hz | RELIABLE | `capstone_track` |
| `/predictions` | `PathPrediction` | 1–2 Hz | RELIABLE | `capstone_predict` |
| `/formation` | `FormationEstimate` | 1–2 Hz | RELIABLE, TRANSIENT_LOCAL | `capstone_predict` |
| `/narration` | `Narration` | event | RELIABLE, TRANSIENT_LOCAL | `capstone_narrate` |
| `/scene_update` | `SceneUpdate` | 10 Hz | RELIABLE, TRANSIENT_LOCAL | `capstone_bridge` |
| `/health/<node>` | `NodeHealth` | 1 Hz | BEST_EFFORT | all |

**Frames:** everything in ROS ENU (REP-103), origin at base station. Conversion
to Three.js Y-up happens **only** inside `capstone_bridge`. No frame maths
anywhere in JS.

---

## Phase 0 exit gate

- [ ] `colcon build` clean; `ros2 interface show` succeeds for all 17 types
      (count corrected from 15 -> 17: FormationEstimate + Narration are pulled in
      as SceneUpdate dependencies and must be typed for capstone_eval regardless
      of who publishes them)
- [ ] Every node in the layout exists as a stub publishing correctly-typed zeros
- [ ] `ros2 topic hz` on each topic matches the table within ±10%
- [ ] `/scene_update` reaches a `roslibjs` client; late-joining client receives
      last message immediately (TRANSIENT_LOCAL verified)
- [ ] 60-second run, zero dropped messages, `ros2 bag record -a` produces a
      replayable bag
- [ ] `use_sim_time:=true` honoured by every node
