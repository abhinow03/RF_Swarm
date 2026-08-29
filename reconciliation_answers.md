# RF_Swarm — Phase 0 reconciliation answers (frozen)

Decisions taken during Phase 0 kickoff, before `capstone_msgs` was frozen.
These reconcile the frozen `capstone_msgs_spec.md` against the real, already-
working viz/rosbridge setup surveyed on `mars-4090`.

## Survey findings (Step 0 / Step 1)

- The AR client (`~/AR-visuliazation`, github abhinow03/AR-visuliazation) is an
  A-Frame/Three.js app that currently runs on **hardcoded demo data** — no
  WebSocket, roslibjs, or rosbridge subscriber. It is not live-wired today.
- The only rosbridge code is `~/rf_rosbridge_test/`, an explicit **connectivity
  placeholder**: a `std_msgs/String` JSON publisher on `/rf/test_emitters` +
  a raw-WS client, built as a colcon overlay on **system ROS 2 Jazzy**
  (`/opt/ros/jazzy`). rosbridge is not in system Jazzy; it was built from source
  in that overlay.
- No conda/mamba/micromamba on this account (only other projects' miniconda).
- `~/rf_env` is a plain venv (Python 3.12.3), the RF-fingerprinting env — untouched.

## Environment decision — Option E (extend, do not parallel-install)

Extend system ROS 2 Jazzy via a colcon overlay workspace inside `RF_Swarm`.
Do **not** install miniforge/RoboStack (would be a second parallel ROS on a
shared machine, and conda isn't even present). Layout:

- `RF_Swarm/capstone_ws/` — colcon overlay on `/opt/ros/jazzy`.
- `RF_Swarm/.venv/` — project venv, created `--system-site-packages` so Jazzy's
  `rclpy` is visible; our extra Python deps live here. Verified: `rclpy` resolves
  to `/opt/ros/jazzy/lib/python3.12/site-packages/rclpy`, node create/init/shutdown OK.
- `RF_Swarm/capstone_ws/activate.sh` — hand-sourced (NOT in `.bashrc`); sources
  Jazzy, activates the venv, exports `ROS_DOMAIN_ID=42`, sources the overlay if built.

### ROS_DOMAIN_ID = 42
`rf_rosbridge_test` sets no domain → runs on default **0**. This project uses
**42** to keep our (zero-filled Phase 0 stub) DDS traffic off the same domain as
the existing setup on this shared box.

## `/scene_update` wire format — Option C

`capstone_bridge` publishes **`std_msgs/String`** on `/scene_update`; `data` is a
JSON string. The client does one `JSON.parse(msg.data)`. No nested typed ROS
parsing in JS (Option A rejected). All ENU->Y-up frame conversion is server-side
in the bridge only.

Envelope (one WS msg per publish, ~10 Hz):
`{ schema_version, scenario_id, k_estimated, pipeline_latency_ms, tracks:[...] }`

Per-track object + provenance from typed `TrackState`:

| wire field | source | notes |
|---|---|---|
| `id` | `global_id` | "d1".."dK" |
| `cls` | constant `"swarm"` for now | valid `CLASSES` key = neutral look. NOT `"unknown"` (reserved as the client's fault-isolation fallback for malformed payloads). Post-STGT: join `FormationEstimate.formation_class` — a **swarm-level** label, same for every track in a scenario, not per-track. |
| `confidence` | join `FingerprintSet.assignments[].confidence` by `global_id` | **emit 0 if no assignment yet** (e.g. STATUS_INITIALIZING before N=120). Never hardcode 1.0 — stub the value the real join computes, never the join logic. Flicker between real value and 0 during coasting is expected. |
| `pos`, `vel` | ENU->Y-up: `three = [enu.x, enu.z, -enu.y]` | applied to both |
| `status` | `TrackState.status` uint8 | `0 init, 1 tracking, 2 coasting, 3 lost` |
| `pos_sigma` | `sqrt((cov[0]+cov[7]+cov[14])/3)` from 6x6 row-major covariance | scalar radius, rotation-invariant, computed pre-rotation; render-only (eval reads full covariance from typed `/tracks`) |
| `gdop` | `TrackState.gdop` | passthrough |

Client parser (`trackToEmitter`, to be written in Phase 6) maps a wire-track to
the client's existing emitter object (`id,cls,confidence,pos[],vel[],driven,mode`)
and carries `status`/`pos_sigma`/`gdop` with defaults `"tracking"/0/0`. Keep the
guard `CLASSES[t.cls] ? t.cls : 'unknown'`. Live tracks: `driven:true`, `mode:'swarm'`.

### Typed `SceneUpdate.msg`
Retained in `capstone_msgs` for schema documentation + `capstone_eval`, but
**not published as-is in Phase 0**. No `/scene_update_typed` topic is added
preemptively — add one only if a real eval requirement appears. A NOTE in
`SceneUpdate.msg`'s comment block records this divergence.

### `/formation` and `/narration`
Stay as their own `TRANSIENT_LOCAL` topics (own small client subscriptions in
Phase 6), NOT folded into the 10 Hz `/scene_update` envelope — different rates
(1-2 Hz / event vs 10 Hz) and latching semantics.

## Message count
`capstone_msgs` defines **17** types (spec exit gate corrected from a stale "15").
All 17 exist regardless of who publishes them, because `SceneUpdate` embeds
`FormationEstimate` + `Narration`, and `capstone_eval` needs typed versions of
everything to score against truth.
