# Phase 2 noise model — physics derivation (PRE-REGISTERED, review before wiring)

**Status:** Step-1 derivation for review. No generator code written yet; no σ
values wired into scenario YAML yet. These numbers are pre-registered here so the
Phase-2 gate can check measured residual σ against them *without* anyone having
adjusted them after seeing results (the forward-only / no-tuning-after-results
discipline).

Forward-only: truth (`/truth/scene`, `/truth/receivers` from Phase 1) is the
input; this phase adds measurement error to produce `/rf_observations` and the
noise-corrupted `/receivers`. Nothing solves backward.

All constants below are sourced from the RFF side (`~/CAPSTONE/DL_model/gen_rff`,
read-only) so the simulated signal stays consistent with the real fingerprint
pipeline. Cited inline.

---

## 0. Protocol tier — THE decision that sets bandwidth (needs your confirm)

The RFF pipeline is **multi-protocol** with three locked tiers
(`gen_rff/Final_demo/separate_and_id.py`):

| tier | sample rate | occupied BW | carrier | window | swarm hw |
|---|---|---|---|---|---|
| **BLE** (GFSK) | 6 MS/s | **~1 MHz** (1–2 MHz ch) | 2.4 GHz | 1850 smp (308 µs) | `esp32_*` |
| WiFi (802.11) | 25 MS/s | ~20 MHz | 2.4/5.8 GHz | 256 smp | — |
| drone OcuSync | 50 MS/s | ~7–10 MHz (hops) | 5.8 GHz | 4096 smp | `drff_airframe_*` |

`EmitterTruth.hardware_uid` examples in the frozen spec are literally
`esp32_lab_03` **and** `drff_airframe_7` — so the tier is a real modelling
choice, and it drives ToA precision through the CRLB bandwidth `B`.

**Recommendation: BLE tier.** Rationale: (a) ESP32/BLE beacons are realistic,
cheap, swarm-able emitters; (b) narrowband → the *hardest* honest ToA case, so we
don't accidentally flatter the multilateration; (c) the slow burst cadence
matches the base station's `N=120 @ ~0.1 Hz` accumulation (see §5). Everything
below is derived for **BLE**; swapping tiers only changes `B` (and carrier), and
the YAML makes it a one-line change.

> ⚠️ Confirm the tier before I wire anything — it changes every σ_toa by up to ~10×.

---

## 1. ToA — `toa_sigma`  (range-dependent, the honest model)

CRLB (simplified form): **σ_toa ≈ 1 / (2π · B · √SNR)**, with `B` the RMS (Gabor)
bandwidth and SNR linear.

- `B` for BLE: rectangular-PSD approx of the ~1 MHz GFSK channel →
  `B_rms = W/√12 = 289 kHz` (W=1 MHz).
- SNR is **not fixed** — it falls with range via path loss (§2), so σ_toa is
  reported as a function of range. This ties ToA precision to RSSI through one
  physical SNR, which is more honest than a single fixed σ.

**LOCKED (decision 2 = convention A):** RFF RSSI model `RSSI(d) = −30 − 10·2.2·
log10(d)`, noise floor −108 dBm, B_rms=289 kHz:

| range | SNR | σ_toa | range-equiv (×c) |
|---|---|---|---|
| 30 m | 46 dB | 2.93 ns | 0.88 m |
| 50 m | 41 dB | 5.13 ns | 1.54 m |
| 100 m | 34 dB | 11.0 ns | 3.30 m |
| 180 m | 28 dB | 21.0 ns | 6.30 m |

Sub-meter-to-few-meter ranging noise — *dwarfed* by the ~30 ns (≈9 m) disciplined
clock nuisance in §4, by design (reconciled from ~100 ns; see §4a).

**Implementation:** compute per-observation SNR from the RSSI model, then
σ_toa(SNR). The YAML pre-registers the *inputs* (B, P_tx, n, NF), not a single
number, since the model is range-dependent.

---

## 2. RSSI — `rssi_dbm`, `rssi_sigma`  (path loss + shadowing)

> **Propagation assumption (explicit, per review):** this model assumes
> **outdoor / open-field LOS propagation** (n=2.2). **Known limitation:** if the
> actual demo venue is **indoor** (lab room, conference hall), multipath pushes
> the path-loss exponent well above 2.2 (typ. 2.7–4+) and adds heavier
> multipath/fading — this model would then *understate* RSSI/ToA error. This is a
> conscious, disclosed simplification, not an undisclosed one; revisit n and the
> shadowing term if the venue is indoors.

- Path-loss exponent **n = 2.2** (outdoor near-LOS; BLE D2 captures were outdoor
  Loc1–4, `gen_rff/Final_demo/README.md`) — and this **matches the RFF side's own
  `demo_router/geometry.py` (`_PATHLOSS_N = 2.2`)**, so the two sides already agree
  on propagation.
- **Link calibration — decision (2) = convention A, WITH honest provenance.**
  `RSSI(d) = −30 − 10·2.2·log10(d)`, adopted from `demo_router/geometry.py`.
  - **What `−30` is NOT:** it is *not* a hardware-calibrated P0 tied to the ESP32
    TX power. That module is self-declared **"MOCKED rssi/aoa/tdoa … DEMO-SIDE —
    NOT PAPER RESULTS"** — it fabricates coherent rssi/aoa/tdoa from made-up 2D
    positions so the AR/multilateration demo layer is self-consistent. Physical
    grounding was explicitly not its purpose. So `−30 dBm@1m` is an **adopted
    reference *scale*** (keeps our RSSI on the same scale as the RFF demo), an
    honest *assumption*, not a measurement.
  - **Why it's still fine for Phase 2:** RSSI is not an identity feature (the
    encoder uses IQ, not RSSI), so the absolute level only sets the SNR/difficulty
    knob — which we pre-register as an assumption anyway. The physically
    load-bearing parts are the **n = 2.2 slope** (independently defensible as
    outdoor LOS, and it happens to match the demo) and the shadowing σ; those
    stand on their own regardless of the `−30` offset.
  - **Rejected alternative B** (`P_tx = 20 dBm` link budget): **20 dBm is not
    traceable** — ESP32-C3 datasheet unverifiable here, collection TX power
    unlogged, real BLE default often 0 dBm. Dropped rather than pre-register an
    untraceable number.
  - **Range definition `d`:** we use **3D Euclidean slant range** (receiver→emitter,
    including altitude) — deliberately diverging from `geometry.py`'s 2D
    `math.hypot(dx,dy)`, because the Phase-1 scene is genuinely 3D (emitters at
    ~50 m). σ tables are functions of range, so this does not change them.
- Log-normal shadowing: **σ_rssi = 3.5 dB** (outdoor LOS) → `rssi_sigma`; additive
  RSSI error `N(0, σ_rssi)`.
- Noise floor: `−174 + 10log10(B=1MHz) + NF(6 dB) = −108 dBm` → feeds SNR in §1.

---

## 3. FDoA — `fdoa_hz`, `fdoa_sigma`  (CFO + Doppler, UNSEPARATED)

Per the frozen spec, `fdoa_hz` is CFO **and** Doppler, not separated at a single
receiver. So `fdoa_hz = CFO_emitter + f_Doppler + N(0, σ_fdoa)`.

- **Doppler** (from Phase-1 truth): `f_d = f_c · v_radial / c`, `v_radial` =
  emitter velocity projected on the receiver LOS. Max ≈ 16 Hz at v=2 m/s.
- **CFO** = per-emitter oscillator offset, a **hardware fingerprint** drawn ONCE
  per emitter, not per burst (RFF side is explicit: *"CFO is fingerprint"*,
  *"CFO is itself a transmitter cue"* — `drone_native.py`, `wisig_manytx.py`).
  **Convention resolved (decision 6):** the RFF `±0.005` is `cfo` in
  `phase = 2π·cfo·t`, `t = arange(WINDOW_SIZE)` → **cycles/sample, normalized by
  sample rate** (verbatim: *"±0.005 cyc/sample is ~1.3 phase cycles across the
  window"*, `wisig_manytx.py`). At BLE 6 MS/s that is **±30 kHz = ±12.5 ppm** of
  the 2.4 GHz carrier. My earlier generic ±20 ppm (±48 kHz) is the **same order of
  magnitude** — the two agree — so I lock the RFF-traceable value in ppm framing:
  **CFO ~ U(±12.5 ppm) ≡ ±0.005 cyc/sample @ 6 MS/s ≡ ±30 kHz**, drawn once per
  emitter. (Caveat: ±0.005 is the encoder's *augmentation* range, not a per-unit
  *measured* CFO — none is recorded in the repo — but ±12.5 ppm is also physically
  plausible for a 40 MHz ESP32 crystal, so it doubles as a defensible physical
  spec.) This CFO *dominates* Doppler (30 kHz ≫ 16 Hz) — physically correct for
  slow emitters, and why the two are honestly lumped in `fdoa_hz`.
- Measurement noise: `σ_fdoa ≈ 1/(2π·T·√SNR)`, T=308 µs, under locked convention
  A → **2.7 Hz @30 m → 19.7 Hz @180 m** (4.8 Hz @50 m, 10.3 Hz @100 m).

---

## 4. Clock — per-receiver NUISANCE STATE (the real TDoA challenge)

Not per-measurement noise: each receiver gets its own persistent bias+drift,
sampled ONCE per run from the scenario seed, stored on
`ReceiverState.clock_bias / clock_drift / clock_bias_sigma`. This is where
Phase-1's `/truth/receivers` vs `/receivers` divergence begins.

Modelling a **synchronised** array (GPS/PTP-disciplined) with a residual.
**RECONCILED 2026-08-26** (Phase-4 observability finding — see §4a; the original
Phase-2 pre-registration is preserved there): the injected spread now MATCHES the
published residual (was `bias_max 100 ns` / `sigma 30 ns` — an overconfident,
inconsistent pair):

- `clock_bias0 ~ U(±30 ns)` per receiver — residual sync offset (≈9 m equiv);
  the half-width equals `clock_bias_sigma` (conservative, never overconfident).
- `clock_drift ~ U(±0.3 ns/s)` per receiver — residual freq error of a *stable*
  disciplined oscillator; `bias(t) = bias0 + drift·t`.
- `clock_bias_sigma = 30 ns` — the receiver's self-reported residual uncertainty
  (published on `/receivers`; it IS the EKF's informative clock prior, see §4a).

The ToA a receiver reports is `true_toa + clock_bias(t) + N(0, σ_toa)`. TDoA
differences see `(bias_i − bias_j)` — a ~30 ns (≈9 m) structured error that
**dominates** the ~1 m σ_toa. That is deliberate: it makes multilateration a
genuine estimate-the-nuisance problem, not a decorative jitter. **Per-receiver
and distinct** (different draw each rx) — the gate checks they are not identical.

**Exposure decision (what the estimator is told):** the TRUE per-receiver bias
lives on `/truth/receivers.clock_bias` (drift on `.clock_drift`, sigma 0 — truth
is exact). `/receivers` — the estimator-visible copy — carries **belief 0** for
`clock_bias`/`clock_drift` and only the stated residual `clock_bias_sigma`. The
estimator is NOT handed the true bias; it must estimate the per-receiver offset as
a nuisance state (that is the whole point). So the ToA residual, per receiver, has
a **mean equal to that receiver's true bias** (the intentional offset), and is
zero-mean only after the known-per-receiver bias is removed — the gate uses this
to tell "designed clock offset" from "bug." This is where Phase-1's
`/truth/receivers` vs `/receivers` divergence begins.

> **PHASE 4 DESIGN REQUIREMENT (do not assume away):** the EKF state vector MUST
> include **per-receiver clock bias (and drift)** as estimated states. `/receivers`
> exposes only `clock_bias_sigma` — the true bias is never readable there, only
> observable through its effect on the ToA residuals being fused. Scoping Phase 4
> as if clock bias were knowable/correctable from the receiver message will either
> silently underperform or leave the state vector short the dimensions it needs.
> The per-rx biases ARE kept as estimated states — but see §4a: they are only
> WEAKLY observable from a compact swarm, so an informative prior is required.

---

## 4a. Phase-4 observability finding + the disciplined-array resolution (2026-08-26)

**What Phase-2 pre-registered (unchanged record):** `bias_max 100 ns`, `drift 1 ns/s`,
`sigma 30 ns`; and the claim (§4 blockquote, original) that "with N≥4 receivers +
one reference clock the per-rx biases are jointly observable from the TDoA set."

**What the Phase-4 observability analysis found (this claim was WRONG for a compact
swarm):** the per-receiver clock bias is **near-UNOBSERVABLE from TDoA of a compact
formation**. A common translation of the whole formation is degenerate with a
receiver-clock-bias shift — in the near field a ~24 m cluster seen from ~130 m
subtends nearly identical angles at every receiver, so sliding the formation looks
just like re-timing the receivers. Proven by SVD of the stacked snapshot Jacobian
(24 measurements × 22 unknowns): σ_min ≈ 0.028, condition ~1.2e4, and the near-null
right-singular-vector is a common formation translation coupled to a clock shift.
Motion + constant-velocity freedom does **not** rescue it (the velocity states
follow the slowly-changing geometry and re-absorb the clock). Dynamically confirmed:
with a flat clock prior the filter fits measurements at the noise floor (~6 m innov)
while position and clock slide along the null to 35 m / 73 m; seeded with the TRUE
clock, it wanders back off.

**Resolution (user-approved): disciplined-array informative prior.** The array is
GPS/PTP time-disciplined; the EKF keeps clock as ESTIMATED states but seeds the
prior from the PUBLISHED `clock_bias_sigma` (mean 0 = belief "synced", σ = stated
residual). This honors "estimate, never read `/truth`" (the value is only on
`/truth/receivers`) while regularizing the null space. Required reconciling the sim
so the injected spread matches the published σ (the `100→30 ns`, `1→0.3 ns/s` change
above) — a deliberate, disclosed re-registration derived from the analysis, NOT a
fit to a target metric. Rejected alternatives: a reference emitter at a surveyed
location (fully observable, meter-level — the lever if better accuracy is needed)
and leaving the array unsynced (localization limited to ~30–50 m).

**Validated outcome (offline gate_k6 replay, 8 CLOCK-noise seeds, faithful
measurement model — `capstone_track/scripts/validate_ekf.py`):** position RMSE
**~17 m mean (8–27 m across clock realizations, p90 ~26 m)**, bounded and stable.
The spread is real: which clock draw you get — how much it projects onto the near-
null space — dominates the error, not measurement noise. The clock is pinned at the
~30 ns prior and, if anything, drifts a bit worse (relative 4-vector ~49 m ≈ prior
level) — TDoA adds little clock info, so the estimate is prior-dominated (do NOT
claim sharp clock recovery). An **iterated EKF** update is required (a single
linearization at the clock-corrupted init is both inaccurate ~32 m AND badly
overconfident, NEES ~130); even iterated, residual optimism remains (position NEES
median ~10, worst-draw ~80 — ~2–3× overconfident in std) so downstream threat
assessment should inflate the reported track covariance ~3×. Process noise
`q_acc=0.02`, `q_bias=1.0` was set for filter consistency, not to hit an RMSE.

---

## 5. Burst cadence + AoA

- **Burst rate: 2 Hz per emitter** (my call). Chosen so K=6 × 2 Hz × 10 s = 120
  windows per base-station fingerprint output at ~0.1 Hz — consistent with
  `FingerprintSet` `N=120 @ 0.1 Hz`. One `RfObservation` per receiver per burst,
  joined by `burst_seq`. `/truth/association` (`BurstAssociationTruth`) publishes
  the true emitter↔burst_seq alongside, for `capstone_eval` only.
- **AoA: `aoa_valid=false`, az/el/σ = NaN** for all observations
  (single-antenna receivers). No fake values behind the flag.

---

## 6. Proposed YAML block (to add to each scenario, pending your review)

```yaml
rf:
  protocol: ble             # tier -> bandwidth/carrier (BLE|wifi|drone_ocusync)
  carrier_hz: 2.4e+9        # +exponent so PyYAML parses float, not str
  bandwidth_hz: 1.0e+6      # occupied GFSK channel; B_rms = bw/sqrt(12) in CRLB
  sample_rate_hz: 6.0e+6    # BLE capture rate (RFF PROTO_NATIVE) — with window_samples gives T
  window_samples: 1850      # BLE window (RFF PROTO_NATIVE); T = 1850/6e6 = 308 us -> sigma_fdoa
  path_loss_exponent: 2.2   # outdoor near-LOS (matches RFF demo_router/geometry.py)
  # --- link calibration: decision (2), shown under recommended convention A ---
  rssi_ref_dbm_at_1m: -30.0 # A: RFF convention P0@1m. (B would instead set
                            #    tx_power_dbm: <real datasheet value> + free-space PL)
  noise_figure_db: 6.0
  rssi_shadow_sigma_db: 3.5
  cfo_ppm: 12.5             # per-emitter, once from seed (= ±0.005 cyc/sample @6MS/s = ±30kHz)
  burst_rate_hz: 2.0        # per emitter
clock:                      # RECONCILED 2026-08-26 (§4a): injected spread == published sigma
  bias_max_ns: 30.0         # per-receiver bias0 ~ U(±bias_max)  (was 100)
  drift_max_ns_per_s: 0.3   # per-receiver drift ~ U(±drift_max) (was 1.0)
  bias_sigma_ns: 30.0       # published residual = EKF informative clock prior
```

`bandwidth_hz`, `carrier_hz` switch with `protocol`; the derivation re-runs from
these. Everything is in the YAML, nothing hardcoded in `capstone_sim`. **The
`rssi_ref_dbm_at_1m` line reflects decision (2) still-pending** — it locks once you
choose convention A or supply the real TX power for B.

---

## 7. Decision status (post-review)

1. **Protocol tier = BLE** — ✅ CONFIRMED (matches the ESP32 fingerprint hardware class).
2. **Link calibration** — ✅ RESOLVED = convention **A** (`RSSI = −30 − 10·2.2·
   log10(d)`), but with **honest provenance (§2):** `−30` is an *adopted reference
   scale* from the RFF demo's self-declared MOCK geometry, NOT a hardware-
   calibrated P0; the load-bearing physics is n=2.2 (defensible outdoor) + the
   shadowing σ. `20 dBm` (B) rejected as untraceable. `d` = **3D slant range**
   (diverges from the demo's 2D on purpose). σ_toa 0.88 m→6.30 m (§1).
3. **Outdoor, n=2.2, shadow 3.5 dB** — ✅ CONFIRMED; outdoor assumption + indoor
   limitation now stated explicitly in §2.
4. **σ_toa range-dependent** — ✅ CONFIRMED (honest model kept).
5. **Clock = disciplined-array residual** — ✅ CONFIRMED, **reconciled 2026-08-26 to
   ±30 ns bias / ±0.3 ns/s drift** (was ±100/±1) after the Phase-4 observability
   finding (§4a): the compact-swarm clock is near-unobservable from TDoA, so the
   injected spread is matched to the published `clock_bias_sigma` and used as the
   EKF's informative prior. The original pre-registration is preserved in §4a.
6. **CFO** — ✅ RESOLVED: RFF `±0.005` = cyc/sample = **±12.5 ppm** @ 2.4 GHz;
   agrees with my ±20 ppm at order of magnitude; locked at **±12.5 ppm** (RFF-
   traceable, ppm framing), see §3.

---

## 8. Multi-protocol extension (Run B/C, pre-registered 2026-08-28, before running)

§0 already cited WiFi and drone_ocusync tier parameters (sample rate, window,
occupied BW, carrier) from the same RFF source (`gen_rff/Final_demo/
separate_and_id.py`'s `PROTO_NATIVE` table) but never carried the §1–§5
derivation past BLE, since §7-1 scoped Phase 2 to BLE only. This section
mechanically applies the SAME locked formulas (§1 CRLB, §3 FDoA) to those
already-cited parameters — no new physics, no new formula, just the two
protocols BLE's derivation skipped.

**Confirmed 2026-08-28** (both re-verified live against the actual RFF repo,
not assumed):
- WiFi CFO: `wisig_manytx.py`'s augmentation range is `CFO_MAX = 0.003`
  cyc/sample (reduced from an earlier 0.005, per the file's own comment: "256-
  sample packets are short ... possibly too aggressive" at 0.005) → **±0.003 ×
  25e6 = ±75 kHz = ±31.25 ppm @ 2.4 GHz**. RFF-traceable, same derivation
  pattern as BLE's §3.
- Drone CFO: **NOT RFF-traceable** — `drone_native.py`'s own augmentation
  explicitly has **NO CFO term** ("CFO is fingerprint" — the DRFF training
  pipeline deliberately treats oscillator offset as a fixed identity cue, not
  a randomized nuisance, so there is no empirical range to cite the way BLE/
  WiFi have one). **Locked value: ±20 ppm @ 5.8 GHz (≈±116 kHz)** — an
  illustrative, physically-plausible consumer-oscillator assumption, explicitly
  NOT RFF-derived. Same honest-disclosure pattern as §2's `−30 dBm` RSSI
  reference (adopted scale, not a measurement).
- Path-loss exponent n=2.2 carries over unchanged — `rffp/geometry.py`'s
  `_PATHLOSS_N` is a single global constant, not protocol-specific, so no
  per-protocol variation exists in the reference implementation either.
- RSSI reference (`rssi_ref_dbm_at_1m = -30.0`) reused unchanged across all
  three protocols — it was already disclosed for BLE as "an adopted reference
  *scale*, not a hardware-calibrated P0" (§2); no more-traceable per-protocol
  TX power data exists for WiFi/drone either, so introducing per-protocol
  values here would manufacture false precision rather than remove an
  assumption. RSSI's role stays what §2 says it is: an SNR/difficulty knob,
  not an identity feature.

**CRLB tables** (same formula as §1/§3, `σ_toa ≈ 1/(2π·B_rms·√SNR)`,
`σ_fdoa ≈ 1/(2π·T·√SNR)`, noise floor `= −174 + 10log10(B) + NF(6dB)`, RSSI/SNR
per the reused §2 link):

| protocol | B (occupied) | B_rms | T (window) | noise floor |
|---|---|---|---|---|
| wifi | 20 MHz | 5.77 MHz | 10.24 µs (256@25MS/s) | −95.0 dBm |
| drone_ocusync | 8.5 MHz (mid of 7–10 MHz hop BW) | 2.45 MHz | 81.92 µs (4096@50MS/s) | −98.7 dBm |

| protocol | range | SNR | σ_toa | σ_fdoa |
|---|---|---|---|---|
| wifi | 30 m | 32.5 dB | 0.65 ns | 368.9 Hz |
| wifi | 50 m | 27.6 dB | 1.15 ns | 647.0 Hz |
| wifi | 100 m | 21.0 dB | 2.46 ns | 1386.9 Hz |
| wifi | 180 m | 15.4 dB | 4.70 ns | 2647.5 Hz |
| drone | 30 m | 36.2 dB | 1.00 ns | 30.1 Hz |
| drone | 50 m | 31.3 dB | 1.76 ns | 52.7 Hz |
| drone | 100 m | 24.7 dB | 3.77 ns | 113.0 Hz |
| drone | 180 m | 19.1 dB | 7.20 ns | 215.7 Hz |

WiFi's much wider occupied bandwidth gives sub-5ns ToA precision even at
180 m — an order of magnitude tighter than BLE's narrowband case (§1: 21 ns
@ 180 m). Drone's larger window (82 µs vs BLE's 308 µs is actually *shorter*
— drone's T is 4x shorter than BLE despite the higher sample rate, because
the native window is only 4096 samples) gives tighter FDoA than BLE. Both are
easier ToA/FDoA regimes than BLE — consistent with BLE being "the hardest
honest case" per §0's original tier recommendation, so Run A (BLE) is not an
easy baseline being compared against harder B/C; if anything the opposite.

**Proposed YAML blocks** (to add to `gate_k6_wifi.yaml` / `gate_k6_drone.yaml`,
`clock:` block unchanged from §6 — clock nuisance is array-side, not
protocol-side):

```yaml
rf:
  protocol: wifi
  carrier_hz: 2.4e+9
  bandwidth_hz: 20.0e+6
  sample_rate_hz: 25.0e+6
  window_samples: 256
  path_loss_exponent: 2.2
  rssi_ref_dbm_at_1m: -30.0
  noise_figure_db: 6.0
  rssi_shadow_sigma_db: 3.5
  cfo_ppm: 31.25
  burst_rate_hz: 2.0
```
```yaml
rf:
  protocol: drone_ocusync
  carrier_hz: 5.8e+9
  bandwidth_hz: 8.5e+6
  sample_rate_hz: 50.0e+6
  window_samples: 4096
  path_loss_exponent: 2.2
  rssi_ref_dbm_at_1m: -30.0
  noise_figure_db: 6.0
  rssi_shadow_sigma_db: 3.5
  cfo_ppm: 20.0
  burst_rate_hz: 2.0
```

**Status: CONFIRMED 2026-08-28** — drone CFO explicitly approved as the
illustrative ±20ppm value (not RFF-traceable, disclosed above) rather than
reusing WiFi's 31.25ppm. Safe to wire into new scenario YAMLs.
