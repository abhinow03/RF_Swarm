"""Turn the raw run capture (_run_raw/*.jsonl) into an organized, navigable demo:
01_inputs / 02_multilateration / 03_model_io / 04_outputs. Pure stdlib + numpy."""
import json, os, csv, sys, shutil
import numpy as np
sys.path.insert(0, "/home/pw26_akp_01/RF_Swarm/capstone_ws/src/capstone_track")
from capstone_track.ekf import tdoa_init, C_MPS

RAW = "/home/pw26_akp_01/RF_Swarm/demo/_run_raw"
DEMO = "/home/pw26_akp_01/RF_Swarm/demo"
FORMATION_NAMES = ["v_shape","encirclement","column","diamond","dispersed","converging","shield","transitioning"]

def load(name): return [json.loads(l) for l in open(os.path.join(RAW, name+".jsonl"))]
def wcsv(path, header, rows):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
def wj(path, obj): json.dump(obj, open(path,"w"), indent=2)
for d in ["01_inputs","02_multilateration","03_model_io","04_outputs","figures"]:
    os.makedirs(os.path.join(DEMO,d), exist_ok=True)

truth   = load("truth_scene")
recv    = load("receivers")
obs     = load("rf_observations")
tassoc  = {r["burst_seq"]: r["truth_emitter_id"] for r in load("truth_association")}
assoc   = load("associations")
tracks  = load("tracks")
forms   = load("formation")
fps     = load("fingerprints")
narr    = load("narration")
preds   = load("predictions")
emb     = load("embeddings")
scene   = load("scene_update")

# ---- gid -> truth emitter (compose blind /associations with /truth/association) ----
from collections import Counter, defaultdict
votes = defaultdict(Counter)
latest_map = assoc[-1]["map"] if assoc else []
for a in latest_map:
    te = tassoc.get(a["burst_seq"])
    if te is not None and a["confidence"] > 0: votes[a["global_id"]][te] += 1
gid2em = {g: c.most_common(1)[0][0] for g,c in votes.items()}

def truth_at(emitter, t):
    best=None; bd=1e9
    for s in truth:
        if abs(s["t"]-t) < bd:
            for e in s["emitters"]:
                if e["id"]==emitter: best=e["pos"]; bd=abs(s["t"]-t)
    return best

# ================= 01_inputs =================
shutil.copy("/home/pw26_akp_01/RF_Swarm/capstone_ws/src/capstone_scenarios/scenarios/gate_k6.yaml",
            os.path.join(DEMO,"01_inputs","scenario.yaml"))
r0 = recv[0]["rx"]
wcsv(os.path.join(DEMO,"01_inputs","receiver_array.csv"),
     ["rx_id","E_m","N_m","U_m","clock_bias_sigma_ns"],
     [[r["id"],r["pos"][0],r["pos"][1],r["pos"][2],round(r["clock_bias_sigma"]*1e9,1)] for r in r0])
# ground-truth emitter trajectories (downsample to ~1 Hz)
gt_rows=[]
for s in truth[::50]:
    for e in s["emitters"]:
        gt_rows.append([round(s["t"],2),e["id"],round(e["pos"][0],3),round(e["pos"][1],3),round(e["pos"][2],3),round(e["cfo_hz"]/1e3,2)])
wcsv(os.path.join(DEMO,"01_inputs","ground_truth_emitters.csv"),
     ["t_s","emitter_id","E_m","N_m","U_m","cfo_khz"], gt_rows)
# raw RF observations the estimators SEE (first 3 bursts, all receivers)
ro=[]
for b in obs[:3]:
    for o in b["obs"]:
        ro.append([b["burst_seq"],round(b["t"],3),o["rx"],f"{o['toa']:.9f}",f"{o['toa_sigma']:.3e}",round(o["rssi_dbm"],1),round(o["fdoa_hz"],1),round(o["snr_db"],1)])
wcsv(os.path.join(DEMO,"01_inputs","rf_observations_sample.csv"),
     ["burst_seq","t_s","rx_id","toa_s","toa_sigma_s","rssi_dbm","fdoa_hz","snr_db"], ro)
# blind embeddings sample (encoder output) — 3 vectors, first 8 dims + norm
es=[]
for e in emb[:3]:
    v=np.array(e["vector"]); es.append([e["burst_seq"],e["rx"],e["dim"],e["sha"],round(float(np.linalg.norm(v)),4)]+[round(x,4) for x in v[:8].tolist()])
wcsv(os.path.join(DEMO,"01_inputs","embeddings_sample.csv"),
     ["burst_seq","rx_id","dim","encoder_sha","L2_norm"]+[f"d{i}" for i in range(8)], es)

# ================= 02_multilateration =================
# worked example: pick a confidently-associated burst with a full 5-rx observation
rx_pos = np.array([r["pos"] for r in r0])
gid_by_burst = {a["burst_seq"]:a["global_id"] for a in latest_map if a["confidence"]>0}
ex=None
for b in obs:
    if len(b["obs"])==5 and b["burst_seq"] in gid_by_burst and b["t"]>40:
        ex=b; break
if ex:
    toas=np.array([next(o["toa"] for o in ex["obs"] if o["rx"]==j) for j in range(5)])
    sig =np.array([next(o["toa_sigma"] for o in ex["obs"] if o["rx"]==j) for j in range(5)])
    gid=gid_by_burst[ex["burst_seq"]]; em=gid2em.get(gid)
    rd=[float(C_MPS*(toas[j]-toas[0])) for j in range(5)]           # range-differences (metres)
    init=tdoa_init(rx_pos, toas, ref=0)
    # EKF track for this gid near this time
    tk=min(tracks, key=lambda m: abs(m["t"]-ex["t"]))
    est=next((t["pos"] for t in tk["tracks"] if t["id"]==gid), None)
    tp=truth_at(em, ex["t"])
    wj(os.path.join(DEMO,"02_multilateration","tdoa_worked_example.json"), {
        "what_this_shows":"one burst -> multilateration. TDoA = range differences from arrival-time differences; closed-form init then refined by the joint EKF (which also estimates receiver clock bias).",
        "burst_seq":ex["burst_seq"],"time_s":round(ex["t"],3),
        "blind_global_id":gid,"(scoring) true_emitter":em,
        "receiver_positions_ENU":rx_pos.tolist(),
        "time_of_arrival_s (receiver clock, bias-corrupted)":[round(float(x),9) for x in toas],
        "toa_sigma_s":[float(x) for x in sig],
        "range_differences_m  c*(toa_j - toa_0)":[round(x,2) for x in rd],
        "closed_form_tdoa_init_ENU":[round(float(x),2) for x in init],
        "EKF_refined_estimate_ENU":[round(float(x),2) for x in est] if est else None,
        "ground_truth_ENU":[round(float(x),2) for x in tp] if tp else None,
        "init_error_m":round(float(np.linalg.norm(init-np.array(tp))),2) if tp else None,
        "EKF_error_m":round(float(np.linalg.norm(np.array(est)-np.array(tp))),2) if (est and tp) else None,
    })

# tracks vs truth over time + errors
tvt=[]; err_by_gid=defaultdict(list)
for m in tracks:
    for t in m["tracks"]:
        em=gid2em.get(t["id"])
        if em is None: continue
        tp=truth_at(em, m["t"])
        if tp is None: continue
        e=float(np.linalg.norm(np.array(t["pos"])-np.array(tp)))
        tvt.append([round(m["t"],2),t["id"],em]+[round(x,2) for x in t["pos"]]+[round(x,2) for x in tp]+[round(e,2),round(t["gdop"],3),t["status"]])
        if m["t"]>40: err_by_gid[t["id"]].append(e)
wcsv(os.path.join(DEMO,"02_multilateration","tracks_vs_truth.csv"),
     ["t_s","global_id","true_emitter","est_E","est_N","est_U","truth_E","truth_N","truth_U","error_m","gdop","status"], tvt[::6])
allerr=[e for v in err_by_gid.values() for e in v]
wj(os.path.join(DEMO,"02_multilateration","localization_summary.json"), {
    "note":"position error = EKF track vs ground-truth emitter (matched via blind fingerprint association). Frame: ENU metres (E,N,U); the client-facing /scene_update separately converts to Three.js Y-up.",
    "position_RMSE_m":round(float(np.sqrt(np.mean(np.square(allerr)))),2),
    "position_mean_m":round(float(np.mean(allerr)),2),
    "position_p90_m":round(float(np.percentile(allerr,90)),2),
    "per_track_mean_error_m":{g:round(float(np.mean(v)),2) for g,v in err_by_gid.items()},
    "gid_to_true_emitter":gid2em,
    "association_bijective":len(set(gid2em.values()))==len(gid2em),
})

# ================= 03_model_io =================
# fingerprint (base station) io
fp=fps[-1]
wj(os.path.join(DEMO,"03_model_io","fingerprint_io.json"), {
    "model":"base station: eigengap K-estimation + centroid-merge on pooled BLE embeddings",
    "INPUT":{"n_accumulated_tracks":fp["n_accumulated"],"encoder_sha":fp["sha"]},
    "OUTPUT":{"k_estimated":fp["k_estimated"],"eigengap_value":round(fp["eigengap_value"],4),
              "eigenvalues":[round(x,4) for x in fp["eigenvalues"]],
              "assignments":[{"global_id":a["global_id"],"confidence":round(a["confidence"],3)} for a in fp["assignments"]]}})
# stgt formation io
f_ex=next((f for f in forms if f["confidence"]>0.8), forms[-1])
wj(os.path.join(DEMO,"03_model_io","stgt_formation_io.json"), {
    "model":"STGT graph-transformer formation classifier (best_model.pt)",
    "INPUT":"pose sequence (50 timesteps x 6 drones x 3 ENU) from EKF /tracks, normalized (x-32.408)/68.177",
    "from_truth":f_ex["from_truth"],
    "OUTPUT":{"formation_class":f_ex["formation_class"],"confidence":round(f_ex["confidence"],4),
              "class_probabilities":{n:round(p,4) for n,p in zip(FORMATION_NAMES,f_ex["class_probabilities"])},
              "member_ids":f_ex["member_ids"]}})
# ekf io (reuse worked example)
if ex:
    wj(os.path.join(DEMO,"03_model_io","ekf_io.json"), {
        "model":"joint iterated EKF (per-track [x y z vx vy vz] + shared per-receiver clock [bias,drift])",
        "INPUT":{"measurement":"z_j = c*(toa_j - toa_0) [range differences, metres]",
                 "range_differences_m":[round(x,2) for x in rd],
                 "receiver_positions_ENU":rx_pos.tolist(),
                 "clock_prior":"disciplined-array clock_bias_sigma = 30 ns (ESTIMATED as a state, never read from /truth)"},
        "OUTPUT":{"track_position_ENU":[round(float(x),2) for x in est] if est else None,
                  "gdop":round(tk["tracks"][0]["gdop"],3),
                  "note":"clock bias is a jointly-estimated state; velocity from the constant-velocity model (TDoA-only)"}})
# llm narration io (ctx in -> narration out) for the real LLM narrations
from capstone_narrate.ctx_builder import build_ctx  # vendored path via activate.sh
llm_ios=[]
prev_class=None
for f in forms:
    cls=f["formation_class"]
    # find an llm narration near this formation time
    prev_class=cls
llm_narr=[n for n in narr if n["source"]=="llm"]
for n in llm_narr:
    # formation just before the narration
    fbefore=[f for f in forms if f["t"]<=n["t"]]
    fc=fbefore[-1] if fbefore else forms[0]
    # previous distinct class for the transition line
    prevs=[f["formation_class"] for f in forms if f["t"]<fc["t"] and f["formation_class"]!=fc["formation_class"]]
    frm=prevs[-1] if prevs else fc["formation_class"]
    snap={"t_start_s":round(n["t"],1),"t_end_s":round(n["t"],1),
          "tracks":[{"global_id":f"d{i}"} for i in range(6)],"k_estimated":6,
          "formation":{"formation_class":fc["formation_class"],"confidence":fc["confidence"],"from":frm,"to":fc["formation_class"]}}
    ctx,kw,meta=build_ctx(snap)
    llm_ios.append({"time_s":round(n["t"],2),
                    "INPUT_ctx (tactical context string fed to Qwen2.5-7B + adapter)":ctx,
                    "OUTPUT_narration (LLM-generated)":n["text"],
                    "template_id":n["template_id"],"source":n["source"]})
wj(os.path.join(DEMO,"03_model_io","llm_narration_io.json"),
   {"model":"Qwen2.5-7B-Instruct (4-bit) + LoRA adapter v5_sft_v5a2, bucket-A path","examples":llm_ios})

# ================= 04_outputs =================
wcsv(os.path.join(DEMO,"04_outputs","tracks.csv"),
     ["t_s","global_id","E","N","U","vE","vN","vU","gdop","status"],
     [[round(m["t"],2),t["id"]]+[round(x,2) for x in t["pos"]]+[round(x,2) for x in t["vel"]]+[round(t["gdop"],3),t["status"]] for m in tracks[::10] for t in m["tracks"]])
wcsv(os.path.join(DEMO,"04_outputs","formation_timeline.csv"),
     ["t_s","formation_class","confidence","from_truth"],
     [[round(f["t"],2),f["formation_class"],round(f["confidence"],3),f["from_truth"]] for f in forms])
p_ex=[p for p in preds if p["t"]>50][:6]
wj(os.path.join(DEMO,"04_outputs","predictions_sample.json"),
   {"model_id":"cv_baseline (constant-velocity extrapolation from EKF track)",
    "examples":[{"global_id":p["global_id"],"horizon_sec":p["horizon_sec"],"waypoints_ENU":[[round(x,2) for x in w] for w in p["waypoints"]],"position_sigma_m":[round(x,2) for x in p["position_sigma"]]} for p in p_ex]})
with open(os.path.join(DEMO,"04_outputs","narration_log.txt"),"w") as f:
    f.write("RF_Swarm narration log (real run)\n"+"="*50+"\n")
    for n in narr:
        f.write(f"[t={n['t']:.1f}s] source={n['source']:8s} template={n['template_id']}\n    {n['text']}\n\n")
wj(os.path.join(DEMO,"04_outputs","scene_update_sample.json"),
   {"note":"the exact JSON the AR/Quest client receives on /scene_update (tracks-only, Option C; pos in Three.js Y-up [E,U,-N]). /formation and /narration arrive as separate latched topics.",
    "sample":scene[len(scene)//2]})
print("demo built. gid2em:", gid2em, "| RMSE:", round(float(np.sqrt(np.mean(np.square(allerr)))),2),"m | llm narrations:", len(llm_narr))
