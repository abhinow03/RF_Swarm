"""Generate the demo PNGs from the raw run capture."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

RAW="/home/pw26_akp_01/RF_Swarm/demo/_run_raw"; FIG="/home/pw26_akp_01/RF_Swarm/demo/figures"
os.makedirs(FIG, exist_ok=True)
def load(n): return [json.loads(l) for l in open(os.path.join(RAW,n+".jsonl"))]
truth=load("truth_scene"); recv=load("receivers"); tracks=load("tracks"); forms=load("formation")
narr=load("narration")
tassoc={r["burst_seq"]:r["truth_emitter_id"] for r in load("truth_association")}
assoc=load("associations")
FN=["v_shape","encirclement","column","diamond","dispersed","converging","shield","transitioning"]
COL=plt.cm.tab10(np.linspace(0,1,10))

# gid->emitter
votes=defaultdict(Counter)
for a in (assoc[-1]["map"] if assoc else []):
    te=tassoc.get(a["burst_seq"]);
    if te is not None and a["confidence"]>0: votes[a["global_id"]][te]+=1
gid2em={g:c.most_common(1)[0][0] for g,c in votes.items()}
def truth_series(em):
    xs=[]
    for s in truth:
        for e in s["emitters"]:
            if e["id"]==em: xs.append((s["t"],e["pos"]))
    return xs
def truth_at(em,t):
    best=None;bd=1e9
    for s in truth:
        if abs(s["t"]-t)<bd:
            for e in s["emitters"]:
                if e["id"]==em: best=e["pos"];bd=abs(s["t"]-t)
    return best

# ---- FIG 1: top-down E-N scene (receivers + truth trajectories + EKF estimates) ----
fig,ax=plt.subplots(figsize=(9,8))
r0=recv[0]["rx"]
for r in r0:
    ax.scatter(r["pos"][0],r["pos"][1],marker="^",s=260,c="k",zorder=5)
    ax.annotate(f"RX{r['id']}",(r["pos"][0],r["pos"][1]),textcoords="offset points",xytext=(6,6),fontsize=9)
gids=sorted(gid2em);
for i,g in enumerate(gids):
    em=gid2em[g]; ts=truth_series(em)
    tx=[p[0] for _,p in ts]; ty=[p[1] for _,p in ts]
    ax.plot(tx,ty,'-',c=COL[i],lw=2.5,label=f"{g} (truth emitter {em})")
    ex=[]; ey=[]
    for m in tracks[::8]:
        for t in m["tracks"]:
            if t["id"]==g: ex.append(t["pos"][0]); ey.append(t["pos"][1])
    ax.scatter(ex,ey,c=[COL[i]],s=8,alpha=0.35)
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
ax.set_title("Scene (top-down ENU): receivers ▲, truth trajectories —, EKF estimates ·\nThe 6-drone swarm flies through the fixed 5-receiver TDoA array")
ax.legend(loc="upper left",fontsize=8); ax.grid(alpha=0.3); ax.set_aspect("equal")
plt.tight_layout(); plt.savefig(f"{FIG}/01_scene_topdown.png",dpi=130); plt.close()

# ---- FIG 2: localization error over time (per track) ----
fig,ax=plt.subplots(figsize=(10,5))
for i,g in enumerate(gids):
    tt=[];ee=[]
    for m in tracks:
        for t in m["tracks"]:
            if t["id"]==g:
                tp=truth_at(gid2em[g],m["t"])
                if tp: tt.append(m["t"]); ee.append(np.linalg.norm(np.array(t["pos"])-np.array(tp)))
    ax.plot(tt,ee,c=COL[i],lw=1.3,label=g,alpha=0.85)
allerr=[]
for m in tracks:
    for t in m["tracks"]:
        tp=truth_at(gid2em.get(t["id"]),m["t"]) if t["id"] in gid2em else None
        if tp and m["t"]>40: allerr.append(np.linalg.norm(np.array(t["pos"])-np.array(tp)))
rmse=np.sqrt(np.mean(np.square(allerr)))
ax.axhline(rmse,ls="--",c="k",label=f"RMSE={rmse:.1f} m (t>40s)")
ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)")
ax.set_title("EKF localization error vs ground truth\n(clock-prior-limited ~15 m — the documented near-field observability limit)")
ax.legend(ncol=4,fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0,None)
plt.tight_layout(); plt.savefig(f"{FIG}/02_localization_error.png",dpi=130); plt.close()

# ---- FIG 3: formation timeline (confidence, colored by class) + narration markers ----
fig,ax=plt.subplots(figsize=(11,5))
classes=sorted(set(f["formation_class"] for f in forms))
cmap={c:plt.cm.Set2(i) for i,c in enumerate(classes)}
for f in forms:
    ax.scatter(f["t"],f["confidence"],c=[cmap[f["formation_class"]]],s=18)
for c in classes: ax.scatter([],[],c=[cmap[c]],label=c)
for n in narr:
    if n["source"]=="llm":
        ax.axvline(n["t"],c="crimson",ls="--",alpha=0.7)
        ax.annotate("LLM narration",(n["t"],1.02),color="crimson",fontsize=8,rotation=90,va="bottom")
ax.axhline(0.6,ls=":",c="grey",label="0.6 (LLM-route threshold)")
ax.set_xlabel("time (s)"); ax.set_ylabel("classifier confidence")
ax.set_title("STGT formation classification over time (fed by real EKF tracks)\nred = a real GPU LLM narration was generated at that moment")
ax.legend(ncol=4,fontsize=8,loc="lower right"); ax.grid(alpha=0.3); ax.set_ylim(0,1.1)
plt.tight_layout(); plt.savefig(f"{FIG}/03_formation_timeline.png",dpi=130); plt.close()

# ---- FIG 4: pipeline dataflow diagram ----
fig,ax=plt.subplots(figsize=(13,3.2)); ax.axis("off")
stages=[("RF sim\n(truth+observables)","/rf_observations"),("BLE encoder\nreplay","/embeddings"),
        ("Base station\n(fingerprint K)","/fingerprints\n/associations"),("Joint EKF\n(TDoA+clock)","/tracks"),
        ("STGT\nformation","/formation"),("LLM\nnarration","/narration"),("Bridge\n(Option-C)","/scene_update\n→ Quest")]
n=len(stages); w=1.0/n
for i,(name,topic) in enumerate(stages):
    x=i*w
    ax.add_patch(plt.Rectangle((x+0.008,0.35),w-0.016,0.4,fc=COL[i%10],ec="k",alpha=0.85))
    ax.text(x+w/2,0.55,name,ha="center",va="center",fontsize=9,weight="bold")
    ax.text(x+w/2,0.22,topic,ha="center",va="top",fontsize=7,color="navy")
    if i<n-1: ax.annotate("",xy=(x+w+0.004,0.55),xytext=(x+w-0.004,0.55),arrowprops=dict(arrowstyle="->",lw=1.6))
ax.text(0.5,0.95,"RF_Swarm pipeline — RF observations → localization → formation/intent → narration → AR",
        ha="center",fontsize=11,weight="bold",transform=ax.transAxes)
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout(); plt.savefig(f"{FIG}/04_pipeline_dataflow.png",dpi=130); plt.close()
print("figures written:", sorted(os.listdir(FIG)))
