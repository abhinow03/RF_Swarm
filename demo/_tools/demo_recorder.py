"""Record a REAL pipeline run — every stage's messages — to raw JSONL for the demo.
Subscribes to the full data path (truth + estimation chain + client output) and
writes one JSONL file per topic. args: <out_dir> <duration_s>"""
import sys, os, json, time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from capstone_msgs.msg import (TruthScene, ReceiverStateArray, RfObservationSet, Embedding,
                               FingerprintSet, BurstAssociationSet, BurstAssociationTruth,
                               TrackArray, FormationEstimate, PathPrediction, Narration)
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
BE=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=10)
RE=QoSProfile(reliability=ReliabilityPolicy.RELIABLE,history=HistoryPolicy.KEEP_LAST,depth=50)
LA=QoSProfile(reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL,history=HistoryPolicy.KEEP_LAST,depth=50)

def st(h): return h.stamp.sec + h.stamp.nanosec*1e-9

class Rec(Node):
    def __init__(self, out):
        super().__init__('demo_recorder')
        os.makedirs(out, exist_ok=True); self.out=out; self.f={}
        self.sub(TruthScene,'/truth/scene',BE,self.truth)
        self.sub(ReceiverStateArray,'/receivers',BE,self.receivers)
        self.sub(RfObservationSet,'/rf_observations',RE,self.obs)
        self.sub(Embedding,'/embeddings',RE,self.emb)
        self.sub(FingerprintSet,'/fingerprints',LA,self.fp)
        self.sub(BurstAssociationSet,'/associations',LA,self.assoc)
        self.sub(BurstAssociationTruth,'/truth/association',RE,self.tassoc)
        self.sub(TrackArray,'/tracks',RE,self.tracks)
        self.sub(FormationEstimate,'/formation',LA,self.form)
        self.sub(PathPrediction,'/predictions',RE,self.pred)
        self.sub(Narration,'/narration',LA,self.narr)
        self.sub(String,'/scene_update',LA,self.scene)
    def sub(self,ty,tp,q,cb): self.create_subscription(ty,tp,cb,q)
    def w(self,name,d):
        if name not in self.f: self.f[name]=open(os.path.join(self.out,name+'.jsonl'),'w')
        self.f[name].write(json.dumps(d)+'\n')
    def truth(self,m): self.w('truth_scene',{'t':st(m.header),'emitters':[{'id':e.emitter_id,'pos':[e.position.x,e.position.y,e.position.z],'vel':[e.velocity.x,e.velocity.y,e.velocity.z],'cfo_hz':e.cfo_hz} for e in m.emitters],'formation_label':m.formation_label})
    def receivers(self,m): self.w('receivers',{'t':st(m.header),'rx':[{'id':r.rx_id,'pos':[r.position.x,r.position.y,r.position.z],'clock_bias_sigma':r.clock_bias_sigma} for r in m.receivers]})
    def obs(self,m): self.w('rf_observations',{'t':st(m.header),'burst_seq':m.burst_seq,'obs':[{'rx':o.rx_id,'toa':st(o.header) if False else (o.toa.sec+o.toa.nanosec*1e-9),'toa_sigma':o.toa_sigma,'rssi_dbm':o.rssi_dbm,'fdoa_hz':o.fdoa_hz,'snr_db':o.snr_db} for o in m.observations]})
    def emb(self,m): self.w('embeddings',{'t':st(m.header),'burst_seq':m.burst_seq,'rx':m.rx_id,'dim':m.dim,'sha':m.encoder_sha256[:12],'vector':[round(float(x),5) for x in m.vector]})
    def fp(self,m): self.w('fingerprints',{'t':st(m.header),'k_estimated':m.k_estimated,'eigenvalues':[float(x) for x in m.eigenvalues],'eigengap_value':m.eigengap_value,'n_accumulated':m.n_accumulated,'assignments':[{'global_id':a.global_id,'confidence':a.confidence,'burst_seq':a.burst_seq} for a in m.assignments],'sha':m.encoder_sha256[:12]})
    def assoc(self,m): self.w('associations',{'t':st(m.header),'k_estimated':m.k_estimated,'map':[{'burst_seq':a.burst_seq,'global_id':a.global_id,'confidence':a.confidence} for a in m.associations]})
    def tassoc(self,m): self.w('truth_association',{'burst_seq':m.burst_seq,'truth_emitter_id':m.truth_emitter_id})
    def tracks(self,m): self.w('tracks',{'t':st(m.header),'tracks':[{'id':t.global_id,'pos':[t.position.x,t.position.y,t.position.z],'vel':[t.velocity.x,t.velocity.y,t.velocity.z],'cov':list(t.covariance),'gdop':t.gdop,'nrx':t.n_receivers_used,'status':t.status} for t in m.tracks]})
    def form(self,m): self.w('formation',{'t':st(m.header),'formation_class':m.formation_class,'confidence':m.confidence,'class_probabilities':list(m.class_probabilities),'member_ids':list(m.member_ids),'from_truth':m.from_truth})
    def pred(self,m): self.w('predictions',{'t':st(m.header),'global_id':m.global_id,'model_id':m.model_id,'horizon_sec':m.horizon_sec,'waypoints':[[p.x,p.y,p.z] for p in m.waypoints],'position_sigma':list(m.position_sigma)})
    def narr(self,m): self.w('narration',{'t':st(m.header),'source':m.source,'template_id':m.template_id,'severity':m.severity,'text':m.text})
    def scene(self,m): self.w('scene_update',json.loads(m.data))

def main():
    out=sys.argv[1]; dur=float(sys.argv[2])
    rclpy.init(); n=Rec(out); t0=time.time()
    while time.time()-t0<dur and rclpy.ok(): rclpy.spin_once(n,timeout_sec=0.2)
    for f in n.f.values(): f.close()
    counts={k.split('/')[-1]:sum(1 for _ in open(os.path.join(out,k.split('/')[-1]))) for k in [os.path.join(out,x) for x in os.listdir(out) if x.endswith('.jsonl')]}
    print('recorded:', {os.path.basename(k):v for k,v in counts.items()})
    n.destroy_node(); rclpy.shutdown()

if __name__=='__main__': main()
