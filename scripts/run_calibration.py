"""Is the occupancy head's probability a probability?

IoU says how often the head is right after you threshold it. It says nothing
about whether 0.7 means 0.7 -- and a planner that consumes an occupancy grid is
consuming the probability, not the threshold. This bins every predicted cell
over all 404 keyframes by confidence and reports the observed frequency in each
bin, plus expected and maximum calibration error. Bin counts only, so the whole
404-frame sweep costs one pass and no storage.
"""
import sys, json, time, numpy as np, torch
sys.path.insert(0,'src')
from opendrivefm.models.model import OpenDriveFM
from fix_trust import remap_trust_keys
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
U='/mnt/user-data/uploads/Projects/opendrivefm/outputs/artifacts'
D=np.load('all_frames.npz',allow_pickle=True)
IMG,CH,ED,OCCGT=D['img'],D['chain'],D['ego_deltas'],D['occ']
N=len(IMG)
ck=torch.load(f'{U}/checkpoints_v11_temporal/best_val_ade.ckpt',map_location='cpu',weights_only=False)
hp=ck['hyper_parameters']
m=OpenDriveFM(d=hp['d'],bev_h=128,bev_w=128,horizon=hp['horizon'],enable_trust=hp['enable_trust'])
sd={k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')}
sd,_,_=remap_trust_keys(sd,m.state_dict()); m.load_state_dict(sd,strict=False); m.eval()

B=15; edges=np.linspace(0,1,B+1)
cnt=np.zeros(B); pos=np.zeros(B); conf=np.zeros(B)
t0=time.time()
with torch.no_grad():
    for i in range(N):
        w=np.transpose(IMG[CH[i]],(1,0,2,3,4))
        x=torch.from_numpy(np.ascontiguousarray(w)).float().div_(255.).permute(0,1,4,2,3).unsqueeze(0)
        out=m(x, ego_deltas=torch.from_numpy(ED[i]).unsqueeze(0))
        occ=out['occupancy'] if isinstance(out,dict) else out[0]
        p=torch.sigmoid(occ).reshape(-1).numpy() if occ.min()<0 or occ.max()>1 else occ.reshape(-1).numpy()
        y=(OCCGT[i].reshape(-1)>0).astype(np.float64)
        if p.size!=y.size:
            k=int(np.sqrt(p.size)); yy=OCCGT[i][0]
            y=(np.asarray(yy,np.float64).reshape(128,128)>0).astype(np.float64)
            import numpy.lib.stride_tricks as _st
            f=128//k; y=y.reshape(k,f,k,f).max(axis=(1,3)).reshape(-1)
        b=np.clip((p*B).astype(int),0,B-1)
        cnt+=np.bincount(b,minlength=B); pos+=np.bincount(b,weights=y,minlength=B)
        conf+=np.bincount(b,weights=p,minlength=B)
        if i%80==0: print(f"  {i}/{N}  {time.time()-t0:.0f}s",flush=True)
nz=cnt>0
acc=np.where(nz,pos/np.maximum(cnt,1),0.0); avg=np.where(nz,conf/np.maximum(cnt,1),0.0)
w=cnt/cnt.sum()
ece=float((w*np.abs(acc-avg)).sum()); mce=float(np.abs(acc-avg)[nz].max())
brier_lo=float((w*(avg-acc)**2).sum())
base=float(pos.sum()/cnt.sum())
bins=[{"lo":round(edges[i],3),"hi":round(edges[i+1],3),"n":int(cnt[i]),
       "confidence":round(float(avg[i]),4),"observed":round(float(acc[i]),4),
       "share":round(float(w[i]),5)} for i in range(B) if cnt[i]>0]
json.dump({"status":"MEASURED",
 "what":"Reliability of the BEV occupancy head over every cell of all 404 keyframes. Confidence is "
        "the head's predicted probability; observed is how often that cell is actually occupied in "
        "the geometric label grid.",
 "keyframes":int(N),"cells_scored":int(cnt.sum()),"bins":B,
 "base_rate":round(base,5),
 "ece":round(ece,4),"mce":round(mce,4),"brier_reliability_term":round(brier_lo,5),
 "reading":("ECE is the share-weighted gap between what the head claims and what happens. A planner "
            "consuming this grid consumes the probability, not the threshold, so a head can post a "
            "usable IoU and still be badly calibrated."),
 "histogram":bins,
 "seconds":round(time.time()-t0,1),
 "caveats":["Labels are the geometric occupancy grid, so this measures agreement with the LiDAR "
            "reconstruction, not with ground truth in an absolute sense.",
            "Occupancy is rare: the base rate below makes any confidence above it informative and "
            "makes accuracy a useless summary.",
            "CPU, single process, batch 1."]},
 open('calibration_report.json','w'),indent=2)
print(f"\nECE {ece:.4f} | MCE {mce:.4f} | base rate {base:.4f} | {time.time()-t0:.0f}s")
