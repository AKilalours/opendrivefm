"""Fault injection across seven conditions, including three weather models.

Rain is re-measured because its implementation changed: opaque two-pixel bars
were replaced by slanted, motion-blurred, semi-transparent streaks with
veiling. The old rain was easier to catch than real rain, so the old number
flattered the trust head.
"""
import sys, json, time, numpy as np, torch, base64, io
sys.path.insert(0,'src'); sys.path.insert(0,'.')
from opendrivefm.models.model import OpenDriveFM
from opendrivefm.robustness.perturbations import PERTURBATIONS
from weather import WEATHER
from fix_trust import remap_trust_keys
from PIL import Image
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
import random; random.seed(0)

U='/mnt/user-data/uploads/Projects/opendrivefm/outputs/artifacts'
D=np.load('all_frames.npz',allow_pickle=True)
IMG,CH,ED,MO,TR,TREL=D['img'],D['chain'],D['ego_deltas'],D['motion'],D['traj'],D['t_rel']
CAMS=["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
def window(i):
    w=np.transpose(IMG[CH[i]],(1,0,2,3,4))
    return torch.from_numpy(np.ascontiguousarray(w)).float().div_(255.).permute(0,1,4,2,3).unsqueeze(0)

ck=torch.load(f'{U}/checkpoints_v11_trustfix2/trust_fixed_v2_cal.ckpt',map_location='cpu',weights_only=False)
m=OpenDriveFM(d=384,bev_h=128,bev_w=128,horizon=12,enable_trust=True)
sd={k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')}
sd,_,_=remap_trust_keys(sd,m.state_dict()); m.load_state_dict(sd,strict=False); m.eval()
print("trust_fixed_v2_cal loaded, calibrated =", bool(m.backbone.trust_scorer.stat_calibrated))

def make(name):
    if name is None: return None
    if name in WEATHER: return WEATHER[name]()
    return PERTURBATIONS[name]()

NR=120; FAULT=0
CONDS=[None,"blur","glare","occlusion","noise","rain","fog","snow"]
def run(name):
    """Same call signature as the validated 404-frame sweep: the model returns
    (occupancy, trajectory residual, trust, _) and needs the velocity prior."""
    P=make(name); TRU=[]; RES=[]
    with torch.no_grad():
        for i in range(NR):
            x=window(i)
            if P is not None:
                v=x[:,FAULT]; B,T,C,H,W=v.shape
                x=x.clone(); x[:,FAULT]=P(v.reshape(B*T,C,H,W)).reshape(B,T,C,H,W).clamp(0,1)
            occ,res,tr,_=m(x,velocity=torch.from_numpy(MO[i][1:3]).unsqueeze(0),
                           ego_deltas=torch.from_numpy(ED[i]).unsqueeze(0))
            TRU.append(tr[0].numpy()); RES.append(res[0].numpy())
    TRU=np.stack(TRU); RES=np.stack(RES)
    dtp=MO[:NR,0:1]; vxy=MO[:NR,1:3]*(dtp>0); cv=TREL[:NR,:,None]*vxy[:,None,:]
    ade3=float(np.linalg.norm((cv+RES)[:,:3]-TR[:NR,:3],axis=-1).mean())
    return {"trust_faulted":float(TRU[:,FAULT].mean()),
            "trust_others":float(np.delete(TRU,FAULT,axis=1).mean()),
            "ade_T3_m":round(ade3,3)}

t0=time.time(); res={}
for c in CONDS:
    k="clean" if c is None else c
    res[k]=run(c); print(f"  {k:10} trust {res[k]['trust_faulted']:.4f} "
                         f"others {res[k]['trust_others']:.4f} ({time.time()-t0:.0f}s)",flush=True)
cl=res["clean"]
for k,v in res.items():
    if k!="clean":
        v["delta_trust_faulted"]=round(v["trust_faulted"]-cl["trust_faulted"],4)
        v["trust_faulted"]=round(v["trust_faulted"],4); v["trust_others"]=round(v["trust_others"],4)
cl["trust_faulted"]=round(cl["trust_faulted"],4); cl["trust_others"]=round(cl["trust_others"],4)
faults=[k for k in res if k!="clean"]
det=[k for k in faults if res[k]["delta_trust_faulted"]<-0.05]
miss=[k for k in faults if res[k]["delta_trust_faulted"]>=-0.05]
sep=float(np.mean([-res[k]["delta_trust_faulted"] for k in faults]))

# the exact perturbed model inputs the head was scored on, for the console
random.seed(0); torch.manual_seed(0)
def png(t):
    a=(t.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
    b=io.BytesIO(); Image.fromarray(a).resize((480,270),Image.LANCZOS).save(b,'PNG')
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
x=window(0); shots={"clean":png(x[0,FAULT,-1])}
for c in CONDS[1:]:
    P=make(c); v=x[:,FAULT].clone(); B,T,C,H,W=v.shape
    shots[c]=png(P(v.reshape(B*T,C,H,W)).reshape(B,T,C,H,W)[0,-1])
json.dump({"frame":0,"camera":"CAM_FRONT",
  "resolution":"90x160 model input, upscaled for display","images":shots},
  open('perturbation_shots.json','w'))

json.dump({"frames":NR,"faulted_camera":"CAM_FRONT",
 "checkpoint":"checkpoints_v11_trustfix2/trust_fixed_v2_cal.ckpt (171/171, stat_calibrated=True)",
 "results":res,"mean_separation":round(sep,4),"detected":det,"missed":miss,
 "verdict":(f"{len(det)} of {len(faults)} conditions move the faulted camera's trust down by more "
            f"than 0.05. {', '.join(miss)} do not, and that is the finding: the head keys on local "
            "variance, so a degradation that REMOVES variance reads as a clean, flat surface."),
 "weather_models":{
   "rain":"slanted motion-blurred semi-transparent streaks + veiling contrast loss",
   "fog":"atmospheric scattering I = I0*t + A*(1-t), t = exp(-beta*d), d approximated by image row",
   "snow":"bright defocused blobs of varying radius + mild veiling"},
 "caveats":[f"{NR} keyframes, one camera faulted at a time, CAM_FRONT.",
   "Perturbations are applied to the model's 90x160 input tensor, not to full-resolution images.",
   "Fog uses image row as a depth proxy because the input tensor carries no depth, so its effect "
   "is a lower bound on real fog at range.",
   "Rain was re-implemented for this run: the previous version painted opaque two-pixel bars, "
   "which a variance-based head detects far more easily than real rain."]},
 open('robustness_report.json','w'),indent=2)
print(f"\nseparation {sep:.4f} | detected {det} | missed {miss} | {time.time()-t0:.0f}s")
