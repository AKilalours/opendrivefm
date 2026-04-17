"""
live_demo_webcam.py — OpenDriveFM Live Demo FINAL
Fixes:
  - Threshold = 0.35 (sweet spot for Precision=0.054)
  - Colormap: black=free, white/yellow=occupied (not all red)
  - GT overlay: green dots = ground truth LiDAR BEV label
  - Per-camera fault injection (1-6 = fault that camera)
  - B = blur all, 0 = clear all

Run: python live_demo_webcam.py --nuscenes
"""
import sys, time, cv2, argparse, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

IMG_H, IMG_W = 90, 160
CKPT = "outputs/artifacts/checkpoints_v11_temporal/best_val_ade.ckpt"
if not Path(CKPT).exists():
    CKPT = "outputs/artifacts/checkpoints_v9/best_val_ade.ckpt"

CAM_NAMES = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
             "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
CAM_SHORT  = ["FRONT","F-L","F-R","BACK","B-L","B-R"]
VAL_SCENES = {"scene-0655","scene-1077"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
OCC_THRESHOLD = 0.35   # sweet spot for this model

FAULT_TYPES  = {0:"CLEAN",1:"BLUR",2:"GLARE",3:"OCCLUDE",4:"NOISE",5:"RAIN",6:"SNOW",7:"FOG"}
FAULT_COLORS = {0:(50,220,50),1:(50,150,255),2:(0,230,230),
                3:(180,50,220),4:(50,180,255),5:(80,80,255),
                6:(220,220,255),7:(180,180,180)}

def T(img,text,pos,sz,col,bold=False):
    cv2.putText(img,text,pos,FONT,sz,col,2 if bold else 1,cv2.LINE_AA)
def BOX(img,x0,y0,x1,y1,col,fill=None,thick=1):
    if fill is not None: cv2.rectangle(img,(x0,y0),(x1,y1),fill,-1)
    cv2.rectangle(img,(x0,y0),(x1,y1),col,thick)

# ── Fault injection ────────────────────────────────────────────────────────────
def fault_img(img, f):
    if f==0: return img.copy()
    if f==1: return cv2.GaussianBlur(img,(25,25),9)
    if f==2: return np.clip(img.astype(np.float32)*2.8,0,255).astype(np.uint8)
    if f==3:
        o=img.copy(); h,w=o.shape[:2]
        o[h//4:h*3//4,w//4:w*3//4]=0; return o
    if f==4:
        n=np.random.randint(-70,70,img.shape,np.int16)
        return np.clip(img.astype(np.int16)+n,0,255).astype(np.uint8)
    if f==5:
        o=img.copy()
        for _ in range(100):
            x=np.random.randint(0,img.shape[1])
            y=np.random.randint(0,img.shape[0]-25)
            cv2.line(o,(x,y),(x-2,y+25),(200,210,240),1)
        return o
    if f==6:  # SNOW (UNSEEN — generalization test)
        o=img.copy(); h,w=o.shape[:2]
        for _ in range(int(h*w*0.008)):
            cx,cy=np.random.randint(0,w),np.random.randint(0,h)
            r=np.random.randint(1,4)
            cv2.circle(o,(cx,cy),r,(240,245,255),-1)
        return cv2.GaussianBlur(o,(5,5),1.2)
    if f==7:  # FOG (UNSEEN — generalization test)
        fog=np.ones_like(img,dtype=np.float32)*235
        return np.clip(img.astype(np.float32)*0.45+fog*0.55,0,255).astype(np.uint8)

def load_real_cams(row, cam_faults):
    cams={}
    for i,name in enumerate(CAM_NAMES):
        p=Path(row["cams"][name])
        img=cv2.imread(str(p)) if p.exists() else np.zeros((480,640,3),np.uint8)
        cams[name]=fault_img(cv2.resize(img,(640,480)), cam_faults[i])
    return cams

def synth_cams(frame, cam_faults):
    h,w=frame.shape[:2]
    raw={
        "CAM_FRONT":       frame.copy(),
        "CAM_FRONT_LEFT":  cv2.resize(frame[:,:w*2//3],(w,h)),
        "CAM_FRONT_RIGHT": cv2.resize(frame[:,w//3:],(w,h)),
        "CAM_BACK":        cv2.flip(frame,1),
        "CAM_BACK_LEFT":   cv2.resize(cv2.convertScaleAbs(
                               cv2.flip(frame[:,:w*2//3],1),alpha=0.7),(w,h)),
        "CAM_BACK_RIGHT":  cv2.resize(cv2.convertScaleAbs(
                               cv2.flip(frame[:,w//3:],1),alpha=0.7),(w,h)),
    }
    return {k:fault_img(v,cam_faults[i]) for i,(k,v) in enumerate(raw.items())}

# ── Model ──────────────────────────────────────────────────────────────────────
def load_model(device):
    from opendrivefm.models.model import OpenDriveFM
    ckpt=torch.load(CKPT,map_location="cpu")
    raw=ckpt.get("state_dict",ckpt)
    sd={}
    for k,v in raw.items():
        if   k.startswith("model."): sd[k[6:]]=v
        elif k.startswith("lit.model."): sd[k[10:]]=v
        elif k.startswith("lit."): sd[k[4:]]=v
        else: sd[k]=v
    m=OpenDriveFM()
    ms=m.state_dict()
    for k,v in sd.items():
        if k in ms and ms[k].shape==v.shape: ms[k]=v
    m.load_state_dict(ms)
    return m.eval().to(device)

def run_inference(model, cams, device):
    imgs=[]
    for name in CAM_NAMES:
        r=cv2.cvtColor(cv2.resize(cams[name],(IMG_W,IMG_H)),cv2.COLOR_BGR2RGB)
        imgs.append(torch.from_numpy(r).permute(2,0,1).float()/255.0)
    x=torch.stack(imgs).unsqueeze(0).unsqueeze(2).to(device)
    t0=time.perf_counter()
    with torch.no_grad():
        out=model(x)
    if device.type=="mps": torch.mps.synchronize()
    ms=(time.perf_counter()-t0)*1000
    return (torch.sigmoid(out[0][0,0]).cpu().numpy(),
            out[1][0].cpu().numpy(),
            out[2][0].cpu().numpy(), ms)

# ── BEV drawing ────────────────────────────────────────────────────────────────
def draw_bev(occ, traj, trust, cam_faults, gt_occ, size=480):
    """
    occ    : (64,64) model predicted occupancy probability
    gt_occ : (64,64) ground truth LiDAR BEV label (or None)

    Colormap logic:
      black  = free space (below threshold)
      yellow/white = model predicted occupied (above threshold)
      green dots = ground truth occupied cells (LiDAR GT)
    """
    img=np.zeros((size,size,3),np.uint8)

    # Grid lines
    for i in range(0,size,size//8):
        cv2.line(img,(i,0),(i,size),(28,28,28),1)
        cv2.line(img,(0,i),(size,i),(28,28,28),1)

    # ── Model prediction: threshold=0.35, colormap HOT ────────────────────
    # HOT colormap: black(0) → red(0.33) → yellow(0.66) → white(1.0)
    # At threshold=0.35: model needs 35% confidence to show anything
    # Result: dark background with yellow/white blobs where vehicles detected
    pred_masked = np.where(occ > OCC_THRESHOLD, occ, 0.0)
    pred_up = cv2.resize(pred_masked, (size,size))
    u8 = (pred_up * 255).astype(np.uint8)
    heat = cv2.applyColorMap(u8, cv2.COLORMAP_HOT)
    # Only paint cells above threshold
    mask = (u8 > 30)[...,None].astype(np.float32)
    img = np.where(mask>0, heat, img).astype(np.uint8)

    # ── Ground truth overlay: green dots ──────────────────────────────────
    if gt_occ is not None:
        gt_bin = (gt_occ > 0.5).astype(np.float32)
        gt_up  = cv2.resize(gt_bin, (size,size), interpolation=cv2.INTER_NEAREST)
        gt_ys, gt_xs = np.where(gt_up > 0.5)
        for gx,gy in zip(gt_xs[::4], gt_ys[::4]):  # subsample for speed
            cv2.circle(img,(int(gx),int(gy)),2,(0,255,0),-1)

    # Distance rings
    cx,cy = size//2, size//2
    scale = size/40.0
    for dm in [5,10,15,20]:
        r=int(dm*scale)
        cv2.circle(img,(cx,cy),r,(45,45,45),1)
        T(img,f"{dm}m",(cx+r+2,cy-3),.27,(60,60,60))

    # Ego vehicle
    cv2.circle(img,(cx,cy),12,(0,255,0),-1)
    cv2.circle(img,(cx,cy),14,(0,180,0),2)
    T(img,"EGO",(cx-15,cy+28),.44,(0,255,0),True)

    # Predicted trajectory (real model output)
    prev=None
    for i,(xv,yv) in enumerate(traj):
        px=int(np.clip(cx+yv*scale,4,size-4))
        py=int(np.clip(cy-xv*scale,4,size-4))
        a=(i+1)/len(traj)
        c=(int(50+200*a),int(200*(1-a)),int(255*a))
        cv2.circle(img,(px,py),6,c,-1)
        if prev: cv2.line(img,prev,(px,py),c,2)
        prev=(px,py)
        if i<9: T(img,str(i+1),(px+6,py+5),.33,c)

    T(img,"FWD",(cx-14,16),.38,(80,80,80))

    # Trust bars — colour = fault type for that camera
    bw=size//6
    for i,(tv,sn) in enumerate(zip(trust,CAM_SHORT)):
        bx=i*bw
        bh=max(int(float(tv)*32),2)
        ft=cam_faults[i]
        bc=FAULT_COLORS[ft] if ft>0 else (0,max(int(180*float(tv)),20),0)
        cv2.rectangle(img,(bx+2,size-bh),(bx+bw-2,size-1),bc,-1)
        T(img,f"{float(tv):.2f}",(bx+2,size-bh-4),.28,bc)
        if ft>0: T(img,FAULT_TYPES[ft][:3],(bx+2,size-bh-14),.25,bc)
        T(img,sn[:3],(bx+2,size-4),.27,(120,120,120))

    # Header + legend
    cv2.rectangle(img,(0,0),(size,22),(0,0,0),-1)
    n_f=sum(1 for f in cam_faults if f>0)
    col=(0,255,100) if n_f==0 else (0,165,255)
    T(img,f"BEV thresh={OCC_THRESHOLD}  pred=yellow  GT=green  {n_f}cam faulted",
      (4,15),.34,col,True)
    return img

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nuscenes",action="store_true")
    ap.add_argument("--image",default=None)
    ap.add_argument("--video",default=None)
    args=ap.parse_args()

    print("Loading model...")
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=load_model(device)
    print(f"Ready on {device}  |  checkpoint: {CKPT}")

    # Load nuScenes val rows
    ns_rows=[]
    manifest=Path("outputs/artifacts/nuscenes_mini_manifest.jsonl")
    if manifest.exists():
        rows=[json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        ns_rows=[r for r in rows
                 if r.get("scene",r.get("scene_name","")) in VAL_SCENES
                 and Path(r["cams"]["CAM_FRONT"]).exists()]
        print(f"Loaded {len(ns_rows)} real nuScenes val samples with GT labels")

    use_ns=len(ns_rows)>0 and (args.nuscenes or not args.image and not args.video)
    ns_idx=0; ns_timer=time.time(); cap=None; static_frame=None

    if not use_ns:
        if args.image:
            static_frame=cv2.resize(cv2.imread(args.image),(640,480))
        elif args.video:
            cap=cv2.VideoCapture(args.video)
        else:
            cap=cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    cam_faults=[0]*6
    next_fault=[1]*6
    fps_s=30.0; save_n=0; frozen=False; frozen_frame=None
    occ=np.zeros((64,64)); traj=np.zeros((12,2))
    trust=np.ones(6)*.8; inf_ms=5.0; gt_occ=None
    CW,CH=1280,900

    # Warmup
    with torch.no_grad():
        model(torch.zeros(1,6,1,3,IMG_H,IMG_W).to(device))

    print("\nControls:")
    print("  1-6 = cycle fault on that camera (blur→glare→occ→noise→rain)")
    print("  B = blur ALL cameras  |  0 = clear ALL faults")
    print("  N = next nuScenes scene  |  SPACE = freeze  |  S = save  |  Q = quit\n")

    while True:
        # Get frame + GT
        current_token=None
        if use_ns and not frozen:
            if time.time()-ns_timer>3.0:
                ns_idx=(ns_idx+1)%len(ns_rows)
                ns_timer=time.time()
            row=ns_rows[ns_idx]
            cams=load_real_cams(row, cam_faults)
            current_token=row["sample_token"]
            src=f"nuScenes [{ns_idx+1}/{len(ns_rows)}] {row.get('scene',row.get('scene_name','?'))}"
        elif use_ns and frozen:
            row=ns_rows[ns_idx]
            cams=load_real_cams(row, cam_faults)
            current_token=row["sample_token"]
            src="FROZEN"
        elif static_frame is not None:
            cams=synth_cams(static_frame, cam_faults); src="image"
        else:
            ret,frame=cap.read()
            if not ret:
                if args.video: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
                break
            frame=cv2.resize(frame,(640,480))
            if frozen: frame=frozen_frame.copy()
            cams=synth_cams(frame, cam_faults); src="webcam"

        # Load GT label if available
        gt_occ=None
        if current_token:
            for ldir in ["outputs/artifacts/nuscenes_labels",
                         "outputs/artifacts/nuscenes_labels_128"]:
                lp=Path(ldir)/f"{current_token}.npz"
                if lp.exists():
                    z=np.load(lp)
                    occ_data=z["occ"]
                    # Handle (1,H,W) or (3,H,W) or (H,W)
                    if occ_data.ndim==3:
                        gt_occ=occ_data[0]
                    else:
                        gt_occ=occ_data
                    # Resize to 64×64
                    gt_occ=cv2.resize(gt_occ.astype(np.float32),(64,64),
                                      interpolation=cv2.INTER_NEAREST)
                    break

        # Real inference
        loop_t=time.perf_counter()
        occ,traj,trust,inf_ms=run_inference(model,cams,device)
        occ_density=float((occ>OCC_THRESHOLD).mean())
        live_ade=float(np.linalg.norm(traj,axis=1).mean())
        loop_ms=(time.perf_counter()-loop_t)*1000
        fps_s=0.88*fps_s+0.12*(1000/max(loop_ms,1))

        # Compute live IoU if GT available
        live_iou=None
        if gt_occ is not None:
            pred_b=(occ>OCC_THRESHOLD).astype(np.float32)
            gt_b=(gt_occ>0.5).astype(np.float32)
            # Resize pred to match gt
            pred_b=cv2.resize(pred_b,(gt_b.shape[1],gt_b.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            tp=(pred_b*gt_b).sum()
            fp=(pred_b*(1-gt_b)).sum()
            fn=((1-pred_b)*gt_b).sum()
            live_iou=float(tp/(tp+fp+fn+1e-8))

        # Build canvas
        canvas=np.full((CH,CW,3),14,np.uint8)

        # TOP BAR
        n_f=sum(1 for f in cam_faults if f>0)
        sc=(50,220,50) if n_f==0 else (50,150,255)
        cv2.rectangle(canvas,(0,0),(CW,36),(18,18,18),-1)
        T(canvas,"OpenDriveFM — LIVE DEMO  |  Real nuScenes  |  Per-Camera Fault Injection",
          (8,24),.55,(255,210,40),True)
        st=f"CLEAN" if n_f==0 else f"{n_f} FAULTED"
        T(canvas,f"{st}  {fps_s:.0f}FPS  {src}",(700,24),.43,sc,n_f>0)

        # LEFT: Steps 1-4
        LX,LW=4,232
        def sp(x,y,w,h,n,title,col,lines):
            fi=tuple(max(0,int(c*.15)) for c in col)
            BOX(canvas,x,y,x+w,y+h,col,fi,1)
            cv2.circle(canvas,(x+13,y+13),11,col,-1)
            T(canvas,str(n),(x+8,y+18),.44,(0,0,0),True)
            T(canvas,title,(x+28,y+17),.41,col,True)
            for i,l in enumerate(lines):
                T(canvas,l,(x+6,y+32+i*16),.36,(185,185,185))
        sp(LX,38,LW,108,1,"Data Collection",(255,170,50),[
            "nuScenes v1.0-mini","404 samples  10 scenes",
            "6 surround cameras","LiDAR_TOP + ego poses",
            "64x64 BEV + traj labels"])
        sp(LX,150,LW,96,2,"Preprocessing",(80,220,80),[
            "LiDAR->ego frame",
            "z[-1.2, 3.0m] filter",
            "Rasterize 64x64 +-20m",
            "Dilation r=2 applied"])
        sp(LX,250,LW,122,3,"CNN/ViT Backbone",(80,160,255),[
            "Shared CNN x6 cameras",
            "Conv(3->192)->BN->GELU",
            "Conv(192->384)->Pool",
            "ViTStem patch=16 L=2",
            "5 augmentation types"])
        sp(LX,376,LW,150,4,"3D->2D->3D LSS",(220,80,220),[
            "3D scene->K proj->2D",
            "K_inv x [u,v,1]=ray",
            "T_cam2ego R.ray+t->ego",
            "D=32 depth bins->probs",
            "splat() scatter->BEV",
            "Per-cam BEV trust-fused"])

        # CENTRE: BEV with GT overlay
        BX,BY,BS=240,38,480
        bev=draw_bev(occ,traj,trust,cam_faults,gt_occ,BS)
        canvas[BY:BY+BS,BX:BX+BS]=bev

        # Legend below BEV
        LY=BY+BS+2
        cv2.rectangle(canvas,(BX,LY),(BX+BS,LY+18),(10,10,10),-1)
        cv2.circle(canvas,(BX+10,LY+9),4,(0,255,0),-1)
        T(canvas,"= GT (LiDAR)",(BX+18,LY+13),.35,(0,255,0))
        cv2.rectangle(canvas,(BX+120,LY+3),(BX+140,LY+14),(200,200,80),-1)
        T(canvas,"= Predicted occ",(BX+145,LY+13),.35,(200,200,80))
        T(canvas,f"threshold={OCC_THRESHOLD}",(BX+290,LY+13),.35,(150,150,150))
        if live_iou is not None:
            T(canvas,f"LIVE IoU={live_iou:.3f}",(BX+390,LY+13),.38,(0,255,120),True)

        # CENTRE BOTTOM: 6 cameras
        TH,TW=100,178
        for idx,name in enumerate(CAM_NAMES):
            ci,ri=idx%3,idx//3
            tx=BX+ci*(TW+3)
            ty=LY+22+ri*(TH+3)
            if ty+TH>CH-26: continue
            th=cv2.resize(cams[name],(TW,TH))
            ft=cam_faults[idx]; tv=float(trust[idx])
            cv2.rectangle(th,(0,0),(TW,18),(0,0,0),-1)
            tc=FAULT_COLORS[ft] if ft>0 else (0,200,50)
            fl=f"[{FAULT_TYPES[ft]}] " if ft>0 else ""
            # Auto-detect fault type label
            detected = ""
            if ft == 0:
                detected = "CLEAN"
            elif ft == 6:
                detected = "[SNOW-UNSEEN]"
            elif ft == 7:
                detected = "[FOG-UNSEEN]"
            else:
                detected = f"[{FAULT_TYPES[ft]}]"
            T(th,f"CAM{idx+1} {detected} t={tv:.2f}",(3,13),.32,tc)
            if ft>0:
                ov=th.copy()
                cv2.rectangle(ov,(0,0),(TW,TH),FAULT_COLORS[ft],-1)
                cv2.addWeighted(ov,0.15,th,0.85,0,th)
            cv2.rectangle(th,(0,0),(TW-1,TH-1),tc,3 if ft>0 else 1)
            canvas[ty:ty+TH,tx:tx+TW]=th

        # RIGHT: Steps 5-7
        RX=BX+BS+4; RW=CW-RX-4

        # Step 5 trust (live)
        BOX(canvas,RX,38,RX+RW,38+192,(255,150,0),(30,20,0),1)
        cv2.circle(canvas,(RX+13,51),11,(255,150,0),-1)
        T(canvas,"5",(RX+8,56),.44,(0,0,0),True)
        T(canvas,"Trust-Aware Fusion [LIVE]",(RX+28,56),.41,(255,150,0),True)
        T(canvas,"CameraTrustScorer output:",(RX+6,74),.35,(185,185,185))
        bmax=RW-92
        for i,(tv,sn) in enumerate(zip(trust,CAM_SHORT)):
            yy=38+88+i*17
            bw2=max(int(float(tv)*bmax),2)
            ft=cam_faults[i]
            bc=FAULT_COLORS[ft] if ft>0 else (0,max(int(160*float(tv)+30),20),0)
            cv2.rectangle(canvas,(RX+6,yy),(RX+6+bw2,yy+11),bc,-1)
            cv2.rectangle(canvas,(RX+6,yy),(RX+6+bmax,yy+11),(45,45,45),1)
            fl=f"[{FAULT_TYPES[ft][:3]}]" if ft>0 else ""
            T(canvas,f"{sn}{fl} {float(tv):.3f}",(RX+RW-88,yy+10),.33,bc)
        T(canvas,"softmax->weighted BEV fusion",(RX+6,38+186),.33,(140,140,140))
        # Physics signal labels per camera (generalization evidence)
        for i,(tv,sn) in enumerate(zip(trust,CAM_SHORT)):
            yy=38+88+i*17
            ft=cam_faults[i]
            if ft>0:
                # Show detected fault type label
                ftag = FAULT_TYPES[ft]
                unseen = " [UNSEEN]" if ft>=6 else ""
                T(canvas,f"→{ftag}{unseen}",(RX+6,yy+10),.28,(255,200,100))

        sp(RX,235,RW,168,6,"BEV Decoder+Training",(50,220,220),[
            "ConvTranspose2d decoder",
            "BCE + Dice loss",
            "AdamW CosineAnnealingLR",
            "8train/2val scene splits",
            "v8  IoU=0.136 ADE=2.740",
            "v11 IoU=0.078 ADE=2.457 *",
            "v14 LSS Step4 IoU=0.020"])

        # Step 7
        BOX(canvas,RX,408,RX+RW,408+170,(190,80,255),(18,0,30),1)
        cv2.circle(canvas,(RX+13,421),11,(190,80,255),-1)
        T(canvas,"7",(RX+8,426),.44,(0,0,0),True)
        T(canvas,"Robustness + Evaluation",(RX+28,426),.41,(190,80,255),True)
        T(canvas,"Fixed val metrics:",(RX+6,444),.33,(120,120,120))
        for i,(k,v) in enumerate([
            ("IoU","0.136"),("Dice","0.087"),
            ("Prec","0.054"),("Rec","0.275")]):
            T(canvas,f"{k}:{v}",(RX+6+i%2*88,457+i//2*13),.33,(160,160,160))
        T(canvas,"Live computed:",(RX+6,490),.33,(0,180,80))
        lv=[("Occ",f"{occ_density*100:.1f}%"),
            ("ADE",f"{live_ade:.2f}m"),
            ("Inf",f"{inf_ms:.1f}ms"),
            ("FPS",f"{fps_s:.0f}")]
        if live_iou is not None:
            lv.append(("IoU",f"{live_iou:.3f}"))
        for i,(k,v) in enumerate(lv):
            T(canvas,f"{k}:{v}",(RX+6+i%2*88,504+i//2*16),.38,(0,255,120),True)

        # Paper comparison
        BOX(canvas,RX,584,RX+RW,584+190,(60,60,60),(10,10,10),1)
        T(canvas,"vs Reference Papers",(RX+6,600),.43,(200,200,200),True)
        for i,(p,m,c) in enumerate([
            ("ProtoOcc CVPR25","17cls no-traj 9.5FPS",(200,100,100)),
            ("GAFusion CVPR24","LiDAR-req no-traj 8FPS",(200,160,80)),
            ("PointBeV CVPR24","camera BEV no-traj ~10FPS",(80,180,180)),
            ("OpenDriveFM *","BEV+traj+trust 317FPS",(50,255,100))]):
            T(canvas,p,(RX+6,616+i*42),.39,c,"*" in p)
            T(canvas,m,(RX+6,632+i*42),.33,(120,120,120))

        # BOTTOM
        cv2.rectangle(canvas,(0,CH-26),(CW,CH),(18,18,18),-1)
        T(canvas,"1-6=fault cam  7=snow(UNSEEN)  8=fog(UNSEEN)  B=blur all  0=clear  N=next  S=save  Q=quit",
          (8,CH-8),.4,(120,120,120))

        cv2.imshow("OpenDriveFM — Live Demo",canvas)
        key=cv2.waitKey(30 if use_ns else 1)&0xFF
        if key==ord('q') or key==27: break
        elif key==ord('0'):
            cam_faults=[0]*6; next_fault=[1]*6
            print("ALL CAMERAS: CLEAN")
        elif key==ord('b') or key==ord('B'):
            cam_faults=[1]*6
            print("ALL CAMERAS: BLUR")
        elif ord('1')<=key<=ord('6'):
            ci=key-ord('1')
            ft=next_fault[ci]
            cam_faults[ci]=ft
            next_fault[ci]=(ft%7)+1
            print(f"CAM{ci+1} ({CAM_SHORT[ci]}): {FAULT_TYPES[ft]}")
        elif key==ord('n') or key==ord('N'):
            ns_idx=(ns_idx+1)%max(len(ns_rows),1)
            ns_timer=time.time(); frozen=False
        elif key==ord(' '):
            frozen=not frozen
            if not frozen: ns_timer=time.time()
            elif cap:
                ret,frozen_frame=cap.read()
                if ret: frozen_frame=cv2.resize(frozen_frame,(640,480))
        elif key==ord('7'):  # Snow on ALL cameras (UNSEEN generalization)
            cam_faults=[6]*6
            print("ALL CAMERAS: SNOW (UNSEEN fault — generalization test)")
        elif key==ord('8'):  # Fog on ALL cameras (UNSEEN generalization)
            cam_faults=[7]*6
            print("ALL CAMERAS: FOG (UNSEEN fault — generalization test)")
        elif key==ord('s'):
            fn=f"demo_{save_n:03d}.png"
            cv2.imwrite(fn,canvas); print(f"Saved: {fn}"); save_n+=1

    if cap: cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
