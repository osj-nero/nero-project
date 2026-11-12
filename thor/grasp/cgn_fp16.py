import os as _ng_os; GRASP_DIR = _ng_os.environ.get("NERO_GRASP_DIR") or _ng_os.path.expanduser("~/grasp")  # shared work dir on PC and Thor (override: NERO_GRASP_DIR)
import sys,time,numpy as np,torch
sys.path.insert(0,"" + GRASP_DIR + "")
import cgn_server as S
EST=S.EST
def load(p):
    d=np.load(p,allow_pickle=True); depth=d["depth_m"].astype(np.float32); k=d["K"].reshape(-1); K=np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]]); seg=d["seg"].astype(np.int32)
    return depth,K,seg,[int(x) for x in d["target_ids"]]
for p in sys.argv[1:]:
    depth,K,seg,ids=load(p); pc_full,pc_segs,_=EST.extract_point_clouds(depth,K,segmap=seg,z_range=[0.2,1.2])
    for sid in ids:
        if sid not in pc_segs: continue
        sp=np.asarray(pc_segs[sid])
        def run(mode):
            torch.cuda.synchronize(); t=time.time()
            with torch.no_grad():
                if mode=="fp16":
                    with torch.autocast("cuda",dtype=torch.float16): g,s,c,o=EST.predict_scene_grasps(pc_full,pc_segments={sid:sp},local_regions=True,filter_grasps=True)
                else: g,s,c,o=EST.predict_scene_grasps(pc_full,pc_segments={sid:sp},local_regions=True,filter_grasps=True)
            torch.cuda.synchronize(); return time.time()-t,g,s
        run("fp32"); run("fp16")      # 워밍업
        ts32=[];ts16=[]
        for _ in range(3):
            t32,g32,s32=run("fp32"); t16,g16,s16=run("fp16"); ts32.append(t32); ts16.append(t16)
        n32=len(np.asarray(g32[sid]).reshape(-1,4,4)) if sid in g32 else 0; n16=len(np.asarray(g16[sid]).reshape(-1,4,4)) if sid in g16 else 0
        m32=float(np.asarray(s32[sid]).max()) if n32 else 0; m16=float(np.asarray(s16[sid]).max()) if n16 else 0
        print("%s sid %d n_pts %d: fp32 %.2fs (%d grasps, max %.3f) | fp16 %.2fs (%d grasps, max %.3f)"%(p.split('/')[-1],sid,len(sp),np.median(ts32),n32,m32,np.median(ts16),n16,m16),flush=True)
