"""按【镜像 seg0】判据评估我方 state.csv。领先=左前, 对角发力=右后, 支撑=左后。
相位用离墙距离对齐到官方 k（官方 k0 在离墙 0.560m, k60 在 -0.074m）。"""
import numpy as np, os, csv, sys, glob
M=np.load(os.path.expanduser("~/s10_logs/ref/ref_seg0_mirror.npz"))
TH=np.load(os.path.expanduser("~/s10_logs/ref/stage_th.npz"))
ACT=[f"{L}_{J}" for L in ("fl","fr","hl","hr") for J in ("hipx","hipy","knee","wheel")]
def kmap(D):
    """我方离墙序列 → 官方 k（按离墙距离单调映射）。"""
    refd=-M['x']                      # 官方各 k 的离墙距离
    return np.array([int(np.argmin(np.abs(refd-d))) for d in D])
def one(p):
    r=list(csv.DictReader(open(p))); g=lambda k:np.array([float(v[k]) for v in r])
    X=g('x'); D=3.0-X; Z=g('z')
    Q=np.stack([g('q%d'%i) for i in range(16)],1); DQ=np.stack([g('dq%d'%i) for i in range(16)],1)
    # 只取本次翻越段：从离墙 0.56 首次到达，到越墙后 0.2m
    s=np.nonzero(D<=0.56)[0]; e=np.nonzero(D<=-0.20)[0]
    if len(s)==0: return None
    s=s[0]; e=e[0] if len(e) else len(D)-1
    sl=slice(s,e+1); K=kmap(D[sl])
    def at(k):
        idx=np.nonzero(K>=k)[0]
        return sl.start+(idx[0] if len(idx) else len(K)-1)
    fl=Q[sl][:,ACT.index("fl_knee")]; fr=Q[sl][:,ACT.index("fr_knee")]
    hr=Q[:,ACT.index("hr_knee")]; hl=Q[:,ACT.index("hl_knee")]
    W=[ACT.index(f"{L}_wheel") for L in ("fl","fr","hl","hr")]
    i16,i40=at(16),at(40); i36,i54=at(36),at(54); i30,i58=at(30),at(58)
    w=DQ[i16:i40+1][:,W]
    o={}
    o['E1领先前膝峰']=float(fl.max()); o['E1另一前膝峰']=float(fr.max())
    o['E1差']=float(fl.max()-fr.max())
    o['E2领先达峰k']=int(K[int(np.argmax(fl))]); o['E2另一达峰k']=int(K[int(np.argmax(fr))])
    o['E3右后膝伸展']=float(hr[i54]-hr[i36]); o['E3左后膝变化']=float(hl[i54]-hl[i36])
    o['E4机身净升']=float(Z[i58]-Z[i30])
    o['E5左前轮均']=float(w[:,0].mean()) if len(w) else np.nan
    o['E5左前最强%']=100*float(np.mean(w[:,0]==w.min(1))) if len(w) else np.nan
    o['P1 领先深度']=o['E1领先前膝峰']>=float(TH['E1_lead_peak'])
    o['P2 领先明显']=o['E1差']>=float(TH['E1_gap'])
    o['P3 顺序正确']=o['E2领先达峰k']<o['E2另一达峰k']
    o['P4 对角发力']=o['E3右后膝伸展']>=float(TH['E3_drive_ext'])
    o['P5 机身推起']=o['E4机身净升']>=float(TH['E4_rise'])
    o['P6 领先轮驱动']=(o['E5左前轮均']<=float(TH['E5_lead_mean'])) and (o['E5左前最强%']>=100*float(TH['E5_lead_best']))
    return o
print(f"{'档':10s} "+" ".join(f"{t:>10s}" for t in sys.argv[1::2]))
res={}
for tag,dd in zip(sys.argv[1::2],sys.argv[2::2]):
    fs=sorted(glob.glob(os.path.expanduser(dd)+"/*.state.csv"))
    res[tag]=[x for x in (one(f) for f in fs) if x]
keys=[k for k in next(iter(res.values()))[0] if not k.startswith('P')]
pks=[k for k in next(iter(res.values()))[0] if k.startswith('P')]
REF={'E1领先前膝峰':2.721,'E1另一前膝峰':1.460,'E1差':1.261,'E2领先达峰k':28,'E2另一达峰k':49,
     'E3右后膝伸展':1.150,'E3左后膝变化':0.029,'E4机身净升':0.315,'E5左前轮均':-26.58,'E5左前最强%':96.0}
for k in keys:
    line=f"{k:10s} "
    for t in res: line+=f"{np.median([r[k] for r in res[t]]):10.3f} "
    print(line+f"  官方镜像 {REF.get(k,''):>8}")
print()
for k in pks:
    line=f"{k:10s} "
    for t in res:
        n=sum(1 for r in res[t] if r[k]); line+=f"{n}/{len(res[t]):<8d} "
    print(line)
