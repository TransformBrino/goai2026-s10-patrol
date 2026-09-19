"""官方运控 200 Hz 录包 → 周期参考（trot_ref.npz 同格式：bins × 50 相位点 × 16 关节（articulation 顺序 hipx×4,hipy×4,knee×4,wheel×4）+ qd + freq + v + clear）。
周期切分：左前轮底离地高度（正运动学，见 official_gait.py 的 Z）峰值为周期起点；按速度档（轮速估计）分箱平均。
踏步：gftbzx1（直线）+ gftb1（转圈）；楼梯：gfstj1（四拍步行，单档）。"""
import numpy as np, sys
perm=np.zeros(16,dtype=int)
for l in range(4):
    for j in range(4): perm[4*j+l]=4*l+j
def load(name):
    A=np.loadtxt(f'/home/robot/s10_logs/night/official_gait/{name}.csv',delimiter=',',skiprows=1)
    return A[:,0],A[:,1:17],A[:,17:33],A[:,49:51],A[:,51:55]
def cycles(name,vlo,vhi,Tlo,Tmax,use_wz=None):
    t,q,qd,rp,Z=load(name); dt=np.median(np.diff(t)); v=(-qd[:,[3,7,11,15]]).mean(1)*0.081
    clear=Z-Z.min(1,keepdims=True); x=clear[:,0]; xs=np.convolve(x,np.ones(9)/9,mode='same')
    peaks=[i for i in range(4,len(xs)-4) if xs[i]>0.04 and xs[i]==xs[i-4:i+5].max()]
    P=[]
    for p in peaks:
        if not P or (p-P[-1])*dt>0.25: P.append(p)
    out=[]
    for a,b in zip(P[:-1],P[1:]):
        T=(b-a)*dt; vm=v[a:b].mean()
        if not (Tlo<T<Tmax) or not (vlo<=vm<=vhi): continue
        ph=np.linspace(0,1,50,endpoint=False); idx=a+np.round(ph*(b-a)).astype(int)
        out.append((q[idx][:,perm],qd[idx][:,perm],1.0/T,vm,clear[idx],rp[idx,1]))
    return out
def pack(cyc):
    Q=np.stack([c[0] for c in cyc]); QD=np.stack([c[1] for c in cyc]); F=np.array([c[2] for c in cyc]); V=np.array([c[3] for c in cyc]); C=np.stack([c[4] for c in cyc]); PI=np.stack([c[5] for c in cyc])
    return dict(q=Q.mean(0),qd=QD.mean(0),q_std=Q.std(0).mean(),freq=F.mean(),v=V.mean(),clear=C.mean(0),n=len(cyc),pitch=PI.mean(0),freq_std=F.std())
def save(path,out):
    np.savez_compressed(path,bins=np.array(list(out.keys())),**{f"{k}_{kk}":np.array(vv) for k,vd in out.items() for kk,vv in vd.items()})
    for k,vd in out.items(): print(f"  {k:12s} 周期 {vd['n']:3d} 个  频率 {vd['freq']:.2f}±{vd['freq_std']:.2f} Hz  速度 {vd['v']:.2f} m/s  抬腿峰 fl/fr/hl/hr {np.round(vd['clear'].max(0),3)}  周期间关节角标准差 {vd['q_std']:.3f}  俯仰均 {np.degrees(vd['pitch']).mean():+.1f}°")
    print('  存',path)
print('== 踏步（官方）')
out={}
out['step_slow']=pack(cycles('gftbzx1',0.25,0.55,0.3,1.3)+cycles('gftb1',0.25,0.55,0.3,1.3))
out['step_mid']=pack(cycles('gftbzx1',0.55,0.85,0.3,1.3)+cycles('gftb1',0.55,0.85,0.3,1.3))
fast=cycles('gftbzx1',0.85,1.4,0.3,1.3)+cycles('gftb1',0.85,1.4,0.3,1.3)
if len(fast)>=3: out['step_fast']=pack(fast)
save('/home/robot/s10_logs/ref/ref_step_official.npz',out)
print('== 爬楼梯（官方，四拍步行）')
st=cycles('gfstj1',0.15,0.6,0.5,1.6)
print('  候选周期',len(st))
out2={'stair_walk':pack(st)}
save('/home/robot/s10_logs/ref/ref_stairwalk_official.npz',out2)
