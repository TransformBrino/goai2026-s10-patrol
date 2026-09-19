"""真机踏步参考：按步态周期切分（fl 离地高度峰值为周期起点），相位归一化到 50 点，按速度档平均，
输出 ref_trot.npz：bins × 50 × 16 关节角（articulation 顺序）、关节速度、周期频率、机身速度、抬腿高度。"""
import numpy as np, math, sys, os
L=0.18
perm=np.zeros(16,dtype=int)
for l in range(4):
    for j in range(4): perm[4*j+l]=4*l+j
segs={"trot_slow":(0.2,0.6),"trot_mid":(0.6,1.0),"trot_fast":(1.5,2.5),"trot_rough":(0.6,1.6)}
out={}
for seg,(vlo,vhi) in segs.items():
    d=np.load(f"/home/robot/桌面/赵文博/实地数据/parsed/{seg}.npz",allow_pickle=True)
    t=d["t"]-d["t"][0]; q=d["q"]; qd=d["qd"]; dt=np.median(np.diff(t))
    v=(-qd[:,3::4]).mean(1)*0.081
    z=[]
    for leg in range(4):
        a1=q[:,leg*4+1]; a2=a1+q[:,leg*4+2]; z.append(-L*(np.cos(a1)+np.cos(a2)))
    z=np.stack(z,1); clear=z-z.min(1,keepdims=True)
    x=clear[:,0]; xs=np.convolve(x,np.ones(3)/3,mode="same")
    # 峰值 = 局部最大且 > 0.04
    peaks=[i for i in range(2,len(xs)-2) if xs[i]>0.04 and xs[i]>=xs[i-1] and xs[i]>=xs[i+1] and xs[i]>=xs[i-2] and xs[i]>=xs[i+2]]
    # 去掉间隔 < 0.25 s 的重复峰
    P=[]
    for p in peaks:
        if not P or (p-P[-1])*dt>0.25: P.append(p)
    cycles=[]
    for a,b in zip(P[:-1],P[1:]):
        T=(b-a)*dt; vm=v[a:b].mean()
        if not (0.3<T<1.2) or not (vlo<=vm<=vhi): continue
        ph=np.linspace(0,1,50,endpoint=False); idx=a+np.round(ph*(b-a)).astype(int)
        cycles.append((q[idx][:,perm], qd[idx][:,perm], 1.0/T, vm, clear[idx]))
    if len(cycles)<3: print(seg,"周期太少",len(cycles)); continue
    Q=np.stack([c[0] for c in cycles]); QD=np.stack([c[1] for c in cycles]); F=np.array([c[2] for c in cycles]); V=np.array([c[3] for c in cycles]); C=np.stack([c[4] for c in cycles])
    out[seg]=dict(q=Q.mean(0), qd=QD.mean(0), q_std=Q.std(0).mean(), freq=F.mean(), v=V.mean(), clear=C.mean(0), n=len(cycles))
    print("%-10s 周期 %d 个  频率 %.2f±%.2f Hz  速度 %.2f m/s  抬腿峰 fl/fr/hl/hr %s  关节角周期间标准差 %.3f rad" % (seg,len(cycles),F.mean(),F.std(),V.mean(),np.round(C.mean(0).max(0),3),Q.std(0).mean()))
os.makedirs("/home/robot/s10_logs/ref",exist_ok=True)
np.savez_compressed("/home/robot/s10_logs/ref/ref_trot.npz", bins=np.array(list(out.keys())), **{f"{k}_{kk}": np.array(vv) for k,vd in out.items() for kk,vv in vd.items()})
print("存 /home/robot/s10_logs/ref/ref_trot.npz")
