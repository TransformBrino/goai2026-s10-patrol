"""楼梯参考第二版：按左前膝屈曲峰分周期（不用轮高，避免台阶高差污染）；每腿抬高按"相对本腿起抬前 0.3 s 最低点"的鼓包算。"""
import numpy as np
perm=np.zeros(16,dtype=int)
for l in range(4):
    for j in range(4): perm[4*j+l]=4*l+j
A=np.loadtxt('/home/robot/s10_logs/night/official_gait/gfstj1.csv',delimiter=',',skiprows=1)
t=A[:,0]; q=A[:,1:17]; qd=A[:,17:33]; rp=A[:,49:51]; Z=A[:,51:55]; dt=np.median(np.diff(t)); v=(-qd[:,[3,7,11,15]]).mean(1)*0.081
knee=q[:,2]; n2=int(2.0/dt); med=np.array([np.median(knee[max(0,i-n2):i+1]) for i in range(len(knee))]); dev=np.abs(knee-med)
xs=np.convolve(dev,np.ones(9)/9,mode='same')
peaks=[i for i in range(4,len(xs)-4) if xs[i]>0.25 and xs[i]==xs[i-4:i+5].max()]
P=[]
for p in peaks:
    if not P or (p-P[-1])*dt>0.4: P.append(p)
print('左前膝屈曲峰',len(P),'个；相邻间隔中位 %.2f s'%np.median(np.diff(P)*dt))
# 每腿鼓包高度（相对本腿过去 0.3 s 最低点）
n03=int(0.3/dt); H=np.zeros_like(Z)
for k in range(4):
    z=Z[:,k]; base=np.array([z[max(0,i-n03):i+1].min() for i in range(len(z))]); H[:,k]=z-base
cyc=[]
for a,b in zip(P[:-1],P[1:]):
    T=(b-a)*dt; vm=v[a:b].mean()
    if not (0.6<T<1.6) or vm<0.1: continue
    ph=np.linspace(0,1,50,endpoint=False); idx=a+np.round(ph*(b-a)).astype(int)
    cyc.append((q[idx][:,perm],qd[idx][:,perm],1.0/T,vm,H[idx],rp[idx,1]))
print('有效周期',len(cyc))
Q=np.stack([c[0] for c in cyc]); QD=np.stack([c[1] for c in cyc]); F=np.array([c[2] for c in cyc]); V=np.array([c[3] for c in cyc]); C=np.stack([c[4] for c in cyc]); PI=np.stack([c[5] for c in cyc])
out={'stair_walk':dict(q=Q.mean(0),qd=QD.mean(0),q_std=Q[:,:,:12].std(0).mean(),freq=F.mean(),v=V.mean(),clear=C.mean(0),n=len(cyc),pitch=PI.mean(0),freq_std=F.std())}
np.savez_compressed('/home/robot/s10_logs/ref/ref_stairwalk_official.npz',bins=np.array(list(out.keys())),**{f"{k}_{kk}":np.array(vv) for k,vd in out.items() for kk,vv in vd.items()})
o=out['stair_walk']; print(f"频率 {o['freq']:.2f}±{o['freq_std']:.2f} Hz  速度 {o['v']:.2f} m/s  腿关节周期间标准差 {o['q_std']:.3f} rad  俯仰均 {np.degrees(o['pitch']).mean():+.1f}°")
c=o['clear']
for k,nm in enumerate(['fl','fr','hl','hr']): print(f'  {nm} 抬高曲线: '+' '.join(f'{c[i,k]:.2f}' for i in range(0,50,5))+f'  峰 {c[:,k].max():.2f} @相位 {c[:,k].argmax()/50:.2f}')
