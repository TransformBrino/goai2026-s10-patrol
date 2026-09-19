#!/usr/bin/env bash
# Trot-Phase-T10-Safe：治自激 + 完成压步频。冒烟 → 400 点 → export246 → **自激门（硬）** → 平地实测
set -u
R=/home/robot/桌面/赵文博/S10决赛交付包-20260908; PY=$HOME/miniconda3/envs/s10_train/bin/python
NV=$HOME/miniconda3/envs/s10_train/lib/python3.12/site-packages/nvidia/cu13/lib
TRAIN='^/home/robot/miniconda3/envs/s10_train/bin/python scripts/reinforcement_learning/rsl_rl/train.py'
export S10_TROTP_W=${S10_TROTP_W:-12} S10_TROTP_YAWW=${S10_TROTP_YAWW:-24} S10_TROTP_LRW=${S10_TROTP_LRW:-8}
TAG=${TAG:-safe1}; ITERS=${ITERS:-400}; LR=${LR:-2026-09-18_21-29-42}; CK=${CK:-model_17192.pt}
cd $R; while pgrep -f "$TRAIN" >/dev/null 2>&1; do sleep 15; done
S=$HOME/s10_logs/smoke_$TAG.log; echo "[$(date +%T)] $TAG 冒烟"
( cd rl_training-main && env LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}" $PY scripts/reinforcement_learning/rsl_rl/train.py \
  --task PerceptS10-TrotPhaseT10Safe-v0 --num_envs 16 --headless --resume --load_run $LR --checkpoint $CK --max_iterations 2 > $S 2>&1 )
OK=1; grep -q Traceback $S && { echo "❌ Traceback"; OK=0; }; grep -q "PhaseT10-Safe" $S || { echo "❌ 无 banner"; OK=0; }
[ "$(grep -ac 'Learning iteration' $S)" -ge 2 ] || { echo "❌ 没跑完迭代"; OK=0; }; grep -q "in_features=246" $S || { echo "❌ 非 246"; OK=0; }
[ $OK -eq 0 ] && { tail -30 $S; exit 1; }; grep -m1 "PhaseT10-Safe" $S | cut -c1-200; echo "✅ 冒烟过 → $ITERS 点"
( cd rl_training-main && env LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}" $PY scripts/reinforcement_learning/rsl_rl/train.py \
  --task PerceptS10-TrotPhaseT10Safe-v0 --num_envs 8192 --headless --resume --load_run $LR --checkpoint $CK --max_iterations $ITERS > $HOME/s10_logs/train_$TAG.log 2>&1 )
L=$HOME/s10_logs/train_$TAG.log; echo "[$(date +%T)] 训练结束"
for k in big_action gp_swing st_yaw_ref wheel_action_lr sg_rate track_lin_vel_xy_exp; do printf "  %-20s %s\n" $k "$(tac $L | grep -a -m1 -oP "Episode_Reward/$k: *\K-?[0-9.]+")"; done
RUN=$(ls -t rl_training-main/logs/rsl_rl/deeprobotics_s10_percept | head -1); CKP=$(ls -t rl_training-main/logs/rsl_rl/deeprobotics_s10_percept/$RUN/model_*.pt | head -1)
echo "NEXT $RUN $(basename $CKP)"; O=$HOME/s10_logs/release/TrotPhase246_$TAG; mkdir -p $O
env LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}" $PY s10_dev/export246.py "$CKP" "$O/$TAG.onnx" 2>&1 | tail -1
echo "######## 自激门（硬）########"; $PY s10_dev/selfexcite_gate.py $O/$TAG.onnx --pass 7 2>/dev/null; echo "  门退出码 $?"
echo "######## 平地实测 ########"; F=$HOME/s10_logs/night/phase246_$TAG; rm -rf $F; mkdir -p $F
env S10_INSTALL=$HOME/s10_fork_install S10_POLICY_PATH=$O/$TAG.onnx S10_SWITCH_DRIVER=$R/s10_dev/g_verify_driver.py S10_CMD_MAX_VX=3.0 S10_GAIT_FREQ=1.10 \
    SCENE=bench_flat ROS_DOMAIN_ID=91 S10_LOG_CSV=$F/flat.state.csv PHASES=1:4,6:2,6v0.5:10,6v1.0:10,6v1.8:8,6v-0.5:5,4:3 \
    timeout 320 bash s10_dev/run_stair_switch.sh ph_$TAG > $F/flat.out 2>&1
$PY - $F/flat.state.csv <<'PYEOF'
import csv,numpy as np,sys
sys.path.insert(0,"/home/robot/桌面/赵文博/S10决赛交付包-20260908/s10_dev"); from mjfk import FK
fk=FK(); r=list(csv.DictReader(open(sys.argv[1]))); g=lambda k: np.array([float(v[k]) for v in r])
t,vx,vy,cx,z=g("t"),g("vx"),g("vy"),g("cmd_vx"),g("z"); rl=np.abs(g("roll")); yaw=np.radians(g("yaw")); Q=np.stack([g("q%d"%i) for i in range(16)],1)
fwd=vx*np.cos(yaw)+vy*np.sin(yaw); tr=[];yw=[]
for c in (0.5,1.0,1.8,-0.5):
    m=np.isclose(cx,c,atol=0.01); s=np.where(m)[0]
    if len(s)<200: continue
    m2=m.copy(); m2[s[:int(0.3*len(s))]]=False; yd=np.degrees(yaw[m2]); tr.append(f"{c:+.1f}→{fwd[m2].mean():+.2f}"); yw.append(f"{yd.max()-yd.min():.0f}°")
bot=np.array([fk.step(g("x")[i],g("y")[i],g("z")[i],np.radians(g("roll")[i]),np.radians(g("pitch")[i]),yaw[i],Q[i])["wheel_bottom"] for i in range(len(r))])
fr=[];rp=[];ks=[]
for lo,hi in [(0.35,0.65),(0.85,1.15),(1.6,2.0)]:
    m=(fwd>lo)&(fwd<hi)
    if m.sum()<300: fr.append("—"); rp.append("—"); ks.append("—"); continue
    idx=np.where(m)[0]; dur=t[idx[-1]]-t[idx[0]]
    fr.append(f"{np.mean([len(np.where((~(bot[idx[0]:idx[-1],w]>0.04)[:-1])&(bot[idx[0]:idx[-1],w]>0.04)[1:])[0])/max(dur,1e-6) for w in range(4)]):.2f}")
    rp.append(f"{np.percentile(rl[m],95):.1f}"); ks.append(f"{Q[m][:,[2,6,10,14]].std(axis=0).min():.2f}")
print(f"  步频 {'/'.join(fr)} Hz ｜ 跟踪 {' '.join(tr)} ｜ yaw跨度 {' '.join(yw)} ｜ roll p95 {'/'.join(rp)} ｜ 膝σ {'/'.join(ks)}  [末端x {g('x')[-1]:.1f}]")
print( "  lr8 参照：步频 1.31/1.35/1.54 ｜ +0.5→+0.36 +1.0→+0.80 +1.8→+1.63 -0.5→-0.56 ｜ yaw 4° 21° 28° 2°")
PYEOF
echo "[$(date +%T)] $TAG 完成"
