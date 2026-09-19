#!/usr/bin/env bash
# 主策略（踏步版）验收（09-13 傍晚）：MAIN=<onnx> OUT=<目录> bash s10_dev/main_accept.sh
#  ① 平地：站 0 速 / 0.5 / 1.0 / 1.8 / 2.3 / 倒退 −0.5 / 原地转 ±0.6（bench_flat，域 99）→ 几何触地统计（是否踏步）+ 速度/转向跟踪
#  ② 摩擦 0.3 平地 0.5 / 1.0 / 1.8（域 98）
#  ③ 园区台阶 bench_park（立面 12/15/18/20、踏面 0.8）主策略自己走 14 s（域 97）→ 看爬到多高、翻没翻
#  ④ 33 cm 平台：主策略走到离墙 1 m 切 ClimbH_8797、登顶后切回主策略（bench_top33，域 95/96，各 2 条）
#  ⑤ 楼梯：主策略走到离第一级 1 m 切楼梯专家 E2_10300（stair_trot.sh TX，域 95/96）
cd "$(dirname "$0")/.."
R=$PWD; OUT=${OUT:-$HOME/s10_logs/night/main_accept}; mkdir -p $OUT; MAIN=${MAIN:?需要 MAIN=<onnx>}
G=$HOME/s10_fork_install_0911i; CL=$HOME/s10_logs/release/s10_pack_20260911/ClimbH_8797/climbh_8797.onnx; PIT=$HOME/s10_logs/release/StairN-E2_10300/stairne2_10300.onnx
S=/tmp/claude-1000/-home-robot--------S10------20260908/321ae045-4f3d-4f3f-847e-392b88e667db/scratchpad
COMV="S10_INSTALL=$G S10_POLICY_PATH=$MAIN S10_SWITCH_DRIVER=$R/s10_dev/g_verify_driver.py S10_CMD_MAX_VX=3.0"
echo "== 主策略验收 $(basename $MAIN) $(md5sum $MAIN | cut -c1-8)  $(date +%T)" | tee $OUT/汇总.txt
# ① ② ③ 并行
( env $COMV SCENE=bench_flat ROS_DOMAIN_ID=99 S10_LOG_CSV=$OUT/flat.state.csv PHASES=1:4,6:2,6v0:3,6v0.5:4,6v1.0:4,6v1.8:4,6v2.3:4,6v-0.5:3,6v0:2,6w0.6:4,6w-0.6:4,4:4 timeout 240 bash s10_dev/run_stair_switch.sh ma_flat > $OUT/flat.out 2>&1 ) &
sleep 2; ( env $COMV SCENE=bench_flat_mu03 ROS_DOMAIN_ID=98 S10_LOG_CSV=$OUT/mu03.state.csv PHASES=1:4,6:2,6v0.5:4,6v1.0:4,6v1.8:4,4:4 timeout 200 bash s10_dev/run_stair_switch.sh ma_mu03 > $OUT/mu03.out 2>&1 ) &
sleep 2; ( env $COMV SCENE=bench_park ROS_DOMAIN_ID=97 S10_LOG_CSV=$OUT/park.state.csv PHASES=1:4,6:2,6v0.5:14,4:4 timeout 200 bash s10_dev/run_stair_switch.sh ma_park > $OUT/park.out 2>&1 ) &
wait
{ echo "## ① 平地"; $HOME/miniconda3/envs/s10_train/bin/python $S/contact_flat.py $OUT/flat.state.csv 2>&1 | grep -v Warning
python3 - $OUT/flat.state.csv <<'EOF'
import csv,sys,statistics as st
r=[dict((k,float(v)) for k,v in x.items()) for x in csv.DictReader(open(sys.argv[1]))]
for lo,hi,nm in ((-0.6,-0.4,'倒退 -0.5'),(0.4,0.6,'0.5'),(0.9,1.1,'1.0'),(1.7,1.9,'1.8'),(2.2,2.4,'2.3')):
    s=[q for q in r if lo<q['cmd_vx']<hi and abs(q['cmd_wz'])<0.05]
    if len(s)>50: print(f"   指令 {nm} → 实际 vx 均 {st.mean(q['vx'] for q in s):+.2f} m/s  偏航角速度均 {st.mean(q['wz'] for q in s):+.2f}  站高均 {st.mean(q['z'] for q in s):.3f}")
# 0 速段按连续片段分别算位移（两段 0 速相隔很远，合起来算会把中间的行程算进去）
zs=[q for q in r if abs(q['cmd_vx'])<0.05 and abs(q['cmd_wz'])<0.05 and q['t']>7]
segs=[]; cur=[]
for q in zs:
    if cur and q['t']-cur[-1]['t']>0.1: segs.append(cur); cur=[]
    cur.append(q)
if cur: segs.append(cur)
for k,sg in enumerate(segs):
    if len(sg)>50: print(f"   0 速站立 #{k+1}（{sg[-1]['t']-sg[0]['t']:.1f} s）：位移 {max(q['x'] for q in sg)-min(q['x'] for q in sg):.2f} m  站高均 {st.mean(q['z'] for q in sg):.3f}")
for nm,cond in (('+0.6',lambda q:q['cmd_wz']>0.3),('-0.6',lambda q:q['cmd_wz']<-0.3)):
    s=[q for q in r if cond(q)]
    if len(s)>50: print(f"   转向 {nm}: 实际 wz 均 {st.mean(q['wz'] for q in s):+.2f} rad/s")
EOF
grep -a "|roll|峰" $OUT/flat.out | grep -o "|roll|峰 [0-9.]*° |pitch|峰 [0-9.]*°" | tr '\n' ' '; echo
echo "## ② 摩擦 0.3"; $HOME/miniconda3/envs/s10_train/bin/python $S/contact_flat.py $OUT/mu03.state.csv 2>&1 | grep -v Warning; grep -a "|roll|峰" $OUT/mu03.out | grep -o "|roll|峰 [0-9.]*° |pitch|峰 [0-9.]*°" | tr '\n' ' '; echo
echo "## ③ 园区台阶 bench_park（12/15/18/20 cm，踏面 0.8）"; python3 - $OUT/park.state.csv <<'EOF'
import csv,sys
r=[dict((k,float(v)) for k,v in x.items()) for x in csv.DictReader(open(sys.argv[1]))]
print(f"   末端 x={r[-1]['x']:.2f} z={r[-1]['z']:.2f}（四级全上 z≈1.07） 最高 z={max(q['z'] for q in r):.2f}  |roll|max {max(abs(q['roll']) for q in r):.0f}°  |pitch|max {max(abs(q['pitch']) for q in r):.0f}°")
EOF
} | tee -a $OUT/汇总.txt
# ④ 33 cm 平台切 ClimbH（2 条）  ⑤ 楼梯切专家（2 条）
for k in 1 2; do ( env S10_INSTALL=$G S10_POLICY_PATH=$MAIN S10_POLICY_CLIMB=$CL S10_POLICY_PIT=$PIT SCENE=bench_top33 N=1 H=0.33 W=1.5 STAIR_X=3.0 S10_STAIR_XR=3.0,4.5 SW_D=1.0 MODE=1 V_APP=0.5 V_CLIMB=0.4 T_MAX=60 ROS_DOMAIN_ID=$((94+k)) REC_CSV=$OUT/top33_$k.rec.csv timeout 200 bash s10_dev/run_stair_switch.sh ma_top33_$k > $OUT/top33_$k.out 2>&1 ) & sleep 2; done; wait
{ echo "## ④ 33 cm 平台（主策略 → ClimbH → 主策略）"; for k in 1 2; do printf "   top33_%d " $k; grep -o "结果 [^⛔❌✅（]*\||roll|峰 [0-9.]*°\||pitch|峰 [0-9.]*°\|用时 [0-9.]* s" $OUT/top33_$k.out | head -4 | tr '\n' ' '; echo; done; } | tee -a $OUT/汇总.txt
rm -rf $OUT/stairs; MAIN=$MAIN OUT=$OUT/stairs DOMS_OVERRIDE="95 96" ONLY="^TX_" bash s10_dev/stair_trot.sh > /dev/null 2>&1
{ echo "## ⑤ 楼梯（主策略走到 1 m 切 E2_10300）"; for f in $OUT/stairs/slp_*.out; do printf "   %s " $(basename $f .out); grep -o "结果 [^⛔❌✅（]*\||roll|峰 [0-9.]*°\|用时 [0-9.]* s" $f | head -3 | tr '\n' ' '; echo; done; echo "$(date +%T) 完成"; } | tee -a $OUT/汇总.txt
