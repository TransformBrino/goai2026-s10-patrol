#!/usr/bin/env bash
# S16 真机楼梯"不起步"闭环复现（09-19）：{滚动接近 / 楼梯前静止顶沿起步} × {主策略 V1H / T10} × {S16 / S8}，n=6
# 真机 09-18 21:34：T10 把车停在第一级跟前 50 s（右前轮贴台沿），再切 S16 从零速起步 → 11 s 不动。09-14 S8 成功是 V1H 类主策略滚动接近。
set -u; R=/home/robot/桌面/赵文博/S10决赛交付包-20260908; cd $R; OUT=$HOME/s10_logs/night/s16_entry; mkdir -p $OUT
until grep -q "safe2 完成" $HOME/s10_logs/night/safe2_launch.out 2>/dev/null; do sleep 30; done      # 评测不与训练并行
echo "[$(date +%T)] 开始"
G=$HOME/s10_fork_install_0911i; REL=$HOME/s10_logs/release
V1H=$REL/s10_pack_20260911/V1H_9196/v1h_9196.onnx; T10=$REL/TrotT10_sweep/t10_15197.onnx
S16=$REL/StairS16_sweep/model_15200.onnx; S8=$REL/StairN-S8_14800/stairns8_14800.onnx; CL=$REL/s10_pack_20260911/ClimbH_8797/climbh_8797.onnx
SIM=$R/s10_ws/src/S10_sdk_deploy/interface/robot/simulation
COM="S10_INSTALL=$G S10_AS_WHEEL_PIT=3 S10_POLICY_CLIMB=$CL STAIR_X=3.0 MODE=2 T_MAX=90 KEEP_EXPERT=1
 HEAD_K=2.0 Y_K=0 WZ_MAX=0.6 HEAD_TGT=fixed HEAD_FB_HZ=10 HEAD_FB_DELAY_MS=50 S10_SIM_WALL_LOG=1 TOP_MARGIN=0.14 Z_BACK=0.14
 S10_LIDAR=1 S10_LIDAR_WORKERS=4 S10_HMAP=1 S10_HMAP_EVERY=20 S10_HMAP_TRUTH=0 S10_HMAP_TTL_S=1.5 S10_HMAP_FILL_HOLES=interp_x
 S10_SIM_PY=$SIM/mujoco_simulation_ros2_airy.py S10_LIDAR_MODEL=airy2 S10_HMAP_TRUTH_ERR=1 SCENE=bench_stairs_site_t55_airy N=20 H=0.092 W=0.55 S10_STAIR_HALF_W=20 S10_STAIR_XR=3.0,13.45"
ROLL="SW_D=3.0 V_APP=0.5 V_CLIMB=0.25"                                     # 起点即切专家、滚动接近（S8 30 条口径）
PARK="S10_START_POS=2.68,0,0.2 SW_D=0.5 V_APP=0.0 PRESTOP_S=5 V_CLIMB=0.25"  # 停在第一级跟前（前轮离台沿 ~1 cm）、主策略静止 5 s、再切专家从零速起步
JOBS=()
for i in 1 2 3 4 5 6; do
  JOBS+=("A_roll_V1H_S16_$i S10_POLICY_PATH=$V1H S10_POLICY_PIT=$S16 $ROLL" "B_roll_T10_S16_$i S10_POLICY_PATH=$T10 S10_POLICY_PIT=$S16 $ROLL"
         "C_park_V1H_S16_$i S10_POLICY_PATH=$V1H S10_POLICY_PIT=$S16 $PARK" "D_park_T10_S16_$i S10_POLICY_PATH=$T10 S10_POLICY_PIT=$S16 $PARK"
         "E_park_V1H_S8_$i S10_POLICY_PATH=$V1H S10_POLICY_PIT=$S8 $PARK"   "F_park_T10_S8_$i S10_POLICY_PATH=$T10 S10_POLICY_PIT=$S8 $PARK")
done
job(){ local name=$1; shift
  env $COM "$@" ROS_DOMAIN_ID=$DOM REC_CSV=$OUT/$name.rec.csv S10_LOG_CSV=$OUT/$name.state.csv timeout 300 bash s10_dev/run_stair_switch.sh $name > $OUT/$name.out 2>&1
  echo "[$(date +%T)] $name $(grep -o '结果 [^⛔❌✅（]*' $OUT/$name.out | head -1)"; }
DOMS=(91 92 93); NW=${#DOMS[@]}
for w in $(seq 0 $((NW-1))); do ( DOM=${DOMS[$w]}; for i in $(seq $w $NW $((${#JOBS[@]}-1))); do job ${JOBS[$i]}; done ) & sleep 3; done
wait
$HOME/miniconda3/envs/s10_train/bin/python - $OUT <<'PY' | tee $OUT/汇总.txt
import csv, glob, os, re, sys, numpy as np
D = sys.argv[1]; rows = {}
for f in sorted(glob.glob(f"{D}/*.state.csv")):
    n = os.path.basename(f)[:-10]; cond = n.rsplit("_", 1)[0]
    r = list(csv.DictReader(open(f))); g = lambda k: np.array([float(v[k]) for v in r])
    t, z, cx = g("t"), g("z"), g("cmd_vx"); q = np.stack([g("q%d" % i) for i in (3, 7, 11, 15)], 1) if "q3" in r[0] else None
    on = np.where(cx > 0.2)[0]; t0 = t[on[0]] if len(on) else np.nan
    up = np.where(z - z[on[0] if len(on) else 0] > 0.10)[0] if len(on) else []
    t_step = (t[up[0]] - t0) if len(up) else np.inf
    res = open(f"{D}/{n}.out", errors="ignore").read(); m = re.search(r"结果 ([^⛔❌✅（\n]*)", res); zmax = z.max() - z[0]
    rows.setdefault(cond, []).append((n, m.group(1).strip() if m else "?", zmax, t_step))
print("条件                | 上楼成功 | 90 s 内没起步(z 没抬 10 cm) | 起步用时 中位/最大 (s) | 各次: 结果 / 最高抬升 / 起步用时")
for cond, L in rows.items():
    ok = sum("成功" in x[1] for x in L); ns = sum(np.isinf(x[3]) for x in L); ts = [x[3] for x in L if not np.isinf(x[3])]
    print(f"{cond:20s} | {ok}/{len(L)}     | {ns}/{len(L)}                    | {np.median(ts) if ts else float('nan'):5.1f} / {max(ts) if ts else float('nan'):5.1f}           | " + "; ".join(f"{x[1][:6]} {x[2]:.2f} {x[3]:.1f}" for x in L))
PY
echo "[$(date +%T)] 完成"; touch $HOME/s10_logs/night/safe3.go
