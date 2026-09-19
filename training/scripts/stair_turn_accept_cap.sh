#!/usr/bin/env bash
# 真实楼梯几何 + 小幅转向验收（09-14 23:2x，作者给出实测：踢面 11 cm 两级 15 cm、踏面 52~60 cm、宽 4 m、整段 S 型弯弧度不大）
#   场景 bench_stairs_site_t55_airy 的立面就是 15 11 11 11 11 11 0 0 15 11 11 11 11 0 0 11 11 11 11 11、踏面 55cm —— 与实测一致。
#   09-14 16:23 我从里程计 Δz/Δx≈0.6 推"真实约 31°"是错的（里程计 z 有漂移，09-13 拆解自己标注过只作参考），
#   据此建的 0.13~0.15×0.30 陡窄场景把 S12/S13 两轮训练带偏了。
# 用法：PIT=<楼梯专家 onnx> OUT=<目录> bash s10_dev/stair_turn_accept.sh
cd "$(dirname "$0")/.."; R=$PWD; OUT=${OUT:-$HOME/s10_logs/night/stair_turn}; mkdir -p $OUT; PIT=${PIT:?}
G=${S10_INSTALL:-$HOME/s10_fork_install_0911i}; MAIN=$HOME/s10_logs/release/s10_pack_20260911/V1H_9196/v1h_9196.onnx
CL=$HOME/s10_logs/release/s10_pack_20260911/ClimbH_8797/climbh_8797.onnx; SIM=$R/s10_ws/src/S10_sdk_deploy/interface/robot/simulation
# 基线严格照抄 stair_accept_full.sh（F 在这套参数下拿到 30/30），只把"转向"作为唯一变量。
# 23:2x/23:3x 两版都白跑：第一版自己拍参数；第二版补齐参数却漏了整个高程图块
#   （S10_HMAP=1 等）——没有高程图策略是瞎的，站在楼梯口不动完全合理。原因不是策略。
# 旧注：第一版我自己拍了一组参数（V_CLIMB 0.25、HEAD_K 0.5、漏掉 S10_STAIR_XR/TOP_MARGIN/Z_BACK），
# 六条全部卡在楼梯口 x=2.64 上不去 —— 换了一堆变量再测，测的是我的脚本不是策略。
COM="S10_INSTALL=$G S10_AS_WHEEL_PIT=3 S10_POLICY_PATH=$MAIN S10_POLICY_CLIMB=$CL S10_POLICY_PIT=$PIT STAIR_X=3.0 V_APP=0.5 MODE=2 T_MAX=140
 HEAD_K=2.0 Y_K=0 WZ_MAX=0.6 HEAD_TGT=fixed HEAD_FB_HZ=10 HEAD_FB_DELAY_MS=50 S10_SIM_WALL_LOG=1 TOP_MARGIN=0.14 Z_BACK=0.14 SW_D=3.0 KEEP_EXPERT=1
 S10_LIDAR=1 S10_LIDAR_WORKERS=4 S10_HMAP=1 S10_HMAP_EVERY=20 S10_HMAP_TRUTH=0 S10_HMAP_TTL_S=1.5 S10_HMAP_FILL_HOLES=interp_x
 S10_SIM_PY=$SIM/mujoco_simulation_ros2_airy.py S10_LIDAR_MODEL=airy2 S10_HMAP_TRUTH_ERR=1 SCENE=bench_stairs_site_t55_airy N=20 H=0.092 W=0.55 S10_STAIR_HALF_W=20 S10_STAIR_XR=3.0,13.45"
# 直行基线 0.5 / 慢档 0.25；起步偏航 ±8°（S 弯入口没对正）；爬楼中航向目标偏 ±8°（作者：弯度不大）
JOBS=("ST05 V_CLIMB=0.5" "ST025 V_CLIMB=0.25"
      "YP8 V_CLIMB=0.5 S10_START_YAW=8" "YN8 V_CLIMB=0.5 S10_START_YAW=-8"
      "TG8P V_CLIMB=0.5 PSI0_DEG=8" "TG8N V_CLIMB=0.5 PSI0_DEG=-8")
job(){ local name=$1; shift; local try
  for try in 1 2 3; do
    env $COM "$@" ROS_DOMAIN_ID=$DOM REC_CSV=$OUT/slp_$name.rec.csv S10_LOG_CSV=$OUT/slp_$name.state.csv timeout 400 bash s10_dev/run_stair_switch.sh slp_$name > $OUT/slp_$name.out 2>&1
    grep -aq "拒绝开测" $OUT/slp_$name.out || return 0
    sleep 20
  done; }
DOMS=(91 92 93); NW=${#DOMS[@]}
for w in $(seq 0 $((NW-1))); do ( DOM=${DOMS[$w]}; for i in $(seq $w $NW $((${#JOBS[@]}-1))); do job ${JOBS[$i]}; done ) & sleep 3; done
wait
{ echo "== $(basename $PIT) $(md5sum $PIT | cut -c1-8)  真实楼梯几何（11/15cm × 55cm）+ 小幅转向  $(date +%T)"
  for j in "${JOBS[@]}"; do n=${j%% *}; printf "  %-7s %s %s\n" "$n" \
    "$(grep -o '结果 [^⛔❌✅（]*' $OUT/slp_$n.out 2>/dev/null | head -1)" \
    "$(grep -o '用时 [0-9.]* s\||roll|峰 [0-9.]*°\|最高 z=[0-9.]*' $OUT/slp_$n.out 2>/dev/null | head -3 | tr '\n' ' ')"
    grep -ao "偏航：切换时 [-+0-9.]*°  爬楼中 [-+0-9.~]*°" $OUT/slp_$n.out 2>/dev/null | head -1 | sed 's/^/          /'
  done
} | tee $OUT/汇总.txt
