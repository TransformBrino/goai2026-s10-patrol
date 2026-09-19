#!/usr/bin/env bash
# 切换上墙端到端测试：通用策略走向墙，离墙 1 m 发 /climb_mode 1 并降速，上台面后切回。
# 环境与启动顺序照搬 run_bench_linux.sh，只把评测驱动换成 switch_stair_driver.py。
# 清理只按 PID 杀，不用 pkill -f（按命令行匹配会误杀调用方的 shell）。
# 用法: SCENE=bench_stairs_w37_15L S10_POLICY_PATH=<通用> S10_POLICY_PIT=<楼梯专家> [WALL_X=3.0 WALL_H=0.33 SW_D=1.0] bash run_switch_test.sh <tag>
set +u
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)
source "$(dirname "$0")/s10_env.sh"
source /opt/ros/jazzy/setup.bash
source "${S10_INSTALL:-$HOME/s10_install}/setup.bash"   # S10_INSTALL 可指定别的编译（新旧 A/B 不必改软链）
PY=/usr/bin/python3; TAG=${1:-test}
WS="$S10_PROJ/s10_ws"; MJCF="$WS/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf"
SIM="${S10_SIM_PY:-$WS/src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py}"   # S10_SIM_PY 可换延迟仿真副本（09-11 晚）
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-91} S10_USE_VIEWER=${S10_USE_VIEWER:-0} S10_REALTIME=1 S10_CMD_SOURCE=ros
# 09-11：本机已连上机器狗 WiFi（10.21.41.x，可路由到 10.21.33.x），机器人/AGX 用 ROS 域 1（包内文档）。
# 仿真只许在本机通信（LOCALHOST），且不许用 0/1 域 —— 否则仿真发的 /cmd_vel /robot_mode /climb_mode /JOINTS_CMD 可能被机器人侧收到。
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_STATIC_PEERS=""
case "$ROS_DOMAIN_ID" in 0|1) echo "!! 拒绝在 ROS 域 $ROS_DOMAIN_ID 上跑仿真（机器人/AGX 用这个域）"; exit 4;; esac
export S10_HMAP_TRUTH=${S10_HMAP_TRUTH:-1} S10_LIDAR=${S10_LIDAR:-0}   # 默认真值高程图；雷达测给 S10_LIDAR=1 S10_LIDAR_WORKERS=4 S10_HMAP=1 S10_HMAP_EVERY=20 S10_HMAP_TRUTH=0
export S10_DEF_HIPX=0.05 S10_DEF_HIPY=0.35 S10_DEF_KNEE=0.65 S10_KD_WHEEL=0.8
export S10_START_POS=${S10_START_POS:-0,0,0.2} S10_CMD_TIMEOUT_MS=500   # 09-12：起点可覆盖（接近偏移测试）
export S10_MUJOCO_XML="$MJCF/${SCENE:-bench_stairs_w37_15L}.xml"
# 开测前必须保证本 ROS 域没有别的 runner / 仿真：同一域里每多一个 runner 就多一路 /JOINTS_CMD，
# 仿真按最后到达的那条执行 —— 09-11 下午 17 个孤儿 runner 因此串台了 16 次测试。不同 ROS_DOMAIN_ID 互不干扰，可并行。
_s10_dom_procs() { local p d out=""; for p in $(pgrep -x rl_deploy) $(pgrep -f 'mujoco_simulation_ro[s]2'); do
    d=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^ROS_DOMAIN_ID=//p'); [ "${d:-0}" = "${ROS_DOMAIN_ID:-0}" ] && out="$out $p"; done; echo "$out"; }
BUSY=""; for _i in $(seq 1 90); do BUSY=$(_s10_dom_procs); [ -z "$BUSY" ] && break; sleep 0.5; done
[ -n "$BUSY" ] && { echo "!! ROS 域 $ROS_DOMAIN_ID 上已有 rl_deploy/仿真进程（PID$BUSY），会串台，拒绝开测"; exit 3; }
echo "开测前检查：ROS 域 $ROS_DOMAIN_ID 无残留 runner/仿真 ✓"
RUN=/tmp/s10to.$$; mkdir -p $RUN
OUTD=$S10_LOGS/bench/switch_$TAG; mkdir -p $OUTD
cleanup(){ [ -n "$DEP" ] && kill $DEP 2>/dev/null; [ -n "$SIMP" ] && kill $SIMP 2>/dev/null; sleep 1
           kill -9 $DEP $SIMP 2>/dev/null; sleep 0.5; cp -f $RUN/*.log $OUTD/ 2>/dev/null
           local L; L=$(_s10_dom_procs); [ -n "$L" ] && { echo "!! 清理后本域仍有残留 PID:$L → SIGKILL"; kill -9 $L 2>/dev/null; }; }
trap cleanup EXIT INT TERM
$PY -u "$SIM" > $RUN/sim.log 2>&1 & SIMP=$!
for i in $(seq 1 120); do grep -q "MuJoCo model loaded" $RUN/sim.log 2>/dev/null && break; kill -0 $SIMP 2>/dev/null || break; sleep 0.5; done
grep -q "MuJoCo model loaded" $RUN/sim.log || { echo "!! 仿真启动失败"; tail -20 $RUN/sim.log; exit 1; }
sleep 2
# 直接起二进制，不经 ros2 run：ros2 run 是 Python 包装进程，杀它带不走真正的 rl_deploy（被 systemd --user 收养后继续
# 订阅 /cmd_vel /climb_mode /robot_mode、发 /JOINTS_CMD）。09-11 查实 17 个孤儿，/JOINTS_CMD 发布者 18 个。
EXE="$(ros2 pkg prefix s10_sdk_deploy 2>/dev/null)/lib/s10_sdk_deploy/rl_deploy"
[ -x "$EXE" ] || { echo "!! 找不到 rl_deploy 可执行文件: $EXE"; exit 1; }
echo "runner 二进制 = $(readlink -f "$EXE")"
"$EXE" > $RUN/deploy.log 2>&1 & DEP=$!
for i in $(seq 1 60); do grep -q "ROS 指令接口已启动" $RUN/deploy.log 2>/dev/null && break; kill -0 $DEP 2>/dev/null || break; sleep 0.5; done
grep -q "ROS 指令接口已启动" $RUN/deploy.log || { echo "!! runner 启动失败"; tail -20 $RUN/deploy.log; exit 1; }
export S10_RUNNER_PID=$DEP   # 09-12：驱动的 STALL_X/STALL_MS 用它对 runner 发 SIGSTOP/SIGCONT（真停顿对照）
$PY -u "${S10_SWITCH_DRIVER:-$(dirname "$0")/switch_stair_driver.py}" | tee $RUN/driver.log   # S10_SWITCH_DRIVER 可换驱动（0911f 模式切换验证用）
