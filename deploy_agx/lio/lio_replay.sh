#!/bin/bash
# S10（09-12）：离线回放点云+IMU，跑 FAST-LIO 看位姿。隔离：ROS_DOMAIN_ID=222 + 只走本机，不碰 domain 0 的在线 KISS/hmap/runner。
# 只在狗空闲时跑：开跑前有 runner 就不跑；回放中发现 runner 起来了立即停。
# 输出 /Odometry（FAST-LIO）和包里原来的 /kiss/odometry 一起录到 s10_data/lio_replay/<标签>，之后用 lio_cmp.py 比。
# 用法：bash lio_replay.sh <点云包> <标签> [倍速，默认 0.7] [配置 yaml]
# 退出码：0 完成；1 参数/标签问题；2 因 runner 在跑而没跑或中途停止
[ $# -ge 2 ] || { echo "用法：bash lio_replay.sh <点云包> <标签> [倍速] [配置]"; exit 1; }
pgrep -x rl_deploy >/dev/null && { echo "[lio_replay] 有 runner 在跑，不回放"; exit 2; }
source /opt/ros/jazzy/setup.bash
source /home/robot/s10_env.sh
source /home/robot/lio_ws/install/setup.bash
export ROS_DOMAIN_ID=222
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
BAG=$1; TAG=$2; RATE=${3:-0.7}; CFG=${4:-/home/robot/lio_ws/src/FAST_LIO/config/s10_airy.yaml}
OUT=/home/robot/s10_data/lio_replay/$TAG
mkdir -p /home/robot/s10_data/lio_replay
[ -e "$OUT" ] && { echo "[lio_replay] 已存在 $OUT，换个标签"; exit 1; }
echo "[lio_replay] $TAG：包 $BAG 倍速 $RATE 域 $ROS_DOMAIN_ID 配置 $CFG"
nice -n 15 python3 /home/robot/lio_ws/s10_imu_relay.py /s10/imu > /tmp/lio_replay_imu_$TAG.log 2>&1 &
I=$!
nice -n 15 /home/robot/lio_ws/install/fast_lio/lib/fast_lio/fastlio_mapping --ros-args --params-file "$CFG" > /tmp/lio_replay_$TAG.log 2>&1 &
K=$!
sleep 4
nice -n 15 ros2 bag record -s mcap -o "$OUT" --topics /Odometry /kiss/odometry > /tmp/lio_replay_rec_$TAG.log 2>&1 &
R=$!
sleep 2
nice -n 15 ros2 bag play "$BAG" --topics /LIDAR/POINTS_MERGED /IMU_DATA /kiss/odometry --rate "$RATE" > /tmp/lio_replay_play_$TAG.log 2>&1 &
PL=$!
ABORT=0
while kill -0 $PL 2>/dev/null; do
  if pgrep -x rl_deploy >/dev/null; then
    echo "[lio_replay] 发现 runner 起来了，立即停止回放"; kill -TERM $PL; ABORT=1; break
  fi
  sleep 0.5
done
wait $PL 2>/dev/null
sleep 2
kill -TERM $R 2>/dev/null; for i in 1 2 3 4 5 6; do kill -0 $R 2>/dev/null || break; sleep 1; done; kill -9 $R 2>/dev/null
kill -TERM $K $I 2>/dev/null; sleep 2; kill -9 $K $I 2>/dev/null
echo "[lio_replay] $TAG 结束（abort=$ABORT）：$(ls $OUT 2>/dev/null | tr '\n' ' ')"
grep -c "" /tmp/lio_replay_$TAG.log >/dev/null && echo "[lio_replay] FAST-LIO 日志尾：" && tail -n 3 /tmp/lio_replay_$TAG.log | cut -c1-160
[ $ABORT = 1 ] && exit 2
exit 0
