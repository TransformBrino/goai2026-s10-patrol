#!/bin/bash
# S10（09-12）：起 IMU 转发 + FAST-LIO。默认 domain 0（真机总线，只订阅点云/IMU，只发 /s10/imu 和 /Odometry 等）；
# 回放测试用 ROS_DOMAIN_ID=222 bash lio_up.sh。PID 写 /tmp/s10_lio/*.pid，日志 /tmp/s10_lio/*.log。不动 KISS/hmap。
set -e
R=/tmp/s10_lio; mkdir -p $R
[ -f $R/lio.pid ] && kill -0 $(cat $R/lio.pid) 2>/dev/null && { echo "!! FAST-LIO 已在跑（$(cat $R/lio.pid)），先 lio_down"; exit 3; }
source /opt/ros/jazzy/setup.bash; source /home/robot/s10_env.sh; source /home/robot/lio_ws/install/setup.bash
CFG=${1:-/home/robot/lio_ws/src/FAST_LIO/config/s10_airy.yaml}
setsid nohup python3 /home/robot/lio_ws/s10_imu_relay.py /s10/imu > $R/imu_relay.log 2>&1 < /dev/null & echo $! > $R/imu.pid
sleep 1
setsid nohup nice -n 5 /home/robot/lio_ws/install/fast_lio/lib/fast_lio/fastlio_mapping --ros-args --params-file "$CFG" > $R/lio.log 2>&1 < /dev/null & echo $! > $R/lio.pid
echo "[lio_up] $(date +%T) domain ${ROS_DOMAIN_ID:-0}  imu_relay pid $(cat $R/imu.pid)  fastlio pid $(cat $R/lio.pid)  cfg $CFG" | tee -a $R/events.log
