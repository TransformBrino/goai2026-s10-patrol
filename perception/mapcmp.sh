#!/usr/bin/env bash
# 雷达累积图 vs 真值射线图 逐格对拍（同一趟里同时算）：机器人以 0.5 m/s 逼近 33cm 立面，看图在墙前哪几格、哪段距离出错。
# 用法: mapcmp.sh [policy_onnx=/root/s10_logs/r2_model_30500.onnx] [fwd_clip=0.6] [lat_clip=0.30]
set +u
POL=${1:-/root/s10_logs/r2_model_30500.onnx}; FWD=${2:-0.6}; LAT=${3:-0.30}
D=/mnt/c/Users/86156/Desktop/机器狗项目/s10_dev
WS=/mnt/c/Users/86156/Desktop/机器狗项目/s10_ws
R=$D/results/mapcmp; mkdir -p "$R"
source /opt/ros/jazzy/setup.bash; source "$HOME/s10_install/setup.bash"
export ROS_DOMAIN_ID=1 S10_USE_VIEWER=0 S10_REALTIME=1 S10_CMD_SOURCE=ros
export S10_MUJOCO_XML=$WS/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/bench_top33.xml
export S10_LIDAR=1 S10_LIDAR_WORKERS=4 S10_HMAP=1 S10_HMAP_EVERY=20 S10_CMD_MAX_VX=2.0
export S10_POLICY_PATH=$POL S10_HMAP_FWD_CLIP=$FWD S10_HMAP_LAT_CLIP=$LAT
pkill -x rl_deploy 2>/dev/null; pkill -f "[m]ujoco_simulation_ros2" 2>/dev/null; pkill -f "[t]ruthcmp.py" 2>/dev/null; sleep 1
python3 -u "$WS/src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py" > "$R/sim.log" 2>&1 &
SIM=$!; sleep 8
ros2 run s10_sdk_deploy rl_deploy > "$R/deploy.log" 2>&1 &
DEP=$!; sleep 6
python3 "$D/nav/bringup_sim.py" >/dev/null || { kill $DEP $SIM; exit 1; }
echo "--- 对拍 45s：策略 $(basename $POL)，裁剪 FWD=$FWD LAT=$LAT，指令 0.5 m/s ---"
timeout -k 5 50 ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}" > /dev/null 2>&1 &
python3 "$D/nav/truthcmp.py" --secs 45 --fwd-clip "$FWD" --lat-clip "$LAT" --wall-x 3.0 2>&1 | tail -14 | tee "$R/cmp_$(basename ${POL%.onnx})_fwd${FWD}.txt"
kill $DEP $SIM 2>/dev/null; sleep 1; pkill -x rl_deploy 2>/dev/null; pkill -f "[m]ujoco_simulation_ros2" 2>/dev/null
echo MAPCMP_DONE
