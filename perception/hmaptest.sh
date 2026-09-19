#!/usr/bin/env bash
# 高程图节点对拍：仿真内部 /height_scan（真值位姿 + 内部累积）vs 外部 hmap_node 从 /front_lidar/points 重建的 /height_scan_ext
# 场景 bench_park（前方有台阶，看得出高程差）。机器人以 0.5 m/s 前进 25s。
set +u
D=/mnt/c/Users/86156/Desktop/机器狗项目/s10_dev
WS=/mnt/c/Users/86156/Desktop/机器狗项目/s10_ws
R=$D/results/hmaptest; mkdir -p "$R"
source /opt/ros/jazzy/setup.bash; source "$HOME/s10_install/setup.bash"
export ROS_DOMAIN_ID=1 S10_USE_VIEWER=0 S10_REALTIME=1 S10_CMD_SOURCE=ros
export S10_MUJOCO_XML=$WS/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/${SCENE:-bench_park}.xml
export S10_LIDAR=1 S10_LIDAR_WORKERS=4 S10_HMAP=1 S10_HMAP_EVERY=20 S10_CMD_MAX_VX=2.0
unset S10_POLICY_PATH S10_HMAP_FWD_CLIP S10_HMAP_LAT_CLIP
pkill -f mujoco_simulation_ros2 2>/dev/null; pkill -x rl_deploy 2>/dev/null; pkill -f hmap_node.py 2>/dev/null; sleep 1
python3 -u "$WS/src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py" > "$R/sim.log" 2>&1 &
SIM=$!; sleep 8; kill -0 $SIM 2>/dev/null || { echo "!! 仿真起不来"; tail -5 "$R/sim.log"; exit 1; }
ros2 run s10_sdk_deploy rl_deploy > "$R/deploy.log" 2>&1 &
DEP=$!; sleep 6
python3 "$D/nav/bringup_sim.py" >/dev/null || { kill $DEP $SIM; exit 1; }
# 外部高程图节点：仿真雷达 body front_lidar_link 在 base (0.2,0,0.08)，俯仰 +0.4363 rad
python3 "$D/nav/hmap_node.py" --cloud-topic /front_lidar/points --pose-source sim --out-topic /height_scan_ext \
  --lidar-xyz 0.2 0 0.08 --lidar-rpy 0 0.436332 0 ${HMAP_ARGS:-} > "$R/hmap.log" 2>&1 &
HM=$!; sleep 3
echo "--- 前进 0.5 m/s 25s，同时对拍 ---"
timeout 26 ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}" > /dev/null 2>&1 &
python3 "$D/nav/hmapcmp.py" --secs 25 2>&1 | tail -3
echo "--- hmap_node 日志尾 ---"; tail -3 "$R/hmap.log"
echo "--- 仿真内部高程图状态 ---"; grep -aE "hmap|高程" "$R/sim.log" | tail -2
kill $HM $DEP $SIM 2>/dev/null; sleep 1; pkill -f mujoco_simulation_ros2 2>/dev/null; pkill -x rl_deploy 2>/dev/null; pkill -f hmap_node.py 2>/dev/null
echo HMAPTEST_DONE
