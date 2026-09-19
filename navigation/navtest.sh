#!/usr/bin/env bash
# 导航节点仿真联调：MuJoCo 平地 + 我方 runner（出厂盲策略，/cmd_vel）+ nav_node（位姿源 /sim/state）
# 用法: navtest.sh [loop|line] [side_or_len=8] [v=0.8]
# 注意：与训练并行时 RTF≈0.9，这里是逻辑联调不是性能评测，可接受；性能数据另测。
set +u
KIND=${1:-loop}; SZ=${2:-8}; V=${3:-0.8}
D=/mnt/c/Users/86156/Desktop/机器狗项目/s10_dev
WS=/mnt/c/Users/86156/Desktop/机器狗项目/s10_ws
R=$D/results/navtest; mkdir -p "$R"
source /opt/ros/jazzy/setup.bash; source "$HOME/s10_install/setup.bash"
export ROS_DOMAIN_ID=1 S10_USE_VIEWER=0 S10_REALTIME=1 S10_CMD_SOURCE=ros
export S10_MUJOCO_XML=$WS/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/bench_flat.xml
export S10_LIDAR=0 S10_HMAP=0 S10_CMD_MAX_VX=2.0 S10_CMD_MAX_WZ=1.5
unset S10_POLICY_PATH S10_HMAP_FWD_CLIP S10_HMAP_LAT_CLIP
pkill -f mujoco_simulation_ros2 2>/dev/null; pkill -x rl_deploy 2>/dev/null; pkill -f nav_node.py 2>/dev/null; sleep 1
WP=$R/wp_${KIND}.txt
if [ "$KIND" = loop ]; then python3 "$D/nav/make_waypoints.py" loop "$WP" --side "$SZ" --v "$V"; else python3 "$D/nav/make_waypoints.py" line "$WP" --len "$SZ" --v "$V"; fi
python3 -u "$WS/src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py" > "$R/sim.log" 2>&1 &
SIM=$!; sleep 6; kill -0 $SIM 2>/dev/null || { echo "!! 仿真起不来"; tail -5 "$R/sim.log"; exit 1; }
ros2 run s10_sdk_deploy rl_deploy > "$R/deploy.log" 2>&1 &
DEP=$!; sleep 6; kill -0 $DEP 2>/dev/null || { echo "!! runner 退出"; tail -10 "$R/deploy.log"; kill $SIM; exit 1; }
python3 "$D/nav/bringup_sim.py" || { kill $DEP $SIM; exit 1; }
echo "--- nav_node 出发（$KIND $SZ m, v=$V）---"
timeout 240 python3 "$D/nav/nav_node.py" --waypoints "$WP" --pose-source sim --vmax "$V" --relative \
  --out "$R/${KIND}.json" --log "$R/${KIND}.csv" --timeout 200 2>&1 | grep -vE "^\[INFO\] \[[0-9.]+\] \[s10_nav\]: $" | sed -E 's/^\[INFO\] \[[0-9.]+\] \[s10_nav\]: //' | tail -30
kill $DEP $SIM 2>/dev/null; sleep 1; pkill -f mujoco_simulation_ros2 2>/dev/null; pkill -x rl_deploy 2>/dev/null
echo "--- 摘要 ---"; python3 -c "import json;d=json.load(open('$R/${KIND}.json'));print({k:d[k] for k in ('result','complete','reached','waypoints','duration_s','retries','handover')})" 2>/dev/null || echo "(无摘要)"
grep -aoE "\[RTF\] t=[0-9]+s.*" "$R/sim.log" | tail -1
