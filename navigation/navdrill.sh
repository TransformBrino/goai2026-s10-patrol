#!/usr/bin/env bash
# 导航节点两项"处理麻烦"演练（仿真）：
#   A 定位退化：平地 20m 直线；第 3s 起标志位=1 持续 5s（应半速），随后 =2 持续 13s（应停车，10s 后记"交给人"），再 =0（应恢复）
#   B 卡死重试：bench_block40（x=3 处 40cm 墙）；直线 8m 穿墙 → 顶墙 → 4s 无进展 → 倒退重试 ×3 → 交给人
set +u
D=/mnt/c/Users/86156/Desktop/机器狗项目/s10_dev
WS=/mnt/c/Users/86156/Desktop/机器狗项目/s10_ws
R=$D/results/navdrill; mkdir -p "$R"
source /opt/ros/jazzy/setup.bash; source "$HOME/s10_install/setup.bash"
export ROS_DOMAIN_ID=1 S10_USE_VIEWER=0 S10_REALTIME=1 S10_CMD_SOURCE=ros
export S10_LIDAR=0 S10_HMAP=0 S10_CMD_MAX_VX=2.0 S10_CMD_MAX_WZ=1.5
unset S10_POLICY_PATH S10_HMAP_FWD_CLIP S10_HMAP_LAT_CLIP
MJ=$WS/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf
launch() {  # scene
  pkill -f mujoco_simulation_ros2 2>/dev/null; pkill -x rl_deploy 2>/dev/null; pkill -f nav_node.py 2>/dev/null; sleep 1
  export S10_MUJOCO_XML=$MJ/$1.xml
  python3 -u "$WS/src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py" > "$R/sim_$1.log" 2>&1 &
  SIM=$!; sleep 6
  ros2 run s10_sdk_deploy rl_deploy > "$R/deploy_$1.log" 2>&1 &
  DEP=$!; sleep 6
  python3 "$D/nav/bringup_sim.py" >/dev/null || { echo "!! 起立失败"; kill $DEP $SIM; return 1; }
}
teardown() { kill $DEP $SIM 2>/dev/null; sleep 1; pkill -f mujoco_simulation_ros2 2>/dev/null; pkill -x rl_deploy 2>/dev/null; }
fmt() { grep -vE "^\[INFO\] \[[0-9.]+\] \[s10_nav\]: $" | sed -E 's/^\[INFO\] \[[0-9.]+\] \[s10_nav\]: //'; }

echo "===== 演练 A：定位退化 ====="
launch bench_flat || exit 1
python3 "$D/nav/make_waypoints.py" line "$R/wp_line.txt" --len 20 --v 0.8 >/dev/null
( sleep 3; timeout 5 ros2 topic pub -r 10 /localization/status std_msgs/msg/UInt8 "{data: 1}" >/dev/null 2>&1
  timeout 13 ros2 topic pub -r 10 /localization/status std_msgs/msg/UInt8 "{data: 2}" >/dev/null 2>&1
  timeout 30 ros2 topic pub -r 10 /localization/status std_msgs/msg/UInt8 "{data: 0}" >/dev/null 2>&1 ) &
timeout 120 python3 "$D/nav/nav_node.py" --waypoints "$R/wp_line.txt" --pose-source sim --vmax 0.8 --relative \
  --out "$R/A.json" --log "$R/A.csv" --timeout 90 2>&1 | fmt | tail -12
python3 - "$R/A.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows: by.setdefault((r["status"], r["mode"]), []).append(float(r["cmd_vx"]))
for k, v in sorted(by.items()): print("  标志位=%s 模式=%-5s 帧数 %4d  指令 vx 均值 %.2f" % (k[0], k[1], len(v), sum(v)/len(v)))
PY
teardown

echo; echo "===== 演练 B：卡死重试（40cm 墙）====="
launch bench_block40 || exit 1
python3 "$D/nav/make_waypoints.py" line "$R/wp_wall.txt" --len 8 --v 0.6 >/dev/null
timeout 120 python3 "$D/nav/nav_node.py" --waypoints "$R/wp_wall.txt" --pose-source sim --vmax 0.6 --relative \
  --out "$R/B.json" --log "$R/B.csv" --timeout 90 2>&1 | fmt | tail -12
teardown
echo NAVDRILL_DONE
