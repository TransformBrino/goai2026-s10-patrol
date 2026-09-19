#!/bin/bash
# S10（09-12）：把 hmap 的位姿源切到 FAST-LIO。只在狗空闲时做：有 runner / 已接管就不做。
# 步骤：起 FAST-LIO（若没在跑）→ 等 /Odometry 出数 → 重启 hmap（hmap_lio_up.sh）→ 等 20 s 地图累积。
# 切回 KISS：bash /home/robot/hmap_rebase.sh（它起的是 hmap_kiss_up.sh）；关 FAST-LIO：bash /home/robot/lio_ws/lio_down.sh
pgrep -x rl_deploy >/dev/null && { echo "[switch] 有 runner，不切"; exit 3; }
A=$(curl -s -m 3 http://127.0.0.1:8089/api/state | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['armed'])" 2>/dev/null)
[ "$A" = False ] || { echo "[switch] 控制台已接管或状态拿不到（$A），不切"; exit 3; }
source /opt/ros/jazzy/setup.bash; source /home/robot/s10_env.sh; source /home/robot/lio_ws/install/setup.bash
R=/tmp/s10_lio
if ! { [ -f $R/lio.pid ] && kill -0 $(cat $R/lio.pid) 2>/dev/null; }; then
  bash /home/robot/lio_ws/lio_up.sh ${1:-/home/robot/lio_ws/src/FAST_LIO/config/s10_airy_light.yaml} || exit 4
fi
echo "[switch] 等 /Odometry 出数…"
python3 - <<'EOF' || { echo "[switch] 12 s 没等到 /Odometry，不重启 hmap"; exit 5; }
import time, rclpy, sys
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
N=[]; rclpy.init(); n=Node("switch_wait_ro")
for rel in (ReliabilityPolicy.BEST_EFFORT, ReliabilityPolicy.RELIABLE):
    n.create_subscription(Odometry, "/Odometry", lambda m: N.append(1), QoSProfile(depth=10, reliability=rel, history=HistoryPolicy.KEEP_LAST))
t0=time.time()
while time.time()-t0<12 and len(N)<10: rclpy.spin_once(n, timeout_sec=0.05)
n.destroy_node(); rclpy.shutdown(); sys.exit(0 if len(N)>=10 else 1)
EOF
echo "[switch] 重启 hmap → FAST-LIO 位姿"
pkill -9 -f hmap_node.py 2>/dev/null
sleep 2
setsid nohup bash /home/robot/lio_ws/hmap_lio_up.sh > /tmp/hmap_lio.log 2>&1 < /dev/null &
sleep 3
pgrep -af hmap_node.py | grep -v 'bash -c' | cut -c1-120
echo "[switch] $(date +%T) hmap 已切到 /Odometry，等 20 s 地图累积后再看 /height_scan（切回：bash /home/robot/hmap_rebase.sh）" | tee -a $R/events.log
