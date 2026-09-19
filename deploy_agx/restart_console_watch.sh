#!/bin/bash
# 09-12：重启控制台和看门狗（换成 LIO 版之后用）。pkill 的模式写在脚本里，调用方命令行不要出现这些文件名。
# 只在无 runner、未接管时做。
pgrep -x rl_deploy >/dev/null && { echo "[restart] 有 runner，不重启控制台"; exit 2; }
A=$(curl -s -m 3 http://127.0.0.1:8089/api/state | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['armed'])" 2>/dev/null)
[ "$A" = False ] || { echo "[restart] 控制台已接管或拿不到状态（$A），不重启"; exit 2; }
source /home/robot/s10_env.sh
export S10_HQ_UNDER_MIN=${S10_HQ_UNDER_MIN:-0.20}   # 09-12 作者拍板 0.20（与 boot_up.sh 一致）
export S10_STAIR_HOLD=0   # 09-12 21:4x 作者：楼梯段仍按住 W 才留在专家（night4o 的不退语义关掉）
export S10_STAIR_POLICY=StairN-E2_10300   # 09-13 作者：楼梯槽默认候选 A
what=${1:-both}
if [ "$what" = both ] || [ "$what" = console ]; then
  pkill -TERM -f "python3 /home/robot/s10_ctrl.py"; sleep 3; pkill -9 -f "python3 /home/robot/s10_ctrl.py"; sleep 1
  cd /home/robot && setsid nohup python3 /home/robot/s10_ctrl.py >> /home/robot/s10_ctrl.log 2>&1 < /dev/null &
  sleep 4; curl -s -m 3 -o /dev/null -w "[restart] 控制台 HTTP %{http_code}\n" http://127.0.0.1:8089/api/state
fi
if [ "$what" = both ] || [ "$what" = watch ]; then
  pkill -TERM -f "python3 /home/robot/lidar_watch.py"; sleep 2; pkill -9 -f "python3 /home/robot/lidar_watch.py"
  setsid nohup bash /home/robot/lidar_watch.sh >> /home/robot/lidar_watch.log 2>&1 < /dev/null &
  sleep 3; pgrep -af "lidar_watch.py" | grep -v "bash -c" | cut -c1-80
fi
echo "[restart] $(date +%T) 完成（$what）"
