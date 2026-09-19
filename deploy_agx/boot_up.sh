#!/bin/bash
# 开机把我们这套东西全拉起来。每一项都先看在不在，避免重复起。
set +e
exec >> /home/robot/boot_up.log 2>&1
echo "===== $(date '+%F %T') boot_up 开始 ====="
source /home/robot/s10_env.sh
python3 -c 'import rclpy' 2>/dev/null || { echo "rclpy 还没就绪，再等 20s"; sleep 20; source /home/robot/s10_env.sh; }
python3 -c 'import rclpy' 2>/dev/null && echo "rclpy OK" || echo "!! rclpy 仍不可用"

up () { pgrep -f "$1" >/dev/null; }

# 2026-09-11 Codex: legacy :8088 disabled; use the single :8089 control console.
sleep 2
export S10_HQ_UNDER_MIN=0.20   # 09-12 17:0x 作者拍板：室外站姿机身下补洞前有效率只有 25~29%，30% 门槛会拦楼梯专家
export S10_STAIR_HOLD=0   # 09-12 21:4x 作者：楼梯段仍按住 W 才留在专家（night4o 的不退语义关掉）
export S10_STAIR_POLICY=StairN-E2_10300   # 09-13 作者：楼梯槽默认候选 A
up "python3 /home/robot/s10_ctrl.py" || { nohup python3 /home/robot/s10_ctrl.py >> /home/robot/s10_ctrl.log 2>&1 & echo "起 s10_ctrl :8089"; }
sleep 3

# 雷达链。机器人没上电时雷达收不到包，起了也是空转，无害。
up "rslidar_sdk_node"        || { nohup bash /home/robot/lidar_up.sh  >> /tmp/lidar_up.log  2>&1 & echo "起 rslidar"; }
sleep 6
up "dual_airy_merger_node"   || { nohup bash /home/robot/merger_up.sh >> /tmp/merger_up.log 2>&1 & echo "起 merger"; }
sleep 4
# 09-11 17:15 换电池：位姿源比雷达数据先起就一直不出位姿。所以等合并点云真有数据再起定位和 hmap。最多等 600 s。
# 定位已经在跑（手动重跑本脚本）就不等。
if ! pgrep -x fastlio_mapping >/dev/null; then
  python3 /home/robot/wait_cloud.py 600 || echo "!! 等 600 s 仍没有合并点云，照旧起 FAST-LIO/hmap"
fi
# 位姿源（09-12 作者拍板）：FAST-LIO（雷达+机器人 IMU），替代纯几何的 KISS-ICP（4 m 宽台阶棱横向无约束，楼梯口跳 0.3~1 m）。
# 厂家的 /LIO_ODOM 永远不会来（106 的 rsdriver 被 drsec 的 TEE 挡死）。KISS 不再开机起；要切回：kiss_up.sh + hmap_kiss_up.sh。
up "fastlio_mapping"         || { nohup bash /home/robot/lio_ws/lio_up.sh >> /tmp/lio_up.log 2>&1 & echo "起 FAST-LIO（含 IMU 转发）"; }
sleep 6
up "hmap_node.py"            || { nohup bash /home/robot/lio_ws/hmap_lio_up.sh >> /tmp/hmap_lio.log 2>&1 & echo "起 hmap(吃 FAST-LIO 位姿)"; }
up "lidar_watch.py"          || { nohup bash /home/robot/lidar_watch.sh >> /home/robot/lidar_watch.log 2>&1 & echo "起 merger 看门狗"; }
echo "===== boot_up 结束 ====="
