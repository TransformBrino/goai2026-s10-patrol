#!/bin/bash
# 只重启 hmap（重建累积地图），不动定位（FAST-LIO；保住里程计世界原点）。
# 什么时候用：狗从趴变站、或者挪到新场地之后 —— 旧地图里的点会把平地基准带偏。
# 用完可以跑 calib.py 复验一下 -0.08±0.01。
pkill -9 -f hmap_node.py 2>/dev/null
sleep 2
setsid nohup bash /home/robot/lio_ws/hmap_lio_up.sh > /tmp/hmap_lio.log 2>&1 < /dev/null &   # 09-12：位姿源 FAST-LIO（切回 KISS 用 hmap_kiss_up.sh）
sleep 3
pgrep -af hmap_node.py | grep -v 'bash -c'
echo "hmap 已重启，等 20 秒地图累积后再看 /height_scan"
