#!/bin/bash
# S10（09-12）：按 PID 文件关 FAST-LIO 和 IMU 转发（不用 pgrep -f）。
R=/tmp/s10_lio
for f in lio imu; do
  [ -f $R/$f.pid ] || continue; p=$(cat $R/$f.pid)
  kill -TERM $p 2>/dev/null; for i in 1 2 3 4 5; do kill -0 $p 2>/dev/null || break; sleep 1; done; kill -9 $p 2>/dev/null; rm -f $R/$f.pid
done
echo "[lio_down] $(date +%T) 已关" | tee -a $R/events.log
