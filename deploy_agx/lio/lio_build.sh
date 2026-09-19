#!/bin/bash
# S10（09-12）：在 AGX 上编 FAST-LIO（ROS2 分支 + S10 补丁）。低优先级，3 个并行，Release。
# 源码在 /home/robot/lio_ws/src/FAST_LIO（从训练机 scp 来，已打 patch_fastlio_s10.py）。日志 /tmp/lio_build.log。
source /opt/ros/jazzy/setup.bash
cd /home/robot/lio_ws
echo "[lio_build] $(date +%T) 开始  $(nproc) 核" | tee -a /tmp/lio_build.log
nice -n 19 colcon build --packages-select fast_lio --parallel-workers 2 --event-handlers console_direct+ \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2" >> /tmp/lio_build.log 2>&1
RC=$?
echo "[lio_build] $(date +%T) 结束 rc=$RC" | tee -a /tmp/lio_build.log
ls -la /home/robot/lio_ws/install/fast_lio/lib/fast_lio/ 2>/dev/null | tee -a /tmp/lio_build.log
exit $RC
