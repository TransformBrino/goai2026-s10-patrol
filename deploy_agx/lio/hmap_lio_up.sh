#!/bin/bash
# S10（09-12）：hmap 用 FAST-LIO 的 /Odometry 当位姿源（机体/IMU 系位姿；FAST-LIO 外参 T=[0,0,-0.0306] 与下面 --lidar-xyz 一致）。
# 其余参数与 hmap_kiss_up.sh 完全一样（--map-ttl 1.5、nice 10）。KISS 照旧在跑，只是 hmap 不再吃它。
source /opt/ros/jazzy/setup.bash
source /home/robot/s10/s10_ws/install/setup.bash
export ROS_DOMAIN_ID=0
cd /home/robot/s10
exec nice -n 10 python3 /home/robot/s10/s10_dev/nav/hmap_node.py \
  --cloud-topic /LIDAR/POINTS_MERGED \
  --pose-source odom --pose-topic /Odometry \
  --lidar-xyz 0 0 -0.0306 --lidar-rpy 0 0 0 \
  --map-ttl 1.5
