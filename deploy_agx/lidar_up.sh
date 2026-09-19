#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/robot/lidar_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/robot/.ros/fastdds_ethernet.xml
cd /home/robot/lidar_ws
exec ros2 run rslidar_sdk rslidar_sdk_node --ros-args \
  -p config_path:=/home/robot/lidar_ws/src/dual_airy_merger/config/airy_dual.yaml
