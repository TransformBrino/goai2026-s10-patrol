#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/robot/lidar_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/robot/.ros/fastdds_ethernet.xml
exec ros2 run dual_airy_merger dual_airy_merger_node
