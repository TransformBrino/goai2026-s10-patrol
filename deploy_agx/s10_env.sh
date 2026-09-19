# S10 recording environment -- self-contained, does NOT depend on the other team's workspace
source /opt/ros/jazzy/setup.bash
export AMENT_PREFIX_PATH=/home/robot/s10_deps/drdds:$AMENT_PREFIX_PATH
export PYTHONPATH=/home/robot/s10_deps/drdds/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/home/robot/s10_deps/drdds/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/home/robot/s10_deps/drdds:$CMAKE_PREFIX_PATH
export ROS_DOMAIN_ID=0
