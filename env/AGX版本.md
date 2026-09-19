# AGX 现场环境（2026-09-19 14:10 只读复核）
- 主机：NVIDIA AGX（机器人加装），用户 robot；Ubuntu 24.04.4 LTS；ROS 2 Jazzy
- runner：官方 S10_sdk_deploy 二次开发 0911i，二进制 `rl_deploy` md5 53d3e0ef39fa8ba98c9b132614f7253f；内置 onnxruntime（third_party/onnxruntime/arm）1.22.0
- 雷达：rslidar_sdk 1.5.19（编译选项 XYZIRT + ENABLE_TRANSFORM，见 `deploy_agx/config/rslidar_sdk-CMakeLists.txt`）；dual_airy_merger 0.1.0（官方），配置 `deploy_agx/config/airy_dual.yaml`
- 定位：FAST-LIO2 ROS 2 分支（Ericsii fork，上游 https://github.com/hku-mars/FAST_LIO.git，本地源码 commit a4743b095409；AGX 编译副本无 .git，源码快照见 perception/FAST_LIO_src），配置 `deploy_agx/lio/config/s10_airy.yaml`（lio_up.sh 默认），备选 s10_airy_light / s10_airy_diag；IMU 转发 `deploy_agx/lio/s10_imu_relay.py`；kiss-icp 1ffa7d7 已停用
- 高程图：`perception/hmap_node.py`（Python 3，rclpy），f2f161ba
- 控制台：`deploy_agx/s10_ctrl.py` night5g 4e08db6c（Python 3 标准库 + rclpy），端口 8089
- 开机自启：crontab `@reboot sleep 25; /home/robot/boot_up.sh`（09-19 起注释，改用官方运控；恢复方法见 deploy_agx/crontab.txt）
- 磁盘：54 GB，已用 45 GB（88%）；录包目录 /home/robot/s10_data
