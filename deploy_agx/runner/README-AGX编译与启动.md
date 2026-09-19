# 244 维 runner（我方 fork 的 S10 SDK）— AGX 编译与启动

> **09-11h 更新**：只改两处，其余和 0911g 一样。一是 RL 运行期间加关节反馈停更检查，默认只告警（阈值 `S10_RL_FB_STALE_MS` 默认 100 ms；设 `S10_RL_STALL_DAMP=1` 才锁阻尼）；二是膝关节限位改成 S10 实际的 ±2.7227。改了 2 个文件，详见 `更新说明-09-11h.md`，补丁见 `runner_0911g_to_0911h.patch`。
>
> **09-11g 更新**：合入 Codex 真机排查出的起立和趴下修复，一共三处：起立时轮子保持当前角度；起立中途卡顿就锁存阻尼；阻尼和趴下的时间改用 double。另外 RL 状态加了停顿监视，默认只告警。一共改了 4 个文件，详见 `更新说明-09-11g.md`。
>
> **09-11f 更新**：只改了 `rl_control_state.hpp` 一个文件。收到 /climb_mode 2 而没加载楼梯策略（S10_POLICY_PIT）时，留在主策略，不再落到 ClimbH，并打印 `!! [CLIMB] …留在主策略（0911f）`。详见 `更新说明-09-11f.md`。
>
> **09-11e 更新**：在 0911d 基础上加了三处。一是电机失控保护：命令力矩很大、实测力矩不到一成，连续 0.3 s 就转阻尼并锁存，要重启 runner 才能恢复。二是 IMU 闸门默认阈值改为 200 ms。三是每 60 s 打印一次控制周期。另外修了 S10Interface 只发指令不存的缺陷。改动 4 个文件，md5 和验证结果见 `更新说明-09-11e.md`，补丁见 `runner_0911d_to_0911e.patch`。

> **09-11d 更新（必看）**：真机上 runner 从来没收到过 IMU（机器人 /IMU_DATA 是 BEST_EFFORT，旧订阅是 RELIABLE）。0911d 已改订阅并加 IMU 闸门（收不到新鲜 IMU 不起立、不进 RL）。要换哪些文件、md5、第一次起 runner 应看到的日志，见 `更新说明-09-11d.md`；补丁见 `runner_0911c_to_0911d.patch`。

> **09-11b 注意：本包不再带 `s10_dev/nav/hmap_node.py`。** 09-11 那份是旧版（md5 `a3df5a0d`，没有滚动窗口、入图三道防线、`--map-ttl`、新鲜度闸门），装上 AGX 会退回假墙和棘轮。**高程图一律用 AGX 上的现行版**（AGX 侧 09-11 备份）。下文凡提到 hmap_node.py 的地方，都指 AGX 现行版。

这就是策略包 README 里说的"runner 本体"：`s10_ws/src/S10_sdk_deploy`（C++，ROS2 Jazzy 包 `s10_sdk_deploy`，可执行文件 `rl_deploy`）。
它比厂家原版多了：观测维度从 onnx 自动读（57 或 244）、`/height_scan` 187 维高程图接入、`/cmd_vel` `/robot_mode` `/climb_mode` ROS 指令源、双策略热切、力矩护栏、默认位形/增益全部可用环境变量配。
**厂家原版 SDK 只有 57 维路径，跑不了我们的四个策略；策略也不能"降级"成 57 维（高程图是它上墙/避障的眼睛）。**

只发了源码，没有 AGX 二进制：这台训练机是 x86，AGX 是 aarch64，必须在 AGX 上编一次（几分钟）。
`third_party/onnxruntime/arm/` 已带 aarch64 的 libonnxruntime，不用另装。

## 0. 包内容
```
s10_ws/src/drdds/            机器人 DDS 消息包（JointsData/ImuData/Steer…）
s10_ws/src/S10_sdk_deploy/   runner 源码（去掉了 x86 onnxruntime 和场景网格）
s10_dev/nav/hmap_node.py     雷达点云 → /height_scan 187 维（244 维策略必需）
s10_dev/nav/nav_node.py      导航层（本轮真机测试不用）
AGX部署步骤.md / README-现场90分钟.md   之前写的现场清单，作参考
```
`s10_dev/` 与 `s10_ws/` **必须同层**：hmap_node 按相对路径 import `s10_ws/src/S10_sdk_deploy/interface/robot/simulation/elevation_map.py`。

## 1. 编译（AGX，系统 python，不要在 conda 里）
```bash
sudo apt install -y ros-jazzy-desktop python3-colcon-common-extensions python3-empy build-essential   # 已装可跳过
which aarch64-linux-gnu-g++ || sudo ln -s /usr/bin/g++ /usr/local/bin/aarch64-linux-gnu-g++           # CMake 里 arm 平台写死这个名字
cd ~/s10/s10_ws && source /opt/ros/jazzy/setup.bash
colcon build --packages-select drdds && source install/setup.bash
colcon build --packages-select s10_sdk_deploy --cmake-args -DBUILD_PLATFORM=arm -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ldd install/s10_sdk_deploy/lib/s10_sdk_deploy/rl_deploy | grep -E "onnxruntime|not found"    # 必须指向 third_party/onnxruntime/arm，且没有 not found
```
坑：PATH 里若有 miniconda，colcon 会用错 python 报 `No module named em`，先 `export PATH=$(echo $PATH | tr ':' '\n' | grep -v conda | paste -sd:)`。

## 2. 启动（策略契约见策略包 README-部署契约.md §2，环境变量原样照抄）
前提：机器人已用授权码切到 SDK 模式、**趴卧**；`ros2 topic hz /JOINTS_DATA` ≈ 50 Hz。
```bash
source /opt/ros/jazzy/setup.bash && source ~/s10/s10_ws/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds_ethernet.xml    # 与雷达/机器人同一 DDS profile
# 终端 A：高程图（先起，等 /height_scan 有 187 个数）
S10_HMAP_FWD_CLIP=0.6 python3 ~/s10/s10_dev/nav/hmap_node.py --cloud-topic /LIDAR/POINTS_MERGED --pose-source odom --pose-topic /localization/pose
ros2 topic echo --once /height_scan | head -3
# 终端 B：runner
export S10_CMD_SOURCE=ros S10_DEF_HIPX=0.05 S10_DEF_HIPY=0.35 S10_DEF_KNEE=0.65 S10_KP_LEG=80 S10_KD_LEG=2 S10_KD_WHEEL=0.8
export S10_TAU_GUARD=1 S10_TAU_LIM_LEG=50 S10_TAU_LIM_WHEEL=14 S10_CMD_MAX_VX=0.3 S10_CMD_MAX_VY=0.2 S10_CMD_MAX_WZ=0.5 S10_CMD_SLEW_V=1.0 S10_CMD_TIMEOUT_MS=500
export S10_POLICY_PATH=~/s10/policies/V1G_7997/v1g_7997.onnx S10_POLICY_CLIMB=~/s10/policies/ClimbG_7598/climbG_7598.onnx
ros2 run s10_sdk_deploy rl_deploy        # 起动时会先发一帧零指令再等 16 个关节都动过：机器人静止不动就手推各腿
```
启动横幅应打印：`观测 244`、`action_scale hipx/hipy/knee/wheel = 0.125 0.25 0.25 5`、`kd_wheel=0.8`。观测不是 244 就停。
**0911c 起还必须看到 IMU 两行**：`[IMU] /IMU_DATA 姿态角按【弧度】读`（没有这行说明装的是旧 runner）；狗趴着起 runner 时还应出现 `[IMU] 开机自检通过：…单位与符号正确`。站平或架空时先打印"暂不能下结论"，机身倾斜 ≥8° 静止后再判。**看到"开机自检失败"就停，不要进 RL。**

## 3. 状态切换（`/robot_mode` std_msgs/UInt8，沿用 RobotMotionState 编码）
```
0 WaitingForStand（趴卧待命）  1 StandingUp（起立）  6 RLControlMode（策略接管）  2 JointDamping（阻尼＝软急停）  4 LieDown（趴下）
```
```bash
ros2 topic pub --once /robot_mode std_msgs/msg/UInt8 "{data: 1}"     # 起立，等站稳 3 s
ros2 topic pub --once /robot_mode std_msgs/msg/UInt8 "{data: 6}"     # 策略接管
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"   # 走；停发 0.5 s 后看门狗归零
ros2 topic pub --once /robot_mode std_msgs/msg/UInt8 "{data: 2}"     # 软急停（阻尼），硬急停按 HES
ros2 topic pub --once /climb_mode std_msgs/msg/UInt8 "{data: 1}"     # 墙前切专家策略；过墙后发 0
```
只允许 0→1→6、6→2、1→4 这些合法转移，其它会被 runner 拒绝并打印。

## 4. 未在 AGX 上验证过的点（编译成功后请回报）
- 官方 09-04 的 S10 控制参数（起立/趴下 IK 用 0.18/0.18 腿长）尚未移植进 fork，fork 起立/趴下仍按 M20 数做样条；MuJoCo 里能站，真机第一次起立看姿态。
- hipx 方向：模型坐标外八 (fl+, fr−, hl−, hr+)，真机第一次站立看四条腿是否都向外，反了就 `S10_DEF_HIPX=-0.05`。
- `/height_scan` 平地基准：站姿 0.42 时应 ≈ −0.07；雷达外参 z/pitch 不对先调 `--lidar-xyz/--lidar-rpy`。
