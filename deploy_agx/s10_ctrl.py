#!/usr/bin/env python3
"""S10 起立与趴下验收控制台 :8089
night3（09-12 凌晨，候选，未装、未上真机）：① 切专家前重查高程图质量（缺失/过期/非法 = 质量未知，不切）；
night3b（09-12 中午）：只改口径——/height_scan_raw 已含 ElevationMap 的 10 cm 邻域补值，不是纯直接观测，相关字段和文字改成"补洞前有效率（含邻域补值）"。
night5g（09-18）：装 DELIVER_20260918 三槽（T10_15197 主策略 / StairS16_15200 楼梯槽 / ClimbR13 爬墙槽），均 244 维不改 runner；新增爬墙槽可选（S10_CLIMB_POLICY、/api/climbpol、页面下拉），EXPERT 不再写死。三者都未真机验收。
night5e（09-14 17:4x）：发布回路自检——/cmd_vel 两次发布间隔 >200 ms 记日志（限频 5 s）并进 LIVE["pub_gap"]（最近 60 s 最大间隔 ms）。
night5d（09-14 17:1x）：录制状态改后台线程刷新，发布回路不再被 s10_rec.sh status（fork+du）卡住（根治 runner 500 ms 指令超时撤专家）。
night5c（09-14 15:4x）：主策略表加 V1HTrotT3_13400（踏步，md5 d408c43b，训练侧建议替代 H7）。
night5b（09-14 14:0x）：楼梯槽加候选 F StairN-S8_14800（能停，档位 0.2～0.5 全管用，默认 0.25，30/30）；表项 seg_vx = 选赛段默认档位。
night5a（09-14 11:2x）：楼梯槽加候选 E StairN-S3_13800（stop_ok：自己能停）；楼梯槽是能停的策略时楼梯赛段自动按全程专家（选中即切、W=档位、松开=0、点平地退），否则仍按 W 切/松开切回。
night4z（09-14 00:5x）：楼梯专家切入后 1 s 内航向纠偏限 ±0.1（STAIR_ENTRY_WZ_S / STAIR_ENTRY_WZ_MAX）。
night4y（09-13 22:1x）：简版页录制状态字段修正（rec.running）。\nnight4x（09-13 22:0x，作者）：楼梯/石笼赛段 = 没按 W 主策略站着、按住 W 自动切专家以档位速度走、松开切回停车（SEG_FULL 默认关）；切入起步值 = 档位。
night4w（09-13 21:4x）：全程专家模式切入不再发 0.5 起步冲（发当时 W 的值，没按就 0）；楼梯专家给 0 时航向保持不动作（只认 Q/E）。
night4v（09-13 21:2x）：楼梯模式默认 0.3（官方约 0.3 m/s），档位 0.1～0.5；石笼同样 W = 档位（0.3～0.4）、松开 = 0；闸门只要求档位 ≥ 0.1。
night4u（09-13 19:4x）：主策略表加 V1HTrotH4_11500 / H6_12099 / H7_11800（候选，未真机验收）；楼梯槽表加候选 D StairN-E5_11200；/api/stairpol 运行时选楼梯槽（下次起 runner 生效）；简版页两个下拉。
night4t（09-13 16:4x/17:0x）：楼梯槽表加候选 B、C。
night4s（09-13 16:3x，作者拍板 A）：楼梯/石笼赛段全程专家——选中即切专家（不用按 W），只有点平地/急停/X/释放/失联/runner 自撤才退；楼梯 W=0.5、松开=0（专家会再走一段）；石笼恒 0.4；急停/释放/失联/X 退专家时赛段回平地。S10_SEG_FULL=0 退回旧语义。
night4r（09-13 13:xx，作者：控制台要简洁）：简版页面做首页（一键接管 / 三种模式 / 只显示当前模式的速度 / 方向 / 录制 / 急停 / 前方对正提示 / 最近日志），完整页挪到 /full；后端接口不改。
night4q（09-13 10:3x）：楼梯槽策略可选，S10_STAIR_POLICY = StairN-C_9300（默认）| StairN-E2_10300（候选：9300 续训 + 官方四拍步行参考，仿真 26/26、侧倾峰 6.6～8°、两前轮同时腾空 7～9%，**未真机验收**）。只加表项与选择逻辑。
night4o（09-12 22:0x，作者拍板）：楼梯赛段里切入专家后松 W / 停刷 / 小指令 / S 都不退专家，一口气到顶；只有急停、倾角急停、X、runner 自己撤、操作员页面心跳断 >3 s 才退（S10_STAIR_HOLD=0 退回旧语义）。石笼赛段与 V 键路径不变。
night4n（09-12 21:3x）：自动对正默认只显示不闭环（S10_SQUARE_AUTO=1 才闭环）；闭环时用 2 s 中位、≥5 帧、离散 ≤10°、≥8 行、左右都检出；页面显示 2 s 中位偏角。依据：训练侧仿真里专家对 wz 基本不响应（目标 0～30° 实际只偏 5～9°），错误目标只会把后轮拧反；逐帧台沿估计会跳。
night4m（09-12 21:2x，作者拍板）：① 切楼梯专家时航向目标改为垂直台沿（台沿左右距离拟合，限 ±15°），爬楼中每 1.5 s 复校（每次 ≤5°；侧倾 >12°、手动转向、地图不可信时不校）；② 侧倾 >12° 冻结航向纠偏，楼梯专家转向上限 0.6→0.3；③ 页面前方栏：左右台沿差 >10 cm 算未正对，显示需左/右转几度。
night4l（09-12 19:1x）：修 night4k 启动 runner 时日志格式串 TypeError（runner 已起但 HTTP 线程崩、脚本收到连接断开）。
night4k（09-12 19:1x）：起 runner 时传 S10_AS_WHEEL_PIT（默认 3，配 0911i 只压楼梯槽）。
night4j（09-12 19:1x，作者拍板）：默认主策略改为 V1H_9196。
night4i（09-12 19:0x）：急停后 3 s 内每 0.2 s 重发 /robot_mode 2 + 零速直到 runner 确认进阻尼，3 s 没确认记警告。
night4h（09-12 18:5x）：策略表加回 V1H_9196（vx ±1.5、wz 1.0，备选主策略；作者：比赛不下台阶、'之前转向好的'是 V1H），默认仍 V1Down。
night4g（09-12 18:1x）：runner 的 S10_AS_WHEEL 可由控制台环境变量覆盖（默认仍 5），起 runner 的日志里显示。
night4f（09-12 18:0x）：前方地形判读（平地/楼梯/台面/障碍、第一级距离、左右是否正对）只显示不动作，用补洞前的图。
night4e（09-12 17:1x，作者现场要求）：选赛段自动调档位（石笼 0.4 / 楼梯 0.5，可调；回平地恢复）；速度控制搬到页面顶栏并显示当前赛段。
night4d（09-12 17:0x）：石笼赛段（ClimbH，按住 W 自动 /climb_mode 1，切入 0.4，其余同楼梯赛段）；页面写进每段的策略/速度/操作表（只提示不限速）。
night4c（09-12 16:5x，作者拍板）：位姿源改 FAST-LIO（订 S10_ODOM_TOPIC，默认 /Odometry），预检/页面文字随之改；HQ_UNDER_MIN 可由 S10_HQ_UNDER_MIN 覆盖。
night4a（09-12 下午）：孤儿 runner 只按可执行文件（/proc/<pid>/exe）判断，不再按命令行文本；其余不动。
night4b（09-12 15:4x，作者指示）：①「楼梯赛段」常驻——按住 W 自动切楼梯专家前进，松 W 交回主策略停车，S 倒退交主策略，赛段里主策略不前进，退出赛段手动选；不记楼梯朝向、不设 5° 闸门（每次切入以当时偏航为目标）。②策略列表只留 V1Down / StairN-C / ClimbH，其余从列表移除（不删资产）。
  ② 空洞率 /hmap_info 过期不再退回数补后的 0；③ 服务端认定唯一操作员页面，其余页面只读（/api/op/claim、/api/op/release）。
只在用户明确操作时发布话题；服务端有看门狗，页面关掉也会自动归零。
安全护栏按《README-部署契约》：首次 0.3 m/s、逐档放开；倾角自动急停 |roll|>60° 或
|pitch|>55°（09-11 作者拍板，所有策略统一，见 TILT_ABORT_*）。
v34（09-11）：09-11 新策略表 + 站姿组；ClimbH 专家只能由操作员手动切入/退出，
控制台唯一替人做的切换是"专家模式下指令 <0.3 先退专家再停车"（R13）与倾角急停（R10）。
"""
import http.server
import json
import math
import os
import re
import shutil
import signal
import sys
import socketserver
import subprocess
import threading
import time
import urllib.parse

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import Log
from std_msgs.msg import String, UInt8, Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from drdds.msg import JointsData, JointsDataCmd, ImuData, BatteryData, Steer
from rclpy.serialization import deserialize_message   # 09-11：/JOINTS_CMD 收原始字节，显示时才解一帧

PORT = 8089
POLDIR = "/home/robot/policies"
WS = "/home/robot/s10/s10_ws"          # 训练侧的 244 维 fork
EXE_PATH = WS + "/install/s10_sdk_deploy/lib/s10_sdk_deploy/rl_deploy"  # 09-11 晚：切回 s10_ws 正式版（0911g：起立防护 + RL 停顿告警 + 0911e 失控保护）
REC_SH = "/home/robot/s10_rec.sh"     # 只订阅、从不发布的录制器
REC_DIR = "/home/robot/s10_data"
NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,47}$")
HMAP = "/home/robot/s10/s10_dev/nav/hmap_node.py"

# /robot_mode 编码（README-AGX编译与启动 §3），只允许这些转移
MODES = {0: "WaitingForStand 趴卧待命", 1: "StandingUp 起立",
         6: "RLControlMode 策略接管", 2: "JointDamping 阻尼(软急停)", 4: "LieDown 趴下"}
DAMP_AUTO_IDLE_S = 3.0                # joint_damping_state.hpp:46 阻尼满 3 秒自动回 kIdle


def mode_allowed(cur, want):
    """逐条复刻 runner 的 RosCommandInterface::OnMode（ros_command_interface.hpp:166-190）。
    注意 runner 根本不处理 want==0 —— kIdle 是状态机自己走回去的，不是能发的指令。
    之前这里手写了一张表，含 (2,0)/(4,0) 这两个 runner 不接受的转移，
    又漏了阻尼之后起立，结果按完急停就卡死在阻尼里出不来。"""
    if want == 2:
        return True, ""                                   # 阻尼：任何状态都收
    if want == 1:
        if cur in (0, 4):
            return True, ""
        return False, ("起立只能从 0 待命 或 4 趴下 进入，我方跟踪到的当前状态是 %d。"
                       "（按过急停的话，阻尼满 3 秒会自动回 0，稍等一下再按）" % cur)
    if want == 6:
        if cur == 1:
            return True, ""
        return False, "策略接管只能从 1 起立 进入，我方跟踪到的当前状态是 %d" % cur
    if want == 4:
        if cur in (1, 6):
            return True, ""
        return False, "趴下只能从 1 起立 或 6 策略接管 进入，我方跟踪到的当前状态是 %d" % cur
    return False, "runner 的 /robot_mode 只接受 1/2/4/6；0 待命是状态机自己回去的，发不了"
NAMES = [f"{l}_{j}" for l in ("fl", "fr", "hl", "hr")
         for j in ("hipx", "hipy", "knee", "wheel")]
OFFS = [-35, -145, 156, 0, 35, -145, 156, 0, -35, 145, -156, 0, 35, 145, -156, 0]
DIRS = [1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1]
R_WHEEL = 0.0825                      # 手册：足部轮胎直径 0.165 m

# 09-11 训练侧新包（s10_pack_20260911，README §3.4 表 + 训练侧《使用说明》09-11 下午版）。
# 键 = /home/robot/policies 下的目录名（adopt_limits 靠目录名认回遗留 runner 的策略）。
# vx/vy/wz 是训练侧实测过的指令范围；runner 的 S10_CMD_MAX_* 也按它给（兜底护栏）。
# stance = 站姿组。runner 的默认位形 S10_DEF_* 必须与训练资产逐位一致：观测是 q−q_default、
#          动作目标是 action×scale+q_default，默认位形错了等于观测和动作工作点一起偏
#          （0.35 蹲姿组膝关节工作点差 48°，不对就摔）。S10_DEF_* 只在 S10PolicyRunner
#          构造函数里读一次（s10_policy_runner.hpp:192-194），两组之间换站姿必须重启 runner。
# expert = 上台面专家。不能当基础策略起 runner（零指令会原地转），只能在 runner 跑着 0.42 组
#          主策略时经 /climb_mode=1 热切（rl_control_state.hpp:88-103 按 reserved_scale 选策略）。
# old    = 09-10 旧版，真机 1.41 Hz 偏航自激（偏航率 std 31°/s），只留作对照。
_OLD = ("已禁用：09-10 旧版，没做延迟随机化。0911d 起 runner 收得到 IMU，仿真里 20 ms 延迟就剧烈摆头"
        "（09-10 真机 1.41 Hz 偏航自激）。只能用上面新的五个")
POLICIES = {
    "V1Down_10695": dict(
        onnx="V1Down_10695/v1down_10695.onnx", label="V1Down 通用线·下坎强化（默认）",
        desc="决赛主策略：平地、接近楼梯、平台上行走都用它。上楼梯不用它——选「楼梯赛段」后按住 W 由楼梯专家爬",
        # 09-11 实测限速：指令 2.0→实际 2.04、2.5→2.46，偏航稳
        vx=[-2.0, 2.5], vy=0.5, wz=2.0, zero_ok=True, stance="0.42", expert=False, old=False),
    "V1H_9196": dict(
        onnx="V1H_9196/v1h_9196.onnx", label="V1H 通用线·转向快/上墙楼梯更强（备选主策略）",
        desc="09-12 作者：比赛不下台阶，V1Down 的低速下坎优势不再重要。V1H：转向 0.6→0.68 rad/s（V1Down 0.35）；后轮占比比 V1Down 高一倍、左右更对称"
             "（1.8 时后轮 63% vs 38%）；单独上现场型楼梯 2/3、宽楼梯 2/2（V1Down 0/2）——楼梯专家的兜底。上限 1.5（1.8 偏航漂 3.7°/s、2.0 会倒退）；"
             "零指令会站住；上墙仍切 ClimbH、上楼梯仍切专家。下 25/31 cm 坎都要 1.8 才过。换不换主策略由作者定。",
        vx=[-1.5, 1.5], vy=0.5, wz=1.0, zero_ok=True, stance="0.42", expert=False, old=False),
    "T10_15197": dict(
        onnx="T10_15197/t10_15197.onnx", label="T10_15197（踏步主策略，09-18 交付，未真机验收）",
        desc="训练侧 DELIVER_20260918（md5 ebef4dd2）。MuJoCo：平地对角同步 65/71/80%、左右反相 -0.94/-0.91（官方 -0.53/-0.75）、10°坡偏航 +7.0°、15°坡 -1.2°、平地 roll p95 6.2°。"
             "已知差距：步频 3.23~3.77 Hz vs 官方 1.10（差 3 倍）。**装机必看：切专家的切换距离 ≤2.0 m，2.5 m 时登顶 0/3（走到 2.07 m 就停）**；1.5/2.0 m 时偏航漂 -13.9/-14.9°（旧 V1H 只有 +1.3/+7.4°）。"
             "指令范围训练侧未给，先按 T3 填（前进 ≤2.0、倒退 ≤0.5、转向 ≤0.6），待确认。",
        vx=[-0.5, 2.0], vy=0.5, wz=0.6, zero_ok=True, stance="0.42", expert=False, old=False),
    "V1HTrotT3_13400": dict(
        onnx="V1HTrotT3_13400/v1htrot_t3_13400.onnx", label="V1H-TrotT3_13400（踏步主策略，训练侧推荐替代 H7，未真机验收）",
        desc="09-14 训练侧（md5 d408c43b）：切 ClimbH 2/2、切 E2 专家 2/2、摩擦 0.3 不摔；0.5/1.0/1.8/2.0 → 0.35/0.92/1.77/2.01，倒退 -0.5 → -0.36 不打转，原地转 ±0.5 → +0.50/-0.45，给 0 站住漂 5 cm；四档干净对角小跑、侧倾 p95 5～6°、踏步站高 0.34。"
             "已知：步频 2.3 Hz（官方 1.1）；10°/15° 坡 1.0 指令会跑到 1.21 且偏航 -21/-45°（0.5 档正常）。控制台上限 前进 ≤2.0、倒退 ≤0.5、转向 ≤0.6。",
        vx=[-0.5, 2.0], vy=0.5, wz=0.6, zero_ok=True, stance="0.42", expert=False, old=False),
    "V1HTrotH7_11800": dict(
        onnx="V1HTrotH7_11800/v1htrot_h7_11800.onnx", label="V1H-TrotH7_11800（踏步主策略，训练侧推荐，未真机验收）",
        desc="09-13 19:40 训练侧新版（md5 231c8437）：治好 H4 的低速左漂和倒退打转——倒退 -0.5 → -0.35 不打转；0.5/1.0/1.8/2.3 → 0.39/0.83/1.73/2.43，直行偏航 ≤0.06 rad/s；0.5～1.8 四轮都踏；"
             "切 ClimbH 2/2、切 E2 专家 2/2、摩擦 0.3 不摔。观测/动作/站姿组同 V1H（站高自己降到 0.34～0.37）。训练范围 vx (-1.0, 2.3)、wz ±1.5；控制台上限 前进 ≤2.3、倒退 ≤0.5、转向 ≤1.0。",
        vx=[-1.0, 2.3], vy=0.5, wz=1.0, zero_ok=True, stance="0.42", expert=False, old=False),
    "V1HTrotH4_11500": dict(
        onnx="V1HTrotH4_11500/v1htrot_h4_11500.onnx", label="V1H-TrotH4_11500（踏步主策略候选，未真机验收）",
        desc="09-13 训练侧踏步线候选（md5 8fd5e3b2）：2.3 指令跑 2.39；切爬台 2/2、切专家 2/2；缺陷：倒退 -0.5 原地打转、1.0 m/s 左漂 +0.1 rad/s、0.5 m/s 右前轮悬着。"
             "训练范围 vx (-0.5, 2.3)、vy ±1.0、wz ±2.0；控制台上限 前进 ≤2.3、倒退 ≤0.5、转向 ≤1.0。",
        vx=[-0.5, 2.3], vy=0.5, wz=1.0, zero_ok=True, stance="0.42", expert=False, old=False),
    "V1HTrotH6_12099": dict(
        onnx="V1HTrotH6_12099/v1htrot_h6_12099.onnx", label="V1H-TrotH6_12099（踏步候选，训练侧不建议，未真机验收）",
        desc="09-13 训练侧（md5 7a17b48a）：**不建议上控制台**——0.5 m/s 直行左漂 +0.4 rad/s，整套验收 0.5 m/s 走不到切换点，切爬台 0/2、切专家 0/2。作者坚持要可选，故列出。训练范围 vx (-1.0, 2.3)、wz ±2.0。",
        vx=[-1.0, 2.3], vy=0.5, wz=1.0, zero_ok=True, stance="0.42", expert=False, old=False),
    "ClimbR13": dict(
        onnx="ClimbR13/climb_r13.onnx", label="爬墙专家 R13（09-18 交付，未真机验收）",
        desc="训练侧 DELIVER_20260918（md5 c72f75ac，run 2026-09-18_05-31-53 / model_18779），替换 09-15 冻结的 C35_14399。"
             "MuJoCo：33 cm 平台 n=12 上台面 12/12、站住 12/12，台上侧倾 中位 18.2°/最差 35.3°（静态翻倒角实算 30.1°，12 条里 1 条越线）；粗糙台面 n=6 6/6 上、6/6 站住、0/6 越翻倒线；配 T10 主策略 n=6 6/6。"
             "已知差距：后腿长变化 -0.043（官方 +0.041，是收腿不是蹬地）；撞墙速度约 2.0 m/s（官方 0.2~0.3）。**切换距离 ≤2.0 m**。零指令同 ClimbH 会原地转。",
        vx=[0.3, 0.6], vy=0.0, wz=0.0, zero_ok=False, stance="0.42", expert=True, old=False),
    "ClimbH_8797": dict(
        onnx="ClimbH_8797/climbh_8797.onnx", label="ClimbH 上台面专家",
        desc="只能在 runner 跑着 0.42 组主策略时热切：按住前进的同时按 C（手柄 Y）。"
             "上 33/36 cm 大平台用，不是窄顶石笼。零指令会原地转！",
        vx=[0.3, 0.6], vy=0.0, wz=0.0, zero_ok=False, stance="0.42", expert=True, old=False),
    "StairN-C_9300": dict(
        onnx="StairN-C_9300/stairnc_9300.onnx", label="楼梯专家（上楼梯）",
        desc="决赛楼梯专家。用法：离第一级约 1 m 正对楼梯 → 点「楼梯赛段」（档位自动调到 0.5）→ 按住 W：控制台自动切入、专家以 0.5 前进，"
             "自己走完这 1 m 平地再爬；松 W = 先退回主策略再停车（专家没练过零速）；再按 W 自动再切；S 倒退交主策略。"
             "航向以每次切入时的偏航为目标（Q/E 微调）。到顶后松 W，点「平地赛段」。",
        vx=[0.2, 0.6], vy=0.0, wz=0.6, zero_ok=False, stance="0.42", expert=True, stair=True, old=False),
    "StairN-E2_10300": dict(
        onnx="StairN-E2_10300/stairne2_10300.onnx", label="楼梯专家 E2（候选，未真机验收）",
        desc="09-13 训练侧候选：9300 续训 1000 步，用官方运控上楼梯 200 Hz 录包做四拍步行参考，罚两前轮同时腾空 / 顶立面 / 轮力矩 >5 N·m。"
             "仿真 26/26（现场型 0.11/0.15，含圆角、低摩擦、斜 15°、途中转向），侧倾峰 6.6～8°，两前轮同时腾空 7～9%（9300 38%）。"
             "用法与 9300 完全一样（S10_STAIR_POLICY=StairN-E2_10300 起控制台后楼梯槽装它）。速度旋钮效果弱，仍不能原地停。",
        vx=[0.2, 0.6], vy=0.0, wz=0.6, zero_ok=False, stance="0.42", expert=True, stair=True, old=False),
    "StairN-E4_10600": dict(
        onnx="StairN-E4_10600/stairne4_10600.onnx", label="楼梯专家 E4（候选 B，未真机验收）",
        desc="09-13 训练侧候选 B。",
        vx=[0.2, 0.6], vy=0.0, wz=0.6, zero_ok=False, stance="0.42", expert=True, stair=True, old=False),
    "StairN-E4_11099": dict(
        onnx="StairN-E4_11099/stairne4_11099.onnx", label="楼梯专家 E4-11099（候选 C，未真机验收）",
        desc="09-13 训练侧候选 C。",
        vx=[0.2, 0.6], vy=0.0, wz=0.6, zero_ok=False, stance="0.42", expert=True, stair=True, old=False),
    "StairN-E5_11200": dict(
        onnx="StairN-E5_11200/stairne5_11200.onnx", label="楼梯专家 E5（候选 D，未真机验收）",
        desc="09-13 晚训练侧候选 D：候选 C 续训到 11200，两只前轮都抬、越台沿净空 4/7 cm、侧倾峰典型 10°。仿真验收过，未真机验收。装法同 9300。",
        vx=[0.2, 0.6], vy=0.0, wz=0.6, zero_ok=False, stance="0.42", expert=True, stair=True, old=False),
    "StairN-S3_13800": dict(
        onnx="StairN-S3_13800/stairns3_13800.onnx", label="楼梯专家 S3（候选 E：能停能倒退，未真机验收）",
        desc="09-14 训练侧：D→E7/E8→S3，楼梯模式里给 0 站 6～8 s 漂 ≤3 cm、能前进/倒退/转向；仿真方棱角 2/2、圆角 1/1，侧倾 6.7～8°；档位 0.3→2.6 s/级、0.4→1.7、0.5→1.0；**0.2 档会停在第 2 级，档位限 0.3～0.5**。左前/左后真抬，右侧贴沿滚。倒退只到六成、左转弱。",
        vx=[0.3, 0.5], vy=0.0, wz=0.6, zero_ok=True, stance="0.42", expert=True, stair=True, old=False, stop_ok=True, seg_vx=0.3),
    "StairS16_15200": dict(
        onnx="StairS16_15200/stairs16_15200.onnx", label="楼梯专家 S16（09-18 交付，未真机验收）",
        desc="训练侧 DELIVER_20260918（md5 faa37c9a）。MuJoCo 真实楼梯几何（踢面 11/15 cm × 踏面 55 cm）+ 小幅转向 6/6 上楼；0.5 档 23.1 s / 侧倾 14.2°，0.25 慢档 52.0 s / 侧倾 10.6°（必保项过）。"
             "最弱 TG8N（目标航向偏 -8°）：侧倾 17.9°、爬楼中偏航 -18.8~+8.0°。已知差距：跟坡 -3~-5° vs 官方 -15.4°。"
             "**零指令能不能站住训练侧没给**，所以控制台按老规矩走：按住 W 才切专家、松开切回主策略停车；确认能停后再加 stop_ok。",
        vx=[0.2, 0.5], vy=0.0, wz=0.6, zero_ok=False, stance="0.42", expert=True, stair=True, old=False, seg_vx=0.25),
    "StairN-S8_14800": dict(
        onnx="StairN-S8_14800/stairns8_14800.onnx", label="楼梯专家 S8（候选 F：能停、档位全管用，未真机验收）",
        desc="09-14 训练侧建议替代 E：30/30 登顶（方棱角/圆角/摩擦 0.6/偏 15°/档位/转向），侧倾峰多数 7～10°；档位 0.2→1.7 s/级、0.25→1.45、0.3→1.25、0.4→0.95、0.5→0.9；平地给 0 站 8 s 漂 ≤2 cm，倒退六成、左转弱。慢档两前轮都抬。真机先 0.25 档，不顺回 A。",
        vx=[0.2, 0.5], vy=0.0, wz=0.6, zero_ok=True, stance="0.42", expert=True, stair=True, old=False, stop_ok=True, seg_vx=0.25),
}
DEFAULT_POLICY = "V1H_9196"        # 09-12 19:0x 作者拍板：主策略换 V1H（比赛不下台阶；转向快、后轮出力高、单独能上楼梯）
# runner 的专家槽（S10_POLICY_CLIMB）只装它。rl_control_state.hpp:169 构造时读一次，进程内唯一。
# night5g：爬墙槽（runner 的 S10_POLICY_CLIMB）也做成可选，默认仍 ClimbH_8797。
CLIMB_TABLE = ("ClimbH_8797", "ClimbR13")
EXPERT = os.environ.get("S10_CLIMB_POLICY", "ClimbH_8797")
if EXPERT not in CLIMB_TABLE:
    EXPERT = "ClimbH_8797"


def set_climb_policy(name):
    """night5g：运行时改爬墙槽策略（下次 runner_start 生效）。返回 (ok, msg)。"""
    global EXPERT
    if name not in CLIMB_TABLE:
        return False, "爬墙槽只能选：%s" % " / ".join(CLIMB_TABLE)
    EXPERT = name
    STATE["climb_choice"] = name
    running = STATE.get("running_climb")
    note = "" if (not STATE.get("runner_pid") or running == name) else "；runner 现在带的是 %s，要先一键结束再一键接管（重起 runner）才生效" % (running or "空")
    logline("爬墙槽策略 → %s%s" % (name, note))
    return True, "爬墙槽策略已选 %s%s" % (name, note)
EXPERT_STANCE = "0.42"                # ClimbH 训练资产是 0.42 站姿，只能配 0.42 组主策略
# ---- 09-11 楼梯专家（作者拍板）：runner 第三槽 S10_POLICY_PIT（代码里叫"坑内"），/climb_mode 2 切入、0 退回。
# 训练指令：vx 0.2~0.6 只前进、没练过零速；vy=0；|wz|≤0.6。训练侧推荐以 0.5 进楼梯。
# night4q：楼梯槽装谁由环境变量选，默认 9300；表里没有的名字 → 回落 9300 并在启动日志里说
STAIR_TABLE = {
    "StairN-C_9300": ("StairN-C_9300/stairnc_9300.onnx", "c0a72923406119b65ebce230a3277557"),
    "StairN-E2_10300": ("StairN-E2_10300/stairne2_10300.onnx", "e0ef6b68"),   # 前 8 位由训练侧 md5.txt 给；下面比对按前缀
    "StairN-E4_10600": ("StairN-E4_10600/stairne4_10600.onnx", "08973219"),   # 候选 B
    "StairN-E4_11099": ("StairN-E4_11099/stairne4_11099.onnx", "62caede9"),
    "StairN-E5_11200": ("StairN-E5_11200/stairne5_11200.onnx", "315c942b"),
    "StairN-S3_13800": ("StairN-S3_13800/stairns3_13800.onnx", "d0ea1578"),   # 候选 E：能停、能前后转向（09-14）
    "StairN-S8_14800": ("StairN-S8_14800/stairns8_14800.onnx", "3a03c3af"),
    "StairS16_15200": ("StairS16_15200/stairs16_15200.onnx", "faa37c9a"),   # 09-18 交付：真实几何 11/15×55 + 小转向 6/6   # 候选 F：能停，档位 0.2～0.5 全管用，30/30（09-14 13:5x，训练侧建议替代 E）   # 候选 D：C 续训，两只前轮都抬、净空 4/7 cm、侧倾峰典型 10°   # 候选 C
}
STAIR = os.environ.get("S10_STAIR_POLICY", "StairN-C_9300")
if STAIR not in STAIR_TABLE:
    STAIR = "StairN-C_9300"
STAIR_ONNX, STAIR_ONNX_MD5 = STAIR_TABLE[STAIR]


def stair_can_stop():
    """night5a：楼梯槽装的策略自己能停（给 0 站住）吗。"""
    return bool(POLICIES.get(STAIR, {}).get("stop_ok"))


def seg_full_now():
    """night5a：楼梯赛段是否按全程专家跑 = 环境变量 SEG_FULL，或楼梯槽策略能停。"""
    return SEG_FULL or (STATE.get("segment") == "stairs" and stair_can_stop())


def set_stair_policy(name):
    """night4u：运行时改楼梯槽策略（下次 runner_start 生效）。返回 (ok, msg)。"""
    global STAIR, STAIR_ONNX, STAIR_ONNX_MD5
    if name not in STAIR_TABLE:
        return False, "楼梯槽只能选：%s" % " / ".join(STAIR_TABLE)
    STAIR = name
    STAIR_ONNX, STAIR_ONNX_MD5 = STAIR_TABLE[name]
    STATE["stair_choice"] = name
    running = STATE.get("running_pit")
    note = "" if (not STATE.get("runner_pid") or running == name) else "；runner 现在带的是 %s，要先一键结束再一键接管（重起 runner）才生效" % (running or "空")
    logline("楼梯槽策略 → %s%s" % (name, note))
    return True, "楼梯槽策略已选 %s%s" % (name, note)
      # 训练侧给的 md5；不符（没传完 / 传错）就不装楼梯槽，坏文件会让 runner 起不来
STAIR_STANCE = "0.42"
STAIR_VX_MIN = 0.1                    # night4v：作者要 0.1/0.15 档（训练范围最低 0.2，低于它策略按自己节奏走，指令照发）
STAIR_VX_MAX = 0.6
STAIR_ENTRY_VX = 0.5                  # 切入速度；楼梯专家期间前进封顶在它（与 ClimbH 封顶 0.4 同一做法）
# night4b 楼梯赛段：按住 W（目标 vx ≥ STAIR_SEG_VX_MIN 且新鲜）就自动切专家；退出专家后至少隔 SEG_REENTER_S 再切；
# 切不进去时前进=0，原因每 SEG_LOG_S 记一次。
STAIR_SEG_VX_MIN = 0.2
STAIR_SEG_DEFAULT_VX = 0.3        # night4v：选楼梯赛段时的默认档位（官方上楼梯 1.75 s/级 ≈ 0.3 m/s）
SEG_REENTER_S = 0.3
SEG_LOG_S = 3.0
SEG = {"t_exit": 0.0, "t_try": 0.0, "t_log": 0.0, "deny": None, "flat_vx": None}   # flat_vx：进专家赛段前的档位，回平地时恢复（night4e）
STAIR_ENTRY_WZ_S = 1.0                # night4z：切入后这么久内纠偏限幅更小
STAIR_ENTRY_WZ_MAX = 0.1
STAIR_WZ_MAX = 0.3                    # night4m：0.6 → 0.3（作者：歪了别拧）；原注：                    # 楼梯专家转向限幅（训练侧仿真验证 ±0.6）
STAIR_K_CHOICES = (2.0, 1.0, 0.0)     # 航向保持增益三档（页面按钮；0 = 关，只剩 Q/E 手动）
STAIR_K_DEFAULT = float(os.environ.get("S10_STAIR_YAW_KP", "2.0"))
STAIR_HEAD_GATE_DEG = 5.0           # 09-11 夜审：续爬时与记下的楼梯朝向差超过它就不让切（训练侧 RG：转到 5° 内再放 13/16，约 9° 放 6/10）
STAIR_IMU_MAX_AGE = 0.5               # /IMU_DATA_10HZ 超过这么久没更新 → 不修正、不许切入
STAIR_ONNX_MIN_BYTES = 100000         # 楼梯 onnx 小于它当成没传完，不装（坏文件会让整个 runner 起不来）
RUNNER_SLEW_V = 1.0                   # 与 runner_start 里 S10_CMD_SLEW_V 一致（runner 侧前进指令 1 m/s² 限斜率）
EXPERT_ENTRY_MAX_RVX = 0.6            # 切专家时 runner 侧前进速度估计的上限（两个专家训练上限都是 0.6）
RVX_MARGIN = 0.06                     # 估计比 runner 早约一个发布周期（≈60 ms × 1 m/s²），闸门留这点余量
IMU_NOW = {"yaw": None, "roll": None, "t": 0.0}     # ros 线程（IMU 回调）写，发布回路 / HTTP 线程读
RVX = {"est": 0.0, "t": 0.0, "last_pub": 0.0}   # runner 侧限斜率后的前进速度估计（发布回路 20 Hz 更新）
# 0911d IMU 闸门上限（runner 默认 100 ms，负载高时擦线；训练侧建议 200 ms，断流照样拦）
IMU_MAX_AGE_MS = "200"
STANCE = {                            # 训练侧《使用说明》§一 / runner 0911b 更新说明 §2
    "0.42": {"S10_DEF_HIPX": "0.05", "S10_DEF_HIPY": "0.35", "S10_DEF_KNEE": "0.65"},
    "0.35": {"S10_DEF_HIPX": "0.05", "S10_DEF_HIPY": "0.72", "S10_DEF_KNEE": "1.44"},
}
# 专家指令语义（R6a/R13）。ClimbH 训练指令范围只有 0.3~0.6；零速度回合占比 0，零指令会原地转。
EXPERT_VX_MIN = 0.3
EXPERT_VX_MAX = 0.6
EXPERT_ENTRY_VX = 0.4                 # 训练侧推荐：离墙约 1 m 切专家，同一个处理里把 vx 设 0.4
# R11/R12 只提示。依据：MuJoCo 离墙 1 m 切专家、0.4 m/s，5 次都在切入后 3.6~3.9 s 上到台面。
LEVEL_DEG = 5.0                       # |roll|、|pitch| 都不超过它算"平"
LEVEL_HOLD_S = 0.5                    # 连续平这么久算"回平"
LEVEL_MIN_S = 3.0                     # 切入满这么久才开始判回平
EXPERT_WARN_S = 8.0                   # 切入满这么久仍未回平 → 红字建议停车/退出
HMAP_AGE_MAX = 0.25                   # /hmap_info[3] 点云年龄上限（与 hmap --cloud-timeout 一致）

HS_TARGET = -0.08                     # 平地基准（官方《相机安装位置确认单》§5）。
HS_TOL = 0.015                        # 【只作标定参考，不是闸门】它只在"狗站平地"
                                      # 这个前提下有意义；狗一蹲/一趴/一上台阶
                                      # 整张图就整体平移。见 hscan_gate 的说明。
HS_UNDER = 0.10                       # |x|<=此值的格子算"身下"，做姿态无关的基准
HS_WALL_REL = 0.20                    # 比身下高出这么多算"墙"
HS_WALL_FRAC = 0.60                   # 六成格子都是墙 = 地图中毒（连身后都是墙，
                                      # 赛道上不可能）。这是 max 棘轮的签名
HS_HOLE_MAX = 0.60                    # 空洞率上限。实测健康值 0.13~0.32（身下自遮挡）
HS_STEP_REL = 0.15                    # 前方中轴判"有坎"的阈值，用于显示

# 训练侧那 187 个格心的权威定义。拿它来分"身下/前方/左右"，别自己猜顺序。
_SIMDIR = "/home/robot/s10/s10_ws/src/S10_sdk_deploy/interface/robot/simulation"
try:
    if _SIMDIR not in sys.path:
        sys.path.insert(0, _SIMDIR)
    from heightmap import CELL_XY as _CELL       # noqa: E402
    _HS_X = [float(v) for v in _CELL[:, 0]]
    _HS_Y = [float(v) for v in _CELL[:, 1]]
except Exception as _e:                          # 拿不到就退化成"只做 NaN/空洞检查"
    _HS_X = _HS_Y = None
    print("[warn] 取不到 heightmap.CELL_XY (%s)，高程图闸门降级为只查 NaN/空洞" % _e)
# R10（作者 09-11 拍板，所有策略统一）：运动中倾角自动急停阈值，两个轴分开给。
# 依据（训练侧 MuJoCo 部署栈实测）：成功上墙本身侧倾最高 40°（40 ms 延迟叠加 20° 入射）、
# ClimbH 单独 43°；俯仰 ClimbH 约 34°，V1H 单独上墙到过 48.6°；上楼梯俯仰也会持续超 15°。
# 旧值 15° 会让每次上墙在半路被自己的急停弄瘫（阻尼 = 狗瘫软，挂在墙上更危险）。
# 翻车时侧倾是 180°（09-10 真机两次 −141°/−146°），60°/55° 仍在翻车之前。
TILT_ABORT_ROLL_DEG = 60.0
TILT_ABORT_PITCH_DEG = 55.0
# 起 runner 前自检"当前姿态"讲的是"起立前机身要平"，与运动中急停是两回事，
# 保留原来的 ok<8 / warn<15 / bad 判法，不随上面的急停阈值放大。
START_TILT_WARN_DEG = 8.0
START_TILT_BAD_DEG = 15.0
CMD_STALE_S = 0.4                     # 页面停止刷新即归零（runner 侧另有 500ms 看门狗）
# 09-12 凌晨（Astra 建议 1/2，night3）：高程图质量。缺失、过期、非法 = "质量未知"，不当成正常。
HQ_HS_MAX_AGE = 0.30        # /height_scan 最后一帧距今上限（hmap 45~50 Hz；点云或 KISS 位姿一断，hmap 整拍不发）
HQ_INFO_MAX_AGE = 0.50      # /hmap_info、/height_scan_raw 最后一帧距今上限（与 /height_scan 同一 tick 发）
HQ_UNDER_X = 0.30           # 机身下方带 |x|<=0.30 m：7 行 × 11 列 = 77 格
ODOM_TOPIC = os.environ.get("S10_ODOM_TOPIC", "/Odometry")   # 09-12 16:5x 作者拍板：位姿源改 FAST-LIO；KISS 停用（S10_ODOM_TOPIC=/kiss/odometry 可切回）
HQ_UNDER_MIN = float(os.environ.get("S10_HQ_UNDER_MIN", "0.30"))   # 机身下方 77 格里"补洞前有效"（直接观测 + 10 cm 邻域中位数补值，不含 v2 插值）的比例下限。暂定：09-11 21:06 平地站立 RL（补洞前录包）
                            # 最小 0.38、p10 0.45、中位 0.64；楼梯上没有数据，作者定
HSRX = {"m": None, "t": 0.0}     # /height_scan 最后一帧（补洞后，策略吃的）和到达时刻
TERR_STEP_MIN = 0.08          # night4f：抬高 ≥ 8 cm 算台阶沿（决赛踢面 11/15 cm，高程图误差 ±2 cm）
TERR_PLAT_MIN = 0.22          # 0.22~0.45 算台面（石笼 33 cm）；≥0.45 算障碍（hmap 45 cm 天花板之上看不出真高度）
TERR_WALL_MIN = 0.45
TERR_SQUARE_M = 0.10          # 左右两侧到抬高沿的距离差 ≤ 它算正对
TERR_X_MIN = 0.25             # 从这一列往前找（更近的格子趴着时是自己的腿）
TERR_ROWS_MIN = 6             # 11 行里至少这么多行看到抬高才算有东西
TERR_HOLD_S = 1.0             # 判到台阶后保持这么久不闪（原始图前排常有洞）
TERR_HOLD = {"t": 0.0, "tr": None}
SQUARE_MAX_DEG = 15.0         # night4m：按台沿修正航向目标的上限（切入时 / 复校时都用）
SQUARE_STEP_DEG = 5.0         # 爬楼中每次复校最多改这么多
SQUARE_PERIOD_S = 1.5         # 复校周期
ROLL_FREEZE_DEG = 12.0        # 侧倾超过它：不纠偏、不复校
SQUARE = {"t": 0.0}
PUBGAP = {"last": 0.0, "max": 0.0, "t_max": 0.0, "t_log": 0.0}   # night5e：发布回路卡顿自检
PUBGAP_WARN_S = 0.2
SQUARE_AUTO = os.environ.get("S10_SQUARE_AUTO", "0") == "1"   # night4n：默认只显示不闭环
SQUARE_MEDIAN_S = 2.0         # 闭环用最近这么久的中位（地形判读每 0.5 s 算一次）
SQUARE_MIN_N = 3              # 至少这么多帧（2 s 里正常 4 帧）
SQUARE_SPREAD_DEG = 10.0      # 这段时间里估计离散超过它 → 不稳，不用
SQUARE_ROWS_MIN = 8           # 11 行里至少这么多行检出台沿才参与估计
SQ_HIST = []                  # (t, yaw_off)，ros/发布回路写，square_estimate 读
STAIR_HOLD_ON = os.environ.get("S10_STAIR_HOLD", "1") == "1"   # night4o：楼梯赛段里松 W 不退专家
STAIR_HOLD_LINK_S = 3.0       # 操作员页面心跳断这么久 → 退专家（唯一的软件保险，急停靠手柄/页面）
SEG_FULL = os.environ.get("S10_SEG_FULL", "0") == "1"   # night4s：楼梯/石笼赛段全程专家（作者 09-13 拍板 A）
HSRAW = {"m": None, "t": 0.0}    # /height_scan_raw 最后一帧（v2 插值前、未裁剪；但已含 ElevationMap 的邻域补值）和到达时刻
HQW = {"q": "ok"}                # 专家途中高程图质量的上一次状态（只用于报警去重，不触发任何动作）
# 09-12 凌晨（Astra 建议 3，night3）：服务端认定唯一操作员页面，其余页面只读。
# 新开 / 刷新的页面一律是旁观；要操作先 /api/op/claim 认领。原操作员页面还有心跳（/api/state?pid=…）时认领不了。
OP = {"pid": "", "hb": 0.0, "since": 0.0, "who": "", "refused": 0}
OPLK = threading.Lock()
OP_STALE_S = 3.0            # 操作员页面多久没心跳算离开（页面每 0.7 s 轮询一次）。只决定"别人能不能认领"，不触发任何动作
SPECTATOR_ESTOP = False     # 旁观页面能不能按急停。默认不能（软急停 = 阻尼，楼梯上也会瘫）；作者定
OP_OPEN = ("/api/state", "/api/rec/status", "/api/rec/list", "/api/preflight", "/api/op/claim", "/api/op/release")
OP_DRY_OK = ("/api/mode", "/api/cmd")   # 只有这两个接口实现了 dry=1（只演算、不发布），旁观也可以用

LIVE = {}
STATE = {"policy": DEFAULT_POLICY, "climb": 0, "max_vx": 0.3, "max_vy": 0.2,
         "max_wz": 0.5, "hscan": False, "runner_pid": None,
         "estop": False, "estop_reason": "", "log": [], "rosout": [],
         "damp_t": 0.0,                # 发出阻尼的时刻，用来模拟 3 秒自动回 0
         "mode_src": "假设",           # 当前 mode 是"我方假设"还是"runner 亲口说的"
         "runner_max": None,           # runner 启动时被冻结的速度上限
         # runner 进程里真正加载的【基础策略】。rl_control_state.hpp:152 在构造函数
         # 里读 S10_POLICY_PATH、:159 建 S10PolicyRunner，qw_state_machine.hpp:80
         # 整个进程只构造一次 —— 所以换基础策略只能重启 runner。
         # 唯一的运行期切换是 /climb_mode（:88-103 按 reserved_scale 选 active_policy_），
         # S10_POLICY_CLIMB 在 0.42 组固定是 ClimbH_8797，0.35 组不装专家。
         "running_policy": None,
         "running_group": None,        # runner 的站姿组（"0.42"/"0.35"；接管来的对不上时为 "?"）
         "running_climb": None,        # runner 专家槽里装的是谁（S10_POLICY_CLIMB 的目录名）
         "running_pit": None,          # runner 楼梯槽里装的是谁（S10_POLICY_PIT 的目录名），09-11
         "pit_loaded": False,          # runner 日志里出现过"已加载坑内策略"（本次 runner），09-11
         "stair_yaw_ref": None,        # 楼梯专家目标朝向（弧度，IMU 偏航），09-11
         "stair_heading": None,        # night4b：不再使用（作者：不记楼梯朝向、不设闸门）；字段保留免得页面/测试引用出错
         "segment": "flat",            # night4b/4d：赛段 flat / gabion（石笼 ClimbH）/ stairs（楼梯）。专家赛段里按住 W 自动切专家，主策略不前进
         "stair_k": STAIR_K_DEFAULT,
    "square_msg": "",              # night4m：最近一次切入时的台沿修正说明
    "climb_choice": "ClimbH_8797",   # night5g：当前选的爬墙槽策略（下次起 runner 生效）
    "stair_choice": STAIR,          # night4u：当前选的楼梯槽策略（下次起 runner 生效）   # 楼梯航向保持增益（页面三档 2/1/0），09-11
         "obs_dump": None,             # 本趟 S10_OBS_DUMP 文件
         "climb_t0": None,             # 切进专家的时刻（R11/R12 计时）
         "climb_cap": None,            # 退专家后的前进限速锁存，见 plan_cmd / _expert_clear
         "climb_rev": 0,               # 专家状态每变一次 +1，页面用它丢掉过时的轮询结果
         "climb_src": "",              # 谁切进来的（kbd/pad/mouse/策略列表）
         "climb_ack": "",              # runner 最后一条 [CLIMB] 回话
         "rec_t0": 0.0, "rec_note": "",
         # armed=False 时一条话题都不往外发（共用总线，未接管不能干扰另一队）
         "armed": False, "mode": 0, "hmap_pid": None}
TGT = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "t": 0.0}
# /api/cmd 的页面序号。ThreadingMixIn 下两个相隔几毫秒的请求可能乱序处理：
# 切专家前发出的"滑块档位"指令晚到，会把刚设好的 0.4 顶掉；松键的零被更早的前进顶掉，
# 专家就多吃 0.4 s。带 pid+seq 的请求，同一页面里序号不增就丢弃；不带的照旧接收。
CMDSEQ = {"pid": "", "seq": -1}
XPL = {"since": None, "seen": False, "peak_p": 0.0, "peak_r": 0.0}   # R11 回平去抖
TILTN = {"n": 0}                      # 侧倾去抖：连续 3 次(1.5s)超限才动作
LK = threading.Lock()
PUB = {}
PROC = {"p": None}                    # runner 的 Popen，停止时要 poll


def effective_policy():
    """此刻真正决定机器人行为的策略。
    runner 没起 → 就按选中的算（还没有既成事实）。
    runner 起了 → 基础策略是它启动时加载的那份；只有 climb=1 时才是专家槽里的 ClimbH。"""
    if not STATE["runner_pid"]:
        return STATE["policy"]
    if STATE["climb"] == 2:
        return STATE.get("running_pit") or STAIR
    if STATE["climb"] == 1:
        return STATE.get("running_climb") or EXPERT
    return STATE["running_policy"] or STATE["policy"]


def eff_limits():
    """当前真正生效的指令范围 = 策略训练范围 ∩ 滑块上限。
    滑块是实时的，所以这里每次发布都重算。"""
    # 用【runner 里真在跑的】策略算范围。用选中的会让页面显示一个
    # 机器人根本不会执行的范围 —— 那正是"界面和实际不一致"。
    pol = POLICIES.get(effective_policy(), {})
    rng = pol.get("vx", [-9.9, 9.9])
    # 下界先跟 0 取小 —— 专家 ClimbH 的 vx 范围是 [0.3, 0.6]，直接拿 0.3 当下界的话，
    # "停止"(vx=0) 会被夹成 +0.3，机器人在你按停止的时候往前爬。
    # 策略的下界只用来限制【后退】，永远不能把零抬起来。
    lo = max(min(rng[0], 0.0), -STATE["max_vx"])
    hi = min(rng[1], STATE["max_vx"])
    if hi < 0.0:
        hi = 0.0
    vy = min(pol.get("vy", 9.9), STATE["max_vx"])
    wz = min(pol.get("wz", 9.9), STATE["max_wz"])
    return lo, hi, vy, wz


def disp_limits():
    """只供显示/记录的"此刻实际能发的范围"（09-11）。专家期间 eff_limits 按策略表算会与 plan_cmd 不一致。
    不用于 clamp_cmd —— 那里下界必须能取 0。"""
    if STATE["climb"] == 2:
        return STAIR_VX_MIN, min(STAIR_ENTRY_VX, STATE["max_vx"]), 0.0, STAIR_WZ_MAX
    if STATE["climb"] == 1:
        return EXPERT_VX_MIN, min(EXPERT_ENTRY_VX, STATE["max_vx"]), 0.0, 0.0
    return eff_limits()


def _approach(cur, tgt, step):
    """复刻 runner ros_command_interface.hpp 的 Approach：每步最多变 step。"""
    if tgt > cur + step:
        return cur + step
    if tgt < cur - step:
        return cur - step
    return tgt


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def imu_yaw_fresh():
    """(偏航弧度, 年龄秒)。拿不到或超过 STAIR_IMU_MAX_AGE → (None, 年龄或 None)。"""
    t = IMU_NOW["t"]
    age = (time.time() - t) if t else None
    y = IMU_NOW["yaw"]
    if y is None or not math.isfinite(y) or age is None or age > STAIR_IMU_MAX_AGE:
        return None, age
    return y, age


def _rl_confirmed():
    """runner 亲口确认处于 RL（mode 6）。专家只能在 RL 里热切；非 RL 时切了，runner 进 RL 的第一拍
    就会按残留的 /climb_mode 直接进专家（rl_control_state OnEnter 先回基础、下一步又按 reserved_scale 选槽）。"""
    return STATE["mode"] == 6 and STATE.get("mode_src") in ("runner 确认", "runner 实测")


def _expert_entry_common():
    """两个专家共同的"此刻能不能切"（09-11 审查确认项）。返回 (ok, why)。"""
    if not _rl_confirmed():
        return False, ("runner 还没确认处于策略接管（控制台记的是 %s，来源 %s）。专家只能在 RL 里热切。"
                       "若狗其实已在 RL（比如刚释放又重新接管），点「1 起立」让 runner 回话确认——RL 中它会拒绝 1，狗不会动；"
                       "否则先 1 起立、再 6 策略接管" % (STATE["mode"], STATE.get("mode_src") or "?"))
    if RVX["est"] + RVX_MARGIN > EXPERT_ENTRY_MAX_RVX + 1e-6:
        return False, ("runner 侧前进速度约 %.2f m/s（它按 %.0f m/s² 限斜率，控制台估算）> %.1f。"
                       "先用主策略把速度降到 %.1f 以下、等它降下来再切，否则专家头几拍吃到训练范围外的指令"
                       % (RVX["est"], RUNNER_SLEW_V, EXPERT_ENTRY_MAX_RVX, EXPERT_ENTRY_MAX_RVX))
    q, qwhy = hmap_quality()      # 09-12 凌晨（Astra 建议 1）：起 runner / 进 RL 时查过，不代表此刻地图还有效
    if q != "ok":
        return False, ("高程图%s：%s。专家是冲着地形去的，地图不可信或不知道可不可信时都不切"
                       "（runner 的平地兜底只适合平地，不适合楼梯 / 上墙）" % ("质量未知" if q == "unknown" else "不可信", qwhy))
    return True, ""


def clamp_cmd(vx, vy, wz):
    """**限幅的权威在这里，不在浏览器。**

    /cmd_vel 全总线只有我们一个发布者，所以在发布前夹一次就是真的夹住了。
    以前只在浏览器 JS 里夹、再靠 runner 启动时冻结的 S10_CMD_MAX_VX 兜底，
    结果就是滑块改了不生效、还得重启 runner —— 那是把限幅放错了层。"""
    lo, hi, vylim, wzlim = eff_limits()
    vx = min(hi, max(lo, vx))
    vy = min(vylim, max(-vylim, vy))
    wz = min(wzlim, max(-wzlim, wz))
    return vx, vy, wz


def logline(s):
    STATE["log"].insert(0, "%s  %s" % (time.strftime("%H:%M:%S"), s))
    del STATE["log"][40:]


def may_publish():
    """**收尾方向**的发布闸门（急停、把狗停下）。
    我们自己的 runner 在跑时也放行 —— 那时关节是我们在控，必须停得下来。"""
    if STATE["armed"] or STATE["runner_pid"]:
        return True
    return False


def may_command():
    """**主动指令**的闸门（切状态机、发按键）——比 may_publish 严格：必须显式"接管"。

    2026-09-10 教训：may_publish 里的 `or STATE["runner_pid"]` 会在控制台接管了
    一个遗留 runner 之后自动打开闸门，于是一个本以为"只读"的状态转移测试
    真的把 /robot_mode 发上了总线。收尾操作可以宽，主动操作必须窄。"""
    return bool(STATE["armed"])


# ----------------------------------------------------------------- 专家（ClimbH）
# 状态：climb=0（基础策略）/ 1（上墙专家 ClimbH）/ 2（楼梯专家，09-11 加，/climb_mode 2，见 stair_enter）。
#   进：只有操作员手动（键盘 C / 手柄 Y / 页面按钮 / 策略列表点 ClimbH）→ climb_enter，
#       必须此刻正按住前进（服务端现场查：最近 0.4 s 内收到过 vx>0 的 /api/cmd）。
#   出：① 手动"退出专家"（X / 手柄 X / 按钮 / 选基础策略）→ 交回主策略，前进限在当时的
#          专家速度，松键再按才恢复档位（不自动加速）；
#       ② R13 停车语义（发布回路）：要发的 vx<0.3、停刷、急停 → 先 /climb_mode 0 再零，锁零到松键；
#       ③ 急停 / 倾角急停 / 释放接管 / 停 runner → 先 /climb_mode 0；
#       ④ runner 自己超时撤专家（/rosout [CLIMB] 指令超时）→ 只同步本地记账。
# R11/R12 的回平、超时只进 LIVE 给页面看，不触发任何发布。
def plan_cmd(vx, vy, wz, stale):
    """本帧 /cmd_vel 该发什么。**只算不发、不改状态**（调用方持 LK）。
    返回 (vx, vy, wz, 退专家原因 或 None)。

    专家模式（R13，训练侧强调）：ClimbH 训练时零速度回合占比 0，零指令会原地打转，
    所以**绝不能把 0 夹成 0.3 继续往前走**。要发的 vx < 0.3（松键、摇杆回中、停刷、急停、
    小指令、滑块上限被拉到 0.3 以下）→ 返回原因，由发布回路先发 /climb_mode 0 再给零。
    夹到 [0.3, 0.6] 只作用于 ≥0.3 的前进指令（给 1.0 → 0.6）。vy、wz 强制 0
    （ClimbH 训练时这两项就是 0）。
    非专家：沿用原来的"停刷/急停归零、按当前滑块夹"，再叠退专家后的限速锁存 climb_cap。"""
    if STATE["climb"] == 2:
        # 楼梯专家（09-11）：训练指令只有前进 0.2~0.6、没练过零速。松键/停刷/急停/要发的 vx 太小 → 返回原因，
        # 由发布回路先 /climb_mode 0 退回主策略再给零（训练侧实测：留在楼梯专家只给 0 停不下；先切 0 能停）。
        if STATE["estop"]:
            return 0.0, 0.0, 0.0, "急停"
        if (seg_full_now() or STAIR_HOLD_ON) and STATE.get("segment") == "stairs":   # night4o/4s：楼梯赛段里松 W / 停刷 / 小指令 / S 都不退
            _hb = OP["hb"] if OP["pid"] else 0.0
            if _hb and time.time() - _hb > STAIR_HOLD_LINK_S:
                return 0.0, 0.0, 0.0, "操作员页面心跳断 %.0f s（失联）" % (time.time() - _hb)
            if seg_full_now():   # night4s：W 按住 = 0.5（受档位限、不低于专家最小 0.2），松开 / S = 0（专家自己会再走一段）
                _fwd = (not stale) and float(vx) >= STAIR_VX_MIN - 1e-6
                _cvx = max(STAIR_VX_MIN, min(STAIR_ENTRY_VX, STATE["max_vx"])) if _fwd else 0.0
                _mw = 0.0 if stale else float(wz)
                return _cvx, 0.0, (stair_wz(_mw) if (_fwd or abs(_mw) > 0.01) else 0.0), None   # night4w：没按 W 时不自动纠偏
            return STAIR_ENTRY_VX, 0.0, stair_wz(0.0 if stale else float(wz)), None
        if stale:
            return 0.0, 0.0, 0.0, "停刷超过 %.1f s（松键 / 页面断连 / 浏览器切走）" % CMD_STALE_S
        cvx = min(float(vx), STAIR_ENTRY_VX, STATE["max_vx"])
        if not (cvx >= STAIR_VX_MIN - 1e-6):          # 顺手挡住 NaN
            return 0.0, 0.0, 0.0, ("楼梯专家下要发的 vx=%.2f < %.1f（松键 / 摇杆回中 / 小指令 / 滑块上限低于 %.1f）"
                                   % (cvx, STAIR_VX_MIN, STAIR_VX_MIN))
        return cvx, 0.0, stair_wz(float(wz)), None    # 航向保持 + 手动微调（stair_wz 会改 stair_yaw_ref）
    if STATE["climb"] == 1:
        if STATE["estop"]:
            return 0.0, 0.0, 0.0, "急停"
        if SEG_FULL and STATE.get("segment") == "gabion":   # night4s：石笼赛段全程 ClimbH，恒 0.4（零指令会原地转），松 W / S 不退
            _hb = OP["hb"] if OP["pid"] else 0.0
            if _hb and time.time() - _hb > STAIR_HOLD_LINK_S:
                return 0.0, 0.0, 0.0, "操作员页面心跳断 %.0f s（失联）" % (time.time() - _hb)
            _fwd = (not stale) and float(vx) >= EXPERT_VX_MIN - 1e-6
            _cvx = max(EXPERT_VX_MIN, min(EXPERT_ENTRY_VX, STATE["max_vx"])) if _fwd else 0.0   # night4v：W = 档位，松开 = 0
            return _cvx, 0.0, 0.0, None
        if stale:
            return 0.0, 0.0, 0.0, "停刷超过 %.1f s（松键 / 页面断连 / 浏览器切走）" % CMD_STALE_S
        # 上界 = min(0.4, 当前滑块)。ClimbH 训练范围 0.3~0.6，但训练侧 MuJoCo 扰动测试验证的是
        # "离墙 1 m 切专家并以 0.4 行驶"（入射 ±30°、延迟 60 ms 全过），专家期间就封顶在 0.4。
        cvx = min(float(vx), EXPERT_ENTRY_VX, STATE["max_vx"])
        if not (cvx >= EXPERT_VX_MIN - 1e-6):          # 顺手挡住 NaN
            return 0.0, 0.0, 0.0, ("要发的 vx=%.2f < %.1f（松键 / 摇杆回中 / 小指令 / 滑块上限低于 %.1f）"
                                   % (cvx, EXPERT_VX_MIN, EXPERT_VX_MIN))
        return cvx, 0.0, 0.0, None
    if stale or STATE["estop"]:
        return 0.0, 0.0, 0.0, None
    vx, vy, wz = clamp_cmd(vx, vy, wz)
    if STATE.get("segment") in ("stairs", "gabion") and vx > 1e-6:    # night4b/4d：专家赛段里前进只由专家执行；切不进专家就不动
        vx = 0.0
    cap = STATE.get("climb_cap")
    if cap is not None:
        if cap <= 0.0:
            return 0.0, 0.0, 0.0, None
        vx = max(-cap, min(cap, vx))
    return vx, vy, wz, None


def _expert_clear(cap):
    """清专家记账（调用方须持 LK）。本来不在专家返回 None，否则返回专家持续秒数。
    cap = 退出后前进速度的锁存上限：0 = 停车语义（锁零）；>0 = 手动退出（交回主策略、
    不自动加速）；None = 不锁。松键（收到全零指令或停刷）后由发布回路解除。"""
    if STATE["climb"] not in (1, 2):
        STATE["climb"] = 0
        return None
    t0 = STATE.get("climb_t0")
    STATE["climb"] = 0
    STATE["climb_t0"] = None
    STATE["climb_cap"] = cap
    STATE["cap_t"] = time.time()   # 09-11 夜审：锁零时刻；只有这之后到的全零帧才算"松键"
    STATE["stair_yaw_ref"] = None
    STATE["climb_rev"] += 1
    SEG["t_exit"] = time.time()        # night4b
    return (time.time() - t0) if t0 else 0.0


def climb_exit(reason, cap):
    if (SEG_FULL or seg_full_now()) and STATE.get("segment") in ("stairs", "gabion") and not str(reason).startswith("换到") and STATE["climb"] in (1, 2):
        _seg_reset("退专家：%s" % reason)     # night4s：全程专家模式下任何退出都回平地，免得 0.3 s 后自动再切进去
    """退出专家：先发 /climb_mode 0；零指令/限速由发布回路按 climb_cap 给。
    收尾方向，沿用 may_publish（我们的 runner 在跑就能发）。本来不在专家也照样发 0
    （幂等，防记账跑偏），但只在真退出时记日志。返回 (刚才是否在专家, 是否发出)。"""
    with LK:
        dur = _expert_clear(cap)
    sent = False
    if may_publish():
        try:
            u = UInt8(); u.data = 0
            PUB["climb"].publish(u)
            sent = True
        except Exception as e:
            logline("发 /climb_mode 0 异常: %r" % e)
    if dur is not None:
        logline("退出专家（%s）→ %s，专家共 %.1f s%s"
                % (reason, "已发 /climb_mode 0" if sent else "未在控狗、只改本地", dur,
                   "" if cap is None else
                   ("；锁零，松开前进再按才恢复" if cap <= 0 else
                    "；交回主策略，前进限在 %.2f m/s（不自动加速），松开前进再按才恢复档位" % cap)))
        obs_event("climb=0 %s" % reason)
    return dur is not None, sent


def expert_out_vx():
    """手动退出时的限速值 = 退出前一刻真正发出去的前进速度（专家范围内）。"""
    co = LIVE.get("cmd_out") or [EXPERT_ENTRY_VX]
    return max(0.0, min(EXPERT_VX_MAX, float(co[0])))


def climb_gate():
    """切专家的静态前提（不含"此刻有没有按住前进"，那个在 climb_enter 里现场查）。
    页面按钮置灰、策略列表的提示都用它。"""
    if not may_command():
        return False, "未接管"
    if STATE["estop"]:
        return False, "急停中，先解除急停"
    if STATE["climb"] == 2:
        return False, "正在楼梯专家里，先按 X 退出楼梯专家"
    if not STATE["runner_pid"]:
        return False, "runner 没起。专家只能在 runner 跑着 0.42 组主策略时经 /climb_mode 1 热切"
    ok, why = _expert_entry_common()   # 09-11：RL 已确认、runner 侧速度 ≤0.6
    if not ok:
        return False, why
    grp = STATE.get("running_group")
    if grp != EXPERT_STANCE:
        return False, ("runner 以站姿组 %s 运行（基础策略 %s）。ClimbH 只能配 0.42 组主策略"
                       "（V1Down / V1H）：0.35 蹲姿组的 runner 没装专家，默认位形也对不上。"
                       "要上台面先停 runner，选 V1Down 或 V1H 再启动"
                       % (grp or "未知", STATE.get("running_policy") or "未知"))
    rc = STATE.get("running_climb")
    if rc != EXPERT:
        return False, ("这个 runner 的专家槽是 %s，不是 %s（多半是旧控制台起的、被接管来的）。"
                       "停 runner 再启动一次" % (rc or "空（没设 S10_POLICY_CLIMB）", EXPERT))
    if STATE["max_vx"] < EXPERT_VX_MIN - 1e-9:
        return False, ("前进上限 %.2f < %.1f —— ClimbH 训练范围 0.3~0.6，比这低只能退出停车。"
                       "先把前进档位调到 ≥ %.1f" % (STATE["max_vx"], EXPERT_VX_MIN, EXPERT_ENTRY_VX))
    return True, ""


def climb_enter(pid, seq, src):
    """切进专家（R6a）。只能由操作员手动触发。

    必须在【按住前进】时切：页面只在按键期间刷 /api/cmd，松键 0.4 s 后服务端判停刷，
    按 R13 先退专家再停车。没按住前进就切，等于切进去 0.4 s 后自己退出 —— 直接拒绝。
    同一个处理里把目标指令设成 0.4/0/0，并立刻按"0.4 → /climb_mode 1 → 0.4"的顺序发出：
    runner 的 in_vx 先变成 0.4 再换专家，紧接着的 /cmd_vel 也是 0.4
    （09-10 那次失败的直接原因就是切专家时给了 1.8）。"""
    if STATE["climb"] == 1:            # 09-11：已在 ClimbH 里再按 C —— 幂等，先于闸门（闸门临时不过也不能弹窗）
        return True, "已经在专家模式"
    if STATE["climb"] == 2:            # 09-11：楼梯专家里误按 C/Y —— 忽略（返回 ok 免得页面弹窗停刷把楼梯专家撤掉）
        return True, "正在楼梯专家里，已忽略切 ClimbH 的请求（要换专家先按 X 退出）"
    ok, why = climb_gate()
    if not ok:
        return False, "不能切专家：" + why
    now = time.time()
    with LK:
        if STATE["climb"] == 1:
            return True, "已经在专家模式"
        age = (now - TGT["t"]) if TGT["t"] else None
        if not (SEG_FULL and STATE.get("segment") == "gabion") and not (age is not None and age <= CMD_STALE_S and TGT["vx"] >= EXPERT_VX_MIN - 1e-6):
            return False, ("切专家必须在【按住前进（≥0.3）】的同时按：键盘按住 W 再按 C；手柄握住 LB/RB/RT、"
                           "左摇杆前推过半再按 Y；用鼠标点按钮时左手要按住 W。服务端现在没收到前进指令"
                           "（%s），切进去 0.4 s 内就会因停刷按停车语义退出，所以不切。"
                           % ("从没收到" if age is None else
                              "最近一条 vx=%.2f，%.1f s 前" % (TGT["vx"], age)))
        if pid and seq is not None:
            if pid == CMDSEQ["pid"]:
                CMDSEQ["seq"] = max(CMDSEQ["seq"], seq)
            else:
                CMDSEQ.update(pid=pid, seq=seq)
        if SEG_FULL and STATE.get("segment") == "gabion":   # night4w：同楼梯，没按 W 就发 0
            _fw = bool(TGT["t"]) and (now - TGT["t"]) <= CMD_STALE_S and TGT["vx"] >= EXPERT_VX_MIN - 1e-6
            vx0 = max(EXPERT_VX_MIN, min(EXPERT_ENTRY_VX, STATE["max_vx"])) if _fw else 0.0
        else:
            vx0 = min(EXPERT_ENTRY_VX, EXPERT_VX_MAX, STATE["max_vx"])   # 闸门保证 ≥0.3
            TGT.update(vx=vx0, vy=0.0, wz=0.0, t=now)
        STATE["climb"] = 1
        STATE["climb_t0"] = now
        STATE["climb_cap"] = None
        STATE["climb_src"] = src
        STATE["climb_rev"] += 1
        XPL.update(since=None, seen=False, peak_p=0.0, peak_r=0.0)
    try:
        t = Twist(); t.linear.x = vx0
        PUB["cmd_vel"].publish(t)
        u = UInt8(); u.data = 1
        PUB["climb"].publish(u)
        t2 = Twist(); t2.linear.x = vx0
        PUB["cmd_vel"].publish(t2)
    except Exception as e:
        climb_exit("切专家时发布异常 %r" % e, 0.0)
        return False, "切专家时发布异常：%r（已回退并发 /climb_mode 0）" % e
    logline("切专家 %s（%s）：已发 /cmd_vel %.2f → /climb_mode 1 → /cmd_vel %.2f。"
            "须一直按住前进；松键即先退专家再停车" % (EXPERT, src, vx0, vx0))
    obs_event("climb=1 %s vx=%.2f" % (src, vx0))
    return True, ("已切专家 %s，前进 %.2f m/s。须一直按住前进；松键即先退专家再停车。"
                  "回平后按 X（手柄 X）手动退出" % (EXPERT, vx0))


def stair_wz(twz):
    """楼梯专家的转向（09-11 作者拍板：自动保持航向 + 手动微调）。调用方持 LK；会改 stair_yaw_ref。
    手动（Q/E，|twz|>0.01）：用操作员转向，限 min(0.6, 转向滑块)；目标朝向跟到当前航向，松手即保持新朝向。
    自动：wz = stair_k ×（目标 − 当前），限 ±STAIR_WZ_MAX；k=0、IMU 过期或没有目标 → 0。"""
    yaw, _ = imu_yaw_fresh()
    if not (twz == twz):          # NaN
        twz = 0.0
    if abs(twz) > 0.01:
        lim = max(0.0, min(STAIR_WZ_MAX, STATE["max_wz"]))
        # 手动时目标朝向跟到当前航向；IMU 拿不到就清掉目标，免得松手后把手动转过的角度一步拉回
        STATE["stair_yaw_ref"] = yaw
        return max(-lim, min(lim, twz))
    ref, k = STATE.get("stair_yaw_ref"), float(STATE.get("stair_k") or 0.0)
    if yaw is None or k <= 0.0:
        return 0.0
    _r = IMU_NOW.get("roll")
    if _r is not None and math.isfinite(_r) and abs(_r) > math.radians(ROLL_FREEZE_DEG):   # night4m：歪了不拧
        return 0.0
    if ref is None:                   # 目标丢了（IMU 断过 / 手动时 IMU 不在）→ 以当前航向为新目标，下一帧起再修
        STATE["stair_yaw_ref"] = yaw
        return 0.0
    _t0 = STATE.get("climb_t0")
    _lim = STAIR_ENTRY_WZ_MAX if (_t0 and time.time() - _t0 < STAIR_ENTRY_WZ_S) else STAIR_WZ_MAX   # night4z
    return max(-_lim, min(_lim, k * _wrap(ref - yaw)))


def stair_gate():
    """切楼梯专家的静态前提（09-11）。页面按钮置灰、自检都用它。"""
    if not may_command():
        return False, "未接管"
    if STATE["estop"]:
        return False, "急停中，先解除急停"
    if STATE["climb"] == 1:
        return False, "正在上墙专家 ClimbH 里，先按 X 退出"
    if not STATE["runner_pid"]:
        return False, "runner 没起。楼梯专家只能在 runner 跑着 %s 组主策略时经 /climb_mode 2 热切" % STAIR_STANCE
    grp = STATE.get("running_group")
    if grp != STAIR_STANCE:
        return False, ("runner 以站姿组 %s 运行（基础策略 %s）。楼梯专家只能配 %s 组主策略（V1Down / V1H），"
                       "停 runner 换策略再启动" % (grp or "未知", STATE.get("running_policy") or "未知", STAIR_STANCE))
    if STATE.get("running_pit") != STAIR:
        return False, ("这个 runner 没带楼梯策略 %s 启动（S10_POLICY_PIT 为 %s）。停 runner 再启动一次"
                       % (STAIR, STATE.get("running_pit") or "空"))
    if not STATE.get("pit_loaded"):
        return False, ("runner 日志里还没出现「已加载坑内策略」，楼梯槽没确认加载，不能切"
                       "（0911e 在楼梯槽没加载时收到 2 会错切到 ClimbH）")
    ok, why = _expert_entry_common()
    if not ok:
        return False, why
    if STATE["max_vx"] < STAIR_VX_MIN - 1e-9:
        return False, ("前进档位 %.2f < %.2f，先把楼梯速度档位调高" % (STATE["max_vx"], STAIR_VX_MIN))
    yaw, age = imu_yaw_fresh()
    if yaw is None:
        return False, ("拿不到新鲜的 IMU 偏航（/IMU_DATA_10HZ %s）—— 楼梯专家要靠它保持航向，不能切"
                       % ("从没收到" if age is None else "已 %.1f s 没更新" % age))
    return True, ""          # night4b：不比楼梯朝向（作者：不过度设计）


def square_estimate(now=None):
    """night4n：最近 SQUARE_MEDIAN_S 内台沿偏角的中位。返回 (度或 None, 说明)。"""
    now = now or time.time()
    vs = [v for t, v in list(SQ_HIST) if now - t <= SQUARE_MEDIAN_S]
    if len(vs) < SQUARE_MIN_N:
        return None, "最近 %.0f s 只有 %d 帧检出台沿（要 ≥%d）" % (SQUARE_MEDIAN_S, len(vs), SQUARE_MIN_N)
    vs.sort()
    if vs[-1] - vs[0] > SQUARE_SPREAD_DEG:
        return None, "台沿偏角在跳（%.0f～%.0f°）" % (vs[0], vs[-1])
    return vs[len(vs) // 2], "2 s 中位 %d 帧" % len(vs)


def square_offset():
    """night4m/4n：切入时按高程图台沿修正航向目标。返回 (修正角度°或 None, 说明)。不闭环时永远返回 None。"""
    tr = LIVE.get("terrain")
    if not tr or tr.get("kind") not in ("楼梯", "台面") or tr.get("yaw_off") is None:
        return None, "台沿看不准（%s），航向目标 = 当时朝向" % ("没判到台阶" if not tr else tr.get("kind"))
    est, why = square_estimate()
    if est is None:
        return None, "台沿偏角不稳（%s），航向目标 = 当时朝向" % why
    lim = max(-SQUARE_MAX_DEG, min(SQUARE_MAX_DEG, float(est)))
    if not SQUARE_AUTO:
        return None, "台沿偏 %+.1f°（只显示未闭环，S10_SQUARE_AUTO=1 才闭环），航向目标 = 当时朝向" % lim
    return lim, "按台沿修正 %+.1f°%s（%s，左 %s / 右 %s，%s）" % (lim, "" if abs(float(est) - lim) < 1e-9 else "（原 %+.1f°，限幅）" % est,
                                                 why, tr.get("dist_l"), tr.get("dist_r"), tr.get("src"))


def stair_enter(pid, seq, src):
    """切进楼梯专家（09-11）。只能由操作员手动触发，必须按住前进（≥ STAIR_VX_MIN）。
    同一个处理里把目标指令设成 STAIR_ENTRY_VX/0/0，按"cmd → /climb_mode 2 → cmd"的顺序发出，
    记下此刻 IMU 偏航为航向保持的目标。"""
    if STATE["climb"] == 1:            # ClimbH 里误按 V —— 忽略（返回 ok 免得页面弹窗停刷把 ClimbH 撤掉）
        return True, "正在上墙专家 ClimbH 里，已忽略切楼梯专家的请求（要换专家先按 X 退出）"
    if STATE["climb"] == 2:            # 已在楼梯专家里再按 V —— 幂等，先于闸门（楼梯中拉低滑块 / IMU 抖动时闸门会临时不过，不能弹窗）
        return True, "已经在楼梯专家"
    ok, why = stair_gate()
    if not ok:
        return False, "不能切楼梯专家：" + why
    now = time.time()
    with LK:
        if STATE["climb"] == 2:
            return True, "已经在楼梯专家"
        age = (now - TGT["t"]) if TGT["t"] else None
        if not (seg_full_now() and STATE.get("segment") == "stairs") and not (age is not None and age <= CMD_STALE_S and TGT["vx"] >= STAIR_VX_MIN - 1e-6):
            return False, ("切楼梯专家必须在【按住前进（≥%.1f）】的同时按：键盘按住 W 再按 V；用鼠标点按钮时左手要按住 W。"
                           "服务端现在没收到够大的前进指令（%s），切进去就会因停刷或指令太小先退回主策略再停车，所以不切。"
                           % (STAIR_VX_MIN, "从没收到" if age is None else "最近一条 vx=%.2f，%.1f s 前" % (TGT["vx"], age)))
        if pid and seq is not None:
            if pid == CMDSEQ["pid"]:
                CMDSEQ["seq"] = max(CMDSEQ["seq"], seq)
            else:
                CMDSEQ.update(pid=pid, seq=seq)
        if seg_full_now() and STATE.get("segment") == "stairs":   # night4w：切入时发当时 W 的值，没按 W 就是 0（不再 0.5 起步冲）
            _fw = bool(TGT["t"]) and (now - TGT["t"]) <= CMD_STALE_S and TGT["vx"] >= STAIR_VX_MIN - 1e-6
            vx0 = max(STAIR_VX_MIN, min(STAIR_ENTRY_VX, STATE["max_vx"])) if _fw else 0.0
        else:
            vx0 = max(STAIR_VX_MIN, min(STAIR_ENTRY_VX, STATE["max_vx"]))   # night4x：起步 = 档位（默认 0.3）
            TGT.update(vx=vx0, vy=0.0, wz=0.0, t=now)
        STATE["climb"] = 2
        STATE["climb_t0"] = now
        STATE["climb_cap"] = None
        STATE["climb_src"] = src
        STATE["climb_rev"] += 1
        _y = imu_yaw_fresh()[0]
        _sq, _sqmsg = square_offset()              # night4m：航向目标 = 垂直台沿（看不准则 = 当时朝向）
        STATE["stair_yaw_ref"] = _y if (_y is None or _sq is None) else _wrap(_y + math.radians(_sq))
        STATE["square_msg"] = _sqmsg; SQUARE["t"] = now
    try:
        t = Twist(); t.linear.x = vx0
        PUB["cmd_vel"].publish(t)
        u = UInt8(); u.data = 2
        PUB["climb"].publish(u)
        t2 = Twist(); t2.linear.x = vx0
        PUB["cmd_vel"].publish(t2)
    except Exception as e:
        climb_exit("切楼梯专家时发布异常 %r" % e, 0.0)
        return False, "切楼梯专家时发布异常：%r（已回退并发 /climb_mode 0）" % e
    ref = STATE.get("stair_yaw_ref")
    logline("切楼梯专家 %s（%s）：已发 /cmd_vel %.2f → /climb_mode 2 → /cmd_vel %.2f。须一直按住前进；松键即先退回主策略再停车。"
            "航向保持目标 %s，增益 %s；%s" % (STAIR, src, vx0, vx0,
                                   "—（IMU 拿不到，不修正）" if ref is None else "%.1f°" % math.degrees(ref),
                                   "关" if not STATE.get("stair_k") else "%.0f" % STATE["stair_k"], STATE.get("square_msg") or "—"))
    obs_event("climb=2 %s vx=%.2f" % (src, vx0))
    return True, ("已切楼梯专家 %s，前进 %.2f m/s，自动保持航向。须一直按住前进；四轮都上顶平台后按 X 退出；"
                  "松键即先退回主策略再停车" % (STAIR, vx0))


def expert_tick(now, roll, pitch, imu_age):
    """R11/R12 的计时与回平判定。**只算、只给 LIVE，绝不发任何话题**
    （作者 09-11 14:55：切换一律由人做）。roll/pitch 为弧度，取绝对值，与 IMU 符号约定无关。
    回平 = 切入满 LEVEL_MIN_S 且 |roll|、|pitch| ≤ LEVEL_DEG 已连续 LEVEL_HOLD_S；
    超时 = 切入满 EXPERT_WARN_S 仍从没出现过回平。"""
    if STATE["climb"] == 2 and STATE.get("climb_t0"):   # 09-11 楼梯专家：计时 + 航向，回平判据不适用
        XPL.update(since=None, seen=False, peak_p=0.0, peak_r=0.0)
        yaw, _ = imu_yaw_fresh()
        ref = STATE.get("stair_yaw_ref")
        return {"on": True, "stair": True, "t": round(now - STATE["climb_t0"], 1),
                "yaw_ref": None if ref is None else round(math.degrees(ref), 1),
                "yaw_err": None if (yaw is None or ref is None) else round(math.degrees(_wrap(ref - yaw)), 1)}
    if STATE["climb"] != 1 or not STATE.get("climb_t0"):
        XPL.update(since=None, seen=False, peak_p=0.0, peak_r=0.0)
        return {"on": False}
    el = now - STATE["climb_t0"]
    ok = roll is not None and pitch is not None and imu_age is not None and imu_age < 0.5
    r = abs(math.degrees(roll)) if ok else None
    p = abs(math.degrees(pitch)) if ok else None
    if ok:
        XPL["peak_r"] = max(XPL["peak_r"], r)
        XPL["peak_p"] = max(XPL["peak_p"], p)
    if ok and r <= LEVEL_DEG and p <= LEVEL_DEG:
        if XPL["since"] is None:
            XPL["since"] = now
    else:
        XPL["since"] = None                   # IMU 断了也当"没回平"
    streak = (now - XPL["since"]) if XPL["since"] is not None else 0.0
    level = el >= LEVEL_MIN_S and streak >= LEVEL_HOLD_S - 1e-9
    if level:
        XPL["seen"] = True
    return {"on": True, "t": round(el, 1),
            "roll": None if r is None else round(r, 1),
            "pitch": None if p is None else round(p, 1),
            "imu_ok": ok, "streak": round(streak, 2), "level": level,
            "leveled_once": XPL["seen"],
            "timeout": el >= EXPERT_WARN_S and not XPL["seen"],
            "peak_roll": round(XPL["peak_r"], 1), "peak_pitch": round(XPL["peak_p"], 1)}


def obs_event(s):
    """往本趟观测转储旁的 .events.txt 追加一行（本地时刻 + epoch + 事件）。
    runner 的转储 CSV 没有时间戳，也不标是哪份策略出的（s10_policy_runner.hpp:535 的
    static ofstream 由基础和专家两份策略共用），训练侧要分"哪几行是专家"只能靠这里。
    写失败不影响控狗。"""
    d = STATE.get("obs_dump")
    if not d:
        return
    try:
        with open(d + ".events.txt", "a", encoding="utf-8") as f:
            f.write("%s %.3f %s\n" % (time.strftime("%H:%M:%S"), time.time(), s))
    except Exception:
        pass


# ----------------------------------------------------------------- ROS 线程
def ros_thread():
    rclpy.init()
    n = rclpy.create_node("s10_ctrl")
    be = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
    rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)

    PUB["cmd_vel"] = n.create_publisher(Twist, "/cmd_vel", rel)
    PUB["key"] = n.create_publisher(String, "/GAMEPAD_KEY", rel)
    PUB["mode"] = n.create_publisher(UInt8, "/robot_mode", rel)
    PUB["climb"] = n.create_publisher(UInt8, "/climb_mode", rel)
    # runner 那边是 create_subscription("/height_scan", 2, ...)，深度数字 = 默认
    # RELIABLE。我们原来用 BEST_EFFORT 发，QoS 不兼容，一帧都到不了 ——
    # 「零填充」复选框其实一直是空操作。hmap_node.py:68 也是 RELIABLE。
    PUB["hscan"] = n.create_publisher(Float32MultiArray, "/height_scan", rel)

    box = {"j": None, "i": None, "o": None, "b": None, "c": None, "s": None, "h": None}
    cnt = {k: 0 for k in ("j", "i", "o", "c", "s", "h")}
    seen = {k: 0.0 for k in cnt}      # 各路最后一帧的到达时刻
    cnt_sum = [0]                     # 总收包数，用来判断"刚才那次 spin 有没有收到东西"

    def bump():
        cnt_sum[0] += 1

    # 显示只需要 2 Hz 刷新，没必要接 200 Hz 的原始流。
    # 接全速时光解包 JointsData 就吃掉 50% CPU，而且在负载下还会漏收，
    # 让控制台自己数出来的频率严重偏低（实测报 23.9 Hz，真实 200.06 Hz），
    # 自检据此误报"阻断"。机器人本来就发了 10Hz 镜像，用它。
    # 真实频率改由 preflight 里实测（见 measure_hz）。
    n.create_subscription(JointsData, "/JOINTS_DATA_10HZ",
                          lambda m: (box.__setitem__("j", m), cnt.__setitem__("j", cnt["j"] + 1), bump(),
                                     seen.__setitem__("j", time.time())), be)
    n.create_subscription(ImuData, "/IMU_DATA_10HZ",
                          lambda m: (box.__setitem__("i", m), cnt.__setitem__("i", cnt["i"] + 1), bump(),
                                     seen.__setitem__("i", time.time()),
                                     IMU_NOW.update(yaw=float(m.data.yaw), roll=float(m.data.roll), t=time.time()) if math.isfinite(float(m.data.yaw)) else None), be)
    # 09-11：位姿源是我们自己的 KISS-ICP。厂家 /LIO_ODOM_HIGH_FREQUENCY 永远不会来
    # （106 的 rsdriver 被 drsec 挡死）。之前盯着它，KISS 静默卡死 18 分钟没人发现。
    # KISS 发 RELIABLE，这里 BEST_EFFORT 订阅照样匹配。
    n.create_subscription(Odometry, ODOM_TOPIC,
                          lambda m: (box.__setitem__("o", m), cnt.__setitem__("o", cnt["o"] + 1), bump()), be)
    n.create_subscription(BatteryData, "/BATTERY_DATA", lambda m: box.__setitem__("b", m), be)
    # 09-11：/JOINTS_CMD 是 200 Hz 的 16 关节整帧。这里只收原始字节：计数、记到达时刻；
    # 显示用的 cw/kp/kd 在 2 Hz 刷新时才解最新一帧（逐帧解包 0.29 ms/帧，200 Hz 约 6% 个核）。
    n.create_subscription(JointsDataCmd, "/JOINTS_CMD",
                          lambda raw: (box.__setitem__("c", raw), cnt.__setitem__("c", cnt["c"] + 1), bump(),
                                       seen.__setitem__("c", time.time())), be, raw=True)
    n.create_subscription(Steer, "/STEER",
                          lambda m: (box.__setitem__("s", m), cnt.__setitem__("s", cnt["s"] + 1), bump()), be)
    hs_med = []                       # 最近若干帧的"每帧中位数"，用来判基准

    def on_hs(m):
        box["h"] = m
        cnt["h"] = cnt["h"] + 1
        bump()
        HSRX.update(m=m.data, t=time.time())   # 09-12 凌晨：切专家闸门要看新鲜度
        v = m.data
        if len(v) >= 20:
            s = sorted(v)
            hs_med.append(s[len(s) // 2])
            if len(hs_med) > 120:
                del hs_med[:60]

    n.create_subscription(Float32MultiArray, "/height_scan", on_hs, be)
    # 09-12 凌晨（Astra 建议 2）：v2 插值前、未裁剪的图，只用来数"机身下方 77 格补洞前有效多少"（含 10 cm 邻域补值，不是纯直接观测）。回调只存引用。
    n.create_subscription(Float32MultiArray, "/height_scan_raw",
                          lambda m: (HSRAW.update(m=m.data, t=time.time()), bump()), be)
    # R7：/hmap_info = [空洞率, 覆盖, 帧, 点云年龄(s), KISS 世界系倾斜(度，新版 hmap 才有)]
    n.create_subscription(Float32MultiArray, "/hmap_info",
                          lambda m: (LIVE.__setitem__("hinfo", [round(float(v), 4) for v in m.data]),
                                     LIVE.__setitem__("hinfo_t", time.time())), be)
    # runner 的 [MODE]/[CLIMB] 判决走 RCLCPP_INFO/WARN，也就是 /rosout。
    # 这是我们唯一能看到状态机【真实】当前状态的通道 ——
    # MotionStateFeedback 不发话题，之前只能靠自己记，记错了就误拦。
    n.create_subscription(Log, "/rosout", on_rosout, rel)

    t_start = time.time()             # 控制台起来的时刻，没收到过 CMD 时拿它算静默时长
    hs = Float32MultiArray()
    hs.data = [0.0] * 187
    last_rate = time.time()
    prev = dict(cnt)
    tick = 0

    last_pub = 0.0
    while rclpy.ok():
        # 先把回调队列排空。spin_once 一次只取一个回调，而总线上 6 路话题
        # 加起来 1200 回调/秒，一轮只 spin 一次必然积压丢帧（实测 /JOINTS_DATA
        # 只收到 86/200 Hz，IMU 同样欠采样，侧倾急停判据会变钝）。
        # 排空到没有新消息为止就停 —— 之前无脑转 64 次，队列空了还在空转，
        # 每秒三万多次 spin_once 把 CPU 干到 82%。现在有多少收多少。
        for _ in range(64):
            before = cnt_sum[0]
            rclpy.spin_once(n, timeout_sec=0.0)
            if cnt_sum[0] == before:
                break
        else:
            pass
        # 09-11：4 ms → LOOP_IDLE_WAIT。rclpy 每次 spin_once 都重建等待集，ARM 上空闲时每秒 250 次光这个就吃 55% 个核。
        # 有消息时立刻返回，不影响收包；20 Hz 发 cmd_vel 由下面的 last_pub 计时，最多晚一个等待周期。
        rclpy.spin_once(n, timeout_sec=LOOP_IDLE_WAIT)   # 队列空了就阻塞等，别空转
        now = time.time()
        tick += 1

        # 20 Hz 发 cmd_vel —— 只在"已接管"时发。
        # 共用一只狗，DDS 域也共用：未接管时一条都不能发，
        # 否则我们的零值会顶掉另一队 runner 的速度指令。
        if now - last_pub >= 0.05:          # 稳定 20 Hz，不再依赖循环节拍
            last_pub = now
            if not STATE["armed"]:
                LIVE["cmd_out"] = None
                if now - RVX["last_pub"] > 0.5:   # 不发了：runner 500 ms 看门狗到点才归零，之后按限斜率降
                    RVX["est"] = _approach(RVX["est"], 0.0, RUNNER_SLEW_V * 0.05)
                RVX["t"] = now
            else:
                if STATE.get("segment") in ("stairs", "gabion") and STATE["climb"] == 0 and not STATE["estop"]:
                    # night4b/4d：专家赛段——按住 W（目标新鲜且 ≥ STAIR_SEG_VX_MIN）就自动切专家（楼梯 /climb_mode 2 以 0.5；石笼 ClimbH /climb_mode 1 以 0.4）。闸门照旧。
                    _seg = STATE.get("segment")
                    with LK:
                        _fresh = bool(TGT["t"]) and (now - TGT["t"]) <= CMD_STALE_S
                        _tvx = TGT["vx"]
                    if (seg_full_now() or SEG_FULL or (_fresh and _tvx >= STAIR_SEG_VX_MIN - 1e-6)) and now - SEG["t_exit"] >= SEG_REENTER_S and now - SEG["t_try"] >= 0.25:   # night4s：全程专家不看 W
                        SEG["t_try"] = now
                        _ok, _msg = (stair_enter if _seg == "stairs" else climb_enter)("", None, "赛段自动")
                        if _ok:
                            SEG["deny"] = None
                        else:
                            SEG["deny"] = _msg
                            if now - SEG["t_log"] >= SEG_LOG_S:
                                SEG["t_log"] = now
                                logline("%s赛段：切不进专家 → 不前进，每 0.25 s 重试。%s" % ("楼梯" if _seg == "stairs" else "石笼", _msg))
                    elif not _fresh or _tvx < STAIR_SEG_VX_MIN - 1e-6:
                        SEG["deny"] = None
                with LK:
                    stale = (now - TGT["t"]) > CMD_STALE_S
                    tvx, tvy, twz = TGT["vx"], TGT["vy"], TGT["wz"]
                    # 退专家后的限速锁存：松键（停刷，或页面发来全零）就解除
                    # 09-11 夜审：全零帧必须晚于锁零时刻 cap_t（触发退专家的那一帧零不算松键；别的页面失焦发的零也解不开）
                    if STATE["climb"] not in (1, 2) and STATE.get("climb_cap") is not None and \
                            (stale or (TGT["t"] > STATE.get("cap_t", 0.0) and
                                       abs(tvx) < 1e-6 and abs(tvy) < 1e-6 and abs(twz) < 1e-6)):
                        STATE["climb_cap"] = None
                    # 全控制台唯一的指令出口：停刷/急停归零、按当前滑块夹、专家语义（R13）
                    vx, vy, wz, why = plan_cmd(tvx, tvy, twz, stale)
                    rev0 = STATE["climb_rev"]
                # R13：专家模式下要发的 vx<0.3 → 同一线程里先 /climb_mode 0（climb_exit 内发），再发零
                if why:
                    climb_exit(why, 0.0)
                    vx, vy, wz = 0.0, 0.0, 0.0
                # 09-11：plan_cmd 在锁内算、这里在锁外发。若中间有人切了专家（climb_rev 变了），
                # 这一帧是按切入前的状态算的，丢掉（退专家的零照发）。
                if why or STATE["climb_rev"] == rev0:
                    t = Twist()
                    t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
                    PUB["cmd_vel"].publish(t)
                    LIVE["cmd_out"] = [round(vx, 3), round(vy, 3), round(wz, 3)]
                    _tp = time.time()                        # night5e：发布间隔自检
                    if PUBGAP["last"]:
                        _gap = _tp - PUBGAP["last"]
                        if _gap > PUBGAP["max"] or _tp - PUBGAP["t_max"] > 60.0:
                            PUBGAP["max"], PUBGAP["t_max"] = _gap, _tp
                        LIVE["pub_gap"] = round(PUBGAP["max"] * 1000.0)
                        if _gap > PUBGAP_WARN_S and _tp - PUBGAP["t_log"] > 5.0:
                            PUBGAP["t_log"] = _tp
                            logline("!! 指令发布间隔 %.0f ms（控制台卡顿；>500 ms 时 runner 会撤专家）" % (_gap * 1000.0))
                    PUBGAP["last"] = _tp
                    dtp = (now - RVX["t"]) if RVX["t"] else 0.05
                    RVX["est"] = _approach(RVX["est"], vx, RUNNER_SLEW_V * min(dtp, 0.5))
                    RVX["t"] = now
                    RVX["last_pub"] = now
                if STATE["hscan"]:
                    PUB["hscan"].publish(hs)

        # 遥测 + 自动急停
        if now - last_rate >= 0.5:
            dt = now - last_rate
            for k in cnt:
                LIVE[k + "_hz"] = round((cnt[k] - prev[k]) / dt, 1)
            prev = dict(cnt)
            last_rate = now

            j = box["j"]
            if j:
                jd = j.data.joints_data
                LIVE["joints"] = [
                    dict(n=NAMES[k],
                         # 轮子是连续旋转关节，"角度"没有意义（会累计到上万度），改显示圈数
                         deg=("%.2f 圈" % (jd[k].position / (2 * math.pi))) if k % 4 == 3
                             else round(math.degrees(jd[k].position * DIRS[k]) + OFFS[k], 1),
                         raw=round(jd[k].position, 4),
                         vel=round(jd[k].velocity, 2),
                         tau=round(jd[k].torque, 2),
                         tm=round(jd[k].motion_temp, 1), td=round(jd[k].driver_temp, 1),
                         sw=jd[k].status_word)
                    for k in range(16)]
                LIVE["tau_max"] = round(max(abs(x.torque) for x in jd), 2)
                LIVE["wheel_rad_s"] = [round(abs(jd[k].velocity), 2) for k in (3, 7, 11, 15)]
                LIVE["wheel_m_s"] = round(
                    sum(abs(jd[k].velocity) for k in (3, 7, 11, 15)) / 4 * R_WHEEL, 2)
                LIVE["frame_id"] = j.header.frame_id
            i = box["i"]
            if i:
                d = i.data
                LIVE["imu"] = dict(roll=round(math.degrees(d.roll), 1),
                                   pitch=round(math.degrees(d.pitch), 1),
                                   yaw=round(math.degrees(d.yaw), 1),
                                   acc_z=round(d.acc_z, 2))
                over = (abs(d.roll) > math.radians(TILT_ABORT_ROLL_DEG) or
                        abs(d.pitch) > math.radians(TILT_ABORT_PITCH_DEG))
                TILTN["n"] = TILTN["n"] + 1 if over else 0
                # 只有在我们自己在控狗时才因侧倾急停。别人开着狗上台阶时 pitch 必然
                # 超 15 度，那时我们发阻尼就是在打断对方，绝对不行。
                if TILTN["n"] >= 3 and not STATE["estop"] and may_publish():
                    do_estop("自动: 倾角 roll=%.1f pitch=%.1f 度，超过急停阈值（roll %.0f° / pitch %.0f°）"
                             % (math.degrees(d.roll), math.degrees(d.pitch),
                                TILT_ABORT_ROLL_DEG, TILT_ABORT_PITCH_DEG))
            o = box["o"]
            if o:
                p, v = o.pose.pose.position, o.twist.twist.linear
                LIVE["odom"] = dict(x=round(p.x, 3), y=round(p.y, 3), z=round(p.z, 3),
                                    vx=round(v.x, 3), vy=round(v.y, 3), vz=round(v.z, 3),
                                    frame=o.header.frame_id)
            b = box["b"]
            if b and len(b.data):
                e = b.data[0]
                LIVE["batt"] = dict(v=round(e.voltage / 100.0, 2), a=round(e.current / 100.0, 2),
                                    soc=e.battery_level)
            c = box["c"]
            if c and (LIVE.get("c_hz") or 0) > 0:
                k0 = deserialize_message(c, JointsDataCmd).data.joints_data[0]   # 原始字节，2 Hz 才解这一帧
                LIVE["cmd_in"] = dict(cw=k0.control_word, kp=round(k0.kp, 1), kd=round(k0.kd, 2))
            elif (LIVE.get("c_hz") or 0) <= 0:
                LIVE["cmd_in"] = None      # 没人在发就别显示上一轮的残值
            s = box["s"]
            if s:
                LIVE["steer"] = dict(x=round(s.data.x, 3), y=round(s.data.y, 3),
                                     yaw=round(s.data.yaw, 3))
            LIVE["nodes"] = []
            try:
                LIVE["nodes"] = [a + b_ for b_, a in n.get_node_names_and_namespaces()]
            except Exception:
                pass
            nn = LIVE.get("nodes", [])
            LIVE["runner_online"] = any("rl_deploy" in x for x in nn)
            # runner 的 /cmd_vel /robot_mode /climb_mode /height_scan 订阅不在 rl_deploy
            # 节点上，而在它另起的 s10_user_command 节点上；而且这个节点要等
            # S10Interface::Start() 收到关节反馈之后才建。所以它在线 = 指令通路真的通了。
            LIVE["cmd_iface"] = any("s10_user_command" in x for x in nn)
            h = box["h"]
            LIVE["hs_len"] = len(h.data) if h is not None else None
            hm = hs_metrics(list(h.data)) if h is not None else None
            if hm is not None:
                hr, hwhy = _hole_rate(hm)   # 09-12 凌晨：/hmap_info 缺失/过期/非法 → None（质量未知），不再退回补后的 0
                hm["hole"] = None if hr is None else round(hr, 3)
                hm["hole_why"] = hwhy
                uc, _ = under_valid_ratio()
                hm["under_valid"] = None if uc is None else round(uc, 3)
            LIVE["hs"] = hm
            try:                                       # night4f：前方地形判读，优先补洞前的图
                _now = time.time()
                _raw = HSRAW["m"] if (HSRAW["m"] is not None and _now - HSRAW["t"] <= HQ_INFO_MAX_AGE) else None
                _tr = terrain_front(list(_raw)) if _raw is not None else None
                _src = "raw"
                if (_tr is None or _tr.get("rows", 0) < TERR_ROWS_MIN) and h is not None:   # 原始图前排有洞 → 退回补洞后的图
                    _tf = terrain_front(list(h.data))
                    if _tf is not None and _tf.get("rows", 0) >= TERR_ROWS_MIN:
                        _tr, _src = _tf, "filled"
                    elif _tr is None:
                        _tr, _src = _tf, "filled"
                if _tr is not None:
                    _tr["src"] = _src
                    if _tr["kind"] != "平地":
                        TERR_HOLD.update(t=_now, tr=dict(_tr))
                    elif TERR_HOLD["tr"] is not None and _now - TERR_HOLD["t"] <= TERR_HOLD_S:   # 判到过台阶 → 保持 1 s 不闪
                        _tr = dict(TERR_HOLD["tr"]); _tr["held"] = True
                if (_tr and not _tr.get("held") and _tr.get("kind") in ("楼梯", "台面") and _tr.get("yaw_off") is not None
                        and _tr.get("rows", 0) >= SQUARE_ROWS_MIN and _tr.get("dist_l") is not None and _tr.get("dist_r") is not None):
                    SQ_HIST.append((_now, float(_tr["yaw_off"])))
                    del SQ_HIST[:-200]
                if _tr is not None:
                    _est, _ = square_estimate(_now)
                    _tr["yaw_med"] = None if _est is None else round(_est, 1)
                LIVE["terrain"] = _tr
            except Exception as _e:
                LIVE["terrain"] = None
            try:                                       # night4m：爬楼中按台沿复校航向目标（每 1.5 s，每次 ≤5°）
                if SQUARE_AUTO and STATE["climb"] == 2 and time.time() - SQUARE["t"] >= SQUARE_PERIOD_S:
                    _tr2 = LIVE.get("terrain"); _r2 = IMU_NOW.get("roll"); _y2 = imu_yaw_fresh()[0]
                    with LK:
                        _twz = TGT["wz"]; _ref = STATE.get("stair_yaw_ref")
                    _est2 = square_estimate()[0]
                    if (_tr2 and not _tr2.get("held") and _tr2.get("kind") in ("楼梯", "台面") and _est2 is not None
                            and _y2 is not None and _ref is not None and abs(_twz) <= 0.01
                            and _r2 is not None and math.isfinite(_r2) and abs(_r2) <= math.radians(ROLL_FREEZE_DEG)):
                        _tgt = _wrap(_y2 + math.radians(max(-SQUARE_MAX_DEG, min(SQUARE_MAX_DEG, float(_est2)))))
                        _d = _wrap(_tgt - _ref); _lim = math.radians(SQUARE_STEP_DEG)
                        _new = _wrap(_ref + max(-_lim, min(_lim, _d)))
                        with LK:
                            if STATE["climb"] == 2 and STATE.get("stair_yaw_ref") == _ref:
                                STATE["stair_yaw_ref"] = _new
                        SQUARE["t"] = time.time()
                        if abs(_d) > math.radians(1.0):
                            logline("楼梯复校航向：台沿偏 %+.1f°（2 s 中位），目标 %.1f° → %.1f°" % (_est2, math.degrees(_ref), math.degrees(_new)))
            except Exception:
                pass
            # hs_ok = 闸门判据（姿态无关）。绝对值偏离 -0.08 **不再进这个判断** ——
            # 见 hscan_gate 的说明：那会在狗正确站上台阶时误伤。
            LIVE["hs_ok"] = None if hm is None else (
                not hm["bad"]
                and hm["hole"] is not None and hm["hole"] <= HS_HOLE_MAX
                and (hm["wall_frac"] is None or hm["wall_frac"] <= HS_WALL_FRAC))
            # hs_flat = 平地标定参考。只在"狗确实站在平地上"时才有意义，
            # 所以它只用于显示和 prep_run.sh 的标定，不用于闸门。
            if len(hs_med) >= 20:
                ss = sorted(hs_med)
                b = ss[len(ss) // 2]
                LIVE["hs_base"] = round(b, 4)
                LIVE["hs_flat"] = abs(b - HS_TARGET) <= HS_TOL
                LIVE["hs_need_z"] = round(HS_TARGET - b, 4)
            else:
                LIVE["hs_base"] = None
                LIVE["hs_flat"] = None
                LIVE["hs_need_z"] = None
            chz = LIVE.get("c_hz") or 0
            if STATE["runner_pid"]:
                LIVE["bus_owner"] = "我们"
            elif chz > 20:
                LIVE["bus_owner"] = "别人"
            elif LIVE.get("runner_online") and (LIVE.get("c_silent_s") or 0) > 30:
                # 节点在、但很久没发指令 —— 多半是进程没干净退出留下的 DDS 残留
                LIVE["bus_owner"] = "残留"
            else:
                LIVE["bus_owner"] = "无人"
            if STATE["runner_pid"] and not runner_alive():
                logline("!! runner pid=%s 已经不在了（自己崩的？看 rl_deploy.log）"
                        % STATE["runner_pid"])
                STATE["runner_pid"] = None
                PROC["p"] = None
                STATE["runner_max"] = None
                STATE["running_group"] = None
                STATE["running_climb"] = None
                STATE["running_pit"] = None
                STATE["stair_heading"] = None   # 09-11 夜审：runner 崩溃，楼梯朝向作废
                STATE["pit_loaded"] = False
                STATE["mode"] = 0
                STATE["mode_src"] = "runner 已停"
                with LK:
                    _expert_clear(None)
                try:                      # nightreview-A：进程没了也补发 /climb_mode 0（幂等），与其他退出路径一致
                    if may_publish():
                        u0 = UInt8(); u0.data = 0
                        PUB["climb"].publish(u0)
                except Exception:
                    pass
            LIVE["may_publish"] = may_publish()
            LIVE["may_command"] = may_command()
            LIVE["rec"] = rec_status()
            LIVE["j_is_mirror"] = True     # j_hz 来自 10Hz 镜像，不是全速流
            # /JOINTS_CMD 静默多久了。用来区分"对方真在控狗"和
            # "上次没退干净留下的 DDS 残留节点"。
            LIVE["c_silent_s"] = (round(now - seen["c"], 0) if seen["c"]
                                  else round(now - t_start, 0))
            LIVE["full"] = {"j": FULL["j"], "c": FULL["c"], "i": FULL["i"],
                            "age": round(now - FULL["t"], 0) if FULL["t"] else None}
            LIVE["running_policy"] = STATE["running_policy"]
            LIVE["effective_policy"] = effective_policy()
            ii = box["i"]
            LIVE["expert"] = expert_tick(now, ii.data.roll if ii else None, ii.data.pitch if ii else None,
                                         (now - seen["i"]) if seen["i"] else None)
            LIVE["climb"] = STATE["climb"]
            gok, gwhy = climb_gate()
            LIVE["climb_gate"] = {"ok": gok, "why": gwhy}
            sok, swhy = stair_gate()
            LIVE["stair_gate"] = {"ok": sok, "why": swhy}
            hq, hqw = hmap_quality()          # 09-12 凌晨：切专家闸门用的高程图质量。页面显示；专家途中变差只报警
            LIVE["hmap_q"] = {"q": hq, "why": hqw}
            if STATE["climb"] in (1, 2):
                if hq != "ok" and HQW["q"] == "ok":
                    logline("!! %s途中高程图%s：%s —— 只报警，不自动动作（要不要停由操作员决定）"
                            % ("楼梯专家" if STATE["climb"] == 2 else "上墙专家",
                               "质量未知" if hq == "unknown" else "不可信", hqw))
                elif hq == "ok" and HQW["q"] != "ok":
                    logline("专家途中高程图恢复正常")
                HQW["q"] = hq
            else:
                HQW["q"] = "ok"
            LIVE["pit_loaded"] = bool(STATE.get("pit_loaded"))
            LIVE["stair_k"] = STATE.get("stair_k")
            LIVE["segment"] = STATE.get("segment")        # night4b
            LIVE["seg_deny"] = SEG["deny"]
            LIVE["stair_heading"] = (round(math.degrees(STATE["stair_heading"]), 1)
                                     if STATE.get("stair_heading") is not None else None)   # 09-11 夜审
            LIVE["rvx_est"] = round(RVX["est"], 2)
            LIVE["running_group"] = STATE.get("running_group")
            ha = LIVE.get("hinfo")
            LIVE["cloud_age"] = ha[3] if ha and len(ha) > 3 else None
            LIVE["hinfo_age"] = round(now - LIVE["hinfo_t"], 1) if LIVE.get("hinfo_t") else None
            LIVE["policy_mismatch"] = bool(
                STATE["runner_pid"] and STATE["policy"] != effective_policy())
            LIVE["j_age"] = round(now - seen["j"], 1) if seen["j"] else None
            lo, hi, vyl, wzl = disp_limits()
            rmx = (STATE["runner_max"] or {}).get("vx")
            LIVE["eff"] = {"vx_lo": round(lo, 2), "vx_hi": round(hi, 2),
                           "vy": round(vyl, 2), "wz": round(wzl, 2),
                           # runner 那道硬上限只有在比滑块还低时才会真正咬住
                           "capped_by_runner": bool(rmx is not None and rmx < hi - 1e-6),
                           "runner_hard_vx": rmx}
            # 复刻 joint_damping_state.hpp:46 —— 阻尼满 3 秒，状态机自己回 kIdle。
            # 不模拟这一步，界面就会永远停在"2 阻尼"，把合法的起立也拦掉。
            if STATE["mode"] == 2 and STATE["damp_t"] and \
                    now - STATE["damp_t"] > DAMP_AUTO_IDLE_S:
                STATE["mode"] = 0
                STATE["mode_src"] = "阻尼超时自动回 0"
                STATE["damp_t"] = 0.0
                logline("阻尼已满 %.0f 秒，状态机自动回到 0 待命，现在可以起立"
                        % DAMP_AUTO_IDLE_S)
    rclpy.shutdown()


REJ = re.compile(r"\[MODE\]\s*拒绝切到\s*(\d+)[^\d]+当前状态\s*(\d+)")
ACCEPT = {"Joint Damping": 2, "Standing Up": 1, "RL Control": 6, "Lie Down": 4}


def on_rosout(m):
    """只收 runner 那两个节点的日志，其余（我们自己、雷达、别队）全丢。"""
    if m.name not in ("rl_deploy", "s10_user_command"):
        return
    txt = m.msg.strip()
    if not (txt.startswith("[MODE]") or txt.startswith("[CLIMB]")
            or "ROS 指令接口已启动" in txt or "策略" in txt):
        return
    STATE["rosout"].insert(0, "%s  %s" % (time.strftime("%H:%M:%S"), txt))
    del STATE["rosout"][30:]
    if txt.startswith("[CLIMB]"):
        STATE["climb_ack"] = txt[:160]
        # runner 0911b：/cmd_vel 超时撤专家并锁存，要重发 /climb_mode 1 才回专家。本地记账跟着清掉，
        # 并锁零到松键 —— 不能让页面以为还在专家、也不能指令一恢复就接着冲。
        if "指令超时" in txt and "撤专家" in txt:
            with LK:
                dur = _expert_clear(0.0)
            if dur is not None:
                logline("runner 超时撤专家（锁存）→ 本地同步 climb=0，锁零到松键。专家共 %.1f s" % dur)
                obs_event("climb=0 runner 超时撤专家")
        # 0911f：楼梯槽没加载时收到 /climb_mode 2，runner 留在主策略并告警 → 立刻退出（发 /climb_mode 0）
        if "没加载" in txt and ("楼梯" in txt or "坑内" in txt):
            STATE["pit_loaded"] = False
            if STATE["climb"] == 2:
                climb_exit("runner 说楼梯策略没加载（0911f）", 0.0)
                logline("!! runner 说楼梯策略没加载、留在主策略 → 已发 /climb_mode 0，锁零到松键")
        return

    mo = REJ.search(txt)
    if mo:
        # runner 亲口报出了它的真实当前状态 —— 拿它覆盖我们的假设
        real = int(mo.group(2))
        if STATE["mode"] != real:
            logline("状态跟踪纠正: 我方以为 %d，runner 说实际是 %d" % (STATE["mode"], real))
        STATE["mode"] = real
        STATE["mode_src"] = "runner 实测"
        return
    for k, v in ACCEPT.items():
        if k in txt:
            STATE["mode"] = v
            STATE["mode_src"] = "runner 确认"
            if v == 2:
                STATE["damp_t"] = time.time()
            break


def do_estop(reason):
    with LK:
        STATE["estop"] = True
        STATE["estop_reason"] = reason
        TGT.update(vx=0.0, vy=0.0, wz=0.0, t=0.0)
        was_expert = _expert_clear(0.0) is not None
    if SEG_FULL or seg_full_now():
        _seg_reset("急停")                  # night4s：急停后赛段回平地，解除急停不会自动再切专家
    if not may_publish():
        logline("急停(仅本地) — %s ；未接管，不向总线发任何话题" % reason)
        return False
    try:
        u0 = UInt8(); u0.data = 0             # 先退专家（R6b）：ClimbH 吃零指令会原地转
        PUB["climb"].publish(u0)
        if was_expert:
            obs_event("climb=0 急停：%s" % reason)
        t = Twist()
        for _ in range(5):
            PUB["cmd_vel"].publish(t)
        u = UInt8(); u.data = 2               # 契约: /robot_mode=2 阻尼（主通道）
        PUB["mode"].publish(u)
        STATE["mode"] = 2
        STATE["damp_t"] = time.time()
        STATE["mode_src"] = "假设"
        m = String(); m.data = "G20_KEY_R2"   # 兼容厂家原版 runner
        PUB["key"].publish(m)
    except Exception as e:
        logline("急停发布异常: %r" % e)
        return False
    logline("!!! 急停 — " + reason)
    # night4i（09-12 18:5x，作者："控制台的急停失效了"）：阻尼指令不只发一次——3 s 内每 0.2 s 重发 /robot_mode 2 + 零速，
    # 直到 runner 亲口确认已进阻尼（mode_src=="runner 确认" 且 mode==2）。runner 退 RL 要等线程 join，第一条可能被拖住；
    # 重发不改变语义（阻尼是幂等的），只是别让一次发布决定生死。硬件急停（手柄/HES）仍是唯一独立于 runner 的保险。
    def _estop_repeat():
        for _ in range(15):
            time.sleep(0.2)
            if not STATE["estop"]:
                return
            if STATE.get("mode") == 2 and STATE.get("mode_src") == "runner 确认":
                return
            if not may_publish():
                return
            try:
                u = UInt8(); u.data = 2
                PUB["mode"].publish(u)
                PUB["cmd_vel"].publish(Twist())
            except Exception:
                return
        logline("!! 急停发出 3 s 后 runner 仍未确认进入阻尼 —— 用手柄急停 / HES")
    threading.Thread(target=_estop_repeat, daemon=True).start()
    return True


# ----------------------------------------------------------------- runner
def runner_alive():
    """runner 自己崩溃时 STATE["runner_pid"] 还留着旧值，会让"谁在控狗"一直显示
    绿色的"我们"，may_publish 也一直开着。每轮遥测探一次活。"""
    pid = STATE["runner_pid"]
    if not pid:
        return False
    pr = PROC.get("p")
    if pr is not None and getattr(pr, "pid", None) == pid:
        try:
            if pr.poll() is not None:     # nightreview-A：自己起的 runner 已退出（poll 顺手回收僵尸）
                return False
        except Exception:
            pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                  # 接管来的 runner 不是我们的子进程、回收不了：僵尸也算没了
        with open("/proc/%d/stat" % pid) as f:
            if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                return False
    except Exception:
        pass
    return True


def stray_runners():
    """按可执行文件找我们自己的 rl_deploy 进程：/proc/<pid>/exe 的真实路径 == EXE_PATH。
    控制台一重启，STATE['runner_pid'] 就没了，但进程还在往真机总线发指令 ——
    2026-09-10 就是这么留下一个孤儿在 domain 0 上跑了两分钟。
    night4a：原来用 pgrep -f <路径> 按命令行文本匹配，命令行里恰好带这条路径的 ssh/bash 命令也会被当成 runner，
    不但拒绝起 runner，收尾时还把它 SIGINT/KILL（09-12 13:39、13:49 两次）。别的用户的进程读不到 exe，跳过。"""
    try:
        target = os.path.realpath(EXE_PATH)
    except Exception:
        return []
    out = []
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        p = int(d)
        if p == me or p == STATE.get("runner_pid"):
            continue
        try:
            exe = os.readlink("/proc/%d/exe" % p)
        except OSError:
            continue
        if exe.endswith(" (deleted)"):           # 二进制重编过、旧进程还在跑
            exe = exe[:-len(" (deleted)")]
        if exe == target:
            out.append(p)
    return out


def read_proc_env(pid):
    """从 /proc/<pid>/environ 读回 runner 启动时的环境变量。
    控制台重启后我们不再有那个 Popen，但内核还留着它启动时的 environ ——
    这是唯一能查出"这个 runner 是用多大速度上限起的"的办法。"""
    try:
        with open("/proc/%d/environ" % pid, "rb") as f:
            raw = f.read().decode("utf-8", "replace")
        d = {}
        for kv in raw.split("\0"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                d[k] = v
        return d
    except Exception:
        return {}


def adopt_limits(pid):
    e = read_proc_env(pid)

    def f(k, dft):
        try:
            return float(e.get(k, dft))
        except ValueError:
            return dft

    if not e:
        return None
    pol = os.path.basename(os.path.dirname(e.get("S10_POLICY_PATH", "")))
    return {"vx": f("S10_CMD_MAX_VX", 1.0), "vy": f("S10_CMD_MAX_VY", 0.6),
            "wz": f("S10_CMD_MAX_WZ", 1.0), "policy": pol or "未知"}


def sweep_strays():
    """控制台启动时【接管】遗留的 runner，而不是杀掉它。

    杀掉是错的：控制台改一行代码就要重启，重启把用户正跑着的 runner 打死，
    狗当场失去控制器。接管同样解决"孤儿在总线上没人管"的问题，且不打断现场。
    真要停它，界面上有「停止 runner」。"""
    s = stray_runners()
    if not s:
        return
    STATE["runner_pid"] = s[0]
    PROC["p"] = None                  # 不是我们 fork 的，只能靠 os.kill 探活
    lim = adopt_limits(s[0])
    if lim:
        STATE["runner_max"] = lim
        STATE["max_vx"] = min(STATE["max_vx"], lim["vx"])
        STATE["max_wz"] = min(STATE["max_wz"], lim["wz"])
        if lim["policy"] in POLICIES:
            STATE["policy"] = lim["policy"]
            STATE["running_policy"] = lim["policy"]
        logline("接管遗留 runner pid=%d —— 它启动时的上限 vx=%.2f wz=%.2f，策略 %s"
                % (s[0], lim["vx"], lim["wz"], lim["policy"]))
    else:
        logline("接管了控制台重启前遗留的 runner pid=%d（读不到它的启动参数）" % s[0])
    if len(s) > 1:
        logline("!! 还有多个 rl_deploy 在跑 %s —— 它们会抢 /JOINTS_CMD，请停掉多余的" % s[1:])


def hs_metrics(data):
    """从一帧 /height_scan 算出**姿态无关**的健康指标。

    空洞的识别：elevation_map.sample() 把没扫到的格子填成"与脚下同高"
    （filled = base_z - OFFSET），于是 obs 恰好等于 0.0。真地形给出精确 0.0
    的概率可以忽略，所以 |obs| < 1e-9 就是空洞。空洞不参与任何判据。
    """
    if data is None or len(data) != 187:
        return None
    hole = [abs(v) < 1e-9 for v in data]
    bad = any((v != v) or v == float("inf") or v == float("-inf") for v in data)
    out = dict(bad=bad, hole=round(sum(hole) / 187.0, 3), base=None,
               wall_frac=None, step_d=None, step_h=None, fwd_max=None)
    if _HS_X is None or bad:
        return out
    und = [data[i] for i in range(187) if abs(_HS_X[i]) <= HS_UNDER and not hole[i]]
    if not und:
        return out                       # 身下全空洞，给不出基准（趴着时会这样）
    und.sort()
    base = und[len(und) // 2]
    out["base"] = round(base, 4)
    # rel = 地形比身下高多少米。正 = 更高。这是姿态无关的量。
    rel = [None if hole[i] else (base - data[i]) for i in range(187)]
    seen = [r for r in rel if r is not None]
    if seen:
        out["wall_frac"] = round(sum(1 for r in seen if r > HS_WALL_REL) / float(len(seen)), 3)
    # 前方中轴（|y|<=0.10）逐行剖面：找最近的抬高
    fwd_max = None
    for x in sorted(set(round(v, 2) for v in _HS_X if v > 0.12)):
        vs = [rel[i] for i in range(187)
              if abs(_HS_X[i] - x) < 0.01 and abs(_HS_Y[i]) <= 0.10 and rel[i] is not None]
        if not vs:
            continue
        vs.sort()
        m = vs[len(vs) // 2]
        fwd_max = m if fwd_max is None else max(fwd_max, m)
        if m >= HS_STEP_REL and out["step_d"] is None:
            out["step_d"], out["step_h"] = x, round(m, 3)
    out["fwd_max"] = None if fwd_max is None else round(fwd_max, 3)
    return out


def terrain_front(data):
    """night4f：前方地形判读（只显示不动作）。data = 一帧 187 维（优先 /height_scan_raw）。
    返回 dict(kind, dist, dist_l, dist_r, h, square, rows) 或 None。距离 = 抬高沿离机体中心（m）。"""
    if data is None or len(data) != 187 or _HS_X is None:
        return None
    hole = [abs(v) < 1e-9 for v in data]
    if any((v != v) for v in data):
        return None
    und = [data[i] for i in range(187) if abs(_HS_X[i]) <= HS_UNDER and not hole[i]]
    if not und:
        return None
    und.sort(); base = und[len(und) // 2]
    rel = [None if hole[i] else (base - data[i]) for i in range(187)]
    ys = sorted(set(round(v, 2) for v in _HS_Y)); xs = sorted(set(round(v, 2) for v in _HS_X if v >= TERR_X_MIN - 1e-6))
    rows = []
    for y in ys:
        found = None
        for x in xs:
            cells = [rel[i] for i in range(187) if abs(_HS_X[i] - x) < 0.01 and abs(_HS_Y[i] - y) < 0.01 and rel[i] is not None]
            if cells and cells[0] >= TERR_STEP_MIN:
                found = (x - 0.05, cells[0]); break
        rows.append((y, found))
    seen = [(y, f) for y, f in rows if f is not None]
    out = dict(kind="平地", dist=None, dist_l=None, dist_r=None, h=None, square=None, rows=len(seen), yaw_off=None)
    if len(seen) < TERR_ROWS_MIN:
        return out
    ds = sorted(f[0] for _, f in seen); hs_ = sorted(f[1] for _, f in seen)
    out["dist"] = round(ds[len(ds) // 2], 2); out["h"] = round(hs_[len(hs_) // 2], 3)
    L = sorted(f[0] for y, f in seen if y >= 0.15 - 1e-6); R = sorted(f[0] for y, f in seen if y <= -0.15 + 1e-6)
    if L: out["dist_l"] = round(L[len(L) // 2], 2)
    if R: out["dist_r"] = round(R[len(R) // 2], 2)
    if L and R:
        out["square"] = abs(out["dist_l"] - out["dist_r"]) <= TERR_SQUARE_M + 1e-9
    # night4m：台沿走向最小二乘拟合 x = a + s·y → 垂直台沿方向相对机头的偏角；>0 = 台沿法线在左、狗要左转
    _ys = [y for y, _ in seen]; _xs = [f[0] for _, f in seen]
    _my = sum(_ys) / len(_ys); _mx = sum(_xs) / len(_xs)
    _den = sum((y - _my) ** 2 for y in _ys)
    if _den > 1e-9:
        _sl = sum((y - _my) * (x - _mx) for y, x in zip(_ys, _xs)) / _den
        out["yaw_off"] = round(math.degrees(-math.atan(_sl)), 1)
    h = out["h"]
    out["kind"] = "障碍" if h >= TERR_WALL_MIN else ("台面" if h >= TERR_PLAT_MIN else "楼梯")
    return out


def _hole_rate(hm):
    """返回 (空洞率 或 None, 原因)。09-12 凌晨改（Astra 建议 2）。
    hmap v2 起 /height_scan 里的空洞已被补掉，数 0 数不出来；真值只在 /hmap_info[0]（补洞之前的空洞率，同一 tick 发）。
    /hmap_info 没收到、过期、非法 → (None, 原因) = 质量未知。**不再退回数补后的 0**
    （旧写法：/hmap_info 过期后返回 0%，原始空洞率 85% 也显示成 0%，闸门放行）。"""
    ha, t = LIVE.get("hinfo"), LIVE.get("hinfo_t")
    if not ha or not t:
        return None, "没收到 /hmap_info"
    age = time.time() - t
    if age > HQ_INFO_MAX_AGE:
        return None, "/hmap_info 已 %.1f s 没更新" % age
    try:
        h0 = float(ha[0])
    except (TypeError, ValueError, IndexError):
        return None, "/hmap_info 数据非法"
    if not (math.isfinite(h0) and 0.0 <= h0 <= 1.0):
        return None, "/hmap_info 空洞率非法（%r）" % (ha[0],)
    h = hm.get("hole") if hm else None
    return max(h0, h if isinstance(h, float) else 0.0), ""


def under_valid_ratio():
    """机身下方 77 格（|x|<=HQ_UNDER_X）里"补洞前有效"的比例，取自 /height_scan_raw。
    注意口径：/height_scan_raw 是 v2 插值之前的图，但已经过 ElevationMap.sample() 的 10 cm 邻域中位数补值，
    所以"有效"= 直接观测 + 邻域补值，不是纯直接观测（hmap 没导出逐格来源标记）。
    返回 (比例 或 None, 原因)：没收到、过期、维数不对、有 NaN、拿不到格心定义 → None。"""
    m, t = HSRAW["m"], HSRAW["t"]
    if m is None or not t:
        return None, "没收到 /height_scan_raw"
    age = time.time() - t
    if age > HQ_INFO_MAX_AGE:
        return None, "/height_scan_raw 已 %.1f s 没更新" % age
    if _HS_X is None:
        return None, "取不到格心定义 heightmap.CELL_XY"
    if len(m) != 187:
        return None, "/height_scan_raw 维数 %d（要 187）" % len(m)
    idx = [i for i in range(187) if abs(_HS_X[i]) <= HQ_UNDER_X + 1e-6]
    good = 0
    for i in idx:
        v = float(m[i])
        if not math.isfinite(v):
            return None, "/height_scan_raw 有 NaN/inf"
        if abs(v) >= 1e-9:
            good += 1
    return good / float(len(idx)), ""


def hmap_quality():
    """切专家前的高程图质量（09-12 凌晨，Astra 建议 1）。返回 (q, 原因)：
    q = "ok" / "bad"（确凿坏了）/ "unknown"（缺失、过期、非法）。
    和 hscan_gate 的分工：hscan_gate 管"起 runner / 进 RL"，没有高程图时放行（runner 按平地兜底）；
    专家（楼梯 / ClimbH）是冲着地形去的，**没有高程图、质量未知都不许切**。每次调用都用最新一帧现算。"""
    if STATE["hscan"]:
        return "bad", "我方正在发 /height_scan 零填充"
    now = time.time()
    if not HSRX["t"] or HSRX["m"] is None:
        return "unknown", "从没收到 /height_scan"
    if now - HSRX["t"] > HQ_HS_MAX_AGE:
        return "unknown", "/height_scan 已 %.2f s 没更新（hmap 停发 = 点云或 KISS 位姿断了）" % (now - HSRX["t"])
    try:
        hm = hs_metrics(list(HSRX["m"]))
    except (TypeError, ValueError):
        hm = None
    if hm is None:
        return "unknown", "/height_scan 维数不对"
    if hm["bad"]:
        return "bad", "/height_scan 里有 NaN/inf"
    hr, hwhy = _hole_rate(hm)
    if hr is None:
        return "unknown", "空洞率未知（%s）" % hwhy
    if hr > HS_HOLE_MAX:
        return "bad", "补洞前空洞率 %.0f%%（上限 %.0f%%）" % (100 * hr, 100 * HS_HOLE_MAX)
    ha = LIVE.get("hinfo") or []
    try:
        cage = float(ha[3])
    except (TypeError, ValueError, IndexError):
        cage = float("nan")
    if not math.isfinite(cage):
        return "unknown", "/hmap_info 缺点云年龄字段"
    if cage > HMAP_AGE_MAX:
        return "bad", "点云年龄 %.2f s（上限 %.2f s），hmap 在用旧点云" % (cage, HMAP_AGE_MAX)
    wf = hm["wall_frac"]
    if wf is not None and wf > HS_WALL_FRAC:
        return "bad", "%.0f%% 的格子比身下高 %.2f m 以上（地图中毒）" % (100 * wf, HS_WALL_REL)
    uc, uwhy = under_valid_ratio()
    if uc is None:
        return "unknown", "机身下方有效率未知（%s）" % uwhy
    if uc < HQ_UNDER_MIN:
        return "bad", ("机身下方 77 格里补洞前有效（含邻域补值）的只有 %.0f%%（下限 %.0f%%），身下主要靠插值猜"
                       % (100 * uc, 100 * HQ_UNDER_MIN))
    return "ok", ""


def hscan_gate():
    """高程图可不可信。返回 (ok, 原因)。**只拦确凿坏了的，不拦"和平地不一样"。**

    没有 /height_scan 时返回 ok —— runner 侧 hscan_max_age_ms_=100，超期就走
    SetHeightScanFallback()（按平地走，注释明确说不是 0），实测是安全降级路径。

    【为什么不能拿绝对值当判据 —— 2026-09-11 的教训】
    obs = base_z - terrain_z - OFFSET，是"机体离地高度减常数"。狗一蹲、一趴、
    一只脚踏上 33cm 石笼，**整张图就整体平移**：实测趴下时全图整体 -0.33。
    旧版判 |中位数 - (-0.08)| <= 0.015，会在狗**正确地站在台阶上**时把
    「起 runner」和「切 mode 6」一起拒掉。那是把测试堵死，不是保护。

    所以判据全部相对**身下那几格的中位数**（hs_metrics 里算的 rel），姿态无关。"""
    hz = LIVE.get("h_hz") or 0
    if hz <= 1:
        return True, ""                       # 没在发，runner 用平地兜底
    if STATE["hscan"]:
        return False, ("我方正在发 /height_scan 零填充。零在策略眼里不是平地"
                       "（runner 自己的平地兜底值是 -0.10），会让它做跨越动作。"
                       "先把「零填充」取消。")
    hm = LIVE.get("hs")
    if hm is None:
        n = LIVE.get("hs_len")
        return False, ("/height_scan 维数是 %s（要 187），或还没收到完整一帧。等几秒再试。"
                       % n)
    if hm["bad"]:
        return False, ("/height_scan 里有 NaN/inf —— 策略吃进去会直接发散。"
                       "先停掉 hmap（runner 会回落到按平地走），再查 hmap.log。")
    if hm["hole"] is None:            # 09-12 凌晨（Astra 建议 2）：不知道空了多少 ≠ 没空
        return False, ("高程图空洞率未知（%s）。hmap 在发 /height_scan，但不知道补洞前真实空了多少，"
                       "不能当成正常。先查 hmap 是否在正常发 /hmap_info（hmap.log）。" % hm.get("hole_why", ""))
    if hm["hole"] > HS_HOLE_MAX:
        return False, ("高程图空洞率 %.0f%%（上限 %.0f%%）—— 六成格子没扫到，等于闭眼。"
                       "先查 /LIDAR/POINTS_MERGED 还在不在："
                       "merger 会死锁（2026-09-11 遇过一次，进程活着但一个 jiffy 不动），"
                       "跑 /home/robot/lidar_watch.sh 或重启 merger。"
                       % (100 * hm["hole"], 100 * HS_HOLE_MAX))
    wf = hm["wall_frac"]
    if wf is not None and wf > HS_WALL_FRAC:
        return False, ("高程图有 %.0f%% 的格子比身下高 %.2f m 以上 —— 连身后也是墙，"
                       "赛道上不可能，这是 max 累积棘轮中毒的签名"
                       "（2026-09-10 夜实测漂到 -0.5681 就是这个样子）。"
                       "跑 /home/robot/hmap_rebase.sh 重建地图（保住 KISS 的世界原点）。"
                       % (100 * wf, HS_WALL_REL))
    return True, ""


# ---- 09-11：runner 的 std::cout/cerr 只写进 rl_deploy.log，不上 /rosout ----
# 把 IMU 相关行和 "!!" 开头的告警搬到页面「runner 判决」栏（前缀 [日志]）。只读、只显示，不参与任何控制。
RLOG_PATH = "/home/robot/rl_deploy.log"
RLOG_KEYS = ("[IMU]", "imu lag warning", "IMU error", "IMU number error", "IMU update overtime",
             "已加载坑内策略", "已加载专家策略", "[周期]", "[policy] 切换到", "terminate called", "what():")   # 09-11


def rlog_pick(line):
    t = line.strip()
    if not t:
        return None
    if t.startswith("!!") or any(k in t for k in RLOG_KEYS):
        return t[:220]
    return None


def rlog_loop():
    pos, tail, last_t, last_e, rep = None, b"", None, None, 0
    while True:
        try:
            try:
                sz = os.path.getsize(RLOG_PATH)
            except OSError:
                sz = 0
            if pos is None:
                pos = sz                      # 控制台启动时从文件尾开始，不翻旧账
            elif sz < pos:
                pos, tail = 0, b""            # 文件被截断或换新，从头读
            if sz > pos:
                with open(RLOG_PATH, "rb") as f:
                    f.seek(pos)
                    data = f.read(min(sz - pos, 1 << 20))
                pos += len(data)
                parts = (tail + data).split(b"\n")
                tail = parts.pop()            # 最后一段可能是半行，留到下次
                if len(tail) > 4096:
                    tail = b""
                for raw in parts:
                    t = rlog_pick(raw.decode("utf-8", "replace"))
                    if t is None:
                        continue
                    if "已加载坑内策略" in t:           # 09-11：楼梯槽确认加载，楼梯按钮才可按
                        STATE["pit_loaded"] = True
                    if "没加载" in t and ("楼梯" in t or "坑内" in t):   # 0911f 告警（若走 stdout）
                        STATE["pit_loaded"] = False
                        if STATE["climb"] == 2:
                            climb_exit("runner 说楼梯策略没加载（0911f）", 0.0)
                            logline("!! runner 说楼梯策略没加载、留在主策略 → 已发 /climb_mode 0，锁零到松键")
                    if "切换到 专家策略" in t and STATE["climb"] == 2:    # 错切到 ClimbH（0911e 的坑）
                        climb_exit("runner 切到了 ClimbH 而不是楼梯专家", 0.0)
                        logline("!! runner 切到了上墙专家 ClimbH 而不是楼梯专家（楼梯槽多半没加载）→ 已发 /climb_mode 0")
                    ro = STATE["rosout"]
                    if t == last_t and ro and ro[0] is last_e:
                        # 同一句连着来（闸门每秒报一次），只更新顶上那条的次数，不刷屏
                        rep += 1
                        last_e = "%s  [日志] %s（×%d）" % (time.strftime("%H:%M:%S"), t, rep)
                        ro[0] = last_e
                    else:
                        rep, last_t = 1, t
                        last_e = "%s  [日志] %s" % (time.strftime("%H:%M:%S"), t)
                        ro.insert(0, last_e)
                        del ro[30:]
        except Exception as e:                # 这条线程只管显示，出错不能拖垮控制台
            try:
                logline("runner 日志转页面出错（只影响显示）: %s" % e)
            except Exception:
                pass
            time.sleep(5.0)
        time.sleep(0.5)


def runner_guard():
    """起 runner = 往真机总线发 /JOINTS_CMD。这是整个控制台唯一真正会动狗的操作，
    闸门必须比其他任何地方都硬。"""
    if not STATE["armed"]:
        return False, ("未接管。先点「接管」—— 接管本身会先检查总线上没有别人在控狗。")
    if STATE["runner_pid"]:
        return False, "我们已经有一个 runner 在跑 (pid=%s)" % STATE["runner_pid"]
    s = stray_runners()
    if s:
        return False, ("已经有我们自己的 rl_deploy 进程 %s 在跑（控制台重启留下的孤儿）。"
                       "先点「停止 runner」清掉，别起第二个。" % s)
    chz = LIVE.get("c_hz") or 0
    if chz > 20:
        return False, ("别人的控制器正在以 %.0f Hz 发 /JOINTS_CMD。"
                       "再起一个就是两套控制器抢同一批关节。" % chz)
    if LIVE.get("runner_online"):
        return False, "总线上有别人的 rl_deploy 节点在线，先确认对方已停。"
    # 注意：LIVE["j_hz"] 是 /JOINTS_DATA_10HZ 镜像的频率，恒等于 10，
    # 拿它跟 150 比会永远拒绝 —— 2026-09-10 就是这么把启动路堵死的。
    ok, why = hscan_gate()
    if not ok:
        return False, why
    jhz = full_hz("j", "/JOINTS_DATA")
    if 50 <= jhz < 150:
        # 这条流存在就说明 SDK 模式是开的；数值偏低是实测时 CPU 忙的欠采样，
        # 不是真的降频（真降频要看自检里的对照）。放行，但记一笔。
        logline("/JOINTS_DATA 实测 %.0f Hz 偏低（标称 200）—— 多半是量的时候机器忙，放行" % jhz)
    elif jhz < 50:
        mir = LIVE.get("j_hz") or 0
        if mir < 3:
            return False, ("/JOINTS_DATA 全速流 %.0f Hz，连 10Hz 镜像也只有 %.0f Hz —— "
                           "机器人可能没上电，或运动主机的 DDS 没起。" % (jhz, mir))
        return False, ("/JOINTS_DATA 全速流只有 %.0f Hz（10Hz 镜像正常，说明机器人活着）"
                       " —— 手柄的 SDK 模式没打开。不打开这条 200Hz 流不会有，"
                       "runner 会卡在「等关节反馈」。" % jhz)
    return True, ""


def _seg_reset(why):
    """night4b：赛段回到平地（runner 停/起时），免得下次按 W 意外切专家。"""
    if STATE.get("segment") != "flat":
        STATE["segment"] = "flat"
        SEG["deny"] = None
        logline("赛段 → 平地（%s）" % why)


def runner_start():
    _seg_reset("起 runner")
    if STATE["runner_pid"]:
        return False, "runner 已在运行"
    exe = EXE_PATH
    if not os.path.exists(exe):
        return False, "找不到 rl_deploy: %s" % exe
    p = POLICIES[STATE["policy"]]
    if p.get("expert"):
        return False, ("%s 是上台面专家，不能当基础策略起 runner（零指令会原地转）。"
                       "选 V1Down 或 V1H 启动，墙前约 1 m 按住 W 再按 C 切专家" % STATE["policy"])
    if p.get("old"):
        # 09-11 作者拍板：旧版策略禁止起 runner（理由见 _OLD）
        return False, "%s %s" % (STATE["policy"], _OLD)
    grp = p.get("stance", "0.42")
    env = dict(os.environ)
    env.pop("S10_STAND_TEST_MS", None)  # formal stand/lie check, native fault guards remain enabled
    # 09-11 晚：0911g 没有 RL_TEST 钩子，不再设 S10_RL_TEST_MS；RL 停顿只告警（作者拍板，不设 S10_RL_STALL_DAMP）
    # 09-11 基础测试模式：不设 S10_RL_TEST_FLAT，用真实高程图
    env["FASTRTPS_DEFAULT_PROFILES_FILE"] = "/home/robot/codex_checks/20260911_1832/diagnostic/fastdds_control_ethernet.xml"
    # 这几个在子进程里必须不生效：S10_CROUCH 会叠加到 S10_DEF_* 上；CLIMB_CLOCK / FWD_CLIP / LAT_CLIP
    # 训练里没有。控制台自身环境里可能带着，所以从字典里删掉（设空串不保险）。
    for k in ("S10_CROUCH", "S10_CLIMB_CLOCK", "S10_HMAP_FWD_CLIP", "S10_HMAP_LAT_CLIP",
              "S10_POLICY_CLIMB", "S10_POLICY_PIT", "S10_OBS_DUMP"):
        env.pop(k, None)
    dump = os.path.join(REC_DIR, "obsdump_%s_%s.csv" % (time.strftime("%Y%m%d_%H%M%S"), STATE["policy"]))
    env.update(STANCE[grp])
    env["S10_IMU_MAX_AGE_MS"] = IMU_MAX_AGE_MS   # 09-11 作者拍板
    env.update({
        "S10_CMD_SOURCE": "ros",
        "S10_KP_LEG": "80", "S10_KD_LEG": "2", "S10_KD_WHEEL": "0.8",
        "S10_AS_HIPX": "0.125", "S10_AS_HIPY": "0.25",
        "S10_AS_KNEE": "0.25", "S10_AS_WHEEL": os.environ.get("S10_AS_WHEEL", "5"),
        "S10_AS_WHEEL_PIT": os.environ.get("S10_AS_WHEEL_PIT", "3"),   # night4k：0911i 起楼梯槽轮子缩放单独给（作者拍板 3；0911g/h 不认这个变量，无害）   # night4g：轮子动作缩放可由环境变量改（训练侧：3 能把 9300 左前轮目标 −65 压到 −32，但全局生效、拖慢主策略；作者定）
        "S10_TAU_GUARD": "1", "S10_TAU_LIM_LEG": "50", "S10_TAU_LIM_WHEEL": "14",
        "S10_POLICY_PATH": os.path.join(POLDIR, p["onnx"]),
        "S10_OBS_DUMP": dump,           # 每趟一个新文件：244 维观测 + 16 维动作，训练侧逐位对照用
        # runner 里的 max_vx_ 只在构造函数读一次、改不了，所以它不该承担
        # "本次想跑多快"这件事 —— 那是滑块的职责，控制台每帧夹。
        # 这里给它策略的【训练范围】，当纯粹的兜底护栏：
        # 控制台真出 bug 发了超范围的值，runner 也不会把策略喂到没见过的指令上。
        "S10_CMD_MAX_VX": str(max(abs(p["vx"][0]), abs(p["vx"][1]))),
        "S10_CMD_MAX_VY": str(p["vy"]),
        "S10_CMD_MAX_WZ": str(p["wz"]),
        "S10_CMD_SLEW_V": "1.0", "S10_CMD_TIMEOUT_MS": "500",
        "ROS_DOMAIN_ID": "0",
    })
    # 专家槽只在 0.42 组装 ClimbH；0.35 蹲姿组不装（默认位形对不上，也就不给切）
    if grp == EXPERT_STANCE:
        env["S10_POLICY_CLIMB"] = os.path.join(POLDIR, POLICIES[EXPERT]["onnx"])
    # 09-11 楼梯专家槽：只在 STAIR_STANCE 组装；onnx 不在或太小（没传完）就不装（楼梯按钮保持置灰）。
    # 坏的 onnx 会让 runner 构造时直接崩、主策略也起不来，所以宁可不装。
    pit = None
    if grp == STAIR_STANCE:
        pp = os.path.join(POLDIR, STAIR_ONNX)
        sz = os.path.getsize(pp) if os.path.exists(pp) else -1
        md5 = None
        if sz >= STAIR_ONNX_MIN_BYTES:
            import hashlib
            with open(pp, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
        if sz >= STAIR_ONNX_MIN_BYTES and md5.startswith(STAIR_ONNX_MD5):
            env["S10_POLICY_PIT"] = pp
            pit = STAIR
        else:
            logline("!! 楼梯策略文件%s：%s —— 这次 runner 不装楼梯槽（坏文件会让 runner 起不来）"
                    % ("不存在" if sz < 0 else ("只有 %d 字节（没传完？）" % sz if sz < STAIR_ONNX_MIN_BYTES
                                                else "md5 %s 与训练侧给的 %s 不符" % (md5, STAIR_ONNX_MD5)), pp))
    log = open("/home/robot/rl_deploy.log", "ab")
    pr = subprocess.Popen([exe], env=env, stdout=log, stderr=log,
                          cwd=WS, start_new_session=True)
    STATE["runner_pid"] = pr.pid
    PROC["p"] = pr
    # S10_CMD_MAX_* 只在 RosCommandInterface 构造函数里读一次
    # （ros_command_interface.hpp:243-245），之后再拖滑块 runner 不认。
    # 把启动时的值记下来，界面上跟当前滑块对照，不一致就警告。
    STATE["runner_max"] = {"vx": max(abs(p["vx"][0]), abs(p["vx"][1])),
                           "vy": p["vy"], "wz": p["wz"], "policy": STATE["policy"]}
    STATE["running_policy"] = STATE["policy"]
    STATE["running_group"] = grp
    STATE["running_climb"] = EXPERT if grp == EXPERT_STANCE else None
    STATE["running_pit"] = pit
    STATE["stair_heading"] = None   # 09-11 夜审：新 runner，楼梯朝向作废
    STATE["pit_loaded"] = False       # 等 runner 日志"已加载坑内策略"
    STATE["mode"] = 0                 # 新 runner 从 kIdle 起步（崩溃后重启不能沿用旧的 6）
    STATE["mode_src"] = "runner 刚启动"
    STATE["obs_dump"] = dump
    with LK:
        _expert_clear(None)
    logline(("启动 runner pid=%d 策略=%s 站姿组 %s（S10_DEF_HIPY=%s KNEE=%s，轮缩放 AS_WHEEL=" + env["S10_AS_WHEEL"] + " 楼梯槽 PIT=" + env["S10_AS_WHEEL_PIT"] + "）专家槽=%s 观测转储 %s")
            % (pr.pid, STATE["policy"], grp, STANCE[grp]["S10_DEF_HIPY"], STANCE[grp]["S10_DEF_KNEE"],
               STATE["running_climb"] or "不装", dump))
    return True, "已启动 pid=%d" % pr.pid


def runner_stop():
    pid = STATE["runner_pid"]
    if not pid:
        s = stray_runners()
        if s:
            for p in s:
                try:
                    os.kill(p, signal.SIGINT)
                except Exception:
                    pass
            time.sleep(3)
            left = stray_runners()
            for p in left:
                try:
                    os.kill(p, signal.SIGKILL)
                except Exception:
                    pass
            logline("清除遗留 runner %s" % s)
            return True, "已清除遗留 runner %s" % s
        return False, "runner 未在运行"
    pr = PROC.get("p")

    def sig(s):
        try:
            os.killpg(os.getpgid(pid), s)
        except Exception:
            try:
                os.kill(pid, s)
            except Exception:
                pass

    # 先 SIGINT 让 rclcpp 正常收摊。这个 SDK 收到 SIGTERM 会 abort 落 core，
    # 而且 HTTP 端口那种"主线程还活着"的半死状态也是 SIGTERM 惹的。
    sig(signal.SIGINT)
    gone = False
    for _ in range(30):
        time.sleep(0.1)
        if pr is not None:
            if pr.poll() is not None:
                gone = True
                break
        else:
            try:
                os.kill(pid, 0)
            except OSError:
                gone = True
                break
    if not gone:
        sig(signal.SIGKILL)
        if pr is not None:
            try:
                pr.wait(timeout=2)
            except Exception:
                pass
        logline("runner 未响应 SIGINT，已 SIGKILL")
    STATE["runner_pid"] = None
    _seg_reset("runner 已停")        # night4b
    PROC["p"] = None
    STATE["runner_max"] = None
    STATE["running_policy"] = None
    STATE["running_group"] = None
    STATE["running_climb"] = None
    STATE["running_pit"] = None
    STATE["stair_heading"] = None   # 09-11 夜审：停 runner，楼梯朝向作废
    STATE["pit_loaded"] = False
    with LK:
        _expert_clear(None)
    STATE["mode"] = 0                 # runner 重启后状态机必回 kIdle
    STATE["mode_src"] = "runner 已停"
    logline("停止 runner pid=%d（%s）" % (pid, "正常退出" if gone else "强杀"))
    return True, "已停止"


# ----------------------------------------------------------------- 录制
_REC_CACHE = {"t": 0.0, "v": None}


def rec_sh(*args, timeout=40):
    try:
        r = subprocess.run([REC_SH] + list(args), capture_output=True,
                           text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return "调用 %s 失败: %r" % (REC_SH, e)


def _rec_parse(out, now):
    st = {"running": out.startswith("RECORDING"), "raw": out,
          "name": None, "size": None, "elapsed": None}
    if st["running"]:
        parts = out.split()
        if len(parts) > 1:
            st["name"] = parts[1]
        for p in parts:
            if p.startswith("size="):
                st["size"] = p[5:]
        t0 = STATE.get("rec_t0") or 0.0
        if not t0 and st["name"]:
            try:                       # 控制台重启过就没有 t0 了，用目录创建时间兜底
                t0 = os.path.getctime(os.path.join(REC_DIR, st["name"]))
            except Exception:
                t0 = 0.0
        st["elapsed"] = round(now - t0, 1) if t0 else None
    return st


def _rec_refresh():
    try:
        out = rec_sh("status", timeout=15).strip()
        _REC_CACHE.update(t=time.time(), v=_rec_parse(out, time.time()))
    finally:
        _REC_CACHE["busy"] = False


def rec_status(force=False):
    """s10_rec.sh status 会 fork + pgrep + du —— night5d：绝不在发布回路里等它。
    force=True（录制开始/停止的 HTTP 处理线程）同步刷；否则缓存过期就丢给后台线程，本次返回旧值。"""
    now = time.time()
    prev = _REC_CACHE["v"]
    ttl = 2.0 if (prev and prev.get("running")) else 15.0
    if force:
        _REC_CACHE["busy"] = True
        _rec_refresh()
        return _REC_CACHE["v"]
    if prev is not None and now - _REC_CACHE["t"] < ttl:
        return prev
    if not _REC_CACHE.get("busy"):
        _REC_CACHE["busy"] = True
        threading.Thread(target=_rec_refresh, daemon=True).start()
    return prev if prev is not None else {"running": False, "raw": "刷新中", "name": None, "size": None, "elapsed": None}

def rec_context(name):
    """把这一条录制当时的测试条件写进 bag 目录，让数据自带说明。
    训练侧拿到 bag 才知道当时跑的是哪份 onnx、上限多少、谁在控。"""
    lo, hi, vyl, wzl = disp_limits()
    ctx = {
        "bag": name,
        "note": STATE.get("rec_note", ""),
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy_selected": STATE["policy"],
        "policy_running_base": STATE["running_policy"],
        "policy_effective": effective_policy(),
        "climb_mode": STATE["climb"],
        "slider_max_vx": STATE["max_vx"], "slider_max_wz": STATE["max_wz"],
        "effective_cmd_range": {"vx": [lo, hi], "vy": vyl, "wz": wzl},
        "runner_pid": STATE["runner_pid"],
        "runner_hard_limits": STATE["runner_max"],
        "armed": STATE["armed"], "mode": STATE["mode"], "mode_src": STATE["mode_src"],
        "battery": LIVE.get("batt"),
        "hz": {k: LIVE.get(k) for k in ("j_hz", "c_hz", "i_hz", "o_hz", "h_hz")},
        "height_scan_source": ("我方零填充" if STATE["hscan"]
                               else ("hmap" if (LIVE.get("h_hz") or 0) > 0 else "无")),
    }
    try:
        d = os.path.join(REC_DIR, name)
        for _ in range(20):            # bag 目录由 ros2 bag 创建，等它出现
            if os.path.isdir(d):
                break
            time.sleep(0.25)
        with open(os.path.join(d, "s10_context.json"), "w", encoding="utf-8") as f:
            json.dump(ctx, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logline("写 s10_context.json 失败: %r" % e)
        return False


def measure_hz(topics, secs=4):
    """真去总线上量一次频率。控制台自己的计数在负载下不可信，
    而"手柄有没有打到 SDK 模式"这个判据必须准 —— 它决定能不能上机。
    自检是手点的，花 6 秒换一个可信数值划算。"""
    parts = []
    for i, t in enumerate(topics):
        # -s INT -k 2：ros2 topic hz 在无数据话题上不理 SIGTERM，普通 timeout 会留孤儿
        # （09-11 在 AGX 上清出 5 个 hz /JOINTS_CMD）。
        parts.append("( timeout -s INT -k 2 %d ros2 topic hz %s 2>/dev/null "
                     "| grep -m1 'average rate' | sed 's|^|%d |' ) &" % (secs, t, i))
    script = ("source /home/robot/s10_env.sh >/dev/null 2>&1; "
              + " ".join(parts) + " wait")
    out = {t: None for t in topics}
    try:
        r = subprocess.run(["bash", "-c", script], capture_output=True,
                           text=True, timeout=secs + 12)
        for line in (r.stdout or "").splitlines():
            m = re.match(r"\s*(\d+)\s+average rate:\s*([\d.]+)", line)
            if m:
                out[topics[int(m.group(1))]] = float(m.group(2))
    except Exception:
        pass
    return out


# 全速流的真实频率。控制台自己只订 10Hz 镜像（省 CPU、也更准），
# 但"SDK 模式开没开"必须看 200Hz 那条流 —— 它是能不能上机的硬判据。
# 后台每 20 秒量一次，闸门和自检直接用，不用现场等。
LOOP_IDLE_WAIT = 0.02                 # 09-11：主循环空闲等待（原 0.004）
FULL_EVERY_S = 60                     # 09-11：全速频率后台采样间隔（原 20 s；full_hz 有效期 75 s）
FULL = {"j": None, "c": None, "i": None, "t": 0.0}


def fullrate_loop():
    while True:
        try:
            m = measure_hz(["/JOINTS_DATA", "/JOINTS_CMD", "/IMU_DATA"])
            FULL.update(j=m.get("/JOINTS_DATA"), c=m.get("/JOINTS_CMD"),
                        i=m.get("/IMU_DATA"), t=time.time())
        except Exception:
            pass
        time.sleep(FULL_EVERY_S)


def full_hz(key, topic, max_age=75.0):
    """拿后台采样值；太旧就现场量一次。"""
    v = FULL.get(key)
    if v is not None and time.time() - FULL["t"] < max_age:
        return v
    v = measure_hz([topic]).get(topic)
    return 0.0 if v is None else v


# ----------------------------------------------------------------- 上机自检
def preflight():
    """全只读：不发布任何话题、不启动任何进程。回答“现在能不能自己上机测”。"""
    items = []

    def add(k, s, v):
        items.append({"k": k, "s": s, "v": v})

    L = LIVE
    # 先把频率量出来 —— 下面"总线归属"和"/JOINTS_DATA"两条都要用。
    # 这一步约 6 秒，自检是手点的，值得。
    meas = {"/JOINTS_DATA": full_hz("j", "/JOINTS_DATA"),
            "/JOINTS_CMD": full_hz("c", "/JOINTS_CMD"),
            "/IMU_DATA": full_hz("i", "/IMU_DATA")}
    meas_age = time.time() - FULL["t"] if FULL["t"] else None
    exe = EXE_PATH
    add("runner 可执行", "ok" if os.path.exists(exe) else "bad",
        exe if os.path.exists(exe) else "缺失 %s —— 先跑 build_runner.sh" % exe)
    stray = stray_runners()
    if stray:
        add("遗留 runner 进程", "bad",
            "按可执行文件发现 %s —— 控制台不认识它（多半是上次控制台重启留下的孤儿），"
            "它正在往真机总线发 /JOINTS_CMD。点「停止 runner」清掉" % stray)

    miss = [k for k, p in POLICIES.items()
            if not os.path.exists(os.path.join(POLDIR, p["onnx"]))]
    add("策略 onnx", "ok" if not miss else "bad",
        "%d 个齐全" % len(POLICIES) if not miss else "缺 " + ", ".join(miss))

    chz = meas.get("/JOINTS_CMD")
    chz = (L.get("c_hz") or 0) if chz is None else chz
    if STATE["runner_pid"]:
        add("总线归属", "ok",
            "我们的 runner pid=%s，/JOINTS_CMD %.0f Hz" % (STATE["runner_pid"], chz))
    elif chz > 20:
        add("总线归属", "bad",
            "别人的控制器正在以 %.0f Hz 发 /JOINTS_CMD —— 现在接管会抢关节，必须等对方停" % chz)
    elif L.get("runner_online"):
        add("总线归属", "warn",
            "有别人的 rl_deploy 节点在线，但当前没发 /JOINTS_CMD（%.0f Hz）。确认对方确实停了" % chz)
    else:
        add("总线归属", "ok", "/JOINTS_CMD 静默（%.0f Hz），没人占着狗" % chz)

    if STATE["runner_pid"]:
        add("指令通路 /cmd_vel", "ok" if L.get("cmd_iface") else "bad",
            "s10_user_command 节点在线，/cmd_vel /robot_mode /climb_mode /height_scan 已订阅"
            if L.get("cmd_iface") else
            "runner 在跑但 s10_user_command 还没建 —— 它卡在等关节反馈（S10Interface::Start）")
    else:
        add("指令通路 /cmd_vel", "warn",
            "runner 没起，通路自然没有。注意：这些订阅在 runner 收到 /JOINTS_DATA 之后才建")

    jhz = meas.get("/JOINTS_DATA")
    jhz = 0 if jhz is None else jhz
    age = L.get("j_age")
    if jhz >= 150:
        add("/JOINTS_DATA", "ok",
            "全速流 %.0f Hz（%s）—— SDK 模式已开。页面上那个 10 Hz 是低频镜像"
            " /JOINTS_DATA_10HZ，不是掉速"
            % (jhz, "%.0f 秒前实测" % meas_age if meas_age else "刚实测"))
    elif jhz >= 50:
        add("/JOINTS_DATA", "warn",
            "全速流 %.0f Hz（标称 200）—— 流是通的，说明 SDK 模式开着；"
            "偏低通常是实测时 CPU 忙的欠采样。要确认真降频，看关节数据有没有卡顿" % jhz)
    elif (L.get("j_hz") or 0) >= 3 and jhz < 30:
        add("/JOINTS_DATA", "bad",
            "全速流没有数据，但 10Hz 镜像 %.0f Hz 正常 —— 机器人活着，"
            "是手柄的 SDK 模式没打开。打开它这条 200Hz 流才会出现"
            % (L.get("j_hz") or 0))
    elif jhz > 30:
        add("/JOINTS_DATA", "bad",
            "只有 %.0f Hz —— 正常 SDK 模式是 200 Hz。状态机和策略都按 200 Hz 跑，"
            "这个频率下策略每帧拿到的是过期观测。检查手柄模式/机器人侧是否降频" % jhz)
    elif jhz > 0.3:
        add("/JOINTS_DATA", "bad",
            "只有 %.1f Hz —— 手柄没打到 SDK 模式。不打开就下不了关节指令" % jhz)
    else:
        add("/JOINTS_DATA", "bad",
            "没有数据%s —— 狗没上电，或手柄不在 SDK 模式"
            % ("" if age is None else "（最后一帧在 %.0f 秒前）" % age))

    ihz = meas.get("/IMU_DATA")
    ihz = (L.get("i_hz") or 0) if ihz is None else ihz
    add("/IMU_DATA", "ok" if ihz >= 100 else "bad",
        "%.0f Hz（总线实测）%s" % (ihz, "" if ihz >= 100 else
                                "  ← 侧倾急停依赖它，太低要查"))

    imu = L.get("imu")
    if imu:
        t = max(abs(imu["roll"]), abs(imu["pitch"]))
        add("当前姿态", "ok" if t < START_TILT_WARN_DEG else ("warn" if t < START_TILT_BAD_DEG else "bad"),
            "roll %.1f°  pitch %.1f°   （起立前机身要平；运动中 roll>%.0f° 或 pitch>%.0f° 持续 1.5 s"
            "且我方在控时自动急停）"
            % (imu["roll"], imu["pitch"], TILT_ABORT_ROLL_DEG, TILT_ABORT_PITCH_DEG))
    else:
        add("当前姿态", "bad", "收不到 IMU")

    ci = L.get("cmd_in")
    if ci:
        kp = ci.get("kp", 0)
        add("关节当前刚度", "ok",
            "kp=%.0f kd=%.2f control_word=%s  → %s"
            % (kp, ci.get("kd", 0), ci.get("cw"),
               "阻尼 / 软（狗是瘫的）" if kp < 1 else "带力（狗正被控制）"))

    js = L.get("joints") or []
    if js:
        sws = sorted(set(j["sw"] for j in js))
        # 单个 sw 值本身不代表故障（空闲时 16 路一致地为 1）。
        # 真正的单关节故障表现为"某几路和别人不一样"。
        if len(sws) == 1 and sws[0] in (0, 1):
            add("关节状态字", "ok", "16 路一致 = %d，没有单关节异常" % sws[0])
        elif len(sws) == 1:
            add("关节状态字", "warn", "16 路一致 = %d（非 0/1，含义待查手册）" % sws[0])
        else:
            odd = {}
            for j in js:
                odd.setdefault(j["sw"], []).append(j["n"])
            worst = min(odd.items(), key=lambda kv: len(kv[1]))
            add("关节状态字", "warn", "16 路不一致：%s —— 少数派 sw=%d 的是 %s"
                % (sws, worst[0], ", ".join(worst[1])))
        tm = max(j["tm"] for j in js)
        td = max(j["td"] for j in js)
        add("关节温度", "ok" if (tm < 60 and td < 70) else "warn",
            "电机最高 %.0f℃   驱动最高 %.0f℃" % (tm, td))
    else:
        add("关节状态字", "bad", "收不到 /JOINTS_DATA")

    b = L.get("batt")
    if b:
        add("电池", "ok" if b["soc"] >= 30 else ("warn" if b["soc"] >= 15 else "bad"),
            "%.2f V   %d%%   %.2f A" % (b["v"], b["soc"], b["a"]))
    else:
        add("电池", "warn", "收不到 /BATTERY_DATA")

    hhz = L.get("h_hz") or 0
    hm = L.get("hs")
    if hhz > 0:
        gok, gwhy = hscan_gate()
        if STATE["hscan"]:
            add("/height_scan", "bad",
                "%.1f Hz 但是我方零填充 —— 零在策略眼里是脚下 0.5m 深的坑"
                "（runner 兜底才是按平地）。取消勾选。" % hhz)
        elif hm is None:
            add("/height_scan", "warn", "%.1f Hz，还没收到完整一帧" % hhz)
        elif not gok:
            add("/height_scan", "bad", "%.1f Hz 但不可信：%s" % (hhz, gwhy))
        else:
            # 姿态无关的读法：前方有没有坎，身下基准多少，空洞多少
            if hm["step_d"] is not None:
                what = "前方 %.2f m 处抬高 %.2f m" % (hm["step_d"], hm["step_h"])
            else:
                what = ("前方 0.12~0.80 m 平整（最大抬高 %+.2f m）" % hm["fwd_max"]
                        if hm["fwd_max"] is not None else "前方全空洞")
            add("/height_scan", "ok",
                "%.1f Hz，%d 维，%s。身下基准 %s，空洞 %.0f%%，墙格 %s"
                % (hhz, L.get("hs_len") or 0, what,
                   ("%+.4f" % hm["base"]) if hm["base"] is not None else "—",
                   100 * hm["hole"],
                   ("%.0f%%" % (100 * hm["wall_frac"])) if hm["wall_frac"] is not None else "—"))
            # 平地标定参考单列一条，明确它**不是**闸门
            fl, b = L.get("hs_flat"), L.get("hs_base")
            if fl is True:
                add("平地标定参考", "ok", "%+.4f 在 %+.3f±%.3f 内（狗站平地时才有意义）"
                    % (b, HS_TARGET, HS_TOL))
            elif fl is False:
                add("平地标定参考", "warn",
                    "%+.4f 不在 %+.3f±%.3f 内。**这不拦你** —— 狗蹲着/趴着/站在台阶上"
                    "本来就会偏。只有当你确认它站在平地上时，这条才说明该重标外参"
                    "（跑 prep_run.sh）" % (b, HS_TARGET, HS_TOL))
    else:
        add("/height_scan", "warn",
            "无数据。244 维策略要它；runner 会按平地兜底（安全但看不见地形），"
            "越障必须用真高程图")

    ohz = L.get("o_hz") or 0
    # 作者 09-11 定：KISS 卡死只报警、不自动重启（重启会让里程计原点归零），由人决定。
    add("定位 " + ODOM_TOPIC, "ok" if ohz >= 5 else ("warn" if ohz > 0 else "bad"),
        ("%.1f Hz" % ohz) if ohz >= 5 else
        ("%.1f Hz —— FAST-LIO 位姿静默（卡死或雷达/IMU 链断）→ hmap 停发 /height_scan，感知全盲。"
         "先看 lidar_watch.log；合并点云正常就人工恢复：bash /home/robot/lio_restart.sh"
         "（里程计原点会归零）" % ohz))
    ha, hage = L.get("hinfo") or [], L.get("hinfo_age")
    if len(ha) > 3 and hage is not None and hage < 2.0:
        add("高程图点云年龄", "ok" if ha[3] <= HMAP_AGE_MAX else "bad",
            "%.3f s（>%.2f s 报红：hmap 在重发旧图）" % (ha[3], HMAP_AGE_MAX))
    else:
        add("高程图点云年龄", "warn", "收不到 /hmap_info 或没有第 4 字段（hmap 没在发 / 旧版）")
    gok, gwhy = climb_gate()
    add("切专家 ClimbH", "ok" if gok else "warn",
        "可以切：离墙约 1 m，按住 W 再按 C（手柄 Y），切入时速度自动设 0.4" if gok else gwhy)
    sok, swhy = stair_gate()
    add("切楼梯专家", "ok" if sok else "warn",
        "可以切：离第一级约 1 m、正对楼梯，按住 W 再按 V，切入速度 %.1f，自动保持航向" % STAIR_ENTRY_VX if sok else swhy)

    try:
        gb = shutil.disk_usage("/home/robot").free / float(1 << 30)
        add("AGX 磁盘", "ok" if gb > 5 else "warn", "剩余 %.1f GB" % gb)
    except Exception:
        pass

    add("速度上限", "ok" if STATE["max_vx"] <= 0.6 else "warn",
        "vx %.2f m/s   wz %.2f rad/s%s"
        % (STATE["max_vx"], STATE["max_wz"],
           "" if STATE["max_vx"] <= 0.6 else "   ← 契约要求首次上机从 0.3 起逐档加"))

    return {"items": items,
            "blocking": sum(1 for i in items if i["s"] == "bad"),
            "warn": sum(1 for i in items if i["s"] == "warn")}


# ----------------------------------------------------------------- HTTP
PAGE = r"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>S10 策略测试台</title>
<style>
*{box-sizing:border-box}
body{margin:0;font:13px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;
background:#12151a;color:#dfe4ea}
h2{margin:0 0 8px;font-size:15px;color:#8fb4ff;font-weight:600}
.wrap{max-width:1180px;margin:0 auto;padding:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:#1a1f27;border:1px solid #2a3240;border-radius:10px;padding:12px}
.estop{width:100%;padding:20px;font-size:20px;font-weight:800;letter-spacing:2px;
border:0;border-radius:10px;background:#c0202c;color:#fff;cursor:pointer;
box-shadow:0 4px 0 #7d1219}
.estop:active{transform:translateY(3px);box-shadow:0 1px 0 #7d1219}
.pol{display:block;width:100%;text-align:left;margin:0 0 6px;padding:9px 11px;
border-radius:8px;border:1px solid #2f3a4a;background:#20262f;color:#dfe4ea;cursor:pointer}
.pol.on{border-color:#4d8dff;background:#1d3557}
.pol b{color:#9dc0ff}.pol i{display:block;color:#8d97a6;font-style:normal;font-size:11.5px;margin-top:2px}
.dpad{display:grid;grid-template-columns:repeat(3,68px);grid-template-rows:repeat(3,58px);
gap:6px;justify-content:center;margin:8px auto}
.dpad button{border:1px solid #33405a;background:#232b38;color:#dfe4ea;border-radius:8px;
font-size:17px;cursor:pointer;-webkit-user-select:none;user-select:none;touch-action:none}
.dpad button:active,.dpad button.on{background:#2f5aa8;border-color:#5d92ff}
table{width:100%;border-collapse:collapse;font-size:11.5px}
th,td{padding:2px 4px;text-align:right;border-bottom:1px solid #232a34}
th{color:#8d97a6;font-weight:500}td:first-child,th:first-child{text-align:left}
.kv{display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #232a34}
.kv span:last-child{font-variant-numeric:tabular-nums;color:#fff}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px}
.ok{background:#14402a;color:#63e08e}.bad{background:#42171b;color:#ff8b95}
.warn{background:#463612;color:#ffd166}
button.act{padding:8px 10px;margin:0 4px 4px 0;border-radius:7px;border:1px solid #33405a;
background:#232b38;color:#dfe4ea;cursor:pointer}
button.act:hover{background:#2b3543}
input[type=range]{width:100%;height:26px}
.gears{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 2px}
.gear{flex:1 1 62px;min-width:62px;padding:12px 0;font-size:16px;font-weight:700;
border-radius:9px;border:2px solid #33405a;background:#232b38;color:#c8d2e0;cursor:pointer;
font-variant-numeric:tabular-nums}
.gear:hover:not(:disabled){background:#2b3543}
.gear.on{background:#2f5aa8;border-color:#7fb0ff;color:#fff;
box-shadow:0 0 0 3px rgba(93,146,255,.25)}
.gear:disabled{opacity:.28;cursor:not-allowed}
.bigval{display:flex;align-items:baseline;gap:10px;margin:10px 0 2px}
.bigval b{font-size:34px;line-height:1;color:#7fb0ff;font-variant-numeric:tabular-nums}
.bigval u{text-decoration:none;color:#8d97a6;font-size:12px}
.lab{color:#8fb4ff;font-size:12px;font-weight:600;margin-top:10px}
#log,#rlog{font:11px ui-monospace,Consolas,monospace;height:120px;overflow:auto;color:#94a3b8;
background:#141922;border-radius:6px;padding:6px}
#rlog{color:#9dc0ff}
.axrow{display:flex;align-items:center;gap:8px;margin:4px 0}
.axrow i{font-style:normal;color:#8d97a6;width:60px;font-size:11.5px}
.axrow b{width:50px;text-align:right;font-variant-numeric:tabular-nums;color:#fff;font-size:12.5px}
.bar{flex:1;height:13px;background:#141922;border-radius:7px;position:relative;overflow:hidden}
.bar em{position:absolute;top:0;bottom:0;left:50%;width:1px;background:#39424f}
.bar s{position:absolute;top:0;bottom:0;left:50%;width:0;background:#4d8dff;text-decoration:none}
.hint{color:#8d97a6;font-size:11.5px;margin-top:8px;line-height:1.75}
input[type=text]{background:#141922;border:1px solid #33405a;color:#dfe4ea;
border-radius:7px;padding:7px 9px;font:13px inherit;width:100%}
input[type=text]:focus{outline:0;border-color:#4d8dff}
.rec{display:inline-block;width:9px;height:9px;border-radius:50%;background:#c0202c;
margin-right:6px;animation:bl 1.1s infinite}
@keyframes bl{0%,49%{opacity:1}50%,100%{opacity:.25}}
#reclist{font:11px ui-monospace,Consolas,monospace;white-space:pre;overflow:auto;
max-height:190px;background:#141922;border-radius:6px;padding:6px;color:#94a3b8;margin-top:8px}
.hint code{background:#141922;padding:1px 5px;border-radius:4px;color:#9dc0ff}
.ban{background:#42171b;border:1px solid #7d1219;color:#ffb3b9;padding:8px 10px;
border-radius:8px;margin-bottom:10px;display:none}
</style><div style="background:#6b4e00;color:white;padding:12px">0911g 复测：runner 为 s10_ws 正式版 0911g（起立防护开，RL 停顿只告警）；RL 不会自动趴下，要手动发趴下；速度按策略范围与滑块，首次 0.3。</div>
<div class=wrap>
<div id=opbar class=ban style="display:none"></div>
<div id=ban class=ban></div>
<div id=speedbar class=card style="margin:0 0 8px 0">
<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap"><h2 style="margin:0">速度</h2><span id=segtop style="color:#ffd166;font-weight:600"></span><span id=terr style="color:#9ad;font-weight:600"></span><span style="color:#8d97a6;font-size:12px">选赛段会自动调档位（石笼 0.4 / 楼梯 0.5，可再调）；专家赛段里专家速度固定，档位只管主策略</span></div>
<div class=bigval><b id=mvxL>0.30</b><u>m/s &nbsp;前进上限</u>
<b id=mwzL style="margin-left:auto;font-size:26px">0.50</b><u>rad/s 转向</u></div>

<div class=lab>前进档位 &nbsp;<span style="color:#8d97a6;font-weight:400">即时生效 · 契约要求从 0.3 起逐档加</span></div>
<div class=gears id=gvx></div>
<input type=range id=mvx min=0 max=3.5 step=0.05 value=0.3
oninput="mvxL.textContent=(+this.value).toFixed(2)" onchange="lim()">

<div class=lab>转向档位</div>
<div class=gears id=gwz></div>
<input type=range id=mwz min=0 max=2 step=0.05 value=0.5
oninput="mwzL.textContent=(+this.value).toFixed(2)" onchange="lim()">

<div id=limwarn style="display:none;background:#463612;border:1px solid #7a5e14;color:#ffd166;
padding:8px 10px;border-radius:7px;margin-top:8px;font-size:11.5px;line-height:1.75"></div>
</div>
<div class=grid>

<div class=card>
<h2>急停</h2>
<button class=estop onclick="api('/api/estop')">■ 急停 &nbsp;/&nbsp; STOP</button>
<div style=margin-top:8px>
<button class=act onclick="api('/api/clear')">解除急停</button>
<button class=act id=armb onclick="api('/api/arm?on='+(armed?0:1))">接管</button>
<button class=act id=forceb onclick="forceArm()"
 style="border-color:#7d4a12;background:#33240f">强制接管</button>
</div>
<div class=lab>状态机 &nbsp;<span style="color:#8d97a6;font-weight:400">/robot_mode · 只允许合法转移</span></div>
<div class=gears style="margin-top:4px">
<button class=gear data-m=1 onclick="mode(1)" style="flex:1 1 88px;font-size:13px">1 起立</button>
<button class=gear data-m=6 onclick="mode(6)" style="flex:1 1 88px;font-size:13px">6 策略接管</button>
<button class=gear data-m=4 onclick="mode(4)" style="flex:1 1 88px;font-size:13px">4 趴下</button>
<button class=gear data-m=2 onclick="mode(2)" style="flex:1 1 88px;font-size:13px">2 阻尼</button>
</div>
<div class=kv><span>当前模式</span><span id=curmode>0 待命</span></div>
<div style="margin-top:6px;display:flex;gap:5px;align-items:center;flex-wrap:wrap">
<span style="color:#8d97a6;font-size:11px">跟踪跑偏了？直接改成实际状态（只改本地，不发话题）</span>
<button class=act style="padding:2px 9px;margin:0" onclick="api('/api/setmode?m=0')">0</button>
<button class=act style="padding:2px 9px;margin:0" onclick="api('/api/setmode?m=1')">1</button>
<button class=act style="padding:2px 9px;margin:0" onclick="api('/api/setmode?m=6')">6</button>
<button class=act style="padding:2px 9px;margin:0" onclick="api('/api/setmode?m=4')">4</button>
</div>
<div style="margin-top:8px;color:#8d97a6;font-size:11.5px;line-height:1.75">
<b style=color:#9dc0ff>runner 的真实转移规则</b>（<code>ros_command_interface.hpp:166</code>）：<br>
<code>2 阻尼</code> 任何状态都收 · <code>1 起立</code> 只能从 0/4 ·
<code>6 接管</code> 只能从 1 · <code>4 趴下</code> 只能从 1/6<br>
<b>没有"切到 0"这条指令</b> —— 阻尼满 3 秒状态机自己回 0
（<code>joint_damping_state.hpp:46</code>）。所以<b>按完急停等 3 秒就能起立</b>。<br>
急停 = <code>/cmd_vel</code> 零 + <code>/robot_mode=2</code> + <code>G20_KEY_R2</code>。
倾角超限（roll 60° / pitch 55°，持续 1.5 s）且我方在控时自动触发，专家模式下先退专家。<b style=color:#ffd166>未接管时一条话题都不发</b>。
</div>
</div>

<div class=card>
<h2>策略</h2>
<div id=pols></div>
<div style="display:flex;gap:8px;margin-top:6px">
<button class=act onclick="startRunner()"
 style="border-color:#7d4a12;background:#33240f">启动 runner</button>
<button class=act onclick="api('/api/runner/stop')">停止 runner</button>
<label style="margin-left:auto;color:#8d97a6"><input type=checkbox id=hs
onchange="api('/api/hscan?on='+(this.checked?1:0))"> 发 /height_scan 零填充</label>
</div>
<div class=lab>上台面专家 ClimbH &nbsp;<span style="color:#8d97a6;font-weight:400">手动切入 / 退出 · 专家期间前进封顶 0.4 · 松键即先退专家再停车</span></div>
<div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap">
<button class=act id=xpin onclick="climb(1,'按钮')" style="border-color:#1f5c38;background:#12301f">切专家（按住 W 再按 C · 手柄 Y）</button>
<button class=act id=xpout onclick="climb(0,'按钮')">退出专家（X · 手柄 X）</button>
</div>
<div id=xpst style="margin-top:6px;font-size:12px;line-height:1.7"></div>
<div class=lab style="margin-top:10px">赛段 &nbsp;<span style="color:#8d97a6;font-weight:400">三选一，常驻，退出手动。专家赛段里：按住 W 自动切专家前进，松 W 先退回主策略再停车（专家不会零速站住），S 倒退交主策略，主策略不会在专家赛段里前进</span></div>
<div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;align-items:center">
<button class=act id=segflat onclick="api('/api/segment?s=flat')">平地赛段（主策略）</button>
<button class=act id=seggab onclick="api('/api/segment?s=gabion')" style="border-color:#1f5c38;background:#12301f">石笼赛段（ClimbH 0.4，离墙 1 m 再点）</button>
<button class=act id=segstair onclick="api('/api/segment?s=stairs')" style="border-color:#1f4b5c;background:#12262f">楼梯赛段（StairN-C 0.5，离第一级 1 m 正对再点）</button>
<span id=segst style="color:#8d97a6;font-size:12px"></span>
</div>
<table style="margin-top:6px;font-size:12px;line-height:1.5;border-collapse:collapse;color:#c9d1d9">
<tr style="color:#8d97a6"><td style="padding:2px 8px 2px 0">赛段 / 地形</td><td style="padding:2px 8px">策略</td><td style="padding:2px 8px">速度</td><td style="padding:2px 8px">操作</td></tr>
<tr><td style="padding:2px 8px 2px 0">起步、草坪、铺装路、砾石、水坑边、终点</td><td>主策略 V1Down</td><td>随意，长直路可到 1.8</td><td>停、倒退、原地转都由主策略做</td></tr>
<tr><td style="padding:2px 8px 2px 0">上草坪 +0.22 / 两级 −0.15 下</td><td>主策略</td><td>1.2~1.8</td><td>—</td></tr>
<tr><td style="padding:2px 8px 2px 0"><b>33 cm 石笼台面（上）</b></td><td><b>石笼赛段 · ClimbH</b></td><td><b>0.4</b>（自动）</td><td>离墙约 1 m 按住 W，上到台面松 W，点「平地赛段」</td></tr>
<tr><td style="padding:2px 8px 2px 0">台面 → 路面 −0.25 下坎</td><td>主策略</td><td><b>≤1.2</b>（1.8 只有 3/4）</td><td>低于 1.2 会停在坎沿不动；翻车都在这道坎，别快</td></tr>
<tr><td style="padding:2px 8px 2px 0">下 31 cm 路缘</td><td>主策略</td><td><b>1.8</b>（靠动量）</td><td>低速会停在坎沿；这是唯一能下 31 cm 的方式</td></tr>
<tr><td style="padding:2px 8px 2px 0"><b>楼梯（上，约 40 级三段，宽 4 m，两平台）</b></td><td><b>楼梯赛段 · StairN-C</b></td><td><b>0.5</b>（自动）</td><td>正对楼梯底、离第一级约 1 m 按住 W，<b>一口气到顶</b>；非停不可只在平台上松 W，<b>再按 W 前先 Q/E 对着下一段</b>；到顶松 W，点「平地赛段」</td></tr>
</table>
<div style="font-size:11px;color:#8d97a6;margin-top:2px">速度栏只是提示，不自动限速。以上依据全是 MuJoCo 仿真；真机只做过 t1~t3。真机待验：急停、上墙一次、楼梯一口气一次、下坎 1.2 / 路缘 1.8 各一次。</div>
<div class=lab style="margin-top:10px">楼梯专家（手动，备用） &nbsp;<span style="color:#8d97a6;font-weight:400">赛段模式下不用碰这里；V = 手动切一次，X = 退出。航向以切入时的偏航为目标（Q/E 微调）</span></div>
<div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;align-items:center">
<button class=act id=stin onclick="climb(2,'按钮')" style="border-color:#1f4b5c;background:#12262f">切楼梯专家（按住 W 再按 V）</button>
<button class=act id=stout onclick="climb(0,'按钮')">退出楼梯专家（X）</button>
<span style="color:#8d97a6;font-size:12px;margin-left:8px">航向保持</span>
<button class=act id=sk2 onclick="api('/api/stairk?k=2')">2</button>
<button class=act id=sk1 onclick="api('/api/stairk?k=1')">1</button>
<button class=act id=sk0 onclick="api('/api/stairk?k=0')">关</button>
<button class=act id=sthc onclick="api('/api/stairhead?clear=1')" title="换一段楼梯时用：清掉第一次切入时记下的楼梯朝向">清除楼梯朝向</button>
<span style="color:#8d97a6;font-size:12px">车头来回摆就调小</span>
</div>
<div id=stst style="margin-top:6px;font-size:12px;line-height:1.7"></div>
</div>

<div class=card>
<h2>方向控制 &nbsp;<span style="color:#8d97a6;font-weight:400">WASD / QE 转向 · 松手即停</span></h2>
<div class=dpad>
<button data-k=q>↰</button><button data-k=w>▲</button><button data-k=e>↱</button>
<button data-k=a>◀</button><button onclick="zero()">■</button><button data-k=d>▶</button>
<button></button><button data-k=s>▼</button><button></button>
</div>
<div class=kv style=margin-top:8px><span>正在下发 /cmd_vel</span><span id=cmdout>0, 0, 0</span></div>
<div class=kv><span>手柄 /STEER</span><span id=steer>—</span></div>
</div>

<div class=card>
<h2>录制 &nbsp;<span style="color:#8d97a6;font-weight:400">只订阅、从不发布 · 数据存 /home/robot/s10_data</span></h2>
<div class=kv><span>状态</span><span id=recstat>—</span></div>
<div style="margin-top:8px">
<input type=text id=recname placeholder="这一条叫什么（字母数字 _ . -，会当目录名）">
</div>
<div style="margin-top:6px">
<input type=text id=recnote placeholder="备注：第几次尝试 / 什么地形 / 成功还是失败（会写进 bag 里）">
</div>
<div style="margin-top:8px;display:flex;gap:6px;align-items:center;flex-wrap:wrap">
<button class=act onclick="autoName()">自动命名</button>
<button class=act id=recgo onclick="recStart()"
 style="border-color:#7d1219;background:#33141a">开始录制</button>
<button class=act onclick="recStop()">停止录制</button>
<select id=recmode style="margin-left:auto;background:#232b38;color:#dfe4ea;
border:1px solid #33405a;border-radius:7px;padding:7px 8px;font:12px inherit">
<option value=extra selected>标准 26 路</option>
<option value=core>精简 4 路（只要关节+IMU+电池）</option>
<option value=cloud>标准 + 点云（约 1 GB/分钟）</option>
</select>
</div>
<div class=hint>
标准档 26 路。其中 SDK 自测必需的 7 路是新加的 ——
<code>/cmd_vel</code>（我们下发了什么）、<code>/height_scan</code> +
<code>/height_scan_raw</code> + <code>/hmap_info</code>（策略看到的地形和它的质量）、
<code>/robot_mode</code> <code>/climb_mode</code>（状态机与策略切换）、
<code>/rosout</code>（runner 的裁决，含它拒绝跳转时报出的真实状态）。<br>
另外 19 路是原有的关节 / IMU / 电池 / tf / 里程计 / 手柄。<br>
开始的一瞬间会往 bag 目录写一份 <code>s10_context.json</code> —— 记下当时跑的哪份 onnx、
上限多少、谁在控、电量、各路频率。训练侧拿到 bag 不用问就知道条件。
</div>
<button class=act style="margin-top:8px" onclick="recList()">列出已有录制</button>
<div id=reclist style="display:none"></div>
</div>

<div class=card>
<h2>手柄 &nbsp;<span style="color:#8d97a6;font-weight:400">USB / 蓝牙游戏手柄 · 按住 LB/RB 才发指令</span></h2>
<div id=padwarn style="display:none;background:#463612;border:1px solid #7a5e14;color:#ffd166;
padding:8px 10px;border-radius:7px;margin-bottom:8px;font-size:11.5px;line-height:1.75"></div>
<div class=kv><span>设备</span><span id=padname>未连接</span></div>
<div class=kv><span>死人开关 LB / RB / RT</span><span id=paddead>&mdash;</span></div>
<div style=margin-top:8px>
<div class=axrow><i>前进 vx</i><div class=bar><em></em><s id=bvx></s></div><b id=nvx>0.00</b></div>
<div class=axrow><i>侧移 vy</i><div class=bar><em></em><s id=bvy></s></div><b id=nvy>0.00</b></div>
<div class=axrow><i>转向 wz</i><div class=bar><em></em><s id=bwz></s></div><b id=nwz>0.00</b></div>
</div>
<div class=hint>
左摇杆 = 前后 / 左右平移 &nbsp;·&nbsp; 右摇杆左右 = 转向 &nbsp;·&nbsp;
<b style=color:#ffd166>必须按住 LB 或 RB 才发指令，松手立即归零</b><br>
<b>B</b> = 急停 &nbsp;·&nbsp; <b>Back</b> = 释放接管 &nbsp;·&nbsp;
<b>Start+A</b> = 起立 &nbsp;·&nbsp; <b>Start+Y</b> = 策略接管 &nbsp;·&nbsp; <b>Start+X</b> = 趴下<br>
摇杆是模拟量：实际下发 = 摇杆量 × 左边的档位上限。插上手柄后<b>按一下手柄上任意键</b>浏览器才认得到。
</div>
</div>

<div class=card>
<h2>状态</h2>
<div class=kv><span>谁在控狗</span><span id=owner>—</span></div>
<div class=kv><span>指令通路 /cmd_vel</span><span id=iface>—</span></div>
<div class=kv><span>高程图基准</span><span id=hsbase>—</span></div>
<div class=kv><span>runner</span><span id=runner>—</span></div>
<div class=kv><span>关节刚度 kp</span><span id=kp>—</span></div>
<div class=kv><span>/JOINTS_DATA</span><span id=jhz>—</span></div>
<div class=kv><span>/JOINTS_CMD</span><span id=chz>—</span></div>
<div class=kv><span>IMU roll / pitch</span><span id=rp>—</span></div>
<div class=kv><span>定位 FAST-LIO</span><span id=od>—</span></div>
<div class=kv><span>轮速均值</span><span id=wsp>—</span></div>
<div class=kv><span>最大关节力矩</span><span id=tau>—</span></div>
<div class=kv><span>电池</span><span id=bat>—</span></div>
</div>

<div class=card style="grid-column:1/-1">
<h2>16 关节</h2>
<table><thead><tr><th>关节</th><th>角度°</th><th>速度</th><th>力矩</th><th>电机℃</th><th>驱动℃</th><th>sw</th></tr></thead>
<tbody id=jt></tbody></table>
</div>

<div class=card style="grid-column:1/-1">
<h2>上机自检 &nbsp;<span style="color:#8d97a6;font-weight:400">全部只读 · 不发任何话题 · 随时可点</span></h2>
<button class=act onclick="pref()">运行自检</button>
<span id=prefsum style="margin-left:10px"></span>
<table style=margin-top:8px><tbody id=preft></tbody></table>
</div>

<div class=card style="grid-column:1/-1">
<h2>runner 判决 &nbsp;<span style="color:#8d97a6;font-weight:400">来自 /rosout —— runner 自己说的，含它的真实状态；「[日志]」开头的是 runner 打进 rl_deploy.log 的 IMU 提示和 !! 告警</span></h2>
<div id=rlog></div></div>

<div class=card style="grid-column:1/-1"><h2>控制台日志</h2><div id=log></div></div>
</div></div>
<script>
const $=s=>document.querySelector(s);
let POL={},cur='',held=new Set(),estop=false,armed=false,MODES={},curmode=0,OPR=false;   // OPR：本页是不是操作员（09-12）
var PAD={on:false,name:'',dead:false,vx:0,vy:0,wz:0};
// nightreview-B：alert/confirm 打开期间页面收不到 keyup。弹窗前先松开所有键并补发一次零，免得关掉弹窗后带着「按住 W」继续前进
const _alert0=window.alert.bind(window), _confirm0=window.confirm.bind(window);
function _dropKeys(){try{if(held.size){held.clear();paint();send()}}catch(e){}}
window.alert=m=>{_dropKeys();return _alert0(m)};
window.confirm=m=>{_dropKeys();return _confirm0(m)};
function api(u){if(!/[?&]pid=/.test(u))u+=(u.includes('?')?'&':'?')+'pid='+CPID;   // 09-12：服务端只认操作员页面的 pid
  return fetch(u).then(r=>r.json()).then(j=>{
  if(j&&j.readonly){roFlash(); poll(); return j}           // 旁观页面：不弹窗（弹窗会松键、会刷屏）
  if(j&&j.msg&&j.ok===false)alert(j.msg); poll(); return j})
  .catch(e=>{offline(true); return {ok:false,msg:'控制台失联'}})}
// 09-12 凌晨（Astra 建议 3）：唯一操作员。新开 / 刷新的页面一律是旁观（只读），要操作先点「我来操作」。
function opPaint(op){const e=$('#opbar'); if(!e)return;
  const sig=(op&&op.you?1:0)+'|'+(op&&op.held?1:0)+'|'+((op&&op.who)||'');
  if(sig!==window.__opsig){window.__opsig=sig; e.style.display='block';   // 只在变了才重建，免得吞掉按钮点击
    if(op&&op.you){e.style.background='#123d2a';e.style.borderColor='#2f8f5b';e.style.color='#b7f5cf';
      e.innerHTML='本页是<b>操作员</b>（服务端只认这一页的指令，其他页面只读）。'+
        '<button class=act style="margin-left:10px" onclick="opRelease()">交出操作权</button>';}
    else{e.style.background='#2a2f3a';e.style.borderColor='#4a5468';e.style.color='#dfe6f3';
      e.innerHTML='<b>旁观（只读）</b> —— 本页的按键、按钮、手柄都不会发给狗。'+
        (op&&op.held?'已有操作员页面在线（来自 '+op.who+'，<span id=opage></span> s 前心跳）。':'现在没有操作员。')+
        '<button class=act style="margin-left:10px" onclick="opClaim()">我来操作</button>';}}
  const a=$('#opage'); if(a&&op&&op.age!=null)a.textContent=op.age;
}
function opClaim(){fetch('/api/op/claim?pid='+CPID).then(r=>r.json()).then(j=>{if(!j.ok)_alert0(j.msg);poll()})
  .catch(()=>offline(true))}
function opRelease(){if(!confirm('交出操作权后本页变成只读。狗若正在走，本页不再发指令，0.4 s 后按停刷处理（专家先退出再停车）。确定？'))return;
  fetch('/api/op/release?pid='+CPID).then(r=>r.json()).then(()=>poll()).catch(()=>offline(true))}
function roFlash(){const e=$('#opbar'); if(e){e.style.outline='2px solid #ffd166'; setTimeout(()=>{e.style.outline=''},600)}}
let OFF=false;
function offline(on){
  if(on===OFF)return; OFF=on;
  const b=$('#ban');
  if(on){
    held.clear();
    b.style.display='block';b.style.background='';b.style.borderColor='';b.style.color='';
    b.innerHTML='<b>控制台失联</b> —— 连不上 '+location.host+
      '。s10_ctrl.py 可能正在重启，页面会自动重连。<br>'+
      '这期间方向键不发指令；runner 若在跑，它自己的 500ms 看门狗会把速度归零。';
  }
}
function key(k){api('/api/key?k='+k)}
function mode(m){api('/api/mode?m='+m)}
const GVX=[0.3,0.5,1.0,1.5,2.0,2.5], GWZ=[0.3,0.5,1.0,1.5,2.0];
function lim(){api('/api/lim?vx='+$('#mvx').value+'&wz='+$('#mwz').value)}
function setg(which,v){
  const el=$(which==='vx'?'#mvx':'#mwz');
  el.value=v;
  $(which==='vx'?'#mvxL':'#mwzL').textContent=(+v).toFixed(2);
  lim();
}
/* poll() 每 700ms 跑一次。以前这里无条件重写 innerHTML，
   把按钮整批销毁重建 —— 鼠标按下和松开之间只要跨过一次 poll，
   click 事件就不会派发（down/up 不在同一个元素上），实测约 15% 的点击被吞。
   用户的表现就是"点了没反应，只好狂点"。
   现在只有内容真的变了才重建。 */
let gsig='';
function gears(){
  const p=POL[cur]||{vx:[0,0],wz:0};
  const cvx=+$('#mvx').value, cwz=+$('#mwz').value;
  const sig=[cur,cvx,cwz,p.vx[1],p.wz].join('|');
  if(sig===gsig)return;
  gsig=sig;
  $('#gvx').innerHTML=GVX.map(v=>
    `<button class="gear${Math.abs(v-cvx)<0.001?' on':''}" ${v>p.vx[1]+1e-6?'disabled':''}`+
    ` onclick="setg('vx',${v})">${v.toFixed(1)}</button>`).join('');
  $('#gwz').innerHTML=GWZ.map(v=>
    `<button class="gear${Math.abs(v-cwz)<0.001?' on':''}" ${v>p.wz+1e-6?'disabled':''}`+
    ` onclick="setg('wz',${v})">${v.toFixed(1)}</button>`).join('');
}
const CPID=Math.random().toString(36).slice(2,10); let CSEQ=0;   // 指令序号：服务端丢弃乱序的旧请求
function zero(){held.clear();send()}
function send(){
  if(estop||!armed||OFF||!OPR)return;   // 09-12：旁观页面一条 /api/cmd 都不发（失焦时的零也不发）
  const p=POL[cur]||{vx:[0,0],vy:0,wz:0};
  const mvx=+$('#mvx').value,mwz=+$('#mwz').value;
  let vx=0,vy=0,wz=0;
  if(PAD.dead){                       // 手柄握住死人开关时，手柄优先，模拟量
    vx=PAD.vx*mvx; vy=PAD.vy*Math.min(mvx,p.vy); wz=PAD.wz*mwz;
  }else{                              // 否则键盘 / 屏幕方向键，开关量
    if(held.has('w'))vx=+mvx; if(held.has('s'))vx=-mvx;
    if(held.has('a'))vy=+Math.min(mvx,p.vy); if(held.has('d'))vy=-Math.min(mvx,p.vy);
    if(held.has('q'))wz=+mwz; if(held.has('e'))wz=-mwz;
  }
  vx=Math.max(Math.min(p.vx[0],0),Math.min(p.vx[1],vx));  // 下界跟 0 取小：
  // ClimbG 的 vx 范围是 [0.3,0.6]，直接用 0.3 当下界会把"停止"夹成 +0.3 前进
  wz=Math.max(-p.wz,Math.min(p.wz,wz));
  vy=Math.max(-p.vy,Math.min(p.vy,vy));
  fetch(`/api/cmd?vx=${vx.toFixed(3)}&vy=${vy.toFixed(3)}&wz=${wz.toFixed(3)}&pid=${CPID}&seq=${++CSEQ}`)
    .catch(()=>offline(true));   // 控制台重启时别抛未捕获的 promise
}
setInterval(()=>{if(held.size&&!PAD.dead)send()},150);

/* ---------------------------------------------------------------- 手柄
   浏览器 Gamepad API。注意它只在【安全上下文】开放：https 或 localhost。
   直接开 http://10.21.33.102:8089 读不到手柄，要用 SSH 隧道走 localhost。 */
const DZ=0.14;                        // 摇杆死区，之后线性重标定到 0..1
function dz(v){const a=Math.abs(v);return a<DZ?0:(a-DZ)/(1-DZ)*(v<0?-1:1)}
let padPrev={},padLast=0,padWasDead=false;
function edge(b,i){                   // 上升沿
  const now=!!(b[i]&&(b[i].pressed||b[i].value>0.5));
  const fired=now&&!padPrev[i]; padPrev[i]=now; return fired;
}
function padPaint(){
  $('#padname').textContent=PAD.on?PAD.name.slice(0,46):'未连接';
  $('#paddead').innerHTML=!PAD.on?'&mdash;':
    (PAD.dead?'<span class="pill ok">按住中</span>':'<span class="pill warn">松开</span>');
  for(const [bs,ns,v] of [['#bvx','#nvx',PAD.vx],['#bvy','#nvy',PAD.vy],['#bwz','#nwz',PAD.wz]]){
    const e=$(bs),w=Math.abs(v)*50;
    e.style.left=(v>=0?50:50-w)+'%'; e.style.width=w+'%';
    e.style.background=PAD.dead?'#4d8dff':'#39424f';
    $(ns).textContent=v.toFixed(2);
  }
}
function padLoop(){
  requestAnimationFrame(padLoop);
  const gs=navigator.getGamepads?navigator.getGamepads():[];
  let g=null; for(const x of gs) if(x&&x.connected){g=x;break}
  if(!g){
    if(PAD.on){PAD.on=false;PAD.dead=false;PAD.vx=PAD.vy=PAD.wz=0;padPaint();send()}
    return;
  }
  PAD.on=true; PAD.name=g.id;
  $('#padwarn').style.display='none';   // 真读到手柄了，收起那条隧道提示
  const b=g.buttons,ax=g.axes;
  const hit=i=>!!(b[i]&&(b[i].pressed||b[i].value>0.5));
  const dead=hit(4)||hit(5)||hit(7);          // LB / RB / RT
  const start=hit(9);
  const ro=!OPR;                               // 09-12：旁观页面照读手柄（边沿照记），但不发任何指令
  if(edge(b,1)&&!ro)api('/api/estop');        // B = 急停，任何时候
  if(edge(b,8)&&!ro)api('/api/arm?on=0');     // Back = 释放接管
  if(start){                                   // Start 当修饰键，防误触
    const e0=edge(b,0),e3=edge(b,3),e2=edge(b,2);
    if(!ro){if(e0)mode(1); if(e3)mode(6); if(e2)mode(4);}
  }else{edge(b,0); const e3=edge(b,3),e2=edge(b,2); if(!ro){if(e3)climb(1,'手柄Y'); if(e2)climb(0,'手柄X');}}
  PAD.dead=dead;
  PAD.vx=dead?dz(-(ax[1]||0)):0;              // 左摇杆上推 = 前进
  PAD.vy=dead?dz(-(ax[0]||0)):0;              // 左摇杆左推 = 向左平移
  PAD.wz=dead?dz(-(ax[2]||0)):0;              // 右摇杆左推 = 左转(+wz)
  padPaint();
  const t=performance.now();
  if(t-padLast>50){padLast=t;                 // 20 Hz；服务端 0.4 s 看门狗兜底
    if(dead||padWasDead)send();               // 松手那一帧也补一次零
    padWasDead=dead;
  }
}
addEventListener('gamepadconnected',()=>padPaint());
if(!window.isSecureContext){
  const w=$('#padwarn'); w.style.display='block';
  w.innerHTML='浏览器只在<b>安全上下文</b>下开放手柄接口。当前是 <code>http://'+location.host+
    '</code>，读不到手柄。<br>在你自己电脑上开个隧道即可：<br>'+
    '<code>ssh -L 8089:localhost:8089 robot@10.21.33.102</code><br>'+
    '然后改开 <code>http://localhost:8089</code>（localhost 算安全上下文）。'+
    '键盘 WASD/QE 不受影响，现在就能用。';
}
if(navigator.getGamepads)padLoop(); else padPaint();

/* ---------------------------------------------------------------- 录制 */
function two(n){return String(n).padStart(2,'0')}
function autoName(){
  const d=new Date(), L=window.__live||{};
  const pol=(L.effective_policy||cur||'x').split('_')[0].toLowerCase();
  const vx=(+$('#mvx').value).toFixed(1).replace('.','p');
  $('#recname').value=`${pol}_vx${vx}_${two(d.getMonth()+1)}${two(d.getDate())}`+
    `_${two(d.getHours())}${two(d.getMinutes())}${two(d.getSeconds())}`;
}
function recStart(){
  const n=$('#recname').value.trim();
  if(!n){alert('先起个名字，或点「自动命名」。');return}
  api(`/api/rec/start?name=${encodeURIComponent(n)}`+
      `&note=${encodeURIComponent($('#recnote').value.trim())}`+
      `&mode=${$('#recmode').value}`)
    .then(j=>{if(j&&j.msg)alert(j.msg)});
}
function recStop(){
  if(!confirm('停止录制并收尾（会等 metadata.yaml 写完，可能十几秒）？'))return;
  api('/api/rec/stop').then(j=>{if(j&&j.msg)alert(j.msg)});
}
function recList(){
  const el=$('#reclist');
  el.style.display='block'; el.textContent='读取中…（要对每个 bag 跑 ros2 bag info，慢）';
  fetch('/api/rec/list').then(r=>r.json()).then(j=>{el.textContent=j.msg||'（空）'})
    .catch(()=>{el.textContent='读取失败'});
}

/* ---------------------------------------------------------------- 自检 */
function pref(){
  $('#prefsum').innerHTML='<span class="pill warn">检查中…要实测总线频率，约 10 秒</span>';
  fetch('/api/preflight').then(r=>r.json()).then(d=>{
    $('#prefsum').innerHTML = d.blocking
      ? `<span class="pill bad">${d.blocking} 项阻断</span>`
      : (d.warn?`<span class="pill warn">${d.warn} 项提醒，可上机</span>`
               :'<span class="pill ok">全部通过</span>');
    $('#preft').innerHTML=d.items.map(i=>
      `<tr><td style="width:170px">${i.k}</td><td style="text-align:left">`+
      `<span class="pill ${i.s==='ok'?'ok':i.s==='warn'?'warn':'bad'}">`+
      `${i.s==='ok'?'通过':i.s==='warn'?'提醒':'阻断'}</span> ${i.v}</td></tr>`).join('');
  });
}
const KEYS=['w','a','s','d','q','e'];
addEventListener('keydown',e=>{const k=e.key.toLowerCase();
  const tg=e.target;   // 09-11：焦点在文本框里打字时不触发热键（空格急停照旧）
  if(k!==' '&&tg&&(tg.tagName==='TEXTAREA'||(tg.tagName==='INPUT'&&/^(text|search|number|)$/.test(tg.type||''))))return;
  if(!OPR){                                    // 09-12：旁观页面的热键一律不发（急停另见服务端 SPECTATOR_ESTOP）
    if(k===' '&&window.__opsp){e.preventDefault();api('/api/estop');return}
    if(k===' '||k==='c'||k==='x'||k==='v'||KEYS.includes(k)){e.preventDefault();roFlash()}
    return}
  if(k===' '){e.preventDefault();api('/api/estop');return}
  if(k==='c'){e.preventDefault();if(!e.repeat)climb(1,'键盘C');return}
  if(k==='x'){e.preventDefault();if(!e.repeat)climb(0,'键盘X');return}
  if(k==='v'){e.preventDefault();if(!e.repeat)climb(2,'键盘V');return}
  if(KEYS.includes(k)&&!held.has(k)){held.add(k);paint();send();e.preventDefault()}});
addEventListener('keyup',e=>{const k=e.key.toLowerCase();
  if(held.delete(k)){paint();send()}});
addEventListener('blur',()=>{const had=held.size>0;held.clear();paint();if(had)zero()});
function paint(){document.querySelectorAll('.dpad button[data-k]').forEach(b=>
  b.classList.toggle('on',held.has(b.dataset.k)))}
document.querySelectorAll('.dpad button[data-k]').forEach(b=>{
  const on=e=>{e.preventDefault();held.add(b.dataset.k);paint();send()};
  const off=e=>{e.preventDefault();held.delete(b.dataset.k);paint();send()};
  b.addEventListener('pointerdown',on);b.addEventListener('pointerup',off);
  b.addEventListener('pointerleave',off);b.addEventListener('pointercancel',off);});
function forceArm(){
  const L=window.__live||{};
  if(!confirm('强制接管会跳过"总线上有没有别人"的检查。\n\n'+
      '此刻总线状态：\n'+
      '  /JOINTS_CMD  '+(L.c_hz??'?')+' Hz\n'+
      '  rl_deploy 节点  '+(L.runner_online?'在线':'不在线')+'\n'+
      '  我们自己的 runner  '+
        (window.__runnerpid?('pid='+window.__runnerpid):'未启动')+'\n\n'+
      '如果对面的控制器还在跑，两边会同时发 /JOINTS_CMD 抢关节。\n'+
      '确认对方已经停了、或者你就是要抢？'))return;
  api('/api/arm?on=1&force=1');
}
function startRunner(){
  if(!armed){alert('先点「接管」。接管时会检查总线上没有别人在控狗。');return}
  const p=POL[cur]||{};
  if(!confirm('要在【真机总线 domain 0】上启动 runner 吗？\n\n'+
      '策略：'+(p.label||cur)+'\n'+
      'vx 上限：'+$('#mvx').value+' m/s   wz 上限：'+$('#mwz').value+' rad/s\n\n'+
      '启动后 runner 立刻开始发 /JOINTS_CMD（先是 kIdle 零刚度，狗不动）。\n'+
      '之后按「1 起立」它才会真的用力。\n\n确认狗周围没有人。'))return;
  api('/api/runner/start?yes=1');
}
function pick(k){api(`/api/policy?p=${k}&pid=${CPID}&seq=${++CSEQ}`)}
function climb(on,src){api(`/api/climb?on=${on}&src=${encodeURIComponent(src)}&pid=${CPID}&seq=${++CSEQ}`)}
function poll(){fetch('/api/state?pid='+CPID).then(r=>r.json()).then(d=>{
  OPR=!!(d.op&&d.op.you); window.__opsp=!!(d.op&&d.op.spectator_estop); opPaint(d.op);   // 09-12：唯一操作员
  if(!OPR&&held.size){held.clear();paint()}
  POL=d.policies;cur=d.state.policy;estop=d.state.estop;
  armed=d.state.armed;MODES=d.modes;curmode=d.state.mode;
  const L0=d.live;
  {const sg=$('#segst'); if(sg){const seg=L0.segment||'flat'; const nm={flat:'平地赛段（主策略）',gabion:'石笼赛段（ClimbH 0.4）',stairs:'楼梯赛段（StairN-C 0.5）'}[seg]||seg;
    sg.textContent='当前：'+nm+(L0.seg_deny?'  !! 切不进专家、不前进：'+L0.seg_deny:'');
    const st2=$('#segtop'); if(st2){st2.textContent='当前赛段：'+nm+(L0.seg_deny?'  !! 切不进专家、不前进':'')}
    const te=$('#terr'); if(te){const T=L0.terrain; if(!T){te.textContent='前方：—'} else if(T.kind==='平地'){te.textContent='前方：平地'} else {
      te.textContent='前方：'+T.kind+' 沿 '+T.dist+' m（左 '+(T.dist_l==null?'—':T.dist_l)+' / 右 '+(T.dist_r==null?'—':T.dist_r)+'）高 '+(100*T.h).toFixed(0)+' cm '+(T.square==null?'':(T.square?'正对':'!! 未正对'+(((T.yaw_med!=null?T.yaw_med:T.yaw_off))==null?'':('，需'+((T.yaw_med!=null?T.yaw_med:T.yaw_off)>0?'左':'右')+'转 '+Math.abs(T.yaw_med!=null?T.yaw_med:T.yaw_off).toFixed(0)+'°'))+'，Q/E 对正'));
      te.style.color=(T.square===false)?'#ff8080':'#9ad'}}
    [['#segflat','flat'],['#seggab','gabion'],['#segstair','stairs']].forEach(([id,v])=>{const e=$(id); if(e)e.classList.toggle('on',seg===v)})}}
  const b=$('#ban');
  if(estop){b.style.display='block';b.textContent='急停生效中 — '+d.state.estop_reason+'（点"解除急停"恢复）'}
  else if(!armed){b.style.display='block';
    b.style.background='#463612';b.style.borderColor='#7a5e14';b.style.color='#ffd166';
    b.textContent='未接管 — 不发送 /cmd_vel。点"接管"后方向键才生效。'}
  else {b.style.display='none';b.style.background='';b.style.borderColor='';b.style.color=''}
  const ab=$('#armb');
  ab.textContent=armed?'✔ 已接管（点此释放）':'接管';
  ab.style.background=armed?'#1d3557':'';ab.style.borderColor=armed?'#4d8dff':'';
  // send() 只在发送时挡住未接管，held 却一直在累积：按着 W 去点"接管"，
  // 接管成功那一刻 150ms 的定时器就按当前档位把 vx 发出去了。这里主动清空。
  if((!armed||estop)&&held.size){held.clear();paint()}
  const SRCC={'runner 确认':'ok','runner 实测':'ok','假设':'warn',
              '人工设定':'warn','阻尼超时自动回 0':'ok'};
  $('#curmode').innerHTML=curmode+' '+((MODES[curmode]||'').split(' ')[1]||'')+
    ` <span class="pill ${SRCC[d.state.mode_src]||'warn'}">${d.state.mode_src}</span>`;
  document.querySelectorAll('.gear[data-m]').forEach(x=>
    x.classList.toggle('on',+x.dataset.m===curmode));
  const RUNP=L0.effective_policy&&d.state.runner_pid?L0.effective_policy:'';
  const psig=Object.keys(POL).join(',')+'|'+cur+'|'+RUNP+'|'+(L0.policy_mismatch?1:0)+'|'+((L0.stair_gate||{}).ok?1:0);
  if(psig!==window.__psig){
    window.__psig=psig;                 // 同样避免每 700ms 重建，吞掉策略按钮的点击
    $('#pols').innerHTML=Object.entries(POL).map(([k,p])=>{
      const running = k===RUNP;
      // ClimbG 是唯一能热切的：/climb_mode=1 经 reserved_scale 进
      // rl_control_state.hpp:88-103，runner 下一个控制周期就换 active_policy_。
      // 其余三个是基础策略，S10_POLICY_PATH 在构造函数里读一次，只能重启换。
      const hot = (k==='ClimbH_8797' || !!(POL[k]&&POL[k].stair&&(L0.stair_gate||{}).ok));   // 09-11：楼梯闸门通过才算可热切
      const badge = running
        ? '<span class="pill ok" style="float:right">runner 正在跑</span>'
        : (!RUNP ? ''
           : hot ? '<span class="pill ok" style="float:right">可热切换</span>'
                 : '<span class="pill warn" style="float:right">需重启 runner</span>');
      return `<button class="pol${k===cur?' on':''}" onclick="pick('${k}')">`+
        `${badge}<b>${p.label}</b>`+
        `<i>${p.desc}<br>vx ${p.vx[0]}~${p.vx[1]} · vy ±${p.vy} · wz ±${p.wz} · 站姿 ${p.stance}`+
        `${p.zero_ok?'':' · <span style=color:#ffd166>零指令会原地转</span>'}</i></button>`;
    }).join('');
  }
  $('#hs').checked=d.state.hscan;
  // 服务端是权威值；正在拖滑块时不覆盖
  for(const [id,lab,key] of [['#mvx','#mvxL','max_vx'],['#mwz','#mwzL','max_wz']]){
    const el=$(id);
    if(document.activeElement!==el && Math.abs(+el.value-d.state[key])>1e-6){
      el.value=d.state[key]; $(lab).textContent=(+d.state[key]).toFixed(2);
    }
  }
  gears();
  const L=d.live;
  if(!estop&&armed&&L.bus_owner==='别人'){
    b.style.display='block';b.style.background='';b.style.borderColor='';b.style.color='';
    b.innerHTML='<b>已接管，但总线上另一方的控制器正在发 /JOINTS_CMD</b> —— '+
      '两边的关节指令会打架。立刻点「接管」释放，或确认对方停机。';
  }
  const OWN={'我们':'ok','无人':'warn','别人':'bad','残留':'warn'};
  const SIL=L.c_silent_s;
  $('#owner').innerHTML=L.bus_owner
    ?`<span class="pill ${OWN[L.bus_owner]||'warn'}">${L.bus_owner}</span>`+
     ` <span style=color:#8d97a6>/JOINTS_CMD ${L.c_hz??'—'} Hz`+
     (L.bus_owner==='残留'
       ? `　有 rl_deploy 节点但已静默 ${SIL} 秒（上次没退干净的残留，接管不受阻）`
       : (L.bus_owner==='无人'&&SIL!=null ? `　静默 ${SIL} 秒` : ''))+
     `</span>`:'—';
  $('#kp').textContent=L.cmd_in
    ?`${L.cmd_in.kp}  (${L.cmd_in.kp<1?'阻尼/瘫软':'带力'})`:'—';
  // 高程图。**读数偏离 -0.08 不再报红** —— 狗蹲着/站在台阶上本来就会偏，
  // 那样报红会让人以为坏了。这里显示的是姿态无关的东西：
  // 前方有没有坎（这就是"狗看见了什么"）、空洞率、墙格比例。
  const hs=L.hs, hok=L.hs_ok, hhz=L.h_hz||0;
  $('#hsbase').innerHTML = (hhz<=1)
    ? '<span class="pill warn">无数据</span>'+
      ' <span style=color:#8d97a6>runner 按平地兜底，安全但看不见地形</span>'
    : (!hs ? '<span class="pill warn">等一帧完整数据</span>'
      : hok===false
        ? `<span class="pill bad">${hs.hole==null?'质量未知':'不可信'}</span>`+
          ` <span style=color:#8d97a6>空洞 ${hs.hole==null?'未知':(100*hs.hole).toFixed(0)+'%'}`+
          `${hs.wall_frac!=null?` / 墙格 ${(100*hs.wall_frac).toFixed(0)}%`:''}`+
          `${hs.bad?' / 有 NaN':''} —— 看自检那条</span>`
        : (hs.step_d!=null
            ? `<span class="pill warn">前方 ${hs.step_d.toFixed(2)}m 有 ${hs.step_h.toFixed(2)}m 的坎</span>`
            : `<span class="pill ok">前方平整</span>`)
          + ` <span style=color:#8d97a6>身下 ${hs.base!=null?hs.base.toFixed(4):'—'}`
          + ` / 空洞 ${hs.hole==null?'未知':(100*hs.hole).toFixed(0)+'%'}</span>`);
  if(L.cloud_age!=null&&(L.hinfo_age??9)<2){const ca=L.cloud_age;
    $('#hsbase').innerHTML+=` <span class="pill ${ca>0.25?'bad':'ok'}">点云年龄 ${ca.toFixed(2)} s</span>`}
  if(L.hs&&L.hs.under_valid!=null)$('#hsbase').innerHTML+=` <span style=color:#8d97a6>身下有效(含邻域) ${(100*L.hs.under_valid).toFixed(0)}%</span>`;
  if(L.hmap_q&&L.hmap_q.q!=='ok')$('#hsbase').innerHTML+=` <span class="pill ${L.hmap_q.q==='bad'?'bad':'warn'}">切专家：高程图${L.hmap_q.q==='bad'?'不可信':'质量未知'}</span> <span style=color:#8d97a6>${(L.hmap_q.why||'').replace(/</g,'')}</span>`;
  $('#iface').innerHTML=L.cmd_iface
    ?'<span class="pill ok">已订阅</span>'
    :(d.state.runner_pid?'<span class="pill bad">未建立（runner 还在等关节反馈）</span>'
                        :'<span class="pill warn">runner 未启动</span>');
  const ro=L.runner_online;
  $('#runner').innerHTML=ro?'<span class="pill ok">在线</span>':'<span class="pill bad">离线</span>'
    +(d.state.runner_pid?` pid=${d.state.runner_pid}`:'');
  $('#jhz').textContent=(L.j_hz??'—')+' Hz';
  $('#chz').textContent=(L.c_hz??'—')+' Hz';
  $('#cmdout').textContent=(L.cmd_out||[0,0,0]).join(', ');
  $('#steer').textContent=L.steer?`x=${L.steer.x} y=${L.steer.y} yaw=${L.steer.yaw}`:'—';
  if(L.imu){const ar=Math.abs(L.imu.roll), ap=Math.abs(L.imu.pitch);
    const cls=(ar>60||ap>55)?'bad':(ar>40||ap>37)?'warn':'ok';
    $('#rp').innerHTML=`<span class="pill ${cls}">${L.imu.roll}° / ${L.imu.pitch}°</span>`+
      ` <span style=color:#8d97a6>急停阈值 roll 60° / pitch 55°</span>`}
  const XP=L.expert||{on:false}, XG=L.climb_gate||{ok:false,why:''};
  const xi=$('#xpin'), xo=$('#xpout'), xs=$('#xpst');
  if(xi){xi.disabled=!XG.ok||XP.on; xi.style.opacity=xi.disabled?.4:1; xi.title=XG.ok?'':XG.why}
  if(xo){xo.disabled=!XP.on; xo.style.opacity=xo.disabled?.4:1}
  if(xs){xs.innerHTML = (XP.on&&XP.stair) ? '<span style=color:#8d97a6>正在楼梯专家里（见下方）。退出：按 X</span>' : XP.on
    ? (XP.timeout ? `<span class="pill bad">专家已 ${XP.t} s 未上台面</span> 建议松键停车（会先退专家再停），或按 X 手动退出`
       : XP.level ? `<span class="pill ok">已回平</span> 专家 ${XP.t} s —— 可以按 X 手动退出专家`
       : `<span class="pill warn">专家中 ${XP.t} s</span> 须一直按住前进；松键即先退专家再停车`)
      + ` <span style=color:#8d97a6>roll ${XP.roll??'—'}° pitch ${XP.pitch??'—'}°（峰值 ${XP.peak_roll}° / ${XP.peak_pitch}°）</span>`
    : (XG.ok ? '<span style=color:#8fe3b0>可以切专家：离墙约 1 m，按住 W 再按 C（手柄：握 LB/RB、左摇杆前推过半再按 Y）。切入时速度自动设 0.4。</span>'
             : `<span style=color:#8d97a6>现在不能切专家：${XG.why}</span>`)}
  const SG=L.stair_gate||{ok:false,why:''}, si=$('#stin'), so=$('#stout'), ss=$('#stst');
  if(si){si.disabled=!SG.ok||XP.on; si.style.opacity=si.disabled?.4:1; si.title=SG.ok?'':SG.why}
  if(so){so.disabled=!(XP.on&&XP.stair); so.style.opacity=so.disabled?.4:1}
  if(ss){ss.innerHTML = (XP.on&&XP.stair)
    ? `<span class="pill warn">楼梯专家中 ${XP.t} s</span> 须一直按住前进；四轮都上顶平台后按 X 退出；松键即先退回主策略再停车`+
      ` <span style=color:#8d97a6>航向目标 ${XP.yaw_ref??'—'}°，偏差 ${XP.yaw_err??'—'}°，转向 ${(L.cmd_out||[0,0,0])[2]} rad/s（增益 ${L.stair_k?L.stair_k:'关'}；Q/E 手动改朝向）</span>`
    : (SG.ok ? '<span style=color:#8fe3b0>可以切楼梯专家：离第一级约 1 m、正对楼梯，按住 W 再按 V。切入速度 0.5，自动保持航向。</span>'
             : `<span style=color:#8d97a6>现在不能切楼梯专家：${SG.why}</span>`)}
  [['#sk2',2],['#sk1',1],['#sk0',0]].forEach(([id,v])=>{const b=$(id); if(b){b.style.outline=(L.stair_k===v)?'2px solid #8fe3b0':'none'}});
  const ohz=L.o_hz||0;
  $('#od').innerHTML=`<span class="pill ${ohz>=5?'ok':ohz>0?'warn':'bad'}">${ohz} Hz</span>`+
    ((ohz>0&&L.odom)?` <span style=color:#8d97a6>z ${L.odom.z} m / vx ${L.odom.vx} m/s</span>`
      :' <span style=color:#8d97a6>定位静默 —— 感知全盲。看 lidar_watch.log；人工恢复 lio_restart.sh</span>');
  $('#wsp').textContent=L.wheel_m_s!=null?`${L.wheel_m_s} m/s (${(L.wheel_rad_s||[]).join(', ')} rad/s)`:'—';
  $('#tau').textContent=L.tau_max!=null?L.tau_max+' N·m':'—';
  $('#bat').textContent=L.batt?`${L.batt.v} V  ${L.batt.a} A  ${L.batt.soc}%`:'—';
  $('#jt').innerHTML=(L.joints||[]).map(j=>
    `<tr><td>${j.n}</td><td>${j.deg}</td><td>${j.vel}</td><td>${j.tau}</td>`+
    `<td>${j.tm}</td><td>${j.td}</td><td>${j.sw}</td></tr>`).join('');
  $('#log').innerHTML=(d.state.log||[]).map(x=>x.replace(/</g,'&lt;')).join('<br>');
  $('#rlog').innerHTML=(d.state.rosout||[]).length
    ?(d.state.rosout).map(x=>{const e=x.replace(/</g,'&lt;');return x.includes('!!')?`<span style=color:#f87171>${e}</span>`:e}).join('<br>')
    :'<span style=color:#6b7280>还没收到 runner 的 [MODE] 日志或 IMU 提示。runner 没起时这里就是空的。</span>';
  // 限幅的权威在服务端发布回路（clamp_cmd），滑块因此是实时的。
  // 只有当这个 runner 是被旧版用低硬上限启动的，才需要提醒重启一次。
  window.__live=L; window.__runnerpid=d.state.runner_pid;
  const rc=L.rec||{}, rs=$('#recstat');
  if(rc.running){
    const e=rc.elapsed;
    rs.innerHTML=`<span class=rec></span><b style=color:#ff8b95>${rc.name||''}</b>`+
      `　${e!=null?Math.floor(e/60)+'分'+two(Math.floor(e%60))+'秒':''}`+
      `　${rc.size||''}`;
    $('#recgo').disabled=true; $('#recgo').style.opacity=.4;
  }else{
    rs.innerHTML='<span style=color:#8d97a6>未录制</span>';
    $('#recgo').disabled=false; $('#recgo').style.opacity=1;
  }
  const fb=$('#forceb');
  fb.style.display=armed?'none':'';
  const lw=$('#limwarn'), eff=L.eff;
  if(!eff){ lw.style.display='none'; }
  else if(eff.capped_by_runner){
    lw.style.display='block';
    lw.style.background='';lw.style.borderColor='';lw.style.color='';
    lw.innerHTML=`<b>这个 runner 的硬上限只有 ${eff.runner_hard_vx.toFixed(2)} m/s，把滑块卡住了。</b><br>`+
      `它是被旧版控制台用 <code>S10_CMD_MAX_VX=${eff.runner_hard_vx.toFixed(2)}</code> 起的，`+
      `而 runner 里的 <code>max_vx_</code> 只在构造函数读一次，跑起来就改不了。<br>`+
      `<b>停止 runner → 再启动</b>一次就好（新版按策略训练范围给硬上限）。`+
      `之后滑块一直是实时的，不用再重启。`;
  } else {
    lw.style.display='block';
    lw.style.background='#12301f';lw.style.borderColor='#1f5c38';lw.style.color='#8fe3b0';
    lw.innerHTML=`实际下发范围　<b>vx ${eff.vx_lo.toFixed(2)} ~ ${eff.vx_hi.toFixed(2)}</b> m/s`+
      `　·　<b>vy ±${eff.vy.toFixed(2)}</b>　·　<b>wz ±${eff.wz.toFixed(2)}</b> rad/s<br>`+
      `<span style=color:#7fb89a>服务端每发一帧都按当前滑块夹一次 —— 拖动即时生效，不用重启 runner。</span>`;
  }
  offline(false);
}).catch(e=>offline(true))}
poll();setInterval(poll,700);
</script>
"""

PAGE2 = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>S10 控制台（简版）</title>
<style>
:root{--bg:#0e1116;--card:#171b22;--line:#2a313c;--fg:#e6e9ef;--dim:#8d97a6;--ok:#3ddc84;--warn:#ffd166;--bad:#ff5d5d;--acc:#4ea1ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,-apple-system,"PingFang SC","Noto Sans CJK SC",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:10px;display:grid;gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;background:#222933;color:var(--dim);font-size:13px;white-space:nowrap}
.pill.ok{background:#173d2a;color:var(--ok)}.pill.warn{background:#3d3416;color:var(--warn)}.pill.bad{background:#4a1c1c;color:var(--bad)}
button{font:inherit;border:1px solid var(--line);background:#222933;color:var(--fg);border-radius:8px;padding:8px 14px;cursor:pointer}
button:disabled{opacity:.45;cursor:default}button.on{background:#1f3b5c;border-color:var(--acc);color:#fff}
button.big{font-size:18px;padding:12px 18px;font-weight:600}
#estop{background:#7a1f1f;border-color:#ff5d5d;color:#fff;font-size:22px;padding:14px 26px;font-weight:700}
#estop.armed{background:#b91c1c}
.seg button{min-width:110px}
.dpad{display:grid;grid-template-columns:repeat(3,64px);grid-template-rows:repeat(2,64px);gap:6px}
.dpad button{font-size:20px;font-weight:700;padding:0}.dpad button.on{background:#1f5c3b;border-color:var(--ok)}
.step{color:var(--warn);min-height:1.4em}.dim{color:var(--dim);font-size:13px}
.log{font-size:13px;color:var(--dim);white-space:pre-wrap;line-height:1.5}
.terr{font-size:16px}.terr.bad{color:var(--bad)}.terr.ok{color:#9ad}
.gear{padding:6px 12px}
@media (max-width:600px){.dpad{grid-template-columns:repeat(3,56px);grid-template-rows:repeat(2,56px)}}
</style>
<div class=wrap>
<div class=card>
  <div class=row>
    <button id=estop onclick="api('/api/estop')">急停（空格）</button>
    <button onclick="api('/api/clear')" id=clearbtn>解除急停</button>
    <span id=pills class=row style="flex:1"></span>
    <a href="/full" class=dim style="margin-left:auto">完整页</a>
  </div>
</div>
<div class=card>
  <div class=row>
    <button class=big onclick="oneKey()" id=k1>▶ 一键接管</button>
    <button class=big onclick="oneEnd()" id=k2>■ 一键结束</button>
    <span class=step id=step></span>
  </div>
  <div class=dim>一键接管 = 认领操作权 → 接管 → 起 runner（等三套策略加载完）→ 起立（等 runner 确认，站稳 5 s）→ 策略接管。一键结束 = 趴下（在专家里会先退专家）→ 停 runner → 释放接管 → 交出操作权。任一步失败即停在那一步并显示原因。</div>
</div>
<div class=card>
  <div class="row seg">
    <b>模式</b>
    <button id=segflat onclick="seg('flat')">平地</button>
    <button id=seggab onclick="seg('gabion')">石笼（翻墙）</button>
    <button id=segstair onclick="seg('stairs')">楼梯</button>
    <span id=speed class=row></span>
  </div>
  <div class=row style="margin-top:6px">
    <span class=dim>平地策略</span><select id=polsel onchange="api('/api/policy?p='+this.value)"></select>
    <span class=dim>楼梯策略</span><select id=stairsel onchange="api('/api/stairpol?p='+this.value)"></select>
    <span class=dim>石笼策略</span><select id=climbsel onchange="api('/api/climbpol?p='+this.value)"></select>
    <span class=dim id=polnote>换策略要重起 runner：先一键结束，再一键接管</span>
  </div>
  <div id=terr class=terr>前方：—</div>
  <div id=segmsg class=dim></div>
</div>
<div class=card>
  <div class=row style="justify-content:space-between;align-items:flex-start">
    <div>
      <div class=dpad>
        <button data-k=q>Q</button><button data-k=w>W</button><button data-k=e>E</button>
        <button data-k=a>A</button><button data-k=s>S</button><button data-k=d>D</button>
      </div>
      <div class=dim>W 前进 / S 后退 / A D 横移 / Q E 转向。键盘同样可用。楼梯、石笼模式里：不按不动，按住 W 走（专家），松开就停。</div>
    </div>
    <div>
      <div class=row>
        <button id=recbtn onclick="recToggle()">● 开始录制</button>
        <span id=recst class=dim>未录制</span>
      </div>
      <div class=dim>录制名自动：模式_时分。录关节、IMU、定位、高程图，不录点云。</div>
    </div>
  </div>
</div>
<div class=card><div id=log class=log>—</div></div>
</div>
<script>
const $=s=>document.querySelector(s);
const CPID=Math.random().toString(36).slice(2,10); let CSEQ=0;
let S={},L={},OP={},POL={},MODES={},OPR=false,OFF=false,busy=false;
const held=new Set(); const KEYS=['w','a','s','d','q','e'];
const SEGN={flat:'平地',gabion:'石笼',stairs:'楼梯'};
function api(u){if(!/[?&]pid=/.test(u))u+=(u.includes('?')?'&':'?')+'pid='+CPID;
  return fetch(u).then(r=>r.json()).then(j=>{if(j&&j.msg&&j.ok===false)setStep('!! '+j.msg);poll();return j}).catch(()=>({ok:false,msg:'控制台失联'}))}
function setStep(t){$('#step').textContent=t}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function waitFor(f,ms,what){const t0=Date.now();while(Date.now()-t0<ms){await poll();if(f())return true;await sleep(300)}throw '等 '+what+' 超时'}
const chk=()=>{if(S.estop)throw '急停中（先解除急停）';if(OFF)throw '控制台失联'};
const guardTripped=()=>(S.rosout||[]).slice(0,6).some(l=>/STAND_GUARD\] 腿关节跟踪误差/.test(l))||(S.log||[]).slice(0,4).some(l=>/起立保护/.test(l));
async function startRunner(){
  setStep('启动 runner…');let j=await api('/api/runner/start?yes=1');if(j.ok===false)throw j.msg;
  setStep('runner 加载策略中（约 10 s）…');await waitFor(()=>S.runner_pid&&L.runner_online&&L.pit_loaded,30000,'runner 加载三套策略');
  setStep('runner 就绪，等 3 s 再起立…');await sleep(3000)}
async function standUp(){
  if(S.mode!==0&&S.mode!==4){await api('/api/setmode?m=0');await sleep(300)}
  setStep('起立…');const m0=(S.rosout||[]).length;let j=await api('/api/mode?m=1');if(j.ok===false)throw j.msg;
  const t0=Date.now();
  while(Date.now()-t0<8000){await poll();
    if(guardTripped()&&(S.mode===0||S.mode===2))return 'guard';
    if(S.mode===1&&/runner/.test(S.mode_src||''))return 'ok';
    await sleep(300)}
  return 'timeout'}
async function oneKey(){
  if(busy)return;busy=true;
  try{
    await poll();chk();
    if(!OPR){setStep('认领操作权…');let j=await api('/api/op/claim');if(j.ok===false)throw j.msg;await waitFor(()=>OPR,3000,'操作权')}
    if(!S.armed){setStep('接管…');let j=await api('/api/arm?on=1');if(j.ok===false)throw j.msg;await waitFor(()=>S.armed,3000,'接管')}
    chk();
    if(!S.runner_pid)await startRunner();
    chk();
    for(let round=0;round<3;round++){
      if(S.mode===2){setStep('阻尼中，等状态机回 0…');await waitFor(()=>S.mode!==2,5000,'阻尼满 3 s 回 0')}
      if(S.mode===6)break;
      if(S.mode!==1){
        let r=await standUp();
        if(r==='guard'){
          if(round>=1)throw 'runner 起立保护又触发（关节没跟上指令，150 ms 内误差 >20°）。最常见：SDK 模式没开；其次：狗没趴正 / 某条腿被卡。检查后再按一键接管';
          setStep('!! runner 起立保护触发（关节没跟上指令）。先检查 SDK 模式是否打开。10 s 后自动重起 runner 再试一次…');await sleep(10000);chk();
          await api('/api/runner/stop');await waitFor(()=>!S.runner_pid,8000,'runner 退出');await sleep(1000);
          await startRunner();continue}
        if(r==='timeout'){setStep('runner 没回话，按 0 待命重发起立…');await api('/api/setmode?m=0');await sleep(300);
          let r2=await standUp();if(r2!=='ok')throw 'runner 没确认起立（'+r2+'）'}
        setStep('起立中，等站稳…');await sleep(5000);chk();
      }
      setStep('策略接管…');let j=await api('/api/mode?m=6');if(j.ok===false)throw j.msg;
      let ok=await waitFor(()=>S.mode===6&&/runner/.test(S.mode_src||''),6000,'').catch(()=>false);
      if(ok)break;
      if(S.mode===0||S.mode===2){setStep('runner 说它不在起立态，回到起立那步重来…');continue}
      setStep('runner 没确认，重发策略接管…');j=await api('/api/mode?m=6');if(j.ok===false)throw j.msg;
      await waitFor(()=>S.mode===6&&/runner/.test(S.mode_src||''),8000,'runner 确认策略接管');break;
    }
    if(S.mode!==6)throw '三轮都没进到策略接管';
    setStep('就绪：按 W 走；点「楼梯」进楼梯模式（不按 W 不动）')
  }catch(e){setStep('中断：'+e)}finally{busy=false;poll()}
}
async function oneEnd(){
  if(busy)return;busy=true;
  try{
    await poll();
    if(S.climb){held.clear();paint();send();if(!confirm('正在专家里（'+(S.climb===2?'楼梯':'上墙')+'）。一键结束会先退回主策略再趴下；狗若在台阶/墙上请改按急停。继续？'))throw '已取消'}
    if(S.armed&&(S.mode===1||S.mode===6)){setStep('趴下…');let j=await api('/api/mode?m=4');if(j.ok===false)throw j.msg;
      await waitFor(()=>S.mode===4||S.mode===0,8000,'runner 确认趴下');setStep('趴下中，等趴稳…');await sleep(4000)}
    if(S.runner_pid){setStep('停 runner…');let j=await api('/api/runner/stop');if(j.ok===false)throw j.msg;await waitFor(()=>!S.runner_pid,8000,'runner 退出')}
    if(S.armed){setStep('释放接管…');let j=await api('/api/arm?on=0');if(j.ok===false)throw j.msg;await waitFor(()=>!S.armed,3000,'释放')}
    if(OPR){setStep('交出操作权…');await api('/api/op/release');await waitFor(()=>!OPR,3000,'交出操作权')}
    setStep('已完全退出：狗趴下、已释放、runner 已停、操作权已交出。看门狗可自动重启定位；厂家遥控可用。')
  }catch(e){setStep('中断：'+e)}finally{busy=false;poll()}
}
function seg(s){held.clear();paint();api('/api/segment?s='+s).then(j=>{if(j&&j.msg)$('#segmsg').textContent=j.msg})}
function lim(vx,wz){api('/api/lim?vx='+vx+'&wz='+wz)}
function send(){
  if(S.estop||!S.armed||OFF||!OPR)return;
  const p=POL[S.policy]||{vx:[0,0],vy:0,wz:0};const mvx=+S.max_vx||0,mwz=+S.max_wz||0;
  let vx=0,vy=0,wz=0;
  if(held.has('w'))vx=+mvx;if(held.has('s'))vx=-mvx;
  if(held.has('a'))vy=+Math.min(mvx,p.vy||0);if(held.has('d'))vy=-Math.min(mvx,p.vy||0);
  if(held.has('q'))wz=+mwz;if(held.has('e'))wz=-mwz;
  vx=Math.max(Math.min(p.vx?p.vx[0]:0,0),Math.min(p.vx?p.vx[1]:vx,vx));
  wz=Math.max(-(p.wz||0),Math.min(p.wz||0,wz));vy=Math.max(-(p.vy||0),Math.min(p.vy||0,vy));
  fetch('/api/cmd?vx='+vx.toFixed(3)+'&vy='+vy.toFixed(3)+'&wz='+wz.toFixed(3)+'&pid='+CPID+'&seq='+(++CSEQ)).catch(()=>{})}
setInterval(()=>{if(held.size)send()},150);
function paint(){document.querySelectorAll('.dpad button').forEach(b=>b.classList.toggle('on',held.has(b.dataset.k)))}
document.querySelectorAll('.dpad button').forEach(b=>{
  const on=e=>{e.preventDefault();held.add(b.dataset.k);paint();send()};
  const off=e=>{e.preventDefault();if(held.delete(b.dataset.k)){paint();send()}};
  b.addEventListener('pointerdown',on);b.addEventListener('pointerup',off);b.addEventListener('pointercancel',off);b.addEventListener('pointerleave',off)});
addEventListener('keydown',e=>{const k=e.key.toLowerCase();const tg=e.target;
  if(k!==' '&&tg&&(tg.tagName==='INPUT'||tg.tagName==='TEXTAREA'))return;
  if(k===' '){e.preventDefault();api('/api/estop');return}
  if(KEYS.includes(k)&&!held.has(k)){held.add(k);paint();send();e.preventDefault()}});
addEventListener('keyup',e=>{const k=e.key.toLowerCase();if(held.delete(k)){paint();send()}});
addEventListener('blur',()=>{if(held.size){held.clear();paint();send();setStep('窗口失焦，已松键')}});
async function recToggle(){
  const r=L.rec||{};
  if(r.running){await api('/api/rec/stop');return}
  const nm=(S.segment||'flat')+'_'+new Date().toTimeString().slice(0,5).replace(':','');
  await api('/api/rec/start?name='+nm+'&mode=extra')}
function pill(t,c){return '<span class="pill '+(c||'')+'">'+t+'</span>'}
function render(){
  const b=L.batt||{},soc=b.soc;
  const opTxt=OP.you?'操作权：本页':(OP.held?'操作权：别的页面':'操作权：空');
  const hq=(L.hmap_q||{}).q,odomOk=L.odom&&L.odom.frame;
  const climbName=S.climb===2?'楼梯专家':S.climb===1?'ClimbH':'';
  $('#pills').innerHTML=[
    pill(opTxt,OP.you?'ok':(OP.held?'warn':'')),
    pill(S.armed?'已接管':'未接管',S.armed?'ok':''),
    pill(S.runner_pid?'runner 在':'runner 无',S.runner_pid?'ok':'warn'),
    pill('模式 '+(MODES[S.mode]||S.mode)+(S.mode_src?'（'+S.mode_src+'）':''),S.mode===6?'ok':''),
    climbName?pill('专家 '+climbName,'ok'):'',
    pill('电量 '+(soc==null?'—':soc+'%'),soc==null?'':(soc<25?'bad':soc<40?'warn':'ok')),
    pill('定位 '+(odomOk?'正常':'无'),odomOk?'ok':'bad'),
    pill('地图 '+(hq==='ok'?'正常':hq==='bad'?'不可信':'未知'),hq==='ok'?'ok':hq==='bad'?'bad':'warn'),
    S.estop?pill('急停中','bad'):''
  ].join('');
  $('#estop').classList.toggle('armed',!!S.armed);
  const sg=S.segment||'flat';
  [['#segflat','flat'],['#seggab','gabion'],['#segstair','stairs']].forEach(([id,v])=>$(id).classList.toggle('on',sg===v));
  let sp='';
  if(sg==='flat'){sp='<span class=dim>前进上限</span>'+[0.3,0.5,0.8,1.0,1.5].map(v=>'<button class="gear'+(Math.abs(v-S.max_vx)<0.01?' on':'')+'" onclick="lim('+v+','+(S.max_wz||0.5)+')">'+v.toFixed(1)+'</button>').join('')
      +'<span class=dim>转向</span>'+[0.5,1.0].map(v=>'<button class="gear'+(Math.abs(v-S.max_wz)<0.01?' on':'')+'" onclick="lim('+(S.max_vx||0.3)+','+v+')">'+v.toFixed(1)+'</button>').join('')}
  else if(sg==='gabion')sp='<span class=dim>石笼速度</span>'+[0.3,0.4].map(v=>'<button class="gear'+(Math.abs(v-S.max_vx)<0.01?' on':'')+'" onclick="lim('+v+','+(S.max_wz||0.5)+')">'+v.toFixed(1)+'</button>').join('')+'<span class=dim>（按住 W 以此速度上墙，松开就停；点「平地」退出石笼模式）</span>';
  else sp='<span class=dim>楼梯速度</span>'+(function(){const q=POL[S.stair_choice]||{};const r=q.vx;return (q.stop_ok&&r)?[0.1,0.15,0.2,0.25,0.3,0.4,0.5].filter(v=>v>=r[0]-1e-6&&v<=Math.min(r[1],0.5)+1e-6):[0.1,0.15,0.2,0.3,0.4,0.5]})().map(v=>'<button class="gear'+(Math.abs(v-S.max_vx)<0.01?' on':'')+'" onclick="lim('+v+','+(S.max_wz||0.5)+')">'+v.toFixed(2).replace(/0$/,'')+'</button>').join('')+'<span class=dim>'+(((POL[S.stair_choice]||{}).stop_ok)?'（该专家能停：进模式狗站住，W 走、松开停；点「平地」退出）':'（官方约 0.5；按住 W 以此速度上楼，松开就停；点「平地」退出楼梯模式）')+'</span>';
  $('#speed').innerHTML=sp;
  const ps=$('#polsel'),ss=$('#stairsel'),cs=$('#climbsel');
  const key=Object.keys(POL).join()+'|'+S.policy+'|'+S.stair_choice+'|'+S.climb_choice;if(ps.dataset.n!==key){ps.dataset.n=key;
    const FLAT=['T10_15197','V1HTrotT3_13400','V1HTrotH7_11800','V1HTrotH4_11500','V1HTrotH6_12099'],CLIMBS=['ClimbR13','ClimbH_8797'],STAIRS=['StairS16_15200','StairN-S8_14800','StairN-S3_13800','StairN-E2_10300','StairN-E5_11200'];   // 作者 09-13：只要 H6_12099 对 H4_11500、A 对 D
    const opt=(list,cur)=>list.concat(cur&&!list.includes(cur)?[cur]:[]).filter(k=>POL[k]).map(k=>'<option value="'+k+'">'+k+(list.includes(k)?'':'（当前，不在备选里）')+'</option>').join('');
    ps.innerHTML=opt(FLAT,S.policy);ss.innerHTML=opt(STAIRS,S.stair_choice);if(cs)cs.innerHTML=opt(CLIMBS,S.climb_choice)}
  if(document.activeElement!==ps)ps.value=S.policy||'';if(document.activeElement!==ss)ss.value=S.stair_choice||'';if(cs&&document.activeElement!==cs)cs.value=S.climb_choice||'';
  ps.disabled=!OPR;ss.disabled=!OPR;if(cs)cs.disabled=!OPR;
  const rp=L.running_policy,rpit=(L.running_pit!=null?L.running_pit:null);
  $('#polnote').textContent=(S.runner_pid&&((rp&&rp!==S.policy)||(S.running_pit&&S.running_pit!==S.stair_choice)))?'!! runner 带的是 '+(rp||'?')+' / '+(S.running_pit||'?')+'，选的没生效：先一键结束再一键接管':'换策略要重起 runner：先一键结束，再一键接管';
  const T=L.terrain;const te=$('#terr');
  if(!T){te.textContent='前方：—';te.className='terr'}
  else if(T.kind==='平地'){te.textContent='前方：平地';te.className='terr'}
  else{const yo=(T.yaw_med!=null?T.yaw_med:T.yaw_off);
    te.textContent='前方：'+T.kind+' 沿 '+T.dist+' m　'+(T.square==null?'':(T.square?'正对':'!! 未正对'+(yo==null?'':('，需'+(yo>0?'左':'右')+'转 '+Math.abs(yo).toFixed(0)+'°')+'，用 Q/E 对正')));
    te.className='terr '+(T.square===false?'bad':'ok')}
  if(L.seg_deny)$('#segmsg').textContent='!! 切不进专家、不前进：'+L.seg_deny;
  const r=L.rec||{};$('#recbtn').textContent=r.running?'■ 停止录制':'● 开始录制';$('#recbtn').classList.toggle('on',!!r.running);
  $('#recst').textContent=r.running?('录制中 '+(r.name||'')+'  '+(r.elapsed!=null?Math.round(r.elapsed)+' s':'')+'  '+(r.size||'')):'未录制';
  $('#log').textContent=(S.log||[]).slice(0,4).join('\n')||'—';
  $('#k1').disabled=busy;$('#k2').disabled=busy;
  document.querySelectorAll('.dpad button').forEach(b=>b.disabled=!OPR||!S.armed);
}
function poll(){return fetch('/api/state?pid='+CPID).then(r=>r.json()).then(d=>{
  OFF=false;S=d.state||{};L=d.live||{};OP=d.op||{};POL=d.policies||{};MODES=d.modes||{};OPR=!!OP.you;
  if(!OPR&&held.size){held.clear();paint()}
  render()}).catch(()=>{OFF=true;$('#pills').innerHTML=pill('控制台失联','bad')})}
setInterval(poll,700);poll();
</script>
"""



def _qint(q, k):
    try:
        return int(q.get(k, [""])[0])
    except (TypeError, ValueError):
        return None


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _s(self, code, body, ct="application/json; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(b)
        except Exception:
            pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        p = u.path

        if p == "/":
            return self._s(200, PAGE2, "text/html; charset=utf-8")
        if p == "/full":
            return self._s(200, PAGE, "text/html; charset=utf-8")

        if p == "/api/state":
            spid = q.get("pid", [""])[0]
            now = time.time()
            if spid and spid == OP["pid"]:
                OP["hb"] = now                  # 09-12 凌晨：只有操作员页面的轮询算心跳
            opv = {"held": bool(OP["pid"]), "you": bool(spid) and spid == OP["pid"],
                   "age": round(now - OP["hb"], 1) if OP["pid"] else None,
                   "who": OP["who"] if OP["pid"] else "", "stale_s": OP_STALE_S,
                   "spectator_estop": SPECTATOR_ESTOP, "refused": OP["refused"]}
            return self._s(200, json.dumps(
                {"state": STATE, "live": LIVE, "policies": POLICIES,
                 "modes": MODES, "op": opv}, ensure_ascii=False))

        if p == "/api/op/claim":            # 09-12 凌晨（Astra 建议 3）：认领操作权
            cpid = q.get("pid", [""])[0]
            if not re.match(r"^[A-Za-z0-9_-]{4,32}$", cpid):
                return self._s(200, json.dumps({"ok": False, "msg": "缺少或非法的页面编号 pid"}, ensure_ascii=False))
            now = time.time()
            who = getattr(self, "client_address", ("?",))[0]
            with OPLK:
                if OP["pid"] == cpid:
                    OP["hb"] = now
                    return self._s(200, json.dumps({"ok": True, "msg": "本页已经是操作员"}, ensure_ascii=False))
                if OP["pid"] and now - OP["hb"] <= OP_STALE_S:
                    return self._s(200, json.dumps(
                        {"ok": False, "msg": "已有操作员页面在线（来自 %s，%.1f s 前还有心跳）。要换人：在那个页面点「交出操作权」，"
                                             "或关掉它等 %.0f 秒再点「我来操作」。" % (OP["who"], now - OP["hb"], OP_STALE_S)},
                        ensure_ascii=False))
                old_age = (now - OP["hb"]) if OP["pid"] else None
                OP.update(pid=cpid, hb=now, since=now, who=who)
            with LK:
                CMDSEQ.update(pid=cpid, seq=-1)   # 新操作员的序号从头算；旧页面的请求在下面的闸门就被拒
            logline("操作权 → 页面 %s…（来自 %s）%s" % (cpid[:4], who,
                    "" if old_age is None else "；原操作员页面已 %.0f s 没心跳" % old_age))
            return self._s(200, json.dumps({"ok": True, "msg": "本页现在是操作员"}, ensure_ascii=False))

        if p == "/api/op/release":          # 09-12 凌晨：交出操作权（只改谁能发指令，不发任何话题）
            cpid = q.get("pid", [""])[0]
            with OPLK:
                if not cpid or cpid != OP["pid"]:
                    return self._s(200, json.dumps({"ok": False, "msg": "本页面不是操作员，没有可交出的操作权"},
                                                   ensure_ascii=False))
                OP.update(pid="", hb=0.0, since=0.0, who="")
            logline("操作员页面交出操作权%s" % ("（仍在接管中：没有页面再发指令，0.4 s 后按停刷处理）" if STATE["armed"] else ""))
            return self._s(200, json.dumps({"ok": True, "msg": "已交出操作权，本页变为只读"}, ensure_ascii=False))

        # 09-12 凌晨（Astra 建议 3）：除只读接口外，只接受操作员页面的请求。旁观页面、刷新后的新 pid、
        # 旧页面迟到的请求、不带 pid 的请求一律只读。dry=1 只对真正实现了演练的两个接口放行（别的接口没有 dry，放行就是绕过）。
        if p.startswith("/api/") and p not in OP_OPEN and not (p in OP_DRY_OK and q.get("dry", ["0"])[0] == "1"):
            rpid = q.get("pid", [""])[0]
            if not (rpid and rpid == OP["pid"]) and not (p == "/api/estop" and SPECTATOR_ESTOP):
                OP["refused"] += 1
                return self._s(200, json.dumps(
                    {"ok": False, "readonly": True,
                     "msg": "本页面是旁观（只读），不接受控制指令%s。要操作先点「我来操作」认领操作权。"
                            % ("" if OP["pid"] else "（现在没有操作员）")}, ensure_ascii=False))


        if p == "/api/arm":
            on = q.get("on", ["0"])[0] == "1"
            force = q.get("force", ["0"])[0] == "1"
            if on and not STATE["runner_pid"] and not force:
                chz = LIVE.get("c_hz") or 0
                if chz > 20:
                    return self._s(200, json.dumps(
                        {"ok": False, "msg": "别人的控制器正在以 %.0f Hz 发 /JOINTS_CMD。"
                         "现在接管会和它抢关节指令。等对方停了再来。" % chz},
                        ensure_ascii=False))
                if LIVE.get("runner_online"):
                    silent = LIVE.get("c_silent_s") or 0
                    if silent > 30:
                        # 进程没干净退出会在 DDS 里留下节点条目，靠节点名判断会把人
                        # 永久挡在外面。以"有没有真在发指令"为准：静默这么久 = 残留。
                        logline("总线上有 rl_deploy 节点，但 /JOINTS_CMD 已静默 %.0f 秒，"
                                "判为上次没退干净的残留，放行接管" % silent)
                    else:
                        return self._s(200, json.dumps(
                            {"ok": False, "msg":
                             "总线上有不是我们启动的 rl_deploy 节点，而且 %.0f 秒前还在发 "
                             "/JOINTS_CMD —— 对方可能随时接管。等它静默满 30 秒会自动放行，"
                             "或者用「强制接管」。" % silent},
                            ensure_ascii=False))
            if not on and STATE["climb"] in (1, 2):
                climb_exit("释放接管", 0.0)     # 趁 may_publish 还开着先发 /climb_mode 0
            STATE["armed"] = on
            with LK:
                TGT.update(vx=0.0, vy=0.0, wz=0.0, t=0.0)
            if not on:
                STATE["mode"] = 0          # 释放后状态机回到"未知/待命"，下次从 0→1 走
                STATE["estop"] = False
                STATE["estop_reason"] = ""
            logline("接管 " + ("开启" if on else "释放"))
            return self._s(200, json.dumps({"ok": True}))

        if p == "/api/mode":
            try:
                m = int(q.get("m", ["-1"])[0])
            except ValueError:
                m = -1
            if m not in MODES:
                return self._s(400, json.dumps({"ok": False, "msg": "非法模式"}, ensure_ascii=False))
            if m not in (1, 2, 4, 6):
                return self._s(200, json.dumps({"ok": False, "msg": "0911g 复测：只接受 1/2/4/6"}, ensure_ascii=False))
            cur = STATE["mode"]
            ok, why = mode_allowed(cur, m)
            if STATE["climb"] in (1, 2) and m in (1, 6):
                # 09-11：专家里不许重新起立 / 进 RL。返回 ok（带 ignored），免得页面弹窗停刷把专家撤掉
                msg = ("正在%s里，已忽略 /robot_mode %d（%s）。先按 X 退出专家"
                       % ("楼梯专家" if STATE["climb"] == 2 else "上墙专家 ClimbH", m, MODES[m]))
                if q.get("dry", ["0"])[0] != "1":
                    logline("!! " + msg)
                    return self._s(200, json.dumps({"ok": True, "ignored": True, "msg": msg}, ensure_ascii=False))
                ok, why = False, msg
            dry = q.get("dry", ["0"])[0] == "1"
            if dry:
                # 演练：把两道闸门的结论都算出来，一条话题都不发。
                # 测状态转移表必须走这条，别再拿真发布去试。
                return self._s(200, json.dumps(
                    {"ok": True, "dry": True, "transition_ok": ok,
                     "would_publish": bool(ok and may_command()),
                     "cur": cur, "want": m,
                     "msg": why or "转移合法"}, ensure_ascii=False))
            if not ok:
                return self._s(200, json.dumps({"ok": False, "msg": why}, ensure_ascii=False))
            if not may_command():
                return self._s(200, json.dumps(
                    {"ok": False, "msg": "未接管 —— 不向共用总线发 /robot_mode。先点「接管」。"},
                    ensure_ascii=False))
            if m == 6:
                # 6 = RLControlMode，策略从这一刻开始吃 244 维观测（含 187 维高程图）
                gok, gwhy = hscan_gate()
                if not gok:
                    return self._s(200, json.dumps(
                        {"ok": False, "msg": "不让切策略接管：" + gwhy}, ensure_ascii=False))
            if STATE["climb"] in (1, 2) and m in (2, 4):
                climb_exit("切状态机 %d（%s）" % (m, MODES[m]), 0.0)   # 09-11：趴下/阻尼前先退专家（先发 /climb_mode 0）
            if m == 6:
                u0 = UInt8(); u0.data = 0
                PUB["climb"].publish(u0)   # 09-11：进 RL 前清掉 runner 里可能残留的专家标志（幂等）
            u = UInt8(); u.data = m
            PUB["mode"].publish(u)
            STATE["mode"] = m
            STATE["mode_src"] = "假设"        # 等 /rosout 回话才算数
            if m == 2:
                STATE["damp_t"] = time.time()
            logline("发 /robot_mode = %d (%s)" % (m, MODES[m]))
            return self._s(200, json.dumps({"ok": True, "msg": MODES[m]}, ensure_ascii=False))

        if p == "/api/cmd":
            if q.get("dry", ["0"])[0] == "1":
                # 演练：只回答"这组指令发出去会被夹成什么"，不碰 TGT、不发布。
                vx, vy, wz = clamp_cmd(float(q.get("vx", [0])[0]),
                                       float(q.get("vy", [0])[0]),
                                       float(q.get("wz", [0])[0]))
                lo, hi, vyl, wzl = disp_limits()
                return self._s(200, json.dumps(
                    {"ok": True, "dry": True, "out": [round(vx, 3), round(vy, 3), round(wz, 3)],
                     "range": {"vx": [round(lo, 2), round(hi, 2)],
                               "vy": round(vyl, 2), "wz": round(wzl, 2)}}, ensure_ascii=False))
            if STATE["estop"]:
                return self._s(200, json.dumps({"ok": False, "msg": "急停中"}, ensure_ascii=False))
            cpid, cseq = q.get("pid", [""])[0], _qint(q, "seq")
            with LK:
                if cpid and cseq is not None:
                    if cpid == CMDSEQ["pid"] and cseq <= CMDSEQ["seq"]:
                        return self._s(200, json.dumps({"ok": True, "dropped": "乱序的旧请求"},
                                                       ensure_ascii=False))
                    CMDSEQ.update(pid=cpid, seq=cseq)
                TGT["vx"] = float(q.get("vx", [0])[0])
                TGT["vy"] = float(q.get("vy", [0])[0])
                TGT["wz"] = float(q.get("wz", [0])[0])
                TGT["t"] = time.time()
            return self._s(200, json.dumps({"ok": True}))

        if p == "/api/estop":
            sent = do_estop("手动")
            if sent:
                return self._s(200, json.dumps(
                    {"ok": True, "sent": True,
                     "msg": "急停已发：/cmd_vel 归零 + /robot_mode=2 + G20_KEY_R2"},
                    ensure_ascii=False))
            # 别让人以为按下去就停了 —— 未接管时我们本来就一条话题都没在发
            return self._s(200, json.dumps(
                {"ok": True, "sent": False,
                 "msg": "未接管，本次急停只记在本地，没有向总线发任何话题。"
                        "我们此刻也没有在发 /cmd_vel，所以没有东西需要停。"
                        "如果狗正在动，那是另一方或手柄在控，用手柄急停。"},
                ensure_ascii=False))

        if p == "/api/setmode":
            # 我们读不到状态机的真实状态（MotionStateFeedback 不发话题），
            # 跟踪跑偏时给用户一个纯本地的纠正入口。这条不发任何话题。
            try:
                m = int(q.get("m", ["0"])[0])
            except ValueError:
                m = -1
            if m not in MODES:
                return self._s(400, json.dumps({"ok": False, "msg": "非法模式"}, ensure_ascii=False))
            STATE["mode"] = m
            STATE["mode_src"] = "人工设定"
            STATE["damp_t"] = time.time() if m == 2 else 0.0
            logline("人工把状态跟踪改成 %d (%s) —— 只改本地，不发话题" % (m, MODES[m]))
            return self._s(200, json.dumps({"ok": True, "msg": MODES[m]}, ensure_ascii=False))

        if p == "/api/clear":
            STATE["estop"] = False
            STATE["estop_reason"] = ""
            logline("解除急停")
            return self._s(200, json.dumps({"ok": True, "msg": "已解除"}, ensure_ascii=False))

        if p == "/api/key":
            k = q.get("k", [""])[0]
            if k not in ("G20_KEY_L1", "G20_KEY_L2", "G20_KEY_R1", "G20_KEY_R2"):
                return self._s(400, json.dumps({"ok": False, "msg": "按键不在白名单"}, ensure_ascii=False))
            if not may_command():
                return self._s(200, json.dumps(
                    {"ok": False, "msg": "未接管 —— 不向共用总线发 /GAMEPAD_KEY。先点「接管」。"},
                    ensure_ascii=False))
            m = String(); m.data = k
            PUB["key"].publish(m)
            logline("发 /GAMEPAD_KEY = " + k)
            return self._s(200, json.dumps({"ok": True, "msg": k}))

        if p == "/api/policy":
            k = q.get("p", [""])[0]
            if k not in POLICIES:
                return self._s(400, json.dumps({"ok": False, "msg": "未知策略"}, ensure_ascii=False))
            pol = POLICIES[k]
            if pol.get("expert"):
                # 点专家 = 请求切专家（同样必须按住前进）；旧版 ClimbG 不在 runner 的专家槽里
                if k == STAIR:            # 09-11 楼梯专家
                    ok, msg = stair_enter(q.get("pid", [""])[0], _qint(q, "seq"), "策略列表")
                    return self._s(200, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False))
                if k != EXPERT:
                    return self._s(200, json.dumps(
                        {"ok": False, "msg": "%s 是旧版专家，runner 的专家槽固定装 %s，不能切入" % (k, EXPERT)},
                        ensure_ascii=False))
                ok, msg = climb_enter(q.get("pid", [""])[0], _qint(q, "seq"), "策略列表")
                return self._s(200, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False))
            if pol.get("old"):
                # 09-11 作者拍板：旧版不能选成基础策略（runner_start 也会拒绝，这里提前说清）
                logline("拒绝选择 %s：%s" % (k, _OLD))
                return self._s(200, json.dumps({"ok": False, "msg": "%s %s" % (k, _OLD)}, ensure_ascii=False))
            if STATE["climb"] in (1, 2):
                climb_exit("选基础策略 %s" % k, expert_out_vx())
            STATE["policy"] = k
            # 换策略后把上限夹到新策略的范围内，避免残留一个过大的档位
            old = (STATE["max_vx"], STATE["max_wz"])
            STATE["max_vx"] = min(STATE["max_vx"], pol["vx"][1])
            STATE["max_wz"] = min(STATE["max_wz"], pol["wz"])
            if (STATE["max_vx"], STATE["max_wz"]) != old:
                logline("上限按 %s 范围夹紧: vx %.2f→%.2f  wz %.2f→%.2f"
                        % (k, old[0], STATE["max_vx"], old[1], STATE["max_wz"]))
            if not STATE["runner_pid"]:
                logline("选择策略 %s（站姿组 %s；runner 还没起，启动时就用它）" % (k, pol["stance"]))
            elif k != STATE["running_policy"]:
                logline("!! 选中 %s（站姿组 %s），但 runner 里跑的基础策略是 %s（站姿组 %s）—— "
                        "基础策略和站姿都在进程启动时加载，换它必须停 runner 再启动%s"
                        % (k, pol["stance"], STATE["running_policy"], STATE.get("running_group") or "?",
                           "（站姿组不同，更不能热切）" if pol["stance"] != STATE.get("running_group") else ""))
            return self._s(200, json.dumps({"ok": True, "msg": k}))

        if p == "/api/segment":          # night4b：赛段选择（操作员）
            sg = q.get("s", [""])[0]
            if sg not in ("flat", "stairs", "gabion"):
                return self._s(400, json.dumps({"ok": False, "msg": "s 只能是 flat / gabion / stairs"}, ensure_ascii=False))
            if sg == "gabion":             # night4d：石笼赛段（ClimbH，/climb_mode 1，切入 0.4）
                if not STATE["runner_pid"]:
                    return self._s(200, json.dumps({"ok": False, "msg": "runner 没起，先起 runner"}, ensure_ascii=False))
                if STATE.get("running_climb") != EXPERT:
                    return self._s(200, json.dumps({"ok": False, "msg": "这个 runner 的专家槽不是 %s（是 %s），停 runner 再起一次" % (EXPERT, STATE.get("running_climb") or "空")}, ensure_ascii=False))
                if STATE.get("running_group") != EXPERT_STANCE:
                    return self._s(200, json.dumps({"ok": False, "msg": "runner 站姿组 %s 不是 %s，ClimbH 只能配 0.42 组主策略" % (STATE.get("running_group") or "?", EXPERT_STANCE)}, ensure_ascii=False))
                if STATE["climb"] == 2:
                    climb_exit("换到石笼赛段", expert_out_vx())
                if STATE.get("segment") == "flat":
                    SEG["flat_vx"] = STATE["max_vx"]
                if abs(STATE["max_vx"] - EXPERT_ENTRY_VX) > 1e-9:
                    logline("档位 vx %.2f → %.2f（石笼赛段默认，滑块仍可调）" % (STATE["max_vx"], EXPERT_ENTRY_VX))
                    STATE["max_vx"] = EXPERT_ENTRY_VX
                STATE["segment"] = "gabion"; SEG["deny"] = None
                logline("赛段 → 石笼（全程专家）：立刻切 ClimbH 并以 0.4 走向墙（零指令会原地转，所以恒 0.4）；停车点「平地」或急停（作者 09-13 拍板 A）" if SEG_FULL else "赛段 → 石笼：离墙约 1 m 按住 W 自动切 ClimbH 以 0.4 上墙，松 W 交回主策略停车，S 倒退交主策略，上到台面后选「平地赛段」")
                return self._s(200, json.dumps({"ok": True, "msg": "石笼赛段：档位已调到 %.1f；离墙约 1 m 按住 W 开始" % STATE["max_vx"]}, ensure_ascii=False))
            if sg == "stairs":
                if not STATE["runner_pid"]:
                    return self._s(200, json.dumps({"ok": False, "msg": "runner 没起，先起 runner（它会带楼梯槽）"}, ensure_ascii=False))
                if STATE.get("running_pit") != STAIR or not STATE.get("pit_loaded"):
                    return self._s(200, json.dumps({"ok": False, "msg": "这个 runner 没带/没加载楼梯策略 %s，停 runner 再起一次" % STAIR}, ensure_ascii=False))
                if STATE["climb"] == 1:
                    climb_exit("换到楼梯赛段", expert_out_vx())
                if STATE.get("segment") == "flat":
                    SEG["flat_vx"] = STATE["max_vx"]
                _pol = POLICIES.get(STAIR, {})
                _dv = float(_pol["seg_vx"]) if _pol.get("seg_vx") else (max(STAIR_SEG_DEFAULT_VX, (_pol.get("vx") or [0.0])[0]) if stair_can_stop() else STAIR_SEG_DEFAULT_VX)
                if abs(STATE["max_vx"] - _dv) > 1e-9:
                    logline("档位 vx %.2f → %.2f（楼梯赛段默认）" % (STATE["max_vx"], _dv))
                    STATE["max_vx"] = _dv
                STATE["segment"] = "stairs"; SEG["deny"] = None
                logline("赛段 → 楼梯（专家 %s 能停）：立刻切楼梯专家，狗原地站住；W = 档位速度、松开 = 停；点「平地」退出楼梯模式" % STAIR if (stair_can_stop() and not SEG_FULL) else "赛段 → 楼梯（全程专家）：立刻切楼梯专家并开始走；W = 0.5、松开 = 0（专家会再走一段）；停车点「平地」或急停（作者 09-13 拍板 A）" if SEG_FULL else ("赛段 → 楼梯：按住 W 自动切楼梯专家前进；切入后松 W 不退专家、一口气爬到顶，只有急停 / X / 页面失联 3 s 才退（作者 09-12 拍板）；到顶后按 X 或选「平地赛段」" if STAIR_HOLD_ON else "赛段 → 楼梯：按住 W 自动切楼梯专家前进，松 W 交回主策略停车，S 倒退交主策略，到顶后选「平地赛段」"))
                return self._s(200, json.dumps({"ok": True, "msg": ("楼梯赛段（专家能停）：已切专家，狗站住；按 W 以 %.1f 上，松开停；点「平地」退出" if (stair_can_stop() and not SEG_FULL) else "楼梯赛段（全程专家）：狗立刻开始走（档位 %.1f）；正对楼梯；停车点「平地」或急停" if SEG_FULL else ("楼梯赛段：档位已调到 %.1f；离第一级约 1 m 正对楼梯，按住 W 开始；切入后松 W 不退专家，出事按空格急停" if STAIR_HOLD_ON else "楼梯赛段：档位已调到 %.1f；离第一级约 1 m 正对楼梯，按住 W 开始")) % STATE["max_vx"]}, ensure_ascii=False))
            was = STATE["climb"] in (1, 2)
            if was:
                climb_exit("退出专家赛段", expert_out_vx())
            STATE["segment"] = "flat"; SEG["deny"] = None
            if SEG.get("flat_vx") is not None:
                if abs(STATE["max_vx"] - SEG["flat_vx"]) > 1e-9:
                    logline("档位 vx %.2f → %.2f（恢复选赛段前的档位）" % (STATE["max_vx"], SEG["flat_vx"]))
                STATE["max_vx"] = SEG["flat_vx"]; SEG["flat_vx"] = None
            logline("赛段 → 平地%s" % ("（已退出专家）" if was else ""))
            return self._s(200, json.dumps({"ok": True, "msg": "平地赛段"}, ensure_ascii=False))

        if p == "/api/climb":
            # 手动切入/退出专家（R6）。只有操作员能触发：键盘 C/X、手柄 Y/X、页面按钮。
            onv = q.get("on", ["0"])[0]
            on = onv == "1"
            src = q.get("src", ["页面"])[0][:16]
            if onv == "2":                # 09-11 楼梯专家
                ok, msg = stair_enter(q.get("pid", [""])[0], _qint(q, "seq"), src)
            elif on:
                ok, msg = climb_enter(q.get("pid", [""])[0], _qint(q, "seq"), src)
            else:
                was, sent = climb_exit("手动退出（%s）" % src, expert_out_vx())
                ok = True
                msg = ("已退出专家，交回主策略（前进限在当前速度、不自动加速；松开前进再按恢复档位）" if was
                       else "本来就不在专家模式" + ("（仍补发了 /climb_mode 0）" if sent else ""))
            return self._s(200, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False))

        if p == "/api/climbpol":          # night5g：运行时选爬墙槽策略
            ok, msg = set_climb_policy(q.get("p", [""])[0])
            return self._s(200, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False))

        if p == "/api/stairpol":          # night4u：运行时选楼梯槽策略
            ok, msg = set_stair_policy(q.get("p", [""])[0])
            return self._s(200, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False))

        if p == "/api/stairhead":         # 09-11 夜审：清除记下的楼梯朝向（换一段楼梯时用；楼梯专家里不许清）
            with LK:
                if STATE["climb"] == 2:
                    return self._s(200, json.dumps({"ok": False, "msg": "楼梯专家里不能清除楼梯朝向，先退出"}, ensure_ascii=False))
                old = STATE.get("stair_heading")
                STATE["stair_heading"] = None
            logline("清除楼梯朝向（原 %s），下次切楼梯专家时重新记" % ("—" if old is None else "%.1f°" % math.degrees(old)))
            return self._s(200, json.dumps({"ok": True, "msg": "已清除楼梯朝向，下次切入时重新记"}, ensure_ascii=False))

        if p == "/api/stairk":            # 09-11 楼梯航向保持增益（作者：页面三档 2 / 1 / 关）
            try:
                k = float(q.get("k", ["-1"])[0])
            except ValueError:
                k = -1.0
            if k not in STAIR_K_CHOICES:
                return self._s(200, json.dumps({"ok": False, "msg": "增益只能是 2 / 1 / 0"}, ensure_ascii=False))
            with LK:
                old = STATE["stair_k"]
                STATE["stair_k"] = k
                if STATE["climb"] == 2 and not old and k:
                    STATE["stair_yaw_ref"] = imu_yaw_fresh()[0]   # 从「关」打开：以当前航向为目标，不一步拉回切入时的朝向
            if k != old:
                logline("楼梯航向保持增益 %s → %s%s" % (("关" if not old else "%.0f" % old), ("关" if not k else "%.0f" % k),
                                                 "（只剩 Q/E 手动转向）" if not k else ""))
            return self._s(200, json.dumps({"ok": True, "msg": "航向保持 %s" % ("关" if not k else "%.0f" % k)},
                                           ensure_ascii=False))

        if p == "/api/lim":
            old = (STATE["max_vx"], STATE["max_wz"])
            STATE["max_vx"] = max(0.0, min(3.5, float(q.get("vx", [0.3])[0])))   # FlatFast16 到 3.5
            STATE["max_wz"] = max(0.0, min(2.0, float(q.get("wz", [0.5])[0])))
            # 只在真的变了才记日志 —— 之前拖一次滑块刷十几行，日志没法看
            if (STATE["max_vx"], STATE["max_wz"]) != old:
                lo, hi, _, wz = disp_limits()
                logline("上限 vx=%.2f wz=%.2f → 实际下发范围 vx %.2f~%.2f wz ±%.2f（即时生效）"
                        % (STATE["max_vx"], STATE["max_wz"], lo, hi, wz))
            return self._s(200, json.dumps({"ok": True}))

        if p == "/api/hscan":
            STATE["hscan"] = q.get("on", ["0"])[0] == "1"
            logline("/height_scan 零填充 %s" % ("开" if STATE["hscan"] else "关"))
            return self._s(200, json.dumps({"ok": True}))

        if p == "/api/rec/status":
            return self._s(200, json.dumps(rec_status(), ensure_ascii=False))

        if p == "/api/rec/start":
            name = (q.get("name", [""])[0] or "").strip()
            STATE["rec_note"] = (q.get("note", [""])[0] or "").strip()[:300]
            if not NAME_OK.match(name):
                return self._s(200, json.dumps(
                    {"ok": False, "msg": "名字只能用字母数字和 _ . -，首字符须是字母或数字，"
                                         "最长 48 —— 它会直接当目录名用"}, ensure_ascii=False))
            if os.path.exists(os.path.join(REC_DIR, name)):
                return self._s(200, json.dumps(
                    {"ok": False, "msg": "已经有叫 %s 的了，换一个名字（不覆盖旧数据）" % name},
                    ensure_ascii=False))
            mode = q.get("mode", ["extra"])[0]
            if mode not in ("core", "extra", "cloud"):
                mode = "extra"
            out = rec_sh("start", name, mode)
            ok = "STARTED" in out
            if ok:
                STATE["rec_t0"] = time.time()
                rec_context(name)
                logline("开始录制 %s（%s）%s" % (name, mode,
                        ("  备注: " + STATE["rec_note"]) if STATE["rec_note"] else ""))
            else:
                logline("录制启动失败: %s" % out.strip().splitlines()[:1])
            rec_status(force=True)
            return self._s(200, json.dumps({"ok": ok, "msg": out.strip()}, ensure_ascii=False))

        if p == "/api/rec/stop":
            out = rec_sh("stop", timeout=90)
            STATE["rec_t0"] = 0.0
            logline("停止录制: %s" % " / ".join(out.strip().splitlines()[:2]))
            rec_status(force=True)
            return self._s(200, json.dumps({"ok": True, "msg": out.strip()}, ensure_ascii=False))

        if p == "/api/rec/list":
            return self._s(200, json.dumps(
                {"ok": True, "msg": rec_sh("list", timeout=120).strip()}, ensure_ascii=False))

        if p == "/api/preflight":
            return self._s(200, json.dumps(preflight(), ensure_ascii=False))

        if p == "/api/runner/start":
            ok, why = runner_guard()
            if not ok:
                logline("拒绝启动 runner：" + why)
                return self._s(200, json.dumps({"ok": False, "msg": why}, ensure_ascii=False))
            if q.get("yes", ["0"])[0] != "1":
                # 单独一次误点不该把 runner 推上真机总线，必须带确认参数
                return self._s(200, json.dumps(
                    {"ok": False, "msg": "缺少确认。启动 runner 会立刻往真机 /JOINTS_CMD 发指令，"
                                         "请从页面按钮走确认弹窗。"}, ensure_ascii=False))
            if not STATE["armed"]:
                return self._s(200, json.dumps({"ok": False, "msg": "启动检查期间已释放接管，取消启动"}, ensure_ascii=False))
            ok, msg = runner_start()
            return self._s(200, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False))

        if p == "/api/runner/stop":
            ok, msg = runner_stop()
            return self._s(200, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False))

        return self._s(404, json.dumps({"ok": False}))


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    threading.Thread(target=ros_thread, daemon=True).start()
    threading.Thread(target=fullrate_loop, daemon=True).start()
    threading.Thread(target=rlog_loop, daemon=True).start()   # 09-11：runner 日志里的 IMU 提示上页面
    time.sleep(2.5)
    sweep_strays()
    logline("控制台启动，策略默认 %s，vx 上限 %.1f m/s（契约首次上机要求）" % (DEFAULT_POLICY, STATE["max_vx"]))
    print("S10 ctrl console on :%d" % PORT, flush=True)
    S(("0.0.0.0", PORT), H).serve_forever()
