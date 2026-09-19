#!/usr/bin/env python3
"""切换上楼梯端到端（部署路径）：起立 → 进 RL → 通用策略以 V_APP 走向楼梯 → 机身离第一级立面 SW_D 时发 /climb_mode MODE
（默认 2 = runner 的 S10_POLICY_PIT 槽）并降到 V_CLIMB → 机身上到顶平台（z > 顶高 + 0.34 且越过最后一级外沿 0.5 m）后
发 /climb_mode 0 切回通用，再走 5 s（场景顶上是 40 m 平台，看切回后会不会掉下来）。
环境变量：STAIR_X（第一级立面 x，默认 3.0）、N、H、W（级数/立面/踏面，默认 8/0.15/0.37）、SW_D（1.0）、V_APP（0.5）、
V_CLIMB（0.4）、MODE（2）、T_MAX（60 s）。
PAUSE_X/PAUSE_S/PAUSE_MODE：楼梯中途停车测试——机身到 PAUSE_X 时给 vx=0 停 PAUSE_S 秒（expert = 保持楼梯专家；main = 先切回主策略），再恢复；estop = 发 /robot_mode 2 阻尼，之后不续爬。/sim/state 布局同 nav_node：pos 0:3、roll 17、pitch 18（度）。"""
import math, os, signal, threading, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, Float32MultiArray
from geometry_msgs.msg import Twist

E = os.environ.get
STAIR_X = float(E("STAIR_X", 3.0)); N = int(E("N", 8)); H = float(E("H", 0.15)); W = float(E("W", 0.37))
SW_D = float(E("SW_D", 1.0)); V_APP = float(E("V_APP", 0.5)); V_CLIMB = float(E("V_CLIMB", 0.4))
MODE = int(E("MODE", 2)); T_MAX = float(E("T_MAX", 60))
# 航向保持（09-11 晚）：训练时 heading_command=True，wz = 0.5 × 航向误差（限 ±0.6）一直在喂；部署时导航层要照做，否则会往一侧漂。
# HEAD_K>0 开（训练值 0.5）；PSI0_DEG 楼梯朝向；Y_K>0 再按横向偏移修目标航向（限 ±20°）；HEAD_APP=1 走近与登顶后也保持（默认只在楼梯段）。
HEAD_K = float(E("HEAD_K", 0) or 0); WZ_MAX = float(E("WZ_MAX", 0.6)); PSI0 = math.radians(float(E("PSI0_DEG", 0))); Y_K = float(E("Y_K", 0) or 0); HEAD_APP = E("HEAD_APP", "0") == "1"
# 控制台条件复现（09-11 晚，AGX 问）：航向反馈只有 HEAD_FB_HZ（如 20 Hz）、再晚 HEAD_FB_DELAY_MS；HEAD_TGT=switch 时以切入瞬间的（反馈）偏航为目标朝向
HEAD_FB_HZ = float(E("HEAD_FB_HZ", 0) or 0); HEAD_FB_DELAY = float(E("HEAD_FB_DELAY_MS", 0) or 0) / 1000.0; HEAD_TGT = E("HEAD_TGT", "fixed")
HIST = []; FB = {"t": -1e9, "psi": None}; PSI_SW = [None]
PRESTOP_S = float(E("PRESTOP_S", 0) or 0)   # >0：走到切换点先停这么多秒（主策略、vx=0），再从静止切进楼梯专家
# 09-12（AGX 侧建议补测"第一次切入的偏航"）：PRE_REALIGN=true 楼梯前停完后，操作员 Q/E 以 REALIGN_WZ 原地转到与楼梯真朝向（PSI0）差 ≤REALIGN_STOP_DEG 再切
#   （做法 a：先对正再按 V；切入时的偏航照常记成楼梯朝向）；PRE_REALIGN=rec 接近途中 x 越过 STAIR_X−PRE_REC_D 时记下偏航作楼梯朝向
#   （做法 b："记下楼梯朝向"按钮），停完后对正到它，切入后也以它为目标。门槛 REALIGN_DEG、最多转 REALIGN_S 秒。默认空 = 不对正（原行为）。
# 09-12 14:2x（AGX 侧 night4 候选规则 (i)）：REALIGN_REF=path 平台停车后操作员 Q/E 对正的参照是楼梯局部朝向（人眼看着下一段对正），RECAPTURE_STAIR=1 切入时以当时
#   偏航重记楼梯朝向（闸门不再比旧朝向，续爬目标也用新的）。二者合起来 = 规则 (i)；HEAD_TGT=path 是它的理想上界。
REALIGN_REF = E("REALIGN_REF", ""); RECAPTURE_STAIR = E("RECAPTURE_STAIR", "0") == "1"
PRE_REALIGN = E("PRE_REALIGN", ""); PRE_REC_D = float(E("PRE_REC_D", 1.0) or 1.0); PSI_REC = [None]
PAUSE_X = float(E("PAUSE_X", 0) or 0); PAUSE_S = float(E("PAUSE_S", 5)); PAUSE_MODE = E("PAUSE_MODE", "expert")
# 09-12 15:2x（作者问"专家能不能停、能不能退"）：PAUSE_VX 停车段发的 vx（默认 0；负值 = 倒退）；TOP_HOLD_S>0 到顶后先保持专家、vx=0 站这么多秒再切回主策略
PAUSE_VX = float(E("PAUSE_VX", 0) or 0); TOP_HOLD_S = float(E("TOP_HOLD_S", 0) or 0)
# 09-11 晚（组合验收发现平台停车时主策略原地转 30~60°、续爬大角度纠偏翻车）：PAUSE_HEAD=1 停车期间航向保持照常工作；
# REALIGN_DEG>0 续爬前 |航向误差| 超过它就先用主策略原地转正到一半以内（最多 REALIGN_S 秒）再切回楼梯专家
PAUSE_HEAD = E("PAUSE_HEAD", "0") == "1"; REALIGN_DEG = float(E("REALIGN_DEG", 0) or 0); REALIGN_S = float(E("REALIGN_S", 6))
# 09-11 晚（AGX 按代码核实控制台 f3bcc5fd）：切回 0 时清掉目标朝向、不再发 wz；续爬时重新取当时的偏航作目标（stair_enter：stair_yaw_ref = imu_yaw_fresh()）。
# HEAD_RECAPTURE=1 照此复现。REALIGN_MODE=op 模拟"续爬闸门 + 操作员按 Q/E 原地转"：以恒定 REALIGN_WZ（rad/s）转到与第一次切入时的
# 楼梯朝向差 ≤ REALIGN_STOP_DEG（默认 REALIGN_DEG/2）；REALIGN_MODE=auto（默认）是原来的"按航向保持自动转"。
HEAD_RECAPTURE = E("HEAD_RECAPTURE", "0") == "1"; REALIGN_MODE = E("REALIGN_MODE", "auto"); REALIGN_WZ = float(E("REALIGN_WZ", 0.3)); REALIGN_STOP_DEG = float(E("REALIGN_STOP_DEG", 0) or 0)
# 09-12（Astra：丢指令 ≠ runner 真卡住）：STALL_X/STALL_MS → 机身越过 STALL_X 时对 runner 整个进程发 SIGSTOP，STALL_MS 毫秒后 SIGCONT
# （推理、发布、DDS 收发全停；电机执行最后一条）。需要 run_stair_switch.sh 导出的 S10_RUNNER_PID。驱动退出时无论如何补发一次 SIGCONT。
STALL_X = float(E("STALL_X", 0) or 0); STALL_MS = float(E("STALL_MS", 0) or 0); RUNNER_PID = int(E("S10_RUNNER_PID", "0") or 0)
STALL = {"done": False}
def _cont():
    if RUNNER_PID:
        try: os.kill(RUNNER_PID, signal.SIGCONT)
        except ProcessLookupError: pass
def stall_check(x):
    if STALL_X > 0 and STALL_MS > 0 and RUNNER_PID and not STALL["done"] and x >= STALL_X:
        STALL["done"] = True
        os.kill(RUNNER_PID, signal.SIGSTOP); threading.Timer(STALL_MS / 1000.0, _cont).start()
        print("[真停顿] 机身 x=%.2f 越过 %.2f：暂停 runner 进程 %d 共 %.0f ms（SIGSTOP → SIGCONT）" % (x, STALL_X, RUNNER_PID, STALL_MS), flush=True)
PSI_STAIR = [None]   # 第一次切入时（控制台看到的）偏航 = 楼梯朝向；续爬闸门、"续爬前航向误差"都按它算
# X_BACK/Z_BACK：切回主策略的位置（默认 = 原逻辑：机身越过最后一级外沿 0.5 m 且高于顶高 +0.34）。负 X_BACK = 提前切回（后轮还在最后几级上）。
X_BACK = float(E("X_BACK", 0.5)); Z_BACK = float(E("Z_BACK", 0.08))
KEEP_EXPERT = E("KEEP_EXPERT", "0") == "1"   # 09-13 作者："不想要切，楼梯模式就全程使用楼梯模式"：到顶不切回通用，保持专家再走 5 s；配 SW_D≥STAIR_X 就是从起点起全程专家
# POST_HOLD_S（09-15 09:1x，作者："台阶注意翻越之后也要能站得稳"）：
#   登顶后那 5 s 行进跑完，再**原地站立** POST_HOLD_S 秒（vx=0、wz=0，模式不变），报位移/姿态/是否掉下去。
#   此前验收在登顶后只有"继续以 V_CLIMB 往前开 5 s"，**从来没测过站住**，默认 0 = 关，不影响既有批次。
POST_HOLD_S = float(E("POST_HOLD_S", 0) or 0)
# 按官方录包对齐接近速度（09-15 11:3x，作者："你要按官方的来"）：
#   `ref_climb34_50hz.npz`（wall_h 0.34，3 段）实测——以 pitch 首次>10° 为 t=0：
#     t −0.6~−0.3  官方 vx 均 **0.087** m/s，我方 0.307（快 3.5 倍）
#     t −0.3~0     官方 0.353，我方 0.658
#   即官方"爬着挪到墙根（≈0.09 m/s）再爆发"，我方匀速 0.4 怼上去。
#   V_NEAR>0 时：机身离第一级立面 < NEAR_D 就把速度指令降到 V_NEAR。
V_NEAR = float(E("V_NEAR", 0) or 0); NEAR_D = float(E("NEAR_D", 0.30))

# ---- 墙前速度剖面（块3，09-16，默认关闭；WALL_PROF=1 打开）----
# 为什么需要：块3 在**训练侧**把 vx 指令按离墙距离改写（mdp/climb_cmd.py），
# 策略是照着「墙前 0.62~0.45 m 指令为 0 / 0.62~0.54 m 指令 −0.15」学的。
# 部署侧若仍发恒定 V_CLIMB，等于问了策略一个和训练不同的问题——**学会了也测不出来**。
# 旧的 V_NEAR 机制做不到：它要求 V_NEAR>0（发不出真正的 0），且没有释放带（进去就出不来）。
# 默认 WALL_PROF=0 → 行为与改动前逐字节一致，既有批次不受影响。
# 各带默认值与训练侧 ClimbApproachVelocityCommandCfg 保持一致，改一边必须改另一边。
WALL_PROF  = int(E("WALL_PROF", 0) or 0)
WP_SLOW_HI = float(E("WP_SLOW_HI", 0.75))   # 减速带上沿
WP_BACK_HI = float(E("WP_BACK_HI", 0.62))   # 后退带上沿（减速带在此降到 0）
WP_STOP_HI = float(E("WP_STOP_HI", 0.54))   # 停住带上沿 = 后退带下沿
WP_RELEASE = float(E("WP_RELEASE", 0.45))   # 释放距离，小于它交回 V_CLIMB
WP_BACK_V  = float(E("WP_BACK_V", -0.15))   # 后退带指令 vx


# 释放条件（09-16 18:2x）：**不能用"到了更近的距离"做释放** —— 探针实测策略在停止带里
# 停死后就再也到不了那个距离，命令永远是 0/−0.15，卡死约 100 s 直到超时。
# 改成「停够了就放行」：进入停止带后，|vx| < WP_V_STOP 持续 WP_HOLD_S 秒即释放；
# 若 WP_TIMEOUT_S 秒内始终没停住，也释放（防死锁，保住登顶）。
WP_HOLD_S    = float(E("WP_HOLD_S", 0.6))
WP_TIMEOUT_S = float(E("WP_TIMEOUT_S", 4.0))
WP_V_STOP    = float(E("WP_V_STOP", 0.10))
WP_BACK_S = float(E("WP_BACK_S", 0.4))
WP_CREEP_V     = float(E("WP_CREEP_V", 0.0))      # >0：蹭近段指令速度（官方 0.022~0.102）
WP_D_CREEP_END = float(E("WP_D_CREEP_END", 0.32)) # 蹭近段结束距离（官方最深处 0.243~0.341）   # 后退段时长（官方 seg0 连续后退 0.405 s）
_wp = {"entered": None, "held": 0.0, "last": None, "released": False, "backing": None}


def _wall_prof(d, vx):
    """按离墙距离 d（m，正 = 还没到墙）与当前 |vx| 给 vx 指令。

    状态机：远处不管 → 进入剖面后计时 → 停够 WP_HOLD_S 或超时 WP_TIMEOUT_S → 永久释放。
    释放是**单向**的：一旦放行就不再回到停止带，否则上墙途中会被反复叫停。
    """
    now = time.time()
    dtp = 0.0 if _wp["last"] is None else max(0.0, now - _wp["last"])
    _wp["last"] = now
    if _wp["released"] or d <= 0.0 or d > WP_SLOW_HI:
        return V_CLIMB
    if _wp["entered"] is None:
        _wp["entered"] = now
    if abs(vx) < WP_V_STOP:
        _wp["held"] += dtp
    else:
        _wp["held"] = 0.0
    # 次序按官方：**先停住，再后退，最后放行**。原先按距离分带会让"后退带"排在"停住带"
    # 外侧——机器人在减速带就停死了，永远进不到后退带（09-16 18:2x 实测：三条全部在
    # 0.91 m 释放，后退 0.000 s）。官方 ref_climb34 是停住(0.60~0.45 m, vx≈+0.02)
    # 期间出现小幅后退（seg0 连续 0.405 s、最低 −0.347、总 0.039 m），不是先退后停。
    # 蹭近段（块10）：WP_CREEP_V>0 时按**距离**结束该段，与训练侧 climb_cmd.step_profile 一致。
    # 官方不是停死，是以 0.022~0.102 m/s 蹭近 0.24~0.25 m 到离墙 0.24~0.34 才起动作。
    done_stop = (d <= WP_D_CREEP_END) if WP_CREEP_V > 0 else (_wp["held"] >= WP_HOLD_S)
    if _wp["backing"] is None and (done_stop or (now - _wp["entered"]) >= WP_TIMEOUT_S):
        _wp["backing"] = now                             # 停够了（或超时）→ 进入后退段
        print("[剖面] 离墙 %.2f m 停住达标（累计 %.2f s，进入后 %.2f s）→ 后退 %.1f s"
              % (d, _wp["held"], now - _wp["entered"], WP_BACK_S), flush=True)
    if _wp["backing"] is not None:
        if (now - _wp["backing"]) < WP_BACK_S:
            return WP_BACK_V                             # 后退
        _wp["released"] = True
        print("[剖面] 离墙 %.2f m 后退完毕，释放" % d, flush=True)
        return V_CLIMB
    if d > WP_BACK_HI:                                   # 减速带：线性降到 0
        span = max(WP_SLOW_HI - WP_BACK_HI, 1e-3)
        return V_CLIMB * max(0.0, min(1.0, (d - WP_BACK_HI) / span))
    return WP_CREEP_V                                    # 蹭近/停住（0 即停死；何时结束见上）
# 09-12 中午（弯楼梯）：S10_STAIR_PATH=<gen_curve_stairs.py 写的 .path.json> → 登顶判据的 X_END 用路径末端 x；HEAD_TGT=path 时目标航向 = 离机身最近那一级的朝向
# （模拟操作员一路把方向带对；直楼梯上等价于 PSI0=0），续爬闸门也和它比。TOP_Z 仍按 N*H。
PATH = None
if E("S10_STAIR_PATH"):
    import json as _json; PATH = _json.load(open(E("S10_STAIR_PATH")))
def path_heading(st):
    if not PATH: return PSI0
    pts = PATH["pts"]; i = min(range(len(pts)), key=lambda k: (pts[k][0] - st[0]) ** 2 + (pts[k][1] - st[1]) ** 2)
    if st[0] < pts[0][0] - 0.3: return pts[0][2]
    return pts[i][2]
TOP_MARGIN = float(E("TOP_MARGIN", 0.08))   # 09-13：E4 登顶后站高 0.33（不是 0.42），2.166 < 1.84+0.42−0.08 被判"没上顶"跑出平台；验收批用 0.14
TOP_Z = N * H; X_END = (PATH["x_end"] if PATH else STAIR_X + N * W); Z_ON = TOP_Z + 0.42 - Z_BACK; X_PAST = X_END + X_BACK; Z_TOP_OK = TOP_Z + 0.42 - TOP_MARGIN

class D(Node):
    def __init__(s):
        super().__init__("switch_stair_driver")
        s.mode = s.create_publisher(UInt8, "/robot_mode", 10)
        s.climb = s.create_publisher(UInt8, "/climb_mode", 10)
        s.cmd = s.create_publisher(Twist, "/cmd_vel", 10)
        s.st = None
        s.create_subscription(Float32MultiArray, "/sim/state", s._on_state, 10)

    def _on_state(s, m):
        s.st = list(m.data); now = time.time(); HIST.append((now, s.st[19]))
        while HIST and HIST[0][0] < now - 2.0:
            HIST.pop(0)

def tw(vx, wz=0.0):
    m = Twist(); m.linear.x = float(vx); m.angular.z = float(wz); return m

def fb_yaw_deg(st):   # 控制台看到的偏航：按 HEAD_FB_HZ 采样保持、再晚 HEAD_FB_DELAY（两者都 0 = 用最新真值）
    if HEAD_FB_HZ <= 0 and HEAD_FB_DELAY <= 0:
        return st[19]
    now = time.time()
    if FB["psi"] is None or HEAD_FB_HZ <= 0 or now - FB["t"] >= 1.0 / HEAD_FB_HZ:
        old = [y for (t, y) in HIST if t <= now - HEAD_FB_DELAY]
        FB["psi"] = old[-1] if old else (HIST[0][1] if HIST else st[19]); FB["t"] = now
    return FB["psi"]

def wz_hold(st, on=True):
    if not on or HEAD_K <= 0 or st is None:
        return 0.0
    base = path_heading(st) if HEAD_TGT == "path" else PSI_SW[0] if (HEAD_TGT == "switch" and PSI_SW[0] is not None) else PSI0
    tgt = base + max(-0.35, min(0.35, -Y_K * st[1])); psi = math.radians(fb_yaw_deg(st))
    e = math.atan2(math.sin(tgt - psi), math.cos(tgt - psi)); return max(-WZ_MAX, min(WZ_MAX, HEAD_K * e))

def main():
    rclpy.init(); n = D(); t0 = time.time()
    while n.st is None and time.time() - t0 < 30:
        rclpy.spin_once(n, timeout_sec=0.1)
    if n.st is None:
        print("!! 收不到 /sim/state"); return
    def run(sec, pub, stop=None):
        t = time.time(); last = 0.0
        while time.time() - t < sec:
            rclpy.spin_once(n, timeout_sec=0.01)
            if time.time() - last >= 0.05:
                pub(); last = time.time()
            if stop and n.st and stop(n.st):
                return True
        return False
    print("[相位] 起立", flush=True); run(4.0, lambda: n.mode.publish(UInt8(data=1)))
    print("[相位] 进 RL，零指令站 3 s", flush=True)
    run(3.0, lambda: (n.mode.publish(UInt8(data=6)), n.climb.publish(UInt8(data=0)), n.cmd.publish(tw(0.0))))
    print("[相位] 通用策略以 %.1f 走向楼梯（%d 级 × %.2f m，踏面 %.2f m，第一级 x=%.2f，离 %.1f m 切 /climb_mode %d）"
          % (V_APP, N, H, W, STAIR_X, SW_D, MODE), flush=True)
    def pub_app():
        if PRE_REALIGN == "rec" and PSI_REC[0] is None and n.st[0] >= STAIR_X - PRE_REC_D:   # 做法 b：行进中正对楼梯时点一下"记下楼梯朝向"
            PSI_REC[0] = math.radians(fb_yaw_deg(n.st)); print("[相位] x=%.2f 记下楼梯朝向 %+.1f°（行进中）" % (n.st[0], math.degrees(PSI_REC[0])), flush=True)
        n.climb.publish(UInt8(data=0)); n.cmd.publish(tw(V_APP, wz_hold(n.st, HEAD_APP)))
    ok = run(30.0, pub_app, stop=lambda s: s[0] >= STAIR_X - SW_D)
    if not ok:
        print("!! 30 s 内没走到切换点，最远 x=%.2f" % n.st[0]); return
    if PRESTOP_S > 0:   # 操作员常见做法：楼梯前停住确认，再按楼梯按钮
        x_ps, y_ps = n.st[0], fb_yaw_deg(n.st)
        print("[相位] x=%.2f 楼梯前停 %.0f s（主策略、vx=0）再切" % (n.st[0], PRESTOP_S), flush=True)
        run(PRESTOP_S, lambda: (n.climb.publish(UInt8(data=0)), n.cmd.publish(tw(0.0))))
        print("楼梯前停车：偏航 %+.1f° → %+.1f°，x %.2f → %.2f" % (y_ps, fb_yaw_deg(n.st), x_ps, n.st[0]), flush=True)
        if PRE_REALIGN in ("true", "rec"):   # 切入前对正（做法 a / b），操作员 Q/E 恒定角速度原地转
            use_rec = PRE_REALIGN == "rec" and PSI_REC[0] is not None; ref = PSI_REC[0] if use_rec else PSI0
            def perr():
                psi = math.radians(fb_yaw_deg(n.st)); return math.degrees(math.atan2(math.sin(ref - psi), math.cos(ref - psi)))
            e0 = perr(); t_al = time.time()
            if abs(e0) > REALIGN_DEG:
                sd = REALIGN_STOP_DEG if REALIGN_STOP_DEG > 0 else REALIGN_DEG * 0.5
                run(REALIGN_S, lambda: (n.climb.publish(UInt8(data=0)), n.cmd.publish(tw(0.0, REALIGN_WZ if perr() > 0 else -REALIGN_WZ))), stop=lambda s_: abs(perr()) <= sd)
            print("切入前对正（参照 = %s）：差 %+.1f° → %+.1f°（转 %.1f s，门槛 %.0f°）  x=%.2f" % (
                "行进中记下的朝向" if use_rec else "楼梯真朝向", e0, perr(), time.time() - t_al, REALIGN_DEG, n.st[0]), flush=True)
            if abs(perr()) > REALIGN_DEG:   # 同上：对正后仍不满足门槛 → 拒绝切入
                print("结果 闸门拒绝切入 ⛔（切入前对正 %.1f s 后仍差 %+.1f° > %.0f°）  末端 x=%.2f z=%.3f" % (time.time() - t_al, perr(), REALIGN_DEG, n.st[0], n.st[2]), flush=True)
                n.destroy_node(); rclpy.shutdown(); return
    PSI_SW[0] = math.radians(fb_yaw_deg(n.st))   # HEAD_TGT=switch：切入瞬间（控制台看到的）偏航 = 楼梯朝向
    PSI_STAIR[0] = PSI_SW[0]
    if PRE_REALIGN == "rec" and PSI_REC[0] is not None:   # 做法 b：楼梯朝向 = 行进中记下的，不是切入时的偏航
        PSI_SW[0] = PSI_STAIR[0] = PSI_REC[0]
    xs = n.st[0]; ts = time.time()
    print("[相位] x=%.2f 发 /climb_mode %d，降到 %.1f" % (xs, MODE, V_CLIMB), flush=True)
    rec = []
    def dump_rec():   # REC_CSV=<文件>：把爬楼段逐拍（50 ms）存下来（t 从切入算起；停车期间不记）
        if E("REC_CSV") and rec:
            with open(E("REC_CSV"), "w") as fo:
                fo.write("t,x,z,roll,pitch,y,yaw\n")
                for r in rec:
                    fo.write("%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%.2f\n" % (r[0] - ts, r[1], r[2], r[7], r[8], r[5], r[6]))
    def pub_c():
        stall_check(n.st[0])
        # 09-15 14:0x 修 bug：原条件 `(STAIR_X - x) < NEAR_D` 在**登顶后 x>STAIR_X** 时差值为负、
        # 永远成立 → 速度被永久钳在 V_NEAR，机器人上了台面就不动了（实测推进仅 +0.07 m）。
        # 必须加下界：只在"还没过墙、且已进入 NEAR_D"时减速。
        _d = STAIR_X - n.st[0]
        vcmd = _wall_prof(_d, n.st[7]) if WALL_PROF else (V_NEAR if (V_NEAR > 0 and 0.0 < _d < NEAR_D) else V_CLIMB)
        n.climb.publish(UInt8(data=MODE)); n.cmd.publish(tw(vcmd, wz_hold(n.st)))
        rec.append((time.time(), n.st[0], n.st[2], abs(n.st[17]), abs(n.st[18]), n.st[1], n.st[19], n.st[17], n.st[18]))
    peak = [0.0, 0.0]   # 每拍都记 |roll|/|pitch| 峰值（rec 只 50 ms 记一次，会漏掉让停下来的那一瞬间）
    def stop_c(s):
        peak[0] = max(peak[0], abs(s[17])); peak[1] = max(peak[1], abs(s[18]))
        return (s[2] > Z_ON and s[0] > X_PAST) or peak[0] > 70 or peak[1] > 70
    if PAUSE_X > 0:
        # 楼梯中途停车（09-11 晚）：机身到 PAUSE_X → vx=0 停 PAUSE_S 秒（expert 保持楼梯专家 / main 先切回主策略）→ 恢复原模式与 V_CLIMB
        run(T_MAX, pub_c, stop=lambda s: stop_c(s) or s[0] >= PAUSE_X)
        if n.st[0] >= PAUSE_X and peak[0] <= 70 and peak[1] <= 70:
            pm = MODE if PAUSE_MODE == "expert" else 0; x0p, z0p = n.st[0], n.st[2]; pk = [0.0, 0.0]; pmin = [x0p, z0p]; est = PAUSE_MODE == "estop"
            print("[相位] x=%.2f z=%.3f 楼梯中途停 %.0f s（%s，vx=0）" % (x0p, z0p, PAUSE_S, "急停：/robot_mode 2 阻尼" if est else "保持楼梯专家" if pm else "先切回主策略"), flush=True)
            def stop_p(s):
                for i, j in ((0, 17), (1, 18)):
                    pk[i] = max(pk[i], abs(s[j])); peak[i] = max(peak[i], abs(s[j]))
                pmin[0] = min(pmin[0], s[0]); pmin[1] = min(pmin[1], s[2])   # 停车中最靠后/最低（滑下去又爬回来，只看首尾会漏）
                return peak[0] > 70 or peak[1] > 70
            prec = []
            def pub_p(wz_on, wz_fixed=None):
                wz = wz_fixed if wz_fixed is not None else (wz_hold(n.st) if wz_on else 0.0)
                if est: n.mode.publish(UInt8(data=2))
                n.climb.publish(UInt8(data=pm)); n.cmd.publish(tw(PAUSE_VX, wz))
                prec.append((time.time() - ts, n.st[0], n.st[2], n.st[17], n.st[18], n.st[1], n.st[19], wz))
            def dump_p():   # 停车（和续爬前对正）期间逐拍另存 .pause.csv（t 从切入算起）
                if E("REC_CSV") and prec:
                    with open(E("REC_CSV").replace(".rec.csv", "") + ".pause.csv", "w") as fo:
                        fo.write("t,x,z,roll,pitch,y,yaw,wz\n")
                        for q in prec: fo.write("%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%.2f,%.3f\n" % q)
            run(PAUSE_S, lambda: pub_p(PAUSE_HEAD and not est and not pm), stop=stop_p)
            print("停车结果 x %.2f→%.2f（%+.2f m）  z %.3f→%.3f（%+.3f m）  停车中 |roll|峰 %.1f°  |pitch|峰 %.1f°  停车中 x 最小 %.2f  z 最低 %.3f" % (
                x0p, n.st[0], n.st[0] - x0p, z0p, n.st[2], n.st[2] - z0p, pk[0], pk[1], pmin[0], pmin[1]), flush=True)
            if est:   # 急停（PAUSE_MODE=estop）：阻尼后不续爬，直接报结果
                print("结果 急停 ⛔（阻尼后不续爬）  末端 x=%.2f z=%.3f  |roll| %.1f°  |pitch| %.1f°" % (n.st[0], n.st[2], abs(n.st[17]), abs(n.st[18])), flush=True)
                dump_rec(); dump_p(); n.destroy_node(); rclpy.shutdown(); return
            def herr():   # 控制台看到的航向误差（度）：楼梯朝向（第一次切入时偏航）− 当前偏航
                base = path_heading(n.st) if (HEAD_TGT == "path" or REALIGN_REF == "path") else PSI_STAIR[0] if PSI_STAIR[0] is not None else PSI0; psi = math.radians(fb_yaw_deg(n.st))
                return math.degrees(math.atan2(math.sin(base - psi), math.cos(base - psi)))
            e0 = herr(); t_al = time.time()
            if REALIGN_DEG > 0 and not pm and abs(e0) > REALIGN_DEG:
                sd = REALIGN_STOP_DEG if REALIGN_STOP_DEG > 0 else REALIGN_DEG * 0.5; xa = [n.st[0], n.st[0]]
                print("[相位] 续爬前对正：与楼梯朝向差 %+.1f° > %.0f°，%s" % (e0, REALIGN_DEG, ("操作员 Q/E 原地转（%.2f rad/s）" % REALIGN_WZ) if REALIGN_MODE == "op" else "主策略按航向保持原地转"), flush=True)
                def pub_a():
                    e = herr(); pub_p(True, (REALIGN_WZ if e > 0 else -REALIGN_WZ) if REALIGN_MODE == "op" else None); xa[1] = min(xa[1], n.st[0])
                run(REALIGN_S, pub_a, stop=lambda s_: abs(herr()) <= sd or stop_p(s_))
                print("对正中 x %.2f → %.2f（最小 %.2f）" % (xa[0], n.st[0], xa[1]), flush=True)
            print("续爬前航向误差 %+.1f° → %+.1f°（对正 %.1f s，门槛 %s）" % (e0, herr(), time.time() - t_al, ("%.0f°" % REALIGN_DEG) if REALIGN_DEG > 0 else "关"), flush=True)
            if REALIGN_DEG > 0 and not pm and abs(herr()) > REALIGN_DEG:   # 09-12 中午（Astra）：对正超时后仍不满足门槛，真实控制台会拒绝切入 → 不再继续爬，单独记
                print("结果 闸门拒绝切入 ⛔（对正 %.1f s 后仍差 %+.1f° > %.0f°，不续爬）  末端 x=%.2f z=%.3f  |roll| %.1f°  |pitch| %.1f°" % (
                    time.time() - t_al, herr(), REALIGN_DEG, n.st[0], n.st[2], abs(n.st[17]), abs(n.st[18])), flush=True)
                dump_rec(); dump_p(); n.destroy_node(); rclpy.shutdown(); return
            if HEAD_RECAPTURE or RECAPTURE_STAIR:   # 控制台行为：续爬时重新取当时的偏航作目标；RECAPTURE_STAIR 还把它记成新的楼梯朝向（规则 (i)）
                PSI_SW[0] = math.radians(fb_yaw_deg(n.st))
                print("续爬时重新取偏航作目标 %+.1f°（与%s差 %+.1f°）" % (math.degrees(PSI_SW[0]), "楼梯局部朝向" if REALIGN_REF == "path" else "楼梯朝向", -herr()), flush=True)
                if RECAPTURE_STAIR:
                    PSI_STAIR[0] = PSI_SW[0]; print("重记楼梯朝向 = %+.1f°（规则 (i)：手动转向后切入以当时偏航为准）" % math.degrees(PSI_STAIR[0]), flush=True)
            dump_p()
    on = run(T_MAX - (time.time() - ts), pub_c, stop=stop_c) if peak[0] <= 70 and peak[1] <= 70 else True
    dt = time.time() - ts
    zmax = max(r[2] for r in rec) if rec else 0.0; rmax = max(r[3] for r in rec) if rec else 0.0; pmax = max(r[4] for r in rec) if rec else 0.0
    rmax = max(rmax, peak[0]); pmax = max(pmax, peak[1])
    flipped = rmax > 70 or pmax > 70
    on = on and not flipped
    k = max(0, min(N, round((zmax - 0.42) / H)))
    fell = False
    if on and TOP_HOLD_S > 0:   # 到顶后保持专家、零速站住
        x0t, z0t, y0t = n.st[0], n.st[2], n.st[19]; pkt = [0.0, 0.0]; mint = [x0t]
        def stop_t(s_):
            pkt[0] = max(pkt[0], abs(s_[17])); pkt[1] = max(pkt[1], abs(s_[18])); mint[0] = min(mint[0], s_[0]); return pkt[0] > 70 or pkt[1] > 70
        run(TOP_HOLD_S, lambda: (n.climb.publish(UInt8(data=MODE)), n.cmd.publish(tw(0.0))), stop=stop_t)
        dy = math.degrees(math.atan2(math.sin(math.radians(n.st[19] - y0t)), math.cos(math.radians(n.st[19] - y0t))))
        print("顶部保持专家零速 %.0f s 结果 x %.2f→%.2f（%+.2f m，最小 %.2f）  z %+.3f  偏航变化 %+.1f°  |roll|峰 %.1f°  |pitch|峰 %.1f°" % (
            TOP_HOLD_S, x0t, n.st[0], n.st[0] - x0t, mint[0], n.st[2] - z0t, dy, pkt[0], pkt[1]), flush=True)
        on = on and pkt[0] <= 70 and pkt[1] <= 70; flipped = flipped or not on
    if on and KEEP_EXPERT:
        print("[相位] 已上顶平台（切换后 %.1f s），不切回：保持楼梯专家以 %.2f 再走 5 s" % (dt, V_CLIMB), flush=True)
        pkt2 = [0.0, 0.0]
        def stop_k(s_):
            pkt2[0] = max(pkt2[0], abs(s_[17])); pkt2[1] = max(pkt2[1], abs(s_[18])); return pkt2[0] > 70 or pkt2[1] > 70
        run(5.0, lambda: (n.climb.publish(UInt8(data=MODE)), n.cmd.publish(tw(V_CLIMB, wz_hold(n.st, HEAD_APP)))), stop=stop_k)
        fell = n.st[2] < Z_TOP_OK - 0.1 or pkt2[0] > 70 or pkt2[1] > 70
    elif on:
        print("[相位] 已上顶平台（切换后 %.1f s），发 /climb_mode 0 切回通用，再走 5 s" % dt, flush=True)
        run(5.0, lambda: (n.climb.publish(UInt8(data=0)), n.cmd.publish(tw(V_APP, wz_hold(n.st, HEAD_APP)))))
        fell = n.st[2] < Z_TOP_OK - 0.1
    if on and not fell and POST_HOLD_S > 0:
        x0h, y0h, z0h = n.st[0], n.st[1], n.st[2]
        pk = [0.0, 0.0]; zlo = [z0h]; xs_h = [x0h]; ys_h = [y0h]
        def pub_h():
            n.climb.publish(UInt8(data=MODE if KEEP_EXPERT else 0)); n.cmd.publish(tw(0.0, 0.0))
        def stop_h(s_):
            pk[0] = max(pk[0], abs(s_[17])); pk[1] = max(pk[1], abs(s_[18]))
            zlo[0] = min(zlo[0], s_[2]); xs_h.append(s_[0]); ys_h.append(s_[1])
            return pk[0] > 70 or pk[1] > 70 or s_[2] < Z_TOP_OK - 0.1
        print("[相位] 登顶后原地站立 %.0f s（vx=0 wz=0，%s）" % (POST_HOLD_S, "保持专家" if KEEP_EXPERT else "切回通用"), flush=True)
        run(POST_HOLD_S, pub_h, stop=stop_h)
        dx = max(xs_h) - min(xs_h); dy = max(ys_h) - min(ys_h)
        # 09-16：站住判据原先**只查高度与翻转**，不查水平位移。
        # 台面长 30 m，机器人在零指令站立测试里自己往后开 2.8 m 也不掉高度 →
        # 被判"站住 ✅"（B2_warm sw0.5/1.0 的 post_2 实测位移 x −2.65/−2.77 m）。
        # 补上水平位移上限（零指令下本该原地不动；C24/C35 实测 ≤0.07 m，取 0.30 m 留足裕量）。
        POST_MAX_DRIFT = float(E("POST_MAX_DRIFT", 0.30) or 0.30)
        drift = max(dx, dy)
        fell = (n.st[2] < Z_TOP_OK - 0.1 or pk[0] > 70 or pk[1] > 70
                or drift > POST_MAX_DRIFT)
        print("站立结果 位移 x %+.2f m（幅度 %.2f） y %+.2f m（幅度 %.2f）  站立中 |roll|峰 %.1f° |pitch|峰 %.1f°  z 最低 %.3f  末端 pitch %+.1f°  %s" % (
            n.st[0] - x0h, dx, n.st[1] - y0h, dy, pk[0], pk[1], zlo[0], n.st[18],
            ("掉下/翻 ❌" if (n.st[2] < Z_TOP_OK - 0.1 or pk[0] > 70 or pk[1] > 70)
             else ("零指令漂移 %.2f m > %.2f ❌" % (drift, POST_MAX_DRIFT)) if fell else "站住 ✅")), flush=True)
    verdict = ("上楼成功 ✅" if on and not fell else ("顶上继续走掉下/翻 ❌" if KEEP_EXPERT else "切回通用后掉下 ❌") if on else "翻车 ❌" if flipped else "%.0f s 内未上完 ❌" % T_MAX)
    print("结果 %s  切换点 x=%.2f  用时 %.1f s  最高 z=%.3f（约第 %d/%d 级，顶上站立约 %.2f）  |roll|峰 %.1f°  |pitch|峰 %.1f°  末端 x=%.2f z=%.3f" % (
        verdict, xs, dt, zmax, k, N, TOP_Z + 0.42, rmax, pmax, n.st[0], n.st[2]), flush=True)
    if rec:   # 横向漂移（决赛楼梯两侧有墙/栏杆时要看）：单独一行，不改上面结果行的格式
        ys = [r[5] for r in rec]
        ps = [r[6] for r in rec]
        print("横向 y：切换时 %+.2f  爬楼中 %+.2f~%+.2f  末端 %+.2f（m）  偏航：切换时 %+.1f°  爬楼中 %+.1f~%+.1f°  爬完 %+.1f°%s" % (
            ys[0], min(ys), max(ys), n.st[1], ps[0], min(ps), max(ps), ps[-1], "  [航向保持 K=%.2f Y_K=%.2f%s]" % (HEAD_K, Y_K, (" 反馈 %.0f Hz/%.0f ms 目标=%s" % (HEAD_FB_HZ, HEAD_FB_DELAY * 1000, HEAD_TGT)) if (HEAD_FB_HZ > 0 or HEAD_FB_DELAY > 0 or HEAD_TGT != "fixed") else "") if HEAD_K > 0 else ""), flush=True)
    dump_rec(); n.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    try:
        main()
    finally:
        _cont()   # STALL：无论怎么退出都补发 SIGCONT，别把 runner 留在暂停态
