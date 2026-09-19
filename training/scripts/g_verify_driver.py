#!/usr/bin/env python3
"""runner 0911g 验证驱动（09-11 晚）：按 PHASES 依次发 /robot_mode（和 /cmd_vel），全程订阅 /JOINTS_CMD，逐段统计：
轮子速度目标最大值、腿 kp 最大值与 kp=0（阻尼）条数、腿位置目标相邻两条的最大跳变（只算 kp>0 的）、指令最大间隔。
PHASES 默认 "1:4,6:2,6v0.5:4,4:5,1:5,2:5,1:5"：起立 → 进 RL → 走 2 m → 趴下 → 再起立 → 阻尼 → 第三次起立。
每段写成 "模式[v速度]:秒"。模式按 RobotMotionState：1 起立、2 阻尼、4 趴下、6 RL。"""
import math, os, re, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, Float32MultiArray
from geometry_msgs.msg import Twist
from drdds.msg import JointsDataCmd, JointsData

PH = []   # 每段 "模式[v前进速度][w转向角速度]:秒"，如 6w0.3:3
for tok in os.environ.get("PHASES", "1:4,6:2,6v0.5:4,4:5,1:5,2:5,1:5").split(","):
    head, sec = tok.split(":"); mm = re.match(r"^(\d+)(?:v([-\d.]+))?(?:y([-\d.]+))?(?:w([-\d.]+))?$", head)   # 09-13 晚加 y = 横移 vy
    PH.append((int(mm.group(1)), float(mm.group(2) or 0), float(sec), float(mm.group(4) or 0), float(mm.group(3) or 0)))
NAMES = {1: "起立", 2: "阻尼", 4: "趴下", 6: "RL"}
# 膝（关节 2/6/10/14）下发口径 → runner 内部角度（= MJCF/URDF 口径）：内部 = 下发 × 方向 + 零位偏置。方向、偏置同仿真桥 JOINT_DIR / POS_OFFSET_DEG
# （±156° = ±2.7227 rad：电机零点就在膝折叠到头的机械限位上）。0911h 起膝限位 ±2.7227，趴下时内部膝目标不应超过它。
KDIR = (-1.0, 1.0, -1.0, 1.0)   # = JOINT_DIR[2, 6, 10, 14]（09-11 晚更正：原来后腿两个写反了）
KOFF = tuple(math.radians(v) for v in (156.0, 156.0, -156.0, -156.0))   # = POS_OFFSET_DEG[2, 6, 10, 14]

def main():
    rclpy.init(); n = Node("g_verify_driver")
    mode = n.create_publisher(UInt8, "/robot_mode", 10); climb = n.create_publisher(UInt8, "/climb_mode", 10)
    cmd = n.create_publisher(Twist, "/cmd_vel", 10)
    st = {"s": None}; cur = {"k": -1, "t0": 0.0}; stats = {}; last = {"t": None, "legpos": None, "kp": 0.0}
    def on_cmd(msg):
        k = cur["k"]
        if k < 0:
            return
        t = time.time(); J = msg.data.joints_data
        d = stats.setdefault(k, {"n": 0, "wv": 0.0, "kp0": 0, "jump": 0.0, "gap": 0.0, "legkp": 0.0, "gaps": []})
        d["n"] += 1
        ws = d.setdefault("wsum", [0.0, 0.0, 0.0, 0.0]); wa = d.setdefault("wabs", [0.0, 0.0, 0.0, 0.0]); wm = d.setdefault("wmax", [0.0, 0.0, 0.0, 0.0])
        for j, i in enumerate((3, 7, 11, 15)):   # 轮子 fl fr hl hr（/JOINTS_CMD 下发口径）
            v = J[i].velocity; ws[j] += v; wa[j] += abs(v); wm[j] = max(wm[j], abs(v))
        d["wv"] = max(d["wv"], max(abs(J[i].velocity) for i in (3, 7, 11, 15)))
        if PH[k][0] in (1, 2, 4) and max(abs(J[i].velocity) for i in (3, 7, 11, 15)) > 2.0:   # 09-12：非 RL 相位里出现的高轮速指令——记它离相位开始多少 ms（分清"在途最后一拍"还是"切完后又发"）
            d.setdefault("late", []).append(((t - cur["t0"]) * 1000, max(abs(J[i].velocity) for i in (3, 7, 11, 15)), legkp))
        legkp = max(J[i].kp for i in range(16) if i % 4 != 3); d["legkp"] = max(d["legkp"], legkp)
        if legkp == 0.0:
            d["kp0"] += 1
        pos = [J[i].position for i in range(16) if i % 4 != 3]
        jv = max(abs(a - b) for a, b in zip(pos, last["legpos"])) if last["legpos"] is not None else 0.0
        if legkp > 0 and last["kp"] > 0:   # 只算相邻两条都在控位置（kp>0）的跳变；kp=0→kp>0 那一下不是真跳变
            d["jump"] = max(d["jump"], jv)
        if last["t"] is not None:
            g = t - last["t"]; d["gap"] = max(d["gap"], g)
            if g > 0.1:   # 长间隔（停顿）之后的第一条指令：kp 是多少、目标跳了多少 —— 用来判断有没有"追赶"
                d["gaps"].append((g * 1000, legkp, jv))
        for j, i in enumerate((2, 6, 10, 14)):   # 只看在控位置（kp>0）的膝目标
            if J[i].kp > 0:
                d["kq"] = max(d.get("kq", 0.0), abs(J[i].position * KDIR[j] + KOFF[j]))
        last["t"] = t; last["legpos"] = pos; last["kp"] = legkp
    n.create_subscription(JointsDataCmd, "/JOINTS_CMD", on_cmd, 200)
    jd = {"t": None, "st": None}
    def on_jd(msg):   # 关节反馈（仿真发出的 /JOINTS_DATA）：到达间隔（墙钟）与时间戳跳变 —— 对照 STAND_GUARD 的"反馈停更/跳变"
        k = cur["k"]; t = time.time(); ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if k >= 0 and jd["t"] is not None:
            d = stats.setdefault(k, {"n": 0, "wv": 0.0, "kp0": 0, "jump": 0.0, "gap": 0.0, "legkp": 0.0, "gaps": []})
            d["jgap"] = max(d.get("jgap", 0.0), t - jd["t"]); d["jst"] = max(d.get("jst", 0.0), ts - jd["st"])
            kt = [abs(msg.data.joints_data[i].torque) for i in (2, 6, 10, 14)]   # 四个膝关节实测/仿真力矩
            d["kn"] = d.get("kn", 0) + 1; d["ksum"] = d.get("ksum", 0.0) + sum(kt) / 4; d["kmax"] = max(d.get("kmax", 0.0), max(kt))
            if t - jd["t"] > 0.08:
                d.setdefault("jlong", []).append((round((t - jd["t"]) * 1000), round((ts - jd["st"]) * 1000)))
        jd["t"] = t; jd["st"] = ts
    n.create_subscription(JointsData, "/JOINTS_DATA", on_jd, 200)
    prev = {"s": None}
    def on_state(m):   # 姿态峰值与水平加速度峰值（20 ms 差分，/sim/state 布局：7:10 世界系线速度、13 仿真时刻、17/18 roll/pitch 度）
        s = list(m.data); st["s"] = s; k = cur["k"]
        if k >= 0:
            d = stats.setdefault(k, {"n": 0, "wv": 0.0, "kp0": 0, "jump": 0.0, "gap": 0.0, "legkp": 0.0, "gaps": []})
            d["roll"] = max(d.get("roll", 0.0), abs(s[17])); d["pitch"] = max(d.get("pitch", 0.0), abs(s[18]))
            if "yaw0" not in d: d["yaw0"] = s[19]; d["t_yaw0"] = s[13]
            d["yaw1"] = s[19]; d["t_yaw1"] = s[13]   # 09-12：本段偏航净变化 → 平均偏航角速度（和 wz 指令比，看转向跟得上不）
            p = prev["s"]
            if p is not None and s[13] - p[13] >= 0.019:
                dt = s[13] - p[13]; a = math.hypot(s[7] - p[7], s[8] - p[8]) / dt; d["acc"] = max(d.get("acc", 0.0), a); prev["s"] = s
        if prev["s"] is None or k < 0:
            prev["s"] = s
    n.create_subscription(Float32MultiArray, "/sim/state", on_state, 10)
    t0 = time.time()
    while st["s"] is None and time.time() - t0 < 30:
        rclpy.spin_once(n, timeout_sec=0.1)
    def run(sec, pub):
        t = time.time(); lp = 0.0
        while time.time() - t < sec:
            rclpy.spin_once(n, timeout_sec=0.005)
            if time.time() - lp >= 0.05:
                pub(); lp = time.time()
    for k, (m, vx, sec, wz, vy) in enumerate(PH):
        cur["k"] = k; cur["t0"] = time.time()
        print("[相位] 第 %d 段：/robot_mode %d（%s）vx %.1f vy %.1f wz %.1f，%.0f s  x=%.2f" % (k + 1, m, NAMES.get(m, m), vx, vy, wz, sec, st["s"][0]), flush=True)
        def pub(m=m, vx=vx, wz=wz, vy=vy):
            mode.publish(UInt8(data=m)); climb.publish(UInt8(data=0)); tw = Twist(); tw.linear.x = vx; tw.linear.y = vy; tw.angular.z = wz; cmd.publish(tw)
        run(sec, pub)
    cur["k"] = -1
    for k, (m, vx, sec, wz, vy) in enumerate(PH):   # 09-14：PH 加了 vy 之后这里漏改，跑完必崩（数据已落盘，只是汇总没打出来）
        d = stats.get(k, {"n": 0, "wv": 0.0, "kp0": 0, "jump": 0.0, "gap": 0.0, "legkp": 0.0, "gaps": []})
        print("统计 第 %d 段 %s%s：指令 %d 条，轮速目标最大 %.2f rad/s，腿 kp 最大 %.0f、kp=0 %d 条，腿目标单步最大 %.3f rad（相邻两条都 kp>0），指令最大间隔 %.0f ms%s"
              % (k + 1, NAMES.get(m, m), (" vx %.1f" % vx) if vx else "", d["n"], d["wv"], d["legkp"], d["kp0"], d["jump"], d["gap"] * 1000,
                 "".join("；长间隔 %.0f ms 后第一条 kp=%.0f、目标跳变 %.3f rad" % g for g in d["gaps"][:3])), flush=True)
        print("      关节反馈：到达最大间隔 %.0f ms、时间戳最大跳变 %.0f ms%s" % (d.get("jgap", 0.0) * 1000, d.get("jst", 0.0) * 1000,
              ("；超过 80 ms 的到达间隔（墙钟 ms, 时间戳 ms）%s" % d["jlong"][:5]) if d.get("jlong") else ""), flush=True)
        if d.get("n"):
            nn = d["n"]; print("      轮速目标（fl fr hl hr，下发口径）均值 %s、|均值| %s、最大 %s rad/s；|roll|峰 %.1f° |pitch|峰 %.1f°；水平加速度峰 %.1f m/s²%s" % (
                " ".join("%+.2f" % (x / nn) for x in d["wsum"]), " ".join("%.2f" % (x / nn) for x in d["wabs"]), " ".join("%.1f" % x for x in d["wmax"]),
                d.get("roll", 0.0), d.get("pitch", 0.0), d.get("acc", 0.0), ("（wz %.1f）" % wz) if wz else ""), flush=True)
        if wz and d.get("t_yaw1", 0) > d.get("t_yaw0", 0):
            dy = (d["yaw1"] - d["yaw0"] + 180) % 360 - 180; dt = d["t_yaw1"] - d["t_yaw0"]
            print("      转向：wz 指令 %.2f rad/s → 实际平均 %.2f rad/s（%.0f° / %.1f s，跟踪 %.0f%%）" % (wz, math.radians(dy) / dt, dy, dt, 100 * math.radians(dy) / dt / wz), flush=True)
        if d.get("late"):
            L = d["late"]; print("      !! 非 RL 相位里高轮速指令 %d 条：离相位开始 %s ms（|v| %s，kp %s）" % (len(L), " ".join("%.0f" % v[0] for v in L[:6]), " ".join("%.1f" % v[1] for v in L[:6]), " ".join("%.0f" % v[2] for v in L[:6])), flush=True)
        if d.get("kn"):
            print("      膝关节力矩：四膝平均 |τ| %.1f N·m、最大 %.1f N·m" % (d["ksum"] / d["kn"], d["kmax"]), flush=True)
            if "kq" in d:
                print("      膝目标（runner 内部角度，kp>0 的指令）最大 |q| %.4f rad（URDF 限位 2.7227）" % d["kq"], flush=True)
    s = st["s"]; print("结束 x=%.2f z=%.3f |roll| %.1f° |pitch| %.1f°" % (s[0], s[2], abs(s[17]), abs(s[18])), flush=True)
    n.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
