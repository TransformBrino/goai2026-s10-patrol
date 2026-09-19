#!/usr/bin/env python3
"""决策层（司机）真机版导航节点 —— 小而干净，不搬 eval_driver 那 3000 行。

职责（《决赛执行计划》§8 七件事）：记路线 / 跟路线 / 控速度 / 对齐 / 处理麻烦 / 保命 / 记账。
输入：位姿（--pose-source sim: /sim/state Float32MultiArray；odom: nav_msgs/Odometry，傅工侧契约）
      定位健康标志 /localization/status UInt8（0 正常 / 1 退化 → 半速 / 2 不可信 → 停车等 ≤10s）
输出：/cmd_vel geometry_msgs/Twist（仿真：我方 runner kRosTopic；真机：steer_pub → /STEER）
航点文件：每行 `序号 x y 限速 [align] [stop]`，# 开头为注释；坐标为地图系（与位姿同源）。

复用自 eval_driver（466s 那套验证过的三段逻辑）：Stanley 循迹 corridor_cmd、掉头锚定 turn_hold、
转弯前减速；不搬的：导航图选路（真机没有 6m 高程图）、专家策略切换、坑内模式。
"""
import argparse
import csv
import json
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, UInt8

try:
    from nav_msgs.msg import Odometry
except Exception:                       # noqa: BLE001
    Odometry = None

SIM_IDX = dict(x=0, y=1, z=2, t=13, roll=17, pitch=18, yaw_deg=19)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(x, y, z, w):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_wps(path):
    wps = []
    for ln in open(path, encoding="utf-8"):
        s = ln.split("#", 1)[0].strip()
        if not s:
            continue
        f = s.split()
        if len(f) < 3:
            raise ValueError("航点行格式：序号 x y [限速] [align] [stop]: %r" % ln)
        wps.append(dict(i=int(f[0]), x=float(f[1]), y=float(f[2]),
                        v=float(f[3]) if len(f) > 3 and f[3][0].isdigit() else 9.9,
                        align="align" in f[3:], stop="stop" in f[3:]))
    if len(wps) < 2:
        raise ValueError("至少要两个航点（起点 + 目标）")
    return wps


class NavNode(Node):
    def __init__(self, a):
        super().__init__("s10_nav")
        self.a = a
        self.wps = load_wps(a.waypoints)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        if a.pose_source == "sim":
            self.create_subscription(Float32MultiArray, a.pose_topic or "/sim/state", self._on_sim, 10)
        else:
            if Odometry is None:
                raise RuntimeError("nav_msgs 不可用，装 ros-jazzy-nav-msgs")
            self.create_subscription(Odometry, a.pose_topic or "/localization/pose", self._on_odom, 10)
        self.create_subscription(UInt8, a.status_topic, self._on_status, 10)
        self.x = self.y = self.yaw = None
        self.pose_wall = -1.0            # 最近一帧位姿的墙钟
        self.pose_t = 0.0                # 位姿自带时间（仿真时间或 header.stamp）
        self.status = 0
        self.status_wall = -1.0
        self.status_seen = False
        self.idx = 1                     # 目标航点下标（wps[0] 视为起点）
        self.anchor = None               # 掉头锚定 (x, y, idx)
        self.t_start = None
        self.rows = []
        self.events = []
        self.retries = 0
        self.retry_until = -1.0
        self.retry_side = 1.0
        self.prog = []                   # (wall, dist) 卡死判据窗口
        self.stop_since = -1.0
        self.handover = None
        self.finished = False
        self.warned_pose = False
        self.start = None                # 出发位姿 (x, y, yaw)：第一段的"前一点"用它，不用文件里的 wp0
        self.dt = 1.0 / a.hz
        self.create_timer(self.dt, self.tick)
        self.get_logger().info("航点 %d 个，vmax=%.2f，位姿源=%s，到点半径=%.2f" % (
            len(self.wps), a.vmax, a.pose_source, a.arrive_r))

    # ------------------------------------------------------------ 输入
    def _on_sim(self, m):
        d = m.data
        if len(d) < 20:
            return
        self.x, self.y = d[SIM_IDX["x"]], d[SIM_IDX["y"]]
        self.yaw = math.radians(d[SIM_IDX["yaw_deg"]])
        self.pose_t = d[SIM_IDX["t"]]
        self.pose_wall = time.monotonic()

    def _on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.pose_t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.pose_wall = time.monotonic()

    def _on_status(self, m):
        self.status = int(m.data)
        self.status_wall = time.monotonic()
        self.status_seen = True

    # ------------------------------------------------------------ 输出
    def send(self, vx=0.0, vy=0.0, wz=0.0):
        t = Twist()
        t.linear.x = float(max(-self.a.vmax, min(self.a.vmax, vx)))
        t.linear.y = float(max(-0.4, min(0.4, vy)))
        t.angular.z = float(max(-self.a.wz_max, min(self.a.wz_max, wz)))
        self.cmd_pub.publish(t)
        return t

    def event(self, s):
        self.events.append((round(time.monotonic() - (self.t_start or time.monotonic()), 2), s))
        self.get_logger().info(s)

    # ------------------------------------------------------------ 控制律（复用 466s 那套）
    def corridor(self, p, q, v_lock, k_e=1.4, k_h=1.6):
        """Stanley 式循迹：航向对齐 + 横向偏差反馈，速度锁定不衰减（eval_driver.corridor_cmd）。"""
        dx, dy = q["x"] - p["x"], q["y"] - p["y"]
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return 0.0, 0.0, 0.0, 0.0
        t = max(0.0, min(1.0, ((self.x - p["x"]) * dx + (self.y - p["y"]) * dy) / L2))
        px, py = p["x"] + t * dx, p["y"] + t * dy
        th = math.atan2(dy, dx)
        e = (self.x - px) * (-math.sin(th)) + (self.y - py) * math.cos(th)   # 左正
        h_err = wrap(th - self.yaw)
        wz = k_h * h_err - math.atan2(k_e * e, max(v_lock, 0.5))
        vx = v_lock if abs(h_err) < math.radians(50) else self.a.crawl_v
        return vx, wz, e, h_err

    def turn_hold(self, h_err, idx):
        """掉头锚定：大航向误差期间锁住起转位置，返回 (vx, vy) 或 None（eval_driver.turn_hold）。"""
        if abs(h_err) < math.radians(35):
            self.anchor = None
            return None
        if self.anchor is None or self.anchor[2] != idx:
            self.anchor = (self.x, self.y, idx)
        ex, ey = self.anchor[0] - self.x, self.anchor[1] - self.y
        if math.hypot(ex, ey) <= self.a.hold_dead:
            return self.a.crawl_v, 0.0
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        bx, by = ex * cy + ey * sy, -ex * sy + ey * cy
        cap, k = self.a.hold_cap, self.a.hold_k
        return max(-cap, min(cap, k * bx)), max(-cap, min(cap, k * by))

    def speed_cap(self, idx, dist):
        """每点限速 + 转弯前减速（eval_driver.terrain_speed 的转角部分）+ 对齐点前减速。"""
        tgt = self.wps[idx]
        v = min(self.a.vmax, tgt["v"])
        if idx + 1 < len(self.wps):
            p, n = self.wps[idx - 1], self.wps[idx + 1]
            ax, ay = tgt["x"] - p["x"], tgt["y"] - p["y"]
            bx, by = n["x"] - tgt["x"], n["y"] - tgt["y"]
            na, nb = math.hypot(ax, ay), math.hypot(bx, by)
            if na > 1e-6 and nb > 1e-6:
                turn = math.acos(max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb))))
                if turn > math.radians(45):
                    k = min(1.0, dist / 3.0)
                    v = min(v, self.a.turn_v + k * (v - self.a.turn_v))
        if tgt["align"] and dist < self.a.align_dist:
            v = min(v, self.a.align_v)
        return v

    # ------------------------------------------------------------ 主循环
    def tick(self):
        now = time.monotonic()
        if self.finished:
            self.send()
            return
        if self.x is None or now - self.pose_wall > self.a.watchdog:
            self.send()                                   # 保命：无位姿一律停
            if self.x is not None and not self.warned_pose:
                self.warned_pose = True
                self.event("!! 位姿超时 %.1fs，停车" % self.a.watchdog)
            return
        self.warned_pose = False
        if self.t_start is None:
            self.t_start = now
            self.start = (self.x, self.y, self.yaw)
            if self.a.relative:          # 航点是相对出发位姿的偏移（仿真联调用；真机用地图系绝对坐标）
                c, s = math.cos(self.yaw), math.sin(self.yaw)
                for w in self.wps:
                    ox, oy = w["x"], w["y"]
                    w["x"], w["y"] = self.x + c * ox - s * oy, self.y + s * ox + c * oy
            self.event("出发：起点 (%.2f, %.2f) 航向 %+.0f°%s" % (
                self.x, self.y, math.degrees(self.yaw), "（航点已按出发位姿平移旋转）" if self.a.relative else ""))
        if now - self.t_start > self.a.timeout:
            self.finish("超时 %.0fs" % self.a.timeout)
            return

        # ---- 定位健康 ----
        st = self.status if (self.status_seen and now - self.status_wall < 1.0) else (0 if not self.status_seen else 2)
        if st == 2:
            if self.stop_since < 0:
                self.stop_since = now
                self.event("定位不可信（标志位 2）→ 停车等待")
            self.send()
            if now - self.stop_since > self.a.status_wait and self.handover is None:
                self.handover = "定位不可信超过 %.0fs" % self.a.status_wait
                self.event("!! 交给人：" + self.handover)
            self.log_row(0, 0, 0, st, "wait")
            return
        if self.stop_since >= 0:
            self.event("定位恢复（标志位 %d）" % st)
            self.stop_since = -1.0

        # ---- 目标航点 ----
        tgt = self.wps[self.idx]
        prev = self.wps[self.idx - 1] if self.idx > 1 else dict(x=self.start[0], y=self.start[1])
        dist = math.hypot(tgt["x"] - self.x, tgt["y"] - self.y)
        bear = math.atan2(tgt["y"] - self.y, tgt["x"] - self.x)
        head_err = wrap(bear - self.yaw)             # 到目标点方位的航向误差（掉头/纯追踪用）
        if dist < self.a.arrive_r:
            self.event("到达 wp%d (%.2f, %.2f)  用时 %.1fs" % (tgt["i"], tgt["x"], tgt["y"], now - self.t_start))
            self.idx += 1
            self.anchor = None
            self.prog = []
            if self.idx >= len(self.wps):
                self.finish("全部航点完成")
            return

        # ---- 卡死重试（退一下换个角度再来）----
        if now < self.retry_until:
            self.send(-self.a.retry_v, self.retry_side * 0.15, 0.0)
            self.log_row(-self.a.retry_v, 0, 0, st, "retry")
            return
        self.prog.append((now, dist, abs(head_err)))
        self.prog = [p for p in self.prog if now - p[0] <= self.a.stuck_time]

        # ---- 速度与转向 ----
        v = self.speed_cap(self.idx, dist)
        k_e = 1.4
        if st == 1:
            v *= 0.5
            k_e = 0.4                   # 退化时只保留弱修正，别跟着抖动的位姿猛打
        vy = 0.0
        if abs(head_err) >= math.radians(35):
            # 目标在侧后方：原地转向目标（掉头），锚定住位置别漂
            vx, wz, e = self.a.crawl_v, max(-self.a.wz_max, min(self.a.wz_max, 1.5 * head_err)), 0.0
            hold = self.turn_hold(head_err, self.idx)
            if hold is not None:
                vx, vy = hold
        else:
            self.anchor = None
            _, wz_c, e, h_seg = self.corridor(prev, tgt, v, k_e=k_e)
            if abs(e) > self.a.offpath:
                # 偏离路径太远（如出发点不在线上）：先纯追踪朝目标点开，别让横向项把航向拧到 50° 卡死
                vx, wz = v, max(-self.a.wz_max, min(self.a.wz_max, 1.5 * head_err))
            else:
                vx, wz = v, wz_c
        if tgt["align"] and dist < self.a.align_dist and abs(head_err) > math.radians(self.a.align_tol):
            vx = 0.0                    # 台阶前先摆正再上
        stalled = (len(self.prog) >= 2 and now - self.prog[0][0] >= self.a.stuck_time * 0.95
                   and (self.prog[0][1] - dist) < self.a.stuck_dist
                   and (self.prog[0][2] - abs(head_err)) < math.radians(10))
        if stalled:
            if self.retries >= self.a.max_retries:
                self.handover = "卡死重试 %d 次仍无进展" % self.a.max_retries
                self.event("!! 交给人：" + self.handover)
                self.finish("卡死")
                return
            self.retries += 1
            self.event("卡死 #%d：%.1fs 内只前进 %.2fm → 倒退 %.1fs" % (
                self.retries, self.a.stuck_time, self.prog[0][1] - dist, self.a.retry_back))
            self.retry_until = now + self.a.retry_back
            self.retry_side = -self.retry_side
            self.prog = []
            self.anchor = None
            return
        self.send(vx, vy, wz)
        self.log_row(vx, vy, wz, st, "go", e=e, h=head_err)

    def log_row(self, vx, vy, wz, st, mode, e=0.0, h=0.0):
        self.rows.append(dict(t=round(time.monotonic() - self.t_start, 3), pose_t=round(self.pose_t, 3),
                              x=round(self.x, 3), y=round(self.y, 3), yaw=round(math.degrees(self.yaw), 1),
                              wp=self.wps[min(self.idx, len(self.wps) - 1)]["i"], mode=mode, status=st,
                              cmd_vx=round(vx, 3), cmd_vy=round(vy, 3), cmd_wz=round(wz, 3),
                              lat_err=round(e, 3), head_err=round(math.degrees(h), 1)))

    def finish(self, why):
        self.finished = True
        self.send()
        dur = time.monotonic() - (self.t_start or time.monotonic())
        summ = dict(result=why, complete=(self.idx >= len(self.wps)), waypoints=len(self.wps) - 1,
                    reached=self.idx - 1, duration_s=round(dur, 1), retries=self.retries,
                    handover=self.handover, events=self.events)
        self.event("结束：%s  到达 %d/%d  %.1fs  重试 %d" % (why, self.idx - 1, len(self.wps) - 1, dur, self.retries))
        if self.a.out:
            json.dump(summ, open(self.a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if self.a.log and self.rows:
            with open(self.a.log, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                w.writeheader()
                w.writerows(self.rows)
        self.done_wall = time.monotonic()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waypoints", required=True)
    ap.add_argument("--pose-source", choices=("sim", "odom"), default="odom")
    ap.add_argument("--pose-topic", default=None)
    ap.add_argument("--status-topic", default="/localization/status")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--wz-max", type=float, default=1.2)
    ap.add_argument("--crawl-v", type=float, default=0.1)
    ap.add_argument("--turn-v", type=float, default=0.4)
    ap.add_argument("--arrive-r", type=float, default=0.35)
    ap.add_argument("--align-dist", type=float, default=1.5)
    ap.add_argument("--align-v", type=float, default=0.4)
    ap.add_argument("--align-tol", type=float, default=10.0, help="度")
    ap.add_argument("--hold-k", type=float, default=1.5)
    ap.add_argument("--hold-cap", type=float, default=0.3)
    ap.add_argument("--hold-dead", type=float, default=0.10)
    ap.add_argument("--watchdog", type=float, default=0.5)
    ap.add_argument("--status-wait", type=float, default=10.0)
    ap.add_argument("--stuck-time", type=float, default=4.0)
    ap.add_argument("--stuck-dist", type=float, default=0.10)
    ap.add_argument("--retry-back", type=float, default=1.5)
    ap.add_argument("--retry-v", type=float, default=0.3)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--offpath", type=float, default=1.0, help="横向偏离超过此值改纯追踪（m）")
    ap.add_argument("--relative", action="store_true", help="航点为相对出发位姿的偏移（仿真联调用）")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    rclpy.init()
    node = NavNode(a)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.finished and time.monotonic() - getattr(node, "done_wall", 1e18) > 1.0:
                break
    except KeyboardInterrupt:
        pass
    node.send()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
