#!/usr/bin/env python3
"""真机侧高程图节点（三缺之一：感知管线）—— 把点云累积成世界系高程图，按策略的 187 格采样发 /height_scan。

复用仿真里验过的 ElevationMap（累积取最高、天花板过滤、邻域中位补值、FWD/LAT 裁剪同一套代码）。
输入：
  点云   sensor_msgs/PointCloud2（真机 /LIDAR/POINTS_MERGED，frame lidar_link；仿真测试 /front_lidar/points）
  位姿   --pose-source odom  : nav_msgs/Odometry（傅工契约 /localization/pose，float64）
         --pose-source sim   : /sim/state Float32MultiArray（仿真真值：pos 0:3, rpy(deg) 17:20）
  外参   --lidar-xyz x y z --lidar-rpy r p y（点云帧 → base_link，弧度）
输出：/height_scan（裁剪后，策略吃）、/height_scan_raw（未裁剪，导航吃）、/hmap_info（hole_rate, coverage, frames）
环境变量（与仿真一致）：S10_HMAP_FWD_CLIP、S10_HMAP_LAT_CLIP、S10_HMAP_FILL
真机上先做「平地基准」验收：站平地 /height_scan 应为 −0.08±0.01（相机安装单里的判据）。
"""
import argparse
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray

SIM_DIR = Path(__file__).resolve().parents[2] / "s10_ws/src/S10_sdk_deploy/interface/robot/simulation"
sys.path.insert(0, str(SIM_DIR))
from elevation_map import ElevationMap  # noqa: E402  （读环境变量的裁剪开关在 import 时生效）
from heightmap import CELL_XY, CLIP, NUM_RAYS, OFFSET  # noqa: E402

# 09-11 作者拍板：机身正下方雷达看不到，sample() 把空洞按"与 base_z−OFFSET 同高"填，编码后正好是 0；
# 而平地实测是 −0.07~−0.14 —— 策略看到的是机身下一道约 8 cm 的沟（x∈[−0.2,0.2] 整个宽度，5×11 格），
# 仿真里射线从上往下打，没有这种输入。实测零速时策略给轮子 3~9 rad/s，前后轮对顶；v1（补全图中位数）实机降到 0.6 rad/s。
# v2（Astra 审查）：①空洞按 sample() 同样的浮点运算逐位认，并与 info["n_hole"] 对账，对不上就不补；
#   ②每条纵向格子沿 x 用前后最近的有效格线性插值（楼梯上补成斜坡，不把上下两个平面搅成一个值），整行无值才用全图中位数。
# 只改发给策略的 /height_scan；/height_scan_raw 和共用的 elevation_map.py 不动。
# S10_HMAP_HOLE_FILL=zero 切回旧行为（空洞=0），用来做 A/B。
_HOLE_FILL = os.environ.get("S10_HMAP_HOLE_FILL", "interp").strip().lower()
_HOLE_MIN_VALID = 20    # 全图有效格太少时（刚起步）不补
_GX = len(np.unique(np.round(CELL_XY[:, 0], 6)))       # 沿 x（前后）的格数，17
_GY = len(np.unique(np.round(CELL_XY[:, 1], 6)))       # 沿 y（左右）的格数，11
assert _GX * _GY == NUM_RAYS
_XS = CELL_XY[:, 0].reshape(_GY, _GX)[0]                # 每行的 x 坐标（CELL_XY 是 (11,17) 按行展平，行=同一 y）
assert np.allclose(CELL_XY[:, 1].reshape(_GY, _GX), CELL_XY[:, 1].reshape(_GY, _GX)[:, :1]), "CELL_XY 行序和假设不符"
print("[hmap] 空洞补法 = %s" % ("0（旧）" if _HOLE_FILL == "zero" else "沿前后就近插值（v2）"), flush=True)


def hole_mask(obs, base_z, n_hole):
    """sample() 里空洞 = clip(base_z − (base_z−OFFSET) − OFFSET)。同样的运算顺序算出这个值逐位比较，
    再和 n_hole 对账；对不上返回 None（不补）。"""
    v = np.clip(base_z - (base_z - OFFSET) - OFFSET, -CLIP, CLIP)
    m = np.asarray(obs) == v
    return m if int(m.sum()) == int(n_hole) else None


def fill_holes(obs, base_z, n_hole):
    """空洞 → 沿前后方向就近插值。返回 (新 obs, 补了几格；-1 表示对账不符没补)。不改入参。"""
    obs = np.asarray(obs, dtype=np.float64)
    if _HOLE_FILL == "zero" or n_hole == 0:
        return obs, 0
    m = hole_mask(obs, base_z, n_hole)
    if m is None:
        return obs, -1
    if NUM_RAYS - int(m.sum()) < _HOLE_MIN_VALID:
        return obs, 0
    out = obs.copy().reshape(_GY, _GX)
    mm = m.reshape(_GY, _GX)
    med = float(np.median(obs[~m]))
    for r in range(_GY):
        h = mm[r]
        if not h.any():
            continue
        ok = ~h
        if ok.any():
            out[r, h] = np.interp(_XS[h], _XS[ok], out[r, ok])   # 两端之外 np.interp 取最近端值
        else:
            out[r, h] = med
    return out.ravel(), int(m.sum())


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_between(a, b):
    """把单位向量 a 转到单位向量 b 的最小旋转（Rodrigues）。"""
    v = np.cross(a, b); s = float(np.linalg.norm(v)); c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = v / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def cloud_xyz(msg):
    """PointCloud2 → (N,3) float32，只认 x/y/z 字段（float32）。"""
    off = {f.name: f.offset for f in msg.fields}
    if not all(k in off for k in ("x", "y", "z")):
        return np.zeros((0, 3), np.float32)
    n = msg.width * msg.height
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    out = np.empty((n, 3), np.float32)
    for i, k in enumerate(("x", "y", "z")):
        out[:, i] = buf[:, off[k]:off[k] + 4].copy().view(np.float32).ravel()
    return out


class HmapNode(Node):
    def __init__(self, a):
        super().__init__("s10_hmap")
        self.a = a
        self.R_bl = rpy_to_R(*a.lidar_rpy); self.t_bl = np.array(a.lidar_xyz)
        def _mk():
            return ElevationMap(resolution=a.res, origin=tuple(a.origin),
                                extent=tuple(a.extent))
        self._mk = _mk
        self.emap = _mk()
        # 影子图：和主图吃同样的帧，但清空时机错开半个周期。
        # 采样永远用主图（较老、已填满），清空的那张先当影子攒够了再换上来。
        self.emap_shadow = _mk() if a.map_ttl > 0 else None
        self.t_swap = time.monotonic()
        self.pose = None            # (t_wall, xyz, R_wb, yaw)
        # --attitude imu 用：IMU 最新横滚/俯仰；KISS 世界系倾斜的低通估计与增量累加的位置
        self.imu = None             # (t_mono, roll, pitch)，弧度，REP-103
        self.g_f = None             # "竖直向上"在 KISS 世界系里的低通方向
        self.t_k_prev = None        # 上一帧 KISS 位置（原始）
        self.t_corr = None          # 转正后累加的位置
        self.t_odom_prev = 0.0
        self.tilt_deg = 0.0         # KISS 世界系相对重力的倾斜（度），进 /hmap_info 第 5 字段
        self.n_imu_stale = 0
        self.t_cloud = 0.0          # 最近一帧被接受的点云时刻（新鲜度闸门用）
        self.n_stale = 0            # 因点云过旧而没发的次数
        self.n_drop = 0             # 被体素点数下限丢掉的点，日志里报比例
        self.n_drop_ceil = 0        # 被"高于地面 --ground-ceil"丢掉的
        self.n_drop_persist = 0     # 被"上一帧没见过"丢掉的
        self.prev_keys = None       # 上一帧攒够点数的体素键（已排序），--persist 用
        self.pub = self.create_publisher(Float32MultiArray, a.out_topic, 10)
        self.pub_raw = self.create_publisher(Float32MultiArray, a.out_topic + "_raw", 10)
        self.pub_info = self.create_publisher(Float32MultiArray, "/hmap_info", 10)
        self.create_subscription(PointCloud2, a.cloud_topic, self.on_cloud, qos_profile_sensor_data)
        if a.pose_source == "sim":
            self.create_subscription(Float32MultiArray, a.pose_topic or "/sim/state", self.on_sim, 10)
        else:
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry, a.pose_topic or "/localization/pose", self.on_odom, 10)
            if a.attitude == "imu":
                from drdds.msg import ImuData
                from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
                self.create_subscription(ImuData, a.imu_topic, self.on_imu,
                                         QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                                    durability=DurabilityPolicy.VOLATILE,
                                                    history=HistoryPolicy.KEEP_LAST))
        self.create_timer(1.0 / a.hz, self.tick)
        self.n_cloud = 0; self.n_pts = 0; self.t_last_log = time.monotonic()
        self.n_fill = 0
        self.get_logger().info("cloud=%s pose=%s 外参 xyz=%s rpy=%s res=%.2f 输出 %s @%.0fHz  滚动窗口=%s" % (
            a.cloud_topic, a.pose_source, a.lidar_xyz, a.lidar_rpy, a.res, a.out_topic, a.hz,
            ("%.1fs（双缓冲，年龄 %.1f~%.1fs）" % (a.map_ttl, a.map_ttl, 2 * a.map_ttl))
            if a.map_ttl > 0 else "无限累积"))
        self.get_logger().info(
            "新鲜度闸门: 点云静默 >%.2fs 即停发 /height_scan（位姿闸门 %.2fs）"
            % (a.cloud_timeout, a.pose_timeout))
        self.get_logger().info(
            "入图防线: 地面天花板 %s | 体素 %.2fx%.2fx%.2f 少于 %d 点丢弃 | 上帧持久性 %s"
            % ("+%.2fm" % a.ground_ceil if a.ground_ceil > 0 else "关",
               a.res, a.res, a.voxel_z, a.min_pts, "开" if a.persist else "关"))
        if a.pose_source == "odom":
            self.get_logger().info(
                ("姿态来源: 横滚/俯仰=IMU（%s，弧度，REP-103，超时 %.2fs 即停止入图）"
                 " + 航向/位置=KISS（世界系倾斜低通 τ=%.1fs，位置按转正后的增量累加）"
                 % (a.imu_topic, a.imu_timeout, a.tilt_tau)) if a.attitude == "imu"
                else "姿态来源: 整套位姿取 KISS（--attitude kiss，旧行为）")

    def on_sim(self, m):
        d = m.data
        if len(d) < 20:
            return
        r, p, y = (math.radians(d[17]), math.radians(d[18]), math.radians(d[19]))
        self.pose = (time.monotonic(), np.array(d[0:3], float), rpy_to_R(r, p, y), y)

    def on_imu(self, m):
        d = m.data
        self.imu = (time.monotonic(), float(d.roll), float(d.pitch))

    def imu_ok(self):
        return self.imu is not None and (time.monotonic() - self.imu[0]) <= self.a.imu_timeout

    def reset_maps(self, why):
        self.get_logger().warn("清空高程图重来：%s" % why)
        self.emap = self._mk()
        self.emap_shadow = self._mk() if self.a.map_ttl > 0 else None
        self.t_swap = time.monotonic()
        self.prev_keys = None

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        R = quat_to_R(q.x, q.y, q.z, q.w)
        t_k = np.array([p.x, p.y, p.z])
        if self.a.attitude != "imu":
            self.pose = (time.monotonic(), t_k, R, math.atan2(R[1, 0], R[0, 0]))
            return
        if not self.imu_ok():
            return                          # 没有新鲜 IMU 就不更新位姿 → 0.5 s 后位姿闸门停发
        now = time.monotonic()
        _, roll, pitch = self.imu
        # IMU 给出的"世界竖直向上"在机体系里的方向：u_b = R_imu^T·ez（R_imu = Rz·Ry(pitch)·Rx(roll)）
        cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
        g_k = R @ np.array([-sp, sr * cp, cr * cp])     # 同一方向在 KISS 世界系里的表达
        if self.t_k_prev is not None and float(np.linalg.norm(t_k - self.t_k_prev)) > 1.0:
            self.reset_maps("KISS 位姿一步跳了 %.2f m（重启或发散）" % float(np.linalg.norm(t_k - self.t_k_prev)))
            self.g_f = None; self.t_k_prev = None
        if self.g_f is None:
            self.g_f = g_k
        else:
            beta = min(1.0, min(max(now - self.t_odom_prev, 0.0), 1.0) / max(self.a.tilt_tau, 1e-3))
            self.g_f = (1.0 - beta) * self.g_f + beta * g_k
        self.g_f = self.g_f / np.linalg.norm(self.g_f)
        C = rot_between(self.g_f, np.array([0.0, 0.0, 1.0]))   # KISS 世界系 → 重力对齐的世界系
        self.t_corr = C @ t_k if self.t_k_prev is None else self.t_corr + C @ (t_k - self.t_k_prev)
        self.t_k_prev = t_k; self.t_odom_prev = now
        Rc = C @ R
        yaw = math.atan2(Rc[1, 0], Rc[0, 0])
        self.tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(self.g_f[2])))))
        self.pose = (now, self.t_corr.copy(), rpy_to_R(roll, pitch, yaw), yaw)

    def denoise(self, pw):
        """丢掉孤立点。pw 是世界系 (N,3)。见文件头 patch 说明。

        用一维整数键做 np.unique，而不是 np.unique(axis=0) ——
        后者是 (N,3) 的 lexsort，94k 点约 15ms；前者是一次 int64 排序，约 4ms。
        键的位宽：ix,iy 各夹到 [0, nx-1]（默认 4000 -> 12 位），
        iz 夹到 [0,1023]（10 位），合起来 < 2^34，int64 装得下。
        """
        k = int(self.a.min_pts)
        if len(pw) == 0:
            return pw, 0
        e = self.emap
        ix = np.clip(np.floor((pw[:, 0] - e.ox) / e.res), 0, e.nx - 1).astype(np.int64)
        iy = np.clip(np.floor((pw[:, 1] - e.oy) / e.res), 0, e.ny - 1).astype(np.int64)
        iz = np.clip(np.floor((pw[:, 2] + 50.0) / self.a.voxel_z), 0, 1023).astype(np.int64)
        key = (ix * np.int64(e.ny) + iy) * np.int64(1024) + iz
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= k if k > 1 else np.ones(len(pw), dtype=bool)

        # 防线 2：体素必须在上一帧也攒够点数。见文件头。
        if self.a.persist:
            good_vox = uk[cnt >= max(k, 1)]          # 本帧"够密"的体素，已排序
            if self.prev_keys is not None and len(self.prev_keys):
                i = np.searchsorted(self.prev_keys, key)
                i = np.clip(i, 0, len(self.prev_keys) - 1)
                seen_before = self.prev_keys[i] == key
            else:
                seen_before = np.zeros(len(pw), dtype=bool)
            n_before = int(keep.sum())
            keep &= seen_before
            self.n_drop_persist += n_before - int(keep.sum())
            self.prev_keys = good_vox                # 下一帧拿它比
        return pw[keep], int(len(pw) - keep.sum())

    def on_cloud(self, msg):
        if self.pose is None:
            return
        if self.a.pose_source == "odom" and self.a.attitude == "imu" and not self.imu_ok():
            self.n_imu_stale += 1
            if self.n_imu_stale in (1, 10) or self.n_imu_stale % 200 == 0:
                self.get_logger().warn("IMU（%s）超过 %.2fs 没来 —— 不入图，/height_scan 将由新鲜度闸门停发。第 %d 次"
                                       % (self.a.imu_topic, self.a.imu_timeout, self.n_imu_stale))
            return
        _, t_wb, R_wb, _ = self.pose
        pts = cloud_xyz(msg)
        if pts.size == 0:
            return
        # 防线 1：相对**当帧实测地面**的天花板。用机体系原点附近的点算地面，
        # 所以必须在转世界系之前做（机体系里雷达就在原点）。
        if self.a.ground_ceil > 0:
            r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
            near = pts[r2 < 0.36, 2]
            if len(near) >= 50:
                gz = float(np.percentile(near, 5))
                m = pts[:, 2] <= gz + self.a.ground_ceil
                self.n_drop_ceil += int(len(pts) - m.sum())
                pts = pts[m]
                if len(pts) == 0:
                    return
        pw = (pts @ self.R_bl.T + self.t_bl) @ R_wb.T + t_wb
        pw, n_drop = self.denoise(pw)
        self.n_drop += n_drop
        self.emap.update(pw, base_z=float(t_wb[2]))
        if self.emap_shadow is not None:
            self.emap_shadow.update(pw, base_z=float(t_wb[2]))
            now = time.monotonic()
            if now - self.t_swap >= self.a.map_ttl:
                # 主图退休（它已经攒了 2*TTL 的历史，棘轮该到头了），
                # 影子顶上（攒了 TTL，够填满 0.6m 窗），新影子从零开始。
                # 主图退休（攒了 2*TTL 的历史），影子顶上（攒了 TTL，
                # 实测 0.3s 就把 187 格填饱和），退下来那张**就地清零**当新影子。
                # 【别改成 self._mk()】默认地图 4000x4000=1600万格=64MB，
                #   TTL=0.4 时重新分配是 160 MB/s 的垃圾。v1 就是这么写的。
                # 【别改成 np.nan】初值是 -inf；累积走 np.maximum.at，
                #   NaN 会永久传播，那格从此写不进值 → 整张图静默变全空洞。
                old, self.emap = self.emap, self.emap_shadow
                old.z[:] = -np.inf
                old.n_seen = 0
                old.n_update = 0
                old.n_points = 0
                self.emap_shadow = old
                self.t_swap = now
        self.t_cloud = time.monotonic()
        self.n_cloud += 1; self.n_pts += len(pts)

    def tick(self):
        if self.pose is None:
            return
        t0, t_wb, _, yaw = self.pose
        if time.monotonic() - t0 > self.a.pose_timeout:
            return                                          # 位姿断了不发（宁缺毋滥）
        # 输入新鲜度闸门。见文件头：不停发，runner 就发现不了地图已经旧了。
        cloud_age = time.monotonic() - self.t_cloud
        if self.a.cloud_timeout > 0 and cloud_age > self.a.cloud_timeout:
            self.n_stale += 1
            if self.n_stale in (1, 10) or self.n_stale % 200 == 0:
                self.get_logger().warn(
                    "点云已静默 %.2fs（上限 %.2fs）—— 停止发布 /height_scan，"
                    "让 runner 回落到按平地走。第 %d 次。查 /LIDAR/POINTS_MERGED"
                    % (cloud_age, self.a.cloud_timeout, self.n_stale))
            return
        if self.n_stale:
            self.get_logger().info("点云恢复，重新发布 /height_scan（之前静默拦了 %d 次）"
                                   % self.n_stale)
            self.n_stale = 0
        obs, info = self.emap.sample(t_wb[:2], float(t_wb[2]), yaw)
        obs, self.n_fill = fill_holes(obs, float(t_wb[2]), info["n_hole"])   # 只影响 /height_scan；info["raw"] 仍是补之前的
        m = Float32MultiArray(); m.data = [float(v) for v in obs]; self.pub.publish(m)
        m2 = Float32MultiArray(); m2.data = [float(v) for v in info["raw"]]; self.pub_raw.publish(m2)
        # 第 4 个字段 = 点云年龄（秒）。控制台用它区分"地图新鲜"和"在重发旧图"。
        mi = Float32MultiArray()
        mi.data = [float(info["hole_rate"]), float(info["map_coverage"]),
                   float(info["frames"]), float(cloud_age), float(self.tilt_deg)]
        self.pub_info.publish(mi)
        if time.monotonic() - self.t_last_log > 5.0:
            self.t_last_log = time.monotonic()
            self.get_logger().info(
                "点云 %d 帧 %d 点  空洞率 %.2f  覆盖 %.4f  obs[中位]=%.3f  地图龄 %.1fs  "
                "孤立点丢弃 %.2f%%(min-pts=%d)" % (
                    self.n_cloud, self.n_pts, info["hole_rate"], info["map_coverage"],
                    float(np.median(obs)),
                    time.monotonic() - self.t_swap + (self.a.map_ttl or 0.0),
                    100.0 * self.n_drop / max(self.n_pts, 1), self.a.min_pts)
                + "  地面天花板丢 %.2f%%  上帧未见丢 %.2f%%" % (
                    100.0 * self.n_drop_ceil / max(self.n_pts, 1),
                    100.0 * self.n_drop_persist / max(self.n_pts, 1))
                + ("  空洞插值 %d 格" % self.n_fill if self.n_fill >= 0 else "  !! 空洞对账不符，本帧没补"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud-topic", default="/LIDAR/POINTS_MERGED")
    ap.add_argument("--pose-source", choices=("odom", "sim"), default="odom")
    ap.add_argument("--pose-topic", default=None)
    ap.add_argument("--pose-timeout", type=float, default=0.5)
    ap.add_argument("--cloud-timeout", type=float, default=0.25,
                    help="点云静默超过此值(秒)就停止发布 /height_scan。"
                         "让 runner 那道 100ms 过期检查能真正生效 —— 它量的是消息"
                         "到达时刻，而我们是 50Hz 定时发、点云只有 10Hz，"
                         "不停发它就永远发现不了地图已经旧了")
    ap.add_argument("--lidar-xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--lidar-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--origin", type=float, nargs=2, default=[-100.0, -100.0])
    ap.add_argument("--extent", type=float, nargs=2, default=[200.0, 200.0])
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--out-topic", default="/height_scan")
    ap.add_argument("--ground-ceil", type=float, default=0.45,
                    help="丢掉高于当帧实测地面超过此值(米)的点。相对地面而非机体，"
                         "所以狗蹲下也不会误滤真台阶。0=关")
    ap.add_argument("--persist", type=int, default=1,
                    help="1=体素必须在当前帧和上一帧都攒够 min-pts 才准入图"
                         "（压掉瞬时稀疏点/粉尘/移动物）。0=关")
    ap.add_argument("--min-pts", type=int, default=3,
                    help="体素点数下限：按 (res, res, --voxel-z) 分箱，点数少于此值的"
                         "箱子整箱丢弃。掐掉单噪点经 np.maximum 造出的假墙。1=不过滤")
    ap.add_argument("--voxel-z", type=float, default=0.10,
                    help="点数统计用的高度分箱（米）。比 res 粗一点，"
                         "让掠射角下的真实表面也能攒够点数")
    ap.add_argument("--map-ttl", type=float, default=0.4,
                    help="滚动窗口秒数。>0 时双缓冲交替清空，地图年龄夹在 "
                         "[TTL, 2*TTL]，掐掉 max 累积的棘轮；0=无限累积（原行为）")
    ap.add_argument("--attitude", choices=("imu", "kiss"), default="imu",
                    help="imu（默认）= 横滚/俯仰取 IMU、航向/位置取 KISS 并做世界系倾斜修正；kiss = 旧行为")
    ap.add_argument("--imu-topic", default="/IMU_DATA_10HZ",
                    help="drdds ImuData。10HZ 镜像约 20 Hz、SDK 模式关着也有；/IMU_DATA 为 200 Hz 但只在 SDK 模式下有")
    ap.add_argument("--imu-timeout", type=float, default=0.3, help="IMU 超过此秒数没来就停止入图")
    ap.add_argument("--tilt-tau", type=float, default=5.0,
                    help="KISS 世界系倾斜估计的低通时间常数（秒）。越大越稳，越小跟得越快")
    a = ap.parse_args()
    rclpy.init(); n = HmapNode(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
