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
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray

SIM_DIR = Path(__file__).resolve().parents[2] / "s10_ws/src/S10_sdk_deploy/interface/robot/simulation"
sys.path.insert(0, str(SIM_DIR))
from elevation_map import ElevationMap  # noqa: E402  （读环境变量的裁剪开关在 import 时生效）
from heightmap import NUM_RAYS  # noqa: E402


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


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
        self.emap = ElevationMap(resolution=a.res, origin=tuple(a.origin), extent=tuple(a.extent))
        self.pose = None            # (t_wall, xyz, R_wb, yaw)
        self.pub = self.create_publisher(Float32MultiArray, a.out_topic, 10)
        self.pub_raw = self.create_publisher(Float32MultiArray, a.out_topic + "_raw", 10)
        self.pub_info = self.create_publisher(Float32MultiArray, "/hmap_info", 10)
        self.create_subscription(PointCloud2, a.cloud_topic, self.on_cloud, 5)
        if a.pose_source == "sim":
            self.create_subscription(Float32MultiArray, a.pose_topic or "/sim/state", self.on_sim, 10)
        else:
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry, a.pose_topic or "/localization/pose", self.on_odom, 10)
        self.create_timer(1.0 / a.hz, self.tick)
        self.n_cloud = 0; self.n_pts = 0; self.t_last_log = time.monotonic()
        self.get_logger().info("cloud=%s pose=%s 外参 xyz=%s rpy=%s res=%.2f 输出 %s @%.0fHz" % (
            a.cloud_topic, a.pose_source, a.lidar_xyz, a.lidar_rpy, a.res, a.out_topic, a.hz))

    def on_sim(self, m):
        d = m.data
        if len(d) < 20:
            return
        r, p, y = (math.radians(d[17]), math.radians(d[18]), math.radians(d[19]))
        self.pose = (time.monotonic(), np.array(d[0:3], float), rpy_to_R(r, p, y), y)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        R = quat_to_R(q.x, q.y, q.z, q.w)
        yaw = math.atan2(R[1, 0], R[0, 0])
        self.pose = (time.monotonic(), np.array([p.x, p.y, p.z]), R, yaw)

    def on_cloud(self, msg):
        if self.pose is None:
            return
        _, t_wb, R_wb, _ = self.pose
        pts = cloud_xyz(msg)
        if pts.size == 0:
            return
        pw = (pts @ self.R_bl.T + self.t_bl) @ R_wb.T + t_wb
        self.emap.update(pw, base_z=float(t_wb[2]))
        self.n_cloud += 1; self.n_pts += len(pts)

    def tick(self):
        if self.pose is None:
            return
        t0, t_wb, _, yaw = self.pose
        if time.monotonic() - t0 > self.a.pose_timeout:
            return                                          # 位姿断了不发（宁缺毋滥）
        obs, info = self.emap.sample(t_wb[:2], float(t_wb[2]), yaw)
        m = Float32MultiArray(); m.data = [float(v) for v in obs]; self.pub.publish(m)
        m2 = Float32MultiArray(); m2.data = [float(v) for v in info["raw"]]; self.pub_raw.publish(m2)
        mi = Float32MultiArray(); mi.data = [float(info["hole_rate"]), float(info["map_coverage"]), float(info["frames"])]; self.pub_info.publish(mi)
        if time.monotonic() - self.t_last_log > 5.0:
            self.t_last_log = time.monotonic()
            self.get_logger().info("点云 %d 帧 %d 点  空洞率 %.2f  覆盖 %.4f  obs[中位]=%.3f" % (
                self.n_cloud, self.n_pts, info["hole_rate"], info["map_coverage"], float(np.median(obs))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud-topic", default="/LIDAR/POINTS_MERGED")
    ap.add_argument("--pose-source", choices=("odom", "sim"), default="odom")
    ap.add_argument("--pose-topic", default=None)
    ap.add_argument("--pose-timeout", type=float, default=0.5)
    ap.add_argument("--lidar-xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--lidar-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--origin", type=float, nargs=2, default=[-100.0, -100.0])
    ap.add_argument("--extent", type=float, nargs=2, default=[200.0, 200.0])
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--out-topic", default="/height_scan")
    a = ap.parse_args()
    rclpy.init(); n = HmapNode(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
