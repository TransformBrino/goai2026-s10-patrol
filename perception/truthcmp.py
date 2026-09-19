#!/usr/bin/env python3
"""雷达累积高程图 vs 真值射线高程图 逐格对拍（同一仿真、同一时刻），定位「图在哪儿错」。

订阅 /sim/state（真值位姿）与 /height_scan（部署栈发给策略的 187 格，含裁剪）；
本进程自己加载同一 MJCF，用 heightmap.TerrainProbe 对地形打射线得到训练等价的 187 格（不裁剪），
再按部署侧同样的 FWD/LAT 裁剪规则裁一份，两两比较。按机身到墙的距离分段统计误差与「假障碍/漏障碍」。
用法：S10_MUJOCO_XML=<场景> python3 truthcmp.py --secs 40 [--fwd-clip 0.6 --lat-clip 0.30] [--wall-x 3.0]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

SIM_DIR = Path(__file__).resolve().parents[2] / "s10_ws/src/S10_sdk_deploy/interface/robot/simulation"
sys.path.insert(0, str(SIM_DIR))
import mujoco  # noqa: E402
from heightmap import CELL_XY, TerrainProbe, encode  # noqa: E402


class Cmp(Node):
    def __init__(self, a):
        super().__init__("truthcmp")
        self.a = a
        self.model = mujoco.MjModel.from_xml_path(os.environ["S10_MUJOCO_XML"])
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.probe = TerrainProbe(self.model)
        self.state = None
        self.rows = []      # (dist_to_wall, |err| per cell, truth, lidar)
        self.create_subscription(Float32MultiArray, "/sim/state", self.on_state, 10)
        self.create_subscription(Float32MultiArray, "/height_scan", self.on_scan, 10)
        self.t0 = time.monotonic()

    def on_state(self, m):
        self.state = np.array(m.data)

    def clip(self, obs):
        obs = obs.copy()
        near = CELL_XY[:, 0] <= 0.1
        if self.a.fwd_clip:
            far = CELL_XY[:, 0] > self.a.fwd_clip
            obs[far] = float(np.median(obs[near]))
        if self.a.lat_clip:
            side = np.abs(CELL_XY[:, 1]) > self.a.lat_clip
            obs[side] = float(np.median(obs[near]))
        return obs

    def on_scan(self, m):
        if self.state is None or len(m.data) != 187:
            return
        s = self.state
        pos = s[0:3]; yaw = np.radians(s[19])
        hz = self.probe.hit_z(self.data, np.asarray(pos[:2], float), float(pos[2]), yaw)
        truth = encode(float(pos[2]), hz)
        truth_c = self.clip(truth)
        lidar = np.array(m.data)
        d = self.a.wall_x - float(pos[0])
        self.rows.append((d, np.abs(lidar - truth_c), truth_c, lidar))

    def report(self):
        if not self.rows:
            print("没有样本"); return
        print("样本 %d 帧。按「机身到墙的距离」分段（墙在 x=%.1f）：" % (len(self.rows), self.a.wall_x))
        print("%-12s %5s %8s %8s %10s %10s" % ("距墙(m)", "帧", "均误差", "p90", "假障碍格%", "漏障碍格%"))
        bins = [(1.5, 9), (1.0, 1.5), (0.7, 1.0), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3), (-9, 0.0)]
        for lo, hi in bins:
            sel = [r for r in self.rows if lo <= r[0] < hi]
            if not sel: continue
            E = np.stack([r[1] for r in sel]); T = np.stack([r[2] for r in sel]); L = np.stack([r[3] for r in sel])
            # obs 定义：base_z - 地形 - OFFSET → 台阶（地形高）使 obs 变小（更负）。假障碍 = 雷达图比真值低 >0.15（多报了高地形）；漏障碍 = 雷达图比真值高 >0.15
            fake = ((L - T) < -0.15).mean() * 100; miss = ((L - T) > 0.15).mean() * 100
            print("%-12s %5d %8.3f %8.3f %10.1f %10.1f" % ("%.1f~%.1f" % (lo, hi), len(sel), E.mean(), np.percentile(E, 90), fake, miss))
        # 最差的格子（按机体系 x,y 列出）
        E = np.stack([r[1] for r in self.rows]).mean(axis=0); idx = np.argsort(-E)[:10]
        print("误差最大的 10 格（机体系 x,y → 均误差）:", [("%.1f,%.1f" % (CELL_XY[i, 0], CELL_XY[i, 1]), round(float(E[i]), 3)) for i in idx])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--secs", type=float, default=40); ap.add_argument("--fwd-clip", type=float, default=0.6)
    ap.add_argument("--lat-clip", type=float, default=0.30); ap.add_argument("--wall-x", type=float, default=3.0)
    a = ap.parse_args(); rclpy.init(); n = Cmp(a)
    while time.monotonic() - n.t0 < a.secs:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.report()


if __name__ == "__main__":
    main()
