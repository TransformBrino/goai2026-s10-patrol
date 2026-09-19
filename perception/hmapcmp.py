#!/usr/bin/env python3
"""对拍：外部 hmap_node 发的 /height_scan_ext 与仿真内部 /height_scan（同一时刻）逐格比较。
用法：python3 hmapcmp.py --secs 30 [--a /height_scan --b /height_scan_ext]
输出：均值绝对误差、p90、>0.1m 的格子占比、两边各自的空洞率（±CLIP 钳位值当作"没看到"）。
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class Cmp(Node):
    def __init__(self, a):
        super().__init__("hmapcmp")
        self.a = a; self.last_a = None; self.last_b = None; self.errs = []; self.n = 0
        self.create_subscription(Float32MultiArray, a.a, self.on_a, 10)
        self.create_subscription(Float32MultiArray, a.b, self.on_b, 10)
        self.t0 = time.monotonic()

    def on_a(self, m):
        self.last_a = (time.monotonic(), np.array(m.data))

    def on_b(self, m):
        self.last_b = (time.monotonic(), np.array(m.data))
        if self.last_a is not None and abs(self.last_a[0] - self.last_b[0]) < 0.05 and len(self.last_a[1]) == len(m.data):
            e = np.abs(self.last_a[1] - self.last_b[1]); self.errs.append(e); self.n += 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", default="/height_scan"); ap.add_argument("--b", default="/height_scan_ext"); ap.add_argument("--secs", type=float, default=30)
    a = ap.parse_args(); rclpy.init(); n = Cmp(a)
    while time.monotonic() - n.t0 < a.secs:
        rclpy.spin_once(n, timeout_sec=0.05)
    if not n.errs:
        print("没有配对样本（两个话题有没有同时在发？）"); return
    E = np.stack(n.errs)
    print("配对帧 %d  均值绝对误差 %.3f m  p90 %.3f  p99 %.3f  >0.10m 格占比 %.1f%%  >0.05m %.1f%%" % (
        len(E), E.mean(), np.percentile(E, 90), np.percentile(E, 99), 100 * (E > 0.10).mean(), 100 * (E > 0.05).mean()))
    worst = E.mean(axis=0); idx = np.argsort(-worst)[:8]
    print("误差最大的格子（flat=iy*17+ix）:", [(int(i), round(float(worst[i]), 3)) for i in idx])


if __name__ == "__main__":
    main()
