#!/usr/bin/env python3
"""造一个像决赛园区的合成点云（PCD ASCII），在真点云到手前把 cloud2map + 点选工具跑通。
地物：40×30m 地面（含轻微起伏）；一条 3m 宽铺装路弯过去；一块抬高 0.38m 的台地（石笼边）；
      台地上高草带（0.5m 高噪点）；一个 0.2m 深的鹅卵石坑；几棵树（1.5~3m 高的柱子）；一堵 2m 高的玻璃幕墙。
用法：python3 make_fake_cloud.py out.pcd [--n 600000]
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=600000)
a = ap.parse_args()
rng = np.random.default_rng(0)
n = a.n
x = rng.uniform(0, 40, n); y = rng.uniform(0, 30, n)
z = 0.03 * np.sin(x / 3.0) * np.cos(y / 4.0) + rng.normal(0, 0.01, n)
# 台地：x 22~40, y 12~30，抬高 0.38；边缘 0.3m 内加碎石噪声（石笼）
terr = (x > 22) & (y > 12)
z[terr] += 0.38
edge = terr & ((x < 22.4) | (y < 12.4))
z[edge] += rng.uniform(-0.06, 0.03, edge.sum())
# 高草带：台地上紧贴边缘 1m 宽
grass = terr & (x > 22.4) & (x < 23.6) & (y > 12.4)
z[grass] += rng.uniform(0.2, 0.55, grass.sum())
# 坑：圆心 (10, 20) 半径 2m，深 0.2，坑底卵石噪声
pit = (x - 10) ** 2 + (y - 20) ** 2 < 4
z[pit] -= 0.2; z[pit] += rng.uniform(0, 0.05, pit.sum())
# 铺装路：沿 y = 8 + 0.15*(x-20)^2/40 的曲线，宽 3m，路面略高 0.02 且更平
yc = 8 + 0.15 * (x - 20) ** 2 / 40
road = np.abs(y - yc) < 1.5
z[road] = 0.02 + rng.normal(0, 0.004, road.sum())
pts = [np.stack([x, y, z], axis=1)]
# 树：柱子
for tx, ty, th in ((6, 5, 2.5), (15, 26, 3.0), (30, 5, 2.0), (36, 20, 2.8)):
    m = 4000; ang = rng.uniform(0, 2 * np.pi, m); zz = rng.uniform(0, th, m)
    pts.append(np.stack([tx + 0.2 * np.cos(ang), ty + 0.2 * np.sin(ang), zz], axis=1))
    m = 8000; r = rng.uniform(0, 2.0, m); ang = rng.uniform(0, 2 * np.pi, m)
    pts.append(np.stack([tx + r * np.cos(ang), ty + r * np.sin(ang), th + rng.uniform(-0.5, 0.5, m)], axis=1))
# 玻璃幕墙：x=0.5, y 0~30，2m 高（稀疏，玻璃回波少）
m = 6000
pts.append(np.stack([np.full(m, 0.5), rng.uniform(0, 30, m), rng.uniform(0, 2.0, m)], axis=1))
P = np.concatenate(pts)
with open(a.out, "w", encoding="ascii") as f:
    f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
    f.write("WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" % (len(P), len(P)))
    np.savetxt(f, P, fmt="%.3f")
print("写出 %s：%d 点" % (a.out, len(P)))
