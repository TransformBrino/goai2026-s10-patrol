#!/usr/bin/env python3
"""在带颜色的点云里找官方喷涂的红点（分块流式读取，1.2GB 全量 LAZ 也能跑）。

判据：地面带（|z−地面|<0.8m）内的饱和红点（R>140 且 R>1.7G 且 R>1.7B），按 res 网格计数；
      格内红点 ≥ min_red 且红占比 > min_frac 的格子聚成团，团大小 ≤ max_cells（直径 ≤ ~0.6m）即候选。
输出：<前缀>_red.csv（x_local, y_local, 格数, 红点数）、<前缀>_red.png（标在网格图上）、<前缀>_rgb.png（真彩俯视）
用法：python3 reddots.py <laz> <前缀> [--res 0.1] [--ground 13.0] [--grid site_grid.png]
"""
import argparse
from collections import deque

import laspy
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("laz"); ap.add_argument("out")
ap.add_argument("--res", type=float, default=0.1); ap.add_argument("--ground", type=float, default=13.0)
ap.add_argument("--x0", type=float, default=214616.5); ap.add_argument("--y0", type=float, default=3358916.7)
ap.add_argument("--min-red", type=int, default=3); ap.add_argument("--min-frac", type=float, default=0.4)
ap.add_argument("--max-cells", type=int, default=40); ap.add_argument("--grid", default=None)
a = ap.parse_args()

with laspy.open(a.laz) as f:
    h = f.header
    W = int(np.ceil((h.maxs[0] - a.x0) / a.res)) + 2; H = int(np.ceil((h.maxs[1] - a.y0) / a.res)) + 2
    cnt = np.zeros(W * H, np.int32); red = np.zeros(W * H, np.int32); rgb = np.zeros((W * H, 3), np.float32)
    total = 0; nred = 0
    for ch in f.chunk_iterator(4_000_000):
        x = np.asarray(ch.x); y = np.asarray(ch.y); z = np.asarray(ch.z)
        R = np.asarray(ch.red).astype(np.float32); G = np.asarray(ch.green).astype(np.float32); B = np.asarray(ch.blue).astype(np.float32)
        if R.max() > 255: R /= 257; G /= 257; B /= 257
        gb = np.abs(z - a.ground) < 0.8
        col = ((x - a.x0) / a.res).astype(int); row = (H - 1 - (y - a.y0) / a.res).astype(int)
        ok = gb & (col >= 0) & (col < W) & (row >= 0) & (row < H)
        cell = (row * W + col)[ok]
        np.add.at(cnt, cell, 1); np.add.at(rgb, cell, np.stack([R[ok], G[ok], B[ok]], 1))
        isred = (R[ok] > 140) & (R[ok] > 1.7 * G[ok]) & (R[ok] > 1.7 * B[ok])
        np.add.at(red, cell[isred], 1)
        total += len(x); nred += int(isred.sum())
print("点数 %d，地面带格子 %d，饱和红点 %d" % (total, int((cnt > 0).sum()), nred))
okc = cnt > 0
img = np.full((W * H, 3), 255, np.uint8); img[okc] = (rgb[okc] / cnt[okc][:, None]).astype(np.uint8)
Image.fromarray(img.reshape(H, W, 3)).save(a.out + "_rgb.png")
frac = np.zeros(W * H, np.float32); frac[okc] = red[okc] / cnt[okc]
mask = ((red >= a.min_red) & (frac > a.min_frac)).reshape(H, W)
print("候选格 %d" % int(mask.sum()))
seen = np.zeros_like(mask); blobs = []
for r, c in zip(*np.where(mask)):
    if seen[r, c]: continue
    q = deque([(r, c)]); seen[r, c] = True; cells = []
    while q:
        p, s = q.popleft(); cells.append((p, s))
        for dp in (-1, 0, 1):
            for ds in (-1, 0, 1):
                p2, s2 = p + dp, s + ds
                if 0 <= p2 < H and 0 <= s2 < W and mask[p2, s2] and not seen[p2, s2]:
                    seen[p2, s2] = True; q.append((p2, s2))
    if len(cells) <= a.max_cells:
        rr = np.mean([p for p, _ in cells]); cc = np.mean([s for _, s in cells])
        blobs.append((cc * a.res, (H - 1 - rr) * a.res, len(cells), int(sum(red[p * W + s] for p, s in cells))))
blobs.sort(key=lambda b: -b[3])
print("小红斑候选 %d 个（按红点数排序，前 40）：" % len(blobs))
for b in blobs[:40]:
    print("  x=%6.1f y=%6.1f  格 %2d  红点 %3d" % b)
np.savetxt(a.out + "_red.csv", np.array(blobs) if blobs else np.zeros((0, 4)), fmt="%.2f,%.2f,%d,%d", header="x_local,y_local,cells,redpts", comments="")
if a.grid:
    im = Image.open(a.grid).convert("RGB"); dr = ImageDraw.Draw(im); gres = 0.2
    for b in blobs[:150]:
        px, py = b[0] / gres, im.size[1] - 1 - b[1] / gres
        dr.ellipse([px - 7, py - 7, px + 7, py + 7], outline=(255, 0, 255), width=2)
    im.save(a.out + "_red.png"); print("已标到", a.out + "_red.png")
