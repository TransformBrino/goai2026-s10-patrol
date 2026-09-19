#!/usr/bin/env python3
"""从点云自动找"立面"（路肩/石笼/台阶）并量高度：回答「石笼到底多高、在哪」。

做法：每格取**低分位**高度当地面（抗树冠/高草），格距 res；边缘 = 8 邻域内最大高差；
      高差在 [hmin, hmax] 的格子标为立面候选；连通域合并，输出每条立面的位置（局部坐标）、长度、高差中位数。
输出：<前缀>_edges.png（地面图上叠洋红色立面）、<前缀>_edges.json（立面清单）、<前缀>_ground.npy（地面高度网格）
用法：python3 stepedges.py <laz/pcd> <前缀> [--res 0.2] [--hmin 0.25] [--hmax 0.6] [--q 20] [--zmin 7 --zmax 17]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cloud2map import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud"); ap.add_argument("out")
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--q", type=float, default=20, help="每格地面高度取的百分位（低分位抗植被）")
    ap.add_argument("--hmin", type=float, default=0.25); ap.add_argument("--hmax", type=float, default=0.60)
    ap.add_argument("--zmin", type=float, default=None); ap.add_argument("--zmax", type=float, default=None)
    ap.add_argument("--minlen", type=int, default=8, help="立面连通域最少格数")
    a = ap.parse_args()
    P = load(a.cloud); P = P[np.isfinite(P).all(axis=1)]
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    W = int(np.ceil((P[:, 0].max() - x0) / a.res)) + 1; H = int(np.ceil((P[:, 1].max() - y0) / a.res)) + 1
    col = ((P[:, 0] - x0) / a.res).astype(int); row = (H - 1 - (P[:, 1] - y0) / a.res).astype(int)
    cell = row * W + col
    order = np.argsort(cell, kind="stable"); cell, z = cell[order], P[order, 2]
    uniq, start = np.unique(cell, return_index=True); ends = np.append(start[1:], len(z))
    g = np.full(W * H, np.nan)
    cnt = ends - start
    # 低分位：格内点数 ≥3 才可信
    for u, s, e in zip(uniq, start, ends):
        if e - s >= 3:
            g[u] = np.percentile(z[s:e], a.q)
    g = g.reshape(H, W)
    np.save(a.out + "_ground.npy", g)
    # 8 邻域最大高差（只在两格都有值时算）
    diff = np.zeros_like(g); diff[:] = 0.0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        A = g[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
        B = g[max(0, dr):H - max(0, -dr) if dr else H, max(0, dc):W - max(0, -dc) if dc else W]
        d = np.abs(A - B); d = np.where(np.isnan(d), 0, d)
        sub = diff[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
        np.maximum(sub, d, out=sub)
        sub2 = diff[max(0, dr):H - max(0, -dr) if dr else H, max(0, dc):W - max(0, -dc) if dc else W]
        np.maximum(sub2, d, out=sub2)
    mask = (diff >= a.hmin) & (diff <= a.hmax)
    # 连通域（4 邻域）
    lab = np.zeros((H, W), dtype=np.int32); n = 0
    segs = []
    from collections import deque
    for r in range(H):
        for c in range(W):
            if mask[r, c] and lab[r, c] == 0:
                n += 1; q = deque([(r, c)]); lab[r, c] = n; cells = []
                while q:
                    rr, cc = q.popleft(); cells.append((rr, cc))
                    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        r2, c2 = rr + dr, cc + dc
                        if 0 <= r2 < H and 0 <= c2 < W and mask[r2, c2] and lab[r2, c2] == 0:
                            lab[r2, c2] = n; q.append((r2, c2))
                if len(cells) >= a.minlen:
                    rs = np.array([p[0] for p in cells]); cs = np.array([p[1] for p in cells])
                    hs = diff[rs, cs]
                    xs = x0 + cs * a.res; ys = y0 + (H - 1 - rs) * a.res
                    segs.append(dict(id=len(segs) + 1, cells=len(cells), len_m=round(len(cells) * a.res, 1),
                                     h_med=round(float(np.median(hs)), 3), h_p90=round(float(np.percentile(hs, 90)), 3),
                                     x_local=[round(float(xs.min() - x0), 1), round(float(xs.max() - x0), 1)],
                                     y_local=[round(float(ys.min() - y0), 1), round(float(ys.max() - y0), 1)],
                                     ground_z=round(float(np.nanmedian(g[rs, cs])), 2)))
    segs.sort(key=lambda s: -s["cells"])
    # 画图：地面图 + 洋红边缘
    zmin = a.zmin if a.zmin is not None else float(np.nanpercentile(g, 2)); zmax = a.zmax if a.zmax is not None else float(np.nanpercentile(g, 98))
    t = np.clip((g - zmin) / max(zmax - zmin, 1e-6), 0, 1)
    img = np.full((H, W, 3), 255, np.uint8); v = ~np.isnan(g)
    img[v, 0] = np.interp(t[v], [0, .35, .65, 1], [20, 30, 230, 200]); img[v, 1] = np.interp(t[v], [0, .35, .65, 1], [40, 170, 210, 30]); img[v, 2] = np.interp(t[v], [0, .35, .65, 1], [120, 60, 40, 30])
    img[mask] = (255, 0, 255)
    try:
        from PIL import Image; Image.fromarray(img).save(a.out + "_edges.png")
    except ImportError:
        import struct, zlib
        raw = b"".join(b"\x00" + img[i].tobytes() for i in range(H))
        def ch(tp, d): return struct.pack(">I", len(d)) + tp + d + struct.pack(">I", zlib.crc32(tp + d) & 0xffffffff)
        Path(a.out + "_edges.png").write_bytes(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw, 9)) + ch(b"IEND", b""))
    meta = dict(x0=float(x0), y0=float(y0), res=a.res, W=W, H=H, hmin=a.hmin, hmax=a.hmax, q=a.q, n_edges=len(segs), edges=segs[:60])
    Path(a.out + "_edges.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("网格 %dx%d @%.2fm，立面候选格 %d，连通域(≥%d格) %d 条" % (W, H, a.res, int(mask.sum()), a.minlen, len(segs)))
    print("最长 15 条（局部坐标 = UTM − (%.1f, %.1f)）：" % (x0, y0))
    for s in segs[:15]:
        print("  #%-3d 长 %5.1fm 高差中位 %.2f (p90 %.2f)  x %s  y %s  地面z %.2f" % (s["id"], s["len_m"], s["h_med"], s["h_p90"], s["x_local"], s["y_local"], s["ground_z"]))


if __name__ == "__main__":
    main()
