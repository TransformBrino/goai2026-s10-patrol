#!/usr/bin/env python3
"""从现场点云地面网格（stepedges 输出的 *_ground.npy，0.1 m）裁出「真实地形训练补丁」。

每个补丁 8×8 m（80×80 格），以指定立面为中心线、机器人出生点（补丁中心）在立面**低侧 2 m**处，
使朝 +x 走 2 m 正对撞上真实的石笼/路肩。高度相对出生点地面归零，空洞用邻域中位数填，超出 ±1.2 m 钳位。
输出：<outdir>/patch_<id>.npy（float32 米，形状 (80,80)，索引 [ix, iy] 与 IsaacLab 高度场一致：ix 沿 +x，iy 沿 +y）
      与 patch_<id>.png 预览、patches.json（来源、立面高、坐标）。
用法：python3 make_patches.py <ground.npy> <edges.json> <outdir> --ids 14 15 6 [--size 8 --approach 2]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def fill_nan(p, r=3):
    out = p.copy()
    nan = np.isnan(out)
    if not nan.any():
        return out
    H, W = out.shape
    ys, xs = np.where(nan)
    for y, x in zip(ys, xs):
        win = p[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        v = win[np.isfinite(win)]
        out[y, x] = np.median(v) if v.size else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ground"); ap.add_argument("edges"); ap.add_argument("outdir")
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--size", type=float, default=8.0); ap.add_argument("--approach", type=float, default=2.0)
    a = ap.parse_args()
    g = np.load(a.ground); meta = json.load(open(a.edges, encoding="utf-8"))
    res = meta["res"]; H, W = g.shape
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    info = []
    n = int(round(a.size / res))
    for e in meta["edges"]:
        if e["id"] not in a.ids:
            continue
        # 立面中心（局部坐标，米）
        wx = (e["x_local"][0] + e["x_local"][1]) / 2; wy = (e["y_local"][0] + e["y_local"][1]) / 2
        # 四个方向逐一试：沿方向 d 从立面中心向「低侧」退 1~3 m 取均值、向高侧进 1~3 m 取均值，
        # 选 (高−低) 最接近立面高、且低侧平（std<0.10）的方向作为低侧方向 low_dir。
        def prof(dxm, dym, lo, hi):
            vals = []
            for dist in np.arange(lo, hi, res):
                xx, yy = wx + dxm * dist, wy + dym * dist
                cc, rr = int(round(xx / res)), int(round(H - 1 - yy / res))
                if 0 <= rr < H and 0 <= cc < W and np.isfinite(g[rr, cc]):
                    vals.append(g[rr, cc])
            return np.array(vals)
        best = None
        for name, (dxm, dym) in {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}.items():
            low = prof(dxm, dym, 1.0, 3.0); high = prof(-dxm, -dym, 1.0, 3.0)
            if len(low) < 8 or len(high) < 8:
                continue
            rise = float(np.median(high) - np.median(low)); flat = float(np.std(low))
            score = abs(rise - e["h_med"]) + (0.5 if flat > 0.10 else 0.0)
            if best is None or score < best[0]:
                best = (score, name, rise, flat)
        if best is None or best[0] > 0.25:
            print("#%d 找不到干净的低侧（%s），跳过" % (e["id"], best)); continue
        low_dir = best[1]
        off = {"+x": (a.approach, 0), "-x": (-a.approach, 0), "+y": (0, a.approach), "-y": (0, -a.approach)}[low_dir]
        sx, sy = wx + off[0], wy + off[1]
        c0 = int(round((sx - a.size / 2) / res)); r0 = int(round(H - 1 - (sy + a.size / 2) / res))
        p = g[r0:r0 + n, c0:c0 + n]
        if p.shape != (n, n):
            print("#%d 越界，跳过" % e["id"]); continue
        p = fill_nan(p)
        # 行 0 是 +y 顶部 → 转成 [ix, iy]（ix 沿 +x，iy 沿 +y）
        arr = np.flipud(p).T.astype(np.float32)
        # 旋转使「低侧→立面」方向成为 +x（机器人朝 +x 撞墙）
        rot = {"+x": 2, "-x": 0, "+y": 1, "-y": 3}[low_dir]   # 低侧在 −x 时已是朝 +x 撞墙（旋 0）
        arr = np.rot90(arr, k=rot)
        c = n // 2
        base = float(np.median(arr[c - 5:c + 5, c - 5:c + 5]))
        arr = np.clip(arr - base, -1.2, 1.2)
        np.save(out / ("patch_%d.npy" % e["id"]), arr)
        # 预览
        try:
            from PIL import Image
            t = np.clip((arr.T[::-1] + 0.2) / 1.2, 0, 1)
            img = np.stack([np.interp(t, [0, .35, .65, 1], [20, 30, 230, 200]), np.interp(t, [0, .35, .65, 1], [40, 170, 210, 30]), np.interp(t, [0, .35, .65, 1], [120, 60, 40, 30])], -1).astype(np.uint8)
            Image.fromarray(img).resize((320, 320), Image.NEAREST).save(out / ("patch_%d.png" % e["id"]))
        except Exception:
            pass
        near = float(np.median(arr[c:c + int(1.2 / res), c - 5:c + 5]))
        far = float(np.median(arr[c + int(2.5 / res):c + int(3.5 / res), c - 5:c + 5]))
        rise = far - near
        if abs(rise - e["h_med"]) > 0.15:
            print("#%d 校验失败：前方 2.5~3.5m 相对 0~1.2m 抬升 %.2f，立面 %.2f，丢弃" % (e["id"], rise, e["h_med"]))
            (out / ("patch_%d.npy" % e["id"])).unlink(missing_ok=True); continue
        info.append(dict(id=e["id"], h_med=e["h_med"], wall_xy=[round(wx, 1), round(wy, 1)], spawn_xy=[round(sx, 1), round(sy, 1)], low_dir=low_dir, rise=round(rise, 3)))
        print("#%-3d 立面 %.2f  低侧 %s  出生 (%.1f, %.1f)  校验：前方抬升 %.2f m ✓" % (e["id"], e["h_med"], low_dir, sx, sy, rise))
    json.dump(info, open(out / "patches.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("写出 %d 个补丁到 %s" % (len(info), out))


if __name__ == "__main__":
    main()
