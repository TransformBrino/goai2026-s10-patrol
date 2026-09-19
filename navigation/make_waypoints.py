#!/usr/bin/env python3
"""生成航点文件（`序号 x y 限速 [align] [stop]`）。

  python3 make_waypoints.py loop out.txt --side 8 --v 0.8        # 平地方形回路：直线 + 三个 90° 弯 + 回起点
  python3 make_waypoints.py line out.txt --len 20 --v 1.0        # 20m 直线（联调第 2 步）
  python3 make_waypoints.py track track_overlay.xml out.txt --v 1.0   # 初赛赛道 XML → 航点（高差>15cm 的段标 align）
点云图上点选的工具（赛日用）另做；本文件只负责仿真联调用的合成路线与赛道转换。
"""
import argparse
import re
from pathlib import Path


def write(path, pts, v, aligns=()):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 序号 x y 限速 [align] [stop]\n")
        for i, (x, y) in enumerate(pts):
            f.write("%d %.3f %.3f %.2f%s\n" % (i, x, y, v, " align" if i in aligns else ""))
    print("写入 %s：%d 个点" % (path, len(pts)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("loop"); p.add_argument("out"); p.add_argument("--side", type=float, default=8.0); p.add_argument("--v", type=float, default=0.8)
    p = sub.add_parser("line"); p.add_argument("out"); p.add_argument("--len", type=float, default=20.0); p.add_argument("--v", type=float, default=1.0)
    p = sub.add_parser("track"); p.add_argument("xml"); p.add_argument("out"); p.add_argument("--v", type=float, default=1.0)
    a = ap.parse_args()
    if a.kind == "loop":
        s = a.side
        write(a.out, [(0, 0), (s, 0), (s, s), (0, s), (0, 0)], a.v)
    elif a.kind == "line":
        write(a.out, [(0, 0), (a.len, 0)], a.v)
    else:
        src = Path(a.xml).read_text(encoding="utf-8")
        hits = re.findall(r'name="track_waypoint_(\d+)_[a-z]+"[^>]*pos="([-0-9. ]+)"', src)
        pts3 = [p for _, p in sorted(((int(i), [float(v) for v in s.split()]) for i, s in hits))]
        aligns = {i for i in range(1, len(pts3)) if abs(pts3[i][2] - pts3[i - 1][2]) > 0.15}
        write(a.out, [(p[0], p[1]) for p in pts3], a.v, aligns)


if __name__ == "__main__":
    main()
