#!/usr/bin/env python3
"""楼梯跑测结局分类（替代 eval_driver 的"通过 N/M 级"）。
场景 gen_steps_scene.py 生成的楼梯默认在最后一级外沿之后是地面（断崖），eval_driver 要"越过最后一级外沿"才记第 N 级，
所以"N/N"其实是开下了断崖、"(N−1)/N 且最后一级停留很久"其实是登顶后停在顶上。本脚本按轨迹判：
  登顶并走上平台 / 登顶并停在顶上 / 登顶后开下断崖 / 登顶后又退下来 / 没登顶（最高到第 k 级）/ 翻车。
--landing L：场景带 L m 顶平台（S10_LANDING_M=L 生成），此时走上平台才算完整上完。
用法: stair_outcome.py --n 16 --h 0.12 --w 0.65 --x0 3.0 [--landing 40] a.csv b.csv ..."""
import argparse, csv, os
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0); ap.add_argument("--h", type=float, default=0.0)
ap.add_argument("--w", type=float, required=True); ap.add_argument("--x0", type=float, default=3.0); ap.add_argument("--stance", type=float, default=0.42)
ap.add_argument("--landing", type=float, default=0.0)
ap.add_argument("--risers", default="", help="逐级立面（逗号分隔，0 = 段间平台踏面），给了就不用 --n/--h")
ap.add_argument("csv", nargs="+"); a = ap.parse_args()
R = [float(v) for v in a.risers.replace(" ", ",").split(",") if v.strip()] if a.risers else [a.h] * a.n
if not R or a.n < 0: ap.error("要给 --risers，或者同时给 --n 和 --h")
a.n = len(R); CUM = [sum(R[:i + 1]) for i in range(len(R))]; K = sum(1 for r in R if r > 1e-6)
def steps_at(z):                                   # 机身高 z 时已上到第几级（只数真立面，段间平台不算）
    hz = z - a.stance + 0.06; return sum(1 for i, r in enumerate(R) if r > 1e-6 and CUM[i] <= hz)
top_z, x_st = CUM[-1], a.x0 + a.n * a.w            # 顶高、最后一级外沿
x_end = x_st + a.landing                            # 顶平台尽头（无平台时 = 最后一级外沿）
up = top_z + a.stance - 0.12                        # 机身高于这个 = 站在顶级/平台上
for f in a.csv:
    r = [x for x in csv.DictReader(open(f)) if x.get("t")]
    if not r: print("%-44s 无数据" % os.path.basename(f)); continue
    X = [float(x["x"]) for x in r]; Z = [float(x["z"]) for x in r]; RO = [abs(float(x["roll"])) for x in r]
    t0 = float(r[0]["t"]); T = [float(x["t"]) - t0 for x in r]
    reached = [i for i in range(len(r)) if X[i] >= a.x0 + (a.n - 1) * a.w and Z[i] >= up]
    flipped = max(RO) > 70; tag = "、翻车" if flipped else ""
    xf, zf = X[-1], Z[-1]
    hi = max(range(len(r)), key=lambda i: Z[i]); k_hi = steps_at(Z[hi])
    if reached:
        i0 = reached[0]
        off = any(X[i] > x_end + 0.15 and Z[i] < top_z for i in range(i0, len(r)))
        plat = [i for i in range(i0, len(r)) if X[i] >= x_st + 0.35 and Z[i] >= up]   # 机身中心过外沿 0.35 m ≈ 后轮也上了平台
        stay = zf >= top_z + a.stance - 0.2 and not flipped
        if off: out = "登顶后开下%s%s（登顶 t=%.0fs，最远 x=%.2f，结束 z=%.2f）" % ("平台尽头" if a.landing > 0 else "断崖", tag, T[i0], max(X), zf)
        elif stay and a.landing > 0 and plat:
            out = "登顶并走上平台（登顶 t=%.0fs，上平台 t=%.0fs，平台上走了 %.1f m，结束 z=%.2f）" % (T[i0], T[plat[0]], max(X) - x_st, zf)
        elif stay: out = "登顶并停在顶上%s（登顶 t=%.0fs，结束 x=%.2f z=%.2f）" % ("，没走上平台" if a.landing > 0 else "", T[i0], xf, zf)
        else: out = "登顶后又退下来%s（登顶 t=%.0fs，结束 x=%.2f z=%.2f，约在第 %d 级）" % (tag, T[i0], xf, zf, steps_at(zf))
    else:
        out = ("翻车，" if flipped else "") + "没登顶：最高到第 %d/%d 级（z=%.2f，t=%.0fs），结束 x=%.2f z=%.2f" % (k_hi, K, Z[hi], T[hi], xf, zf)
    print("%-44s %s" % (os.path.basename(f).replace(".csv", ""), out))
