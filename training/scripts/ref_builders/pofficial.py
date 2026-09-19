"""从一批 segment 跑测的 JSON 里算官方栈上的单次翻越概率 p。

为什么要单独算：训练环境 n=200 量到最优条件格 p=0.40，官方栈第一批只有 ~0.04。
那一批的差距后来查明主要是 eval_driver 自己的缺陷（后撤横向误差恒为 0，
"正对壁面"这个前提从未满足），所以修完之后必须重新量，而且要**按尝试次数**
而不是按跑测次数算 —— 一次跑测里有十几次尝试，用"几次跑测成功"会把 p 严重低估。

同时核算"重试到底是不是独立采样"：后撤到位率、横向抖动的实际散布、
每次恢复动作产生的位移。这三个数只要有一个塌了，p 就不是独立重复试验的 p。
"""
import glob
import json
import math
import sys

pat = sys.argv[1] if len(sys.argv) > 1 else (
    "/mnt/c/Users/86156/Desktop/机器狗项目/s10_dev/logs/s15_*.json")
files = sorted(glob.glob(pat))
if not files:
    raise SystemExit(f"没有匹配的结果文件: {pat}")

runs, att, cross, arrive = 0, 0, 0, 0
at_start_ok, at_start_n, moved_all, offs = 0, 0, [], []
reasons = {}
for f in files:
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print(f"  跳过 {f}: {type(e).__name__}")
        continue
    runs += 1
    rt = d.get("retries", [])
    # 一次跑测里的"尝试"= 首次助跑 + 每次重试各一次
    att += len(rt) + 1
    if d.get("crossed_at") is not None:
        cross += 1
    if d.get("finished"):
        arrive += 1
    reasons[d.get("reason", "finished")] = reasons.get(d.get("reason", "finished"), 0) + 1
    for r in rt:
        if "at_start" in r:
            at_start_n += 1
            at_start_ok += int(bool(r["at_start"]))
        if "moved" in r:
            moved_all.append(r["moved"])
        if "off" in r:
            offs.append(r["off"])

print(f"跑测 {runs} 次，累计尝试 {att} 次")
print(f"  至少翻越一次的跑测: {cross}/{runs}")
print(f"  到达 wp16 的跑测:   {arrive}/{runs}")
if att:
    p = cross / att
    # Wilson 区间，小样本下比正态近似靠谱
    z = 1.96
    den = 1 + z * z / att
    c = (p + z * z / (2 * att)) / den
    hw = z * math.sqrt(p * (1 - p) / att + z * z / (4 * att * att)) / den
    print(f"  单次翻越概率 p ≈ {p:.3f}  95%CI [{max(0,c-hw):.3f}, {min(1,c+hw):.3f}]")
    print(f"  （偏保守：翻越成功后剩余的尝试也计入了分母）")
print(f"  结束原因: {reasons}")

print("\n重试是不是独立采样：")
if at_start_n:
    print(f"  后撤到位率 {at_start_ok}/{at_start_n} = {at_start_ok/at_start_n:.2f}"
          f"   （低说明机器人根本没退到助跑起点，'正对壁面'的前提不成立）")
if moved_all:
    ms = sorted(moved_all)
    print(f"  恢复动作位移 中位 {ms[len(ms)//2]:.2f}m  最小 {ms[0]:.2f}  最大 {ms[-1]:.2f}"
          f"   （<0.5m 的算无效恢复）")
    print(f"  其中 <0.5m 的: {sum(1 for m in moved_all if m < 0.5)}/{len(moved_all)}")
if offs:
    print(f"  横向抖动实际散布 {min(offs):+.2f} ~ {max(offs):+.2f}"
          f"  （设定 ±0.25，铺不开就不是独立采样）")

if att and cross:
    p = cross / att
    for n in (10, 12, 15, 20):
        print(f"  按 p={p:.3f} 重试 {n} 次的完赛概率: {1-(1-p)**n:.3f}")
