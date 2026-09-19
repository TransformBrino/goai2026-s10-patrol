#!/usr/bin/env python3
"""动作自激门（09-19）。T10_15197 真机 1 s 内动作 0.9 → 1e14 的复现与验收门。

机理：观测里含「上一动作」16 维；训练侧对它 clip ±100、runner 不钳位 → 正反馈环在真机上没有闸。
做法：其余观测取标称、只改 roll，把上一帧动作喂回观测迭代 N 步取不动点。
打限位线 |a| = 2.7227 / 0.25 = 10.9（腿缩放 0.25 rad）。

用法: selfexcite_gate.py <onnx> [--pass 7.0]   退出码 0=过 1=不过
09-18 基线（roll 0/20/30/45）：R13 3.3/2.0/1.6/2.0  S16 2.5/2.1/3.1/5.8  V1H 2.4/3.4/4.3/6.3
                              T10 1.8/11.4/13.6/16.2 ✗   246-lr8 4.2/12.5/15.4/18.9 ✗
"""
import sys, numpy as np, onnxruntime as ort

path = sys.argv[1]
thr = float(sys.argv[sys.argv.index("--pass") + 1]) if "--pass" in sys.argv else 7.0   # V1H（真机 09-12/13 走过）roll45 为 6.31，以它为健康参照
s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
nm, on = s.get_inputs()[0].name, s.get_outputs()[0].name
n = s.get_inputs()[0].shape[-1]

def obs(roll_deg, last, ph):
    o = np.zeros((1, n), dtype=np.float32); r = np.radians(roll_deg)
    o[0, 3:6] = [np.sin(r), 0.0, -np.cos(r)]; o[0, 6:9] = [0.3, 0.0, 0.0]; o[0, 41:57] = last
    if n == 246: o[0, 244], o[0, 245] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)
    return o

# 09-19 补：同时给「单帧」(上一动作=0) 与「增益」= 不动点/单帧。
# 爬墙专家本来动作就大（ClimbH_8797 roll0 单帧就高），绝对值门会误伤；真正的危险信号是**增益**——
# T10 roll20 单帧 2.10 → 不动点 11.39（×5.4），环在放大；健康策略增益 ≈1~2。
fp, sf = [], []
for rd in (0, 20, 30, 45):
    worst, single = 0.0, 0.0
    for ph in (0.0, 0.25, 0.5, 0.75):
        a = np.zeros(16, dtype=np.float32)
        a1 = s.run([on], {nm: obs(rd, a, ph)})[0][0]; single = max(single, float(np.abs(a1).max()))
        a = a1
        for _ in range(14):
            a = s.run([on], {nm: obs(rd, a, ph)})[0][0]
        worst = max(worst, float(np.abs(a).max()))
    fp.append(worst); sf.append(single)
gain = [f / max(s_, 1e-6) for f, s_ in zip(fp, sf)]
ok_abs = max(fp) < thr
ok_gain = max(gain) < 2.5 and max(fp) < 10.9        # 增益门：任一档放大 ≥2.5× 或越限位线 → 不过
ok = ok_abs or ok_gain                              # 两个口径任一通过即放行（绝对值门保主策略，增益门保动作本来就大的专家）
print(f"  {path.split('/')[-1]:28s} ({n}维) 不动点 {fp[0]:5.2f}/{fp[1]:5.2f}/{fp[2]:5.2f}/{fp[3]:5.2f}"
      f"  单帧 {sf[0]:4.2f}/{sf[1]:4.2f}/{sf[2]:4.2f}/{sf[3]:4.2f}  增益 {gain[0]:3.1f}/{gain[1]:3.1f}/{gain[2]:3.1f}/{gain[3]:3.1f}"
      f"  绝对门<{thr:g} {'✓' if ok_abs else '✗'}  增益门<2.5&<10.9 {'✓' if ok_gain else '✗'}  → {'✓ 过' if ok else '✗ 不过'}"
      f"{'（越限位线）' if max(fp) > 10.9 else ''}")
sys.exit(0 if ok else 1)
