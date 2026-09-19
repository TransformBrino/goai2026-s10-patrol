"""ref_climb34.npz（200 Hz，robot 腿序）→ ref_climb34_50hz.npz（50 Hz，关节按仿真 articulation 顺序 hipx×4,hipy×4,knee×4,wheel×4）。
含每帧：t(相对压墙事件)、q16、qd16、pitch(抬头正)、pitch_rate、z、x_rel、vx、vz、roll、seg。"""
import numpy as np, sys
src = sys.argv[1] if len(sys.argv) > 1 else "/home/robot/s10_logs/ref/ref_climb34.npz"
dst = sys.argv[2] if len(sys.argv) > 2 else "/home/robot/s10_logs/ref/ref_climb34_50hz.npz"
d = np.load(src, allow_pickle=True); n = int(d["n_segs"])
art = [4 * j + l for l in range(4) for j in range(4)]           # robot 顺序 (4l+j) → articulation 顺序下标
inv = np.argsort(art)                                           # art_q = q_robot[inv]? 计算：art 位置 p=4j+l 应取 robot 下标 4l+j
perm = np.zeros(16, dtype=int)
for l in range(4):
    for j in range(4):
        perm[4 * j + l] = 4 * l + j
out = {k: [] for k in ("t", "q", "qd", "pitch", "pitch_rate", "z", "x", "vx", "vz", "roll", "seg")}
for i in range(n):
    t = d[f"seg{i}_t"]; q = d[f"seg{i}_q"][:, perm]; qd = d[f"seg{i}_qd"][:, perm]
    pitch = d[f"seg{i}_pitch"]; z = d[f"seg{i}_z"]; x = d[f"seg{i}_x"]; roll = d[f"seg{i}_roll"]
    # 平滑后差分得速度（200 Hz → 用 11 点滑窗）
    def sm(v, w=11): return np.convolve(v, np.ones(w) / w, mode="same")
    zs, xs, ps = sm(z), sm(x), sm(pitch)
    vz = np.gradient(zs, t); vx = np.gradient(xs, t); pr = np.gradient(ps, t)
    tt = np.arange(-0.5, 1.2001, 0.02)
    for k in tt:
        j = np.searchsorted(t, k)
        if j >= len(t): break
        out["t"].append(k); out["q"].append(q[j]); out["qd"].append(qd[j]); out["pitch"].append(pitch[j]); out["pitch_rate"].append(pr[j])
        out["z"].append(zs[j]); out["x"].append(xs[j]); out["vx"].append(vx[j]); out["vz"].append(vz[j]); out["roll"].append(roll[j]); out["seg"].append(i)
np.savez_compressed(dst, **{k: np.array(v) for k, v in out.items()}, wall_h=float(d["wall_h"]), n_segs=n)
o = {k: np.array(v) for k, v in out.items()}
print("帧数", len(o["t"]), "段数", n, "wall_h", float(d["wall_h"]))
for k in (0, 25, 35, 45, 55, 65, 75, 85):
    if k < len(o["t"]): print("t=%+.2f seg%d z=%.2f x=%+.2f vx=%+.2f vz=%+.2f pitch=%+.0f° rate=%+.1f  q(art)=%s" % (o["t"][k], o["seg"][k], o["z"][k], o["x"][k], o["vx"][k], o["vz"][k], np.degrees(o["pitch"][k]), o["pitch_rate"][k], np.round(o["q"][k][:12], 2)))
print("存", dst)
