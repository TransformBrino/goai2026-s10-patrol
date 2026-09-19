"""把真机上墙段整理成参考轨迹 ref_climb.npz：以"前轮开始压墙驱动"为相位零点，反推机身高度（FK+IMU）、离墙距离（右前轮到顶沿事件锚定 + 轮式里程积分）。
用法: python3 build_climb_ref.py parsed/climb33_2.npz [parsed/climb33_1.npz ...] --wall_h 0.34 --out ref/ref_climb34.npz"""
import argparse, math, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("npz", nargs="+"); ap.add_argument("--wall_h", type=float, default=0.34); ap.add_argument("--out", default=""); ap.add_argument("--pre", type=float, default=1.5); ap.add_argument("--post", type=float, default=1.0)
a = ap.parse_args()
L, R, HIP_X = 0.18, 0.081, 0.2277
segs = []
for path in a.npz:
    d = np.load(path, allow_pickle=True); t = d["t"] - d["t"][0]; q = d["q"]; qd = d["qd"]; tau = d["tau"]; imu = d["imu"]
    ti = imu[:, 0] - d["t"][0]; pitch_up = -np.interp(t, ti, imu[:, 2]); roll = np.interp(t, ti, imu[:, 1])
    wv = -qd[:, 3::4]                                    # 轮前向角速度
    W = []
    for leg in range(4):
        a1 = q[:, leg * 4 + 1]; a2 = a1 + q[:, leg * 4 + 2]
        x = L * (np.sin(a1) + np.sin(a2)); z = -L * (np.cos(a1) + np.cos(a2))
        hx = HIP_X if leg < 2 else -HIP_X
        xw = (hx + x) * np.cos(pitch_up) - z * np.sin(pitch_up); zw = (hx + x) * np.sin(pitch_up) + z * np.cos(pitch_up)
        W.append((xw, zw, np.hypot(x, z)))
    Wx = np.stack([w[0] for w in W], 1); Wz = np.stack([w[1] for w in W], 1); Wr = np.stack([w[2] for w in W], 1)
    zmin = Wz.min(1)
    rel = Wz - zmin[:, None]
    # 事件：press = 右前轮速首次 >5 rad/s（之后 1 s 内右前轮相对高度超过 0.3）；fr_top = 右前相对高度首次 ≥ wall_h−0.02；level = 峰值之后抬头首次 <5°
    k_top = np.argmax(rel[:, 1] >= a.wall_h - 0.02)
    k_press = k_top
    while k_press > 0 and wv[k_press, 1] > 3.0: k_press -= 1
    k_peak = k_press + np.argmax(pitch_up[k_press:k_top + 200])
    k_level = k_peak + np.argmax(np.degrees(pitch_up[k_peak:]) < 5.0)
    k_hind_up = k_peak + np.argmax((rel[k_peak:, 2] > 0.10) | (rel[k_peak:, 3] > 0.10))
    print("%s: press t=%.2f  fr_top t=%.2f  peak pitch %.0f° t=%.2f  hind_up t=%.2f  level t=%.2f  总时长(press→level) %.2f s" % (
        path, t[k_press], t[k_top], math.degrees(pitch_up[k_peak]), t[k_peak], t[k_hind_up], t[k_level], t[k_level] - t[k_press]))
    # 机身高度：地面参考在后轮抬起前 = 0，抬起后 = wall_h（中间按前轮已在台上处理：用前轮作参考）
    z_body = np.where(np.arange(len(t)) < k_hind_up, -zmin + R, a.wall_h - Wz[:, :2].min(1) + R)
    # 离墙距离：锚点 fr_top 时右前轮心 x_rel = −R+0.02；机身速度 = 着地轮轮速×R（后轮抬起前用后轮均值，之后用前轮均值）
    v_body = np.where(np.arange(len(t)) < k_hind_up, wv[:, 2:].mean(1), wv[:, :2].mean(1)) * R
    dt = np.gradient(t); x_body = np.cumsum(v_body * dt)
    x_body = x_body - x_body[k_top] + (-R + 0.02 - Wx[k_top, 1])
    k0 = np.searchsorted(t, t[k_press] - a.pre); k1 = np.searchsorted(t, t[k_level] + a.post)
    sl = slice(k0, k1)
    segs.append(dict(name=path, t=t[sl] - t[k_press], q=q[sl], qd=qd[sl], tau=tau[sl], pitch=pitch_up[sl], roll=roll[sl], z=z_body[sl], x=x_body[sl],
                     wheel_x=Wx[sl], wheel_z=Wz[sl], leg_r=Wr[sl], wv=wv[sl], events=dict(press=0.0, fr_top=t[k_top] - t[k_press], peak=t[k_peak] - t[k_press], hind_up=t[k_hind_up] - t[k_press], level=t[k_level] - t[k_press])))
    s = segs[-1]
    for tt in (-1.0, -0.5, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
        k = np.searchsorted(s["t"], tt)
        if k >= len(s["t"]): break
        print("   t=%+.1f 抬头 %+5.1f° 机身 z=%.2f x_rel=%+.2f 腿长 %s 轮相对高 %s 轮速 %s" % (s["t"][k], math.degrees(s["pitch"][k]), s["z"][k], s["x"][k], np.round(s["leg_r"][k], 2), np.round(rel[sl][k], 2), np.round(s["wv"][k], 0)))
if a.out:
    import os; os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, wall_h=a.wall_h, n_segs=len(segs), **{f"seg{i}_{k}": (np.array(v) if not isinstance(v, dict) else np.array([v[e] for e in ("press", "fr_top", "peak", "hind_up", "level")])) for i, s in enumerate(segs) for k, v in s.items() if k != "name"}, names=np.array([s["name"] for s in segs]))
    print("已存", a.out)
