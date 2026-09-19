#!/usr/bin/env python3
"""MuJoCo 里校验 self_filter.py（训练侧用，AGX 不需要）：
① 正运动学：随机关节角下，四条腿各节原点位置对 MuJoCo 真值；
② 射线实测：两台 Airy 按 09-11 修正后的装法（前雷达在 x=+0.22341 圆顶朝 +x，后雷达在 −0.22341 朝 −x；半球 1°×1°；
   盲区 0.1 m），用 mj_multiRay 打真网格（机器人视觉网格 + 碰撞体 + 场景），把命中点标成"打到自己 / 打到世界"，
   关节角先转成 /JOINTS_DATA 发布值再喂滤波（顺带校验换算）。统计：自身点删掉比例、世界点误删比例、
   以及按训练网格（17×11、0.1 m、随航向）算出的逐格高度：原始（含自身点）vs 滤后 vs 真值（只有世界点）。"""
import sys, csv, math, time, pathlib, numpy as np, mujoco
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import self_filter as SF
MJ = HERE.parent.parent / "s10_ws/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf"

def quat_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])

def robot_ids(m):
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    isr = np.zeros(m.nbody, bool)
    for b in range(m.nbody):
        r = b
        while r not in (0, root): r = m.body_parentid[r]
        isr[b] = (r == root)
    return root, isr

def fk_check(n=500, seed=0):
    m = mujoco.MjModel.from_xml_path(str(MJ / "S10.xml")); d = mujoco.MjData(m); rng = np.random.default_rng(seed)
    sf = SF.SelfFilter(); err = 0.0
    jadr = {f"{l}_{p}": m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{l}_{p}')]] for l in SF.LEGS for p in ("hipx", "hipy", "knee", "wheel")}
    for _ in range(n):
        raw = np.zeros(16)
        for k, l in enumerate(SF.LEGS):
            for j, p in enumerate(("hipx", "hipy", "knee")):
                jid = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{l}_{p}")]
                raw[4 * k + j] = rng.uniform(*m.jnt_range[jid])
        d.qpos[:] = 0; d.qpos[3] = 1.0
        for k, l in enumerate(SF.LEGS):
            for j, p in enumerate(("hipx", "hipy", "knee")): d.qpos[jadr[f"{l}_{p}"]] = raw[4 * k + j]
        mujoco.mj_forward(m, d)
        pub = (raw - SF.JOINT_OFFSET) * SF.JOINT_DIR        # 模型角 → 发布值（仿真桥的换算），再让滤波自己换回来
        sf.set_joints_published(pub); F = sf.frames()
        base = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")]; Rb = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")].reshape(3, 3)
        for link, (R, t) in F.items():
            if link == "base": continue
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
            tb = Rb.T @ (d.xpos[b] - base); Rl = Rb.T @ d.xmat[b].reshape(3, 3)
            err = max(err, float(np.abs(tb - t).max()))
            if not link.endswith("_wheel"): err = max(err, float(np.abs(Rl - R).max()))
    return err

def hemisphere(axis):
    th = np.radians(np.arange(0, 90.5, 1.0)); ph = np.radians(np.arange(0, 360, 1.0))
    T, P = np.meshgrid(th, ph, indexing="ij")
    loc = np.stack([np.cos(T), np.sin(T) * np.cos(P), np.sin(T) * np.sin(P)], -1).reshape(-1, 3)   # 局部 x = 圆顶轴
    if axis < 0: loc = loc * np.array([-1, -1, 1])                                                      # 朝 −x：绕 z 转 180°
    return np.unique(np.round(loc, 9), axis=0)

LIDARS = [(np.array([0.22341, 0.0, -0.0001]), hemisphere(+1)), (np.array([-0.22341, 0.0, -0.0001]), hemisphere(-1))]

def scan(m, d, root, isr):
    pb, Rb = d.xpos[root].copy(), d.xmat[root].reshape(3, 3).copy()
    P, L = [], []
    for o, dirs_b in LIDARS:
        pnt = pb + Rb @ o; dw = dirs_b @ Rb.T; n = len(dw)
        gid = np.zeros(n, np.int32); dist = np.zeros(n)
        mujoco.mj_multiRay(m, d, pnt, dw.flatten(), None, 1, root, gid, dist, None, n, 25.0)
        ok = (gid >= 0) & (dist >= 0.10)
        P.append(pnt + dw[ok] * dist[ok, None]); L.append(isr[m.geom_bodyid[gid[ok]]])
    return np.concatenate(P), np.concatenate(L), pb, Rb

def hscan(pw, pb, Rb):
    yaw = math.atan2(Rb[1, 0], Rb[0, 0]); c, s = math.cos(yaw), math.sin(yaw)
    rel = pw[:, :2] - pb[:2]; xl = c * rel[:, 0] + s * rel[:, 1]; yl = -s * rel[:, 0] + c * rel[:, 1]
    ix = np.round((xl + 0.8) / 0.1).astype(int); iy = np.round((yl + 0.5) / 0.1).astype(int)
    ok = (ix >= 0) & (ix <= 16) & (iy >= 0) & (iy <= 10)
    g = np.full((11, 17), -np.inf); np.maximum.at(g, (iy[ok], ix[ok]), pw[ok, 2])
    return g

def evaluate(name, m, d, root, isr, sf, raw):
    mujoco.mj_forward(m, d)
    pw, lab, pb, Rb = scan(m, d, root, isr)
    p_base = (pw - pb) @ Rb
    sf.set_joints_published((raw - SF.JOINT_OFFSET) * SF.JOINT_DIR)
    t0 = time.perf_counter(); keep = sf.keep_mask(p_base); dt = (time.perf_counter() - t0) * 1e3
    near = np.linalg.norm(p_base[:, :2], axis=1) < 0.6
    g_true, g_raw, g_flt = hscan(pw[~lab], pb, Rb), hscan(pw, pb, Rb), hscan(pw[keep], pb, Rb)
    have = np.isfinite(g_true)
    fake_raw = int(((g_raw - g_true) > 0.05)[have].sum()); fake_flt = int(((g_flt - g_true) > 0.05)[have].sum())
    lost = int((have & ~np.isfinite(g_flt)).sum()); low = int(((g_true - g_flt) > 0.05)[have & np.isfinite(g_flt)].sum())
    extra = int((~have & np.isfinite(g_flt)).sum())
    ns, nw = int(lab.sum()), int((~lab).sum())
    return dict(name=name, n_self=ns, self_rm=100.0 * (~keep[lab]).sum() / max(ns, 1), n_world=nw, world_rm=100.0 * (~keep[~lab]).sum() / max(nw, 1),
                world_rm_near=100.0 * (~keep[~lab & near]).sum() / max((~lab & near).sum(), 1), fake_raw=fake_raw, fake_flt=fake_flt,
                lost=lost, low=low, extra=extra, ms=dt, npts=len(pw))

def set_robot(m, d, root_q, raw, base_xyz=None, base_quat=None):
    fj = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")]; a = m.jnt_qposadr[fj]
    if base_xyz is not None: d.qpos[a:a + 3] = base_xyz
    if base_quat is not None: d.qpos[a + 3:a + 7] = base_quat
    for k, l in enumerate(SF.LEGS):
        for j, p in enumerate(("hipx", "hipy", "knee", "wheel")):
            jid = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{l}_{p}")]
            d.qpos[m.jnt_qposadr[jid]] = raw[4 * k + j]

def stance(hy, kn, hx=0.05):
    return np.array([hx, -hy, kn, 0, -hx, -hy, kn, 0, -hx, hy, -kn, 0, hx, hy, -kn, 0], float)

def main(margins=(0.02, 0.03, 0.04), geoms=("mesh", "coll")):
    print("① 正运动学对 MuJoCo 真值（500 组随机关节角，经发布值换算）：最大误差 %.2e（位置 m / 旋转元素）" % fk_check())
    mf = mujoco.MjModel.from_xml_path(str(MJ / "bench_flat.xml")); df = mujoco.MjData(mf); rootf, isrf = robot_ids(mf)
    mc = mujoco.MjModel.from_xml_path(str(MJ / "bench_top33.xml")); dc = mujoco.MjData(mc); rootc, isrc = robot_ids(mc)
    poses = []
    rng = np.random.default_rng(1)
    for nm, raw in [("站姿0.42", stance(0.35, 0.65)), ("蹲姿0.35", stance(0.72, 1.44))] + \
                   [("随机腿%d" % i, stance(0.35, 0.65) + np.concatenate([np.r_[rng.uniform(-0.3, 0.3), rng.uniform(-0.6, 0.6), rng.uniform(-0.8, 0.8), 0] for _ in range(4)])) for i in range(4)]:
        set_robot(mf, df, rootf, raw, [0, 0, 1.0], [1, 0, 0, 0]); mujoco.mj_forward(mf, df)
        wz = min(df.xpos[mujoco.mj_name2id(mf, mujoco.mjtObj.mjOBJ_BODY, f"{l}_wheel")][2] for l in SF.LEGS)
        set_robot(mf, df, rootf, raw, [0, 0, 1.0 - (wz - 0.081)], [1, 0, 0, 0])
        poses.append(("flat", nm, raw, df.qpos.copy()))
    rows = [r for r in csv.DictReader(open(pathlib.Path.home() / "s10_logs/bench/tq_climb33.csv")) if r.get("t")]
    t0 = float(rows[0]["t"]); pick = [r for r in rows if float(r["x"]) >= 1.8][::60][:14]
    for r in pick:
        raw = np.array([float(r["q%d" % i]) for i in range(16)])
        q = quat_rpy(math.radians(float(r["roll"])), math.radians(float(r["pitch"])), math.radians(float(r["yaw"])))
        set_robot(mc, dc, rootc, raw, [float(r["x"]), float(r["y"]), float(r["z"])], q)
        poses.append(("top33", "上墙 t=%.1f x=%.2f" % (float(r["t"]) - t0, float(r["x"])), raw, dc.qpos.copy()))
    for g in geoms:
        for mg in margins:
            sf = SF.SelfFilter(margin=mg, geometry=g); res = []
            for scene, nm, raw, qpos in poses:
                m, d, root, isr = (mf, df, rootf, isrf) if scene == "flat" else (mc, dc, rootc, isrc)
                d.qpos[:] = qpos; res.append(evaluate(nm, m, d, root, isr, sf, raw))
            S = lambda k: np.mean([x[k] for x in res])
            print("\n=== 包围体=%s  外扩 %.2f m：自身点删掉 %.2f%%（最差 %.2f%%）  世界点误删 %.2f%%（0.6 m 内 %.2f%%）  假障碍格 原始 %d → 滤后 %d  丢格 %d  读低 %d  用时 %.1f ms/帧（%d 点）" % (
                g, mg, S("self_rm"), min(x["self_rm"] for x in res), S("world_rm"), S("world_rm_near"),
                sum(x["fake_raw"] for x in res), sum(x["fake_flt"] for x in res), sum(x["lost"] for x in res), sum(x["low"] for x in res), S("ms"), S("npts")))
            if g == "mesh" and mg == 0.03:
                for x in res:
                    print("   %-18s 自身 %5d 删 %6.2f%% | 世界 %6d 误删 %5.2f%%（近 %5.2f%%）| 假障碍格 %2d→%d 丢格 %d 读低 %d" % (
                        x["name"], x["n_self"], x["self_rm"], x["n_world"], x["world_rm"], x["world_rm_near"], x["fake_raw"], x["fake_flt"], x["lost"], x["low"]))

if __name__ == "__main__":
    main()
