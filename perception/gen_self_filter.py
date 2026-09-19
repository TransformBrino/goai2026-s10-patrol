#!/usr/bin/env python3
"""从 S10.xml 生成 self_filter.py（几何常数写死在文件里，运行时只要 numpy，不要 MuJoCo）。
用法：python3 gen_self_filter.py   → 覆盖同目录 self_filter.py"""
import pathlib, numpy as np, mujoco

HERE = pathlib.Path(__file__).resolve().parent
XML = HERE.parent.parent / "s10_ws/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml"
m = mujoco.MjModel.from_xml_path(str(XML))
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
LEGS, PARTS = ("fl", "fr", "hl", "hr"), ("hipx", "hipy", "knee", "wheel")

def qmat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)

E = np.eye(3)
def perm_z(ax):  # 局部 z 指向第 ax 轴、右手系
    return np.column_stack([E[(ax + 1) % 3], E[(ax + 2) % 3], E[ax]])

POS, AX = {}, {}
for leg in LEGS:
    POS[leg] = {}
    for p in PARTS:
        b = bid(f"{leg}_{p}")
        assert np.allclose(m.body_quat[b], [1, 0, 0, 0]), f"{leg}_{p} 姿态不是单位阵，本生成器不支持"
        POS[leg][p] = [float(v) for v in m.body_pos[b]]
        if p != "wheel":
            AX[(leg, p)] = [float(v) for v in m.jnt_axis[m.body_jntadr[b]]]
ax_hipx = {AX[(l, "hipx")] and tuple(AX[(l, "hipx")]) for l in LEGS}; ax_pitch = {tuple(AX[(l, p)]) for l in LEGS for p in ("hipy", "knee")}
assert len(ax_hipx) == 1 and len(ax_pitch) == 1, "四条腿关节轴不一致"

def geoms(body_name):
    b = bid(body_name); gm, gc = [], []
    wheel = body_name.endswith("_wheel")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b: continue
        Rg, pg, t = qmat(m.geom_quat[g]), m.geom_pos[g].astype(float), int(m.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; v = m.mesh_vert[a:a + m.mesh_vertnum[mid]]
            lo, hi = v.min(0), v.max(0); c = pg + Rg @ ((lo + hi) / 2); half = (hi - lo) / 2
            if wheel:
                ax = int(np.argmin(half)); r = float(max(half[(ax + 1) % 3], half[(ax + 2) % 3]))
                gm.append(("cyl", c, Rg @ perm_z(ax), [r, float(half[ax])]))
            else:
                gm.append(("box", c, Rg, [float(x) for x in half]))
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            gc.append(("box", pg, Rg, [float(x) for x in m.geom_size[g][:3]]))
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            gc.append(("cyl", pg, Rg, [float(m.geom_size[g][0]), float(m.geom_size[g][1])]))
    return gm, gc

GM, GC = {}, {}
for ln in ["base_link"] + [f"{l}_{p}" for l in LEGS for p in PARTS]:
    k = "base" if ln == "base_link" else ln
    GM[k], GC[k] = geoms(ln)

def lit(G):
    rows = []
    for k, prims in G.items():
        items = ", ".join("(%r, %s, %s, %s)" % (kind, np.round(c, 6).tolist(), np.round(R, 6).tolist(), np.round(sz, 6).tolist())
                          for kind, c, R, sz in prims)
        rows.append("    %r: [%s]," % (k, items))
    return "{\n" + "\n".join(rows) + "\n}"

def fk_ref(raw):
    d = mujoco.MjData(m); d.qpos[:] = 0; d.qpos[3] = 1.0
    for k, leg in enumerate(LEGS):
        for j, p in enumerate(PARTS):
            jid = m.body_jntadr[bid(f"{leg}_{p}")]; d.qpos[m.jnt_qposadr[jid]] = raw[4 * k + j]
    mujoco.mj_forward(m, d)
    b0 = bid("base_link"); R0 = d.xmat[b0].reshape(3, 3)
    return {f"{leg}_{p}": np.round(R0.T @ (d.xpos[bid(f"{leg}_{p}")] - d.xpos[b0]), 9).tolist() for leg in LEGS for p in PARTS}
def stance(hy, kn, hx=0.05):
    return [hx, -hy, kn, 0, -hx, -hy, kn, 0, -hx, hy, -kn, 0, hx, hy, -kn, 0]
rng = np.random.default_rng(7)
rnd = (np.array(stance(0.35, 0.65)) + rng.uniform(-0.5, 0.5, 16) * np.tile([0.5, 1, 1, 0], 4)).tolist()
REF = [(r, fk_ref(r)) for r in (stance(0.35, 0.65), stance(0.72, 1.44), rnd)]
tmpl = (HERE / "self_filter.tmpl.py").read_text()
out = tmpl.replace("@@POS@@", repr(POS)).replace("@@AX_HIPX@@", repr(list(ax_hipx.pop()))).replace("@@AX_PITCH@@", repr(list(ax_pitch.pop()))) \
          .replace("@@GEOM_MESH@@", lit(GM)).replace("@@GEOM_COLL@@", lit(GC)).replace("@@XML@@", str(XML.name)).replace("@@REF@@", repr(REF))
(HERE / "self_filter.py").write_text(out)
print("已生成 self_filter.py：%d 个 link；网格包围体 %d 个、碰撞体 %d 个" % (len(GM), sum(map(len, GM.values())), sum(map(len, GC.values()))))
