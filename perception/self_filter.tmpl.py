#!/usr/bin/env python3
"""S10 雷达自遮挡滤波（训练侧 2026-09-11 写，MuJoCo 校验；由 gen_self_filter.py 从 @@XML@@ 生成，勿手改常数）。

为什么要滤：训练里高程扫描（IsaacLab RayCaster）只打地面，机器人本身不是射线目标，所以机身下方和四条腿周围
的格子读的都是地面。真机雷达会打到自己的腿和轮子，在高程图里变成"脚边 30~40 cm 的假障碍"（AGX 09-11 实测：
x=+0.2、y≈−0.2~−0.3 两格读约 38 cm，是右前腿和轮子）。这在训练分布之外。

怎么滤：按关节角做正运动学，算出机身、四条腿每一节（hipx、大腿、小腿）和轮子的位置；每一节用一个包围体
（默认取视觉网格的有向包围盒，轮子取网格外径的圆柱），外扩 margin；落在任一包围体里的点删掉。
不用固定的机身框：前轮顶到台阶时台阶立面就在 x≈+0.31，固定框会把策略最需要看的立面一起删掉。

用法（纯 numpy）：
    sf = SelfFilter()                          # margin 默认 0.03 m
    sf.set_joints_published(pos16)             # /JOINTS_DATA 或 /JOINTS_DATA_10HZ 的 16 个 position（发布值）
    keep = sf.keep_mask(p_base)                # p_base: (N,3) 机体系点（base_link：x 前、y 左、z 上）
    p_base = p_base[keep]
关节顺序：fl, fr, hl, hr × (hipx, hipy, knee, wheel)，与 /JOINTS_DATA 相同。
"""
import math
import time
import numpy as np

# /JOINTS_DATA 发布值 → 模型关节角：raw = pub·DIR + OFFSET（与 runner s10_interface.hpp、MuJoCo 仿真桥、AGX 控制台一致）
JOINT_DIR = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1], float)
JOINT_OFFSET = np.radians([-35, -145, 156, 0, 35, -145, 156, 0, -35, 145, -156, 0, 35, 145, -156, 0])
HIPY_KNEE = (1, 2, 5, 6, 9, 10, 13, 14)     # 编码器可能差整圈，runner 开机时按 ±360° 修；这里折回 (−π, π]

LEGS = ("fl", "fr", "hl", "hr")
POS = @@POS@@                                # 每节 body 相对父 body 的位置（姿态均为单位阵）
AX_HIPX = @@AX_HIPX@@
AX_PITCH = @@AX_PITCH@@
# 包围体：(类型, 中心, 旋转(列 = 局部轴在 link 系里的方向), 尺寸)；box 尺寸为半边长，cyl 为 (半径, 半长)，轴为局部 z
GEOM_MESH = @@GEOM_MESH@@
GEOM_COLL = @@GEOM_COLL@@
# 自检参考：MuJoCo 在 3 组关节角（模型角）下算出的各节原点（机体系）
REF = @@REF@@


def _rot(axis, q):
    x, y, z = axis
    c, s = math.cos(q), math.sin(q)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class SelfFilter:
    def __init__(self, margin=0.03, geometry="mesh", stale_s=0.5, stale_extra=0.03):
        self.margin = float(margin)
        self.geom = GEOM_MESH if geometry == "mesh" else GEOM_COLL
        self.stale_s, self.stale_extra = float(stale_s), float(stale_extra)
        self.q = None
        self.t_q = None
        self.n_in = self.n_self = self.n_calls = self.n_nojoint = self.n_stale = 0
        self._prims = [(link, kind, np.asarray(c, float), np.asarray(R, float), np.asarray(sz, float))
                       for link, lst in self.geom.items() for kind, c, R, sz in lst]

    # ---- 关节角 ----
    def set_joints_published(self, pos16, t=None):
        raw = np.asarray(pos16, float) * JOINT_DIR + JOINT_OFFSET
        self.set_joints_raw(raw, t)

    def set_joints_raw(self, raw16, t=None):
        q = np.array(raw16, float)
        for i in HIPY_KNEE:
            q[i] = (q[i] + math.pi) % (2.0 * math.pi) - math.pi
        self.q = q
        self.t_q = time.monotonic() if t is None else float(t)

    # ---- 正运动学：link 系 → 机体系 (R, t) ----
    def frames(self):
        F = {"base": (np.eye(3), np.zeros(3))}
        for k, leg in enumerate(LEGS):
            q0, q1, q2 = self.q[4 * k], self.q[4 * k + 1], self.q[4 * k + 2]
            P = POS[leg]
            R = _rot(AX_HIPX, q0); t = np.array(P["hipx"], float); F[leg + "_hipx"] = (R, t)
            t = t + R @ np.array(P["hipy"]); R = R @ _rot(AX_PITCH, q1); F[leg + "_hipy"] = (R, t)
            t = t + R @ np.array(P["knee"]); R = R @ _rot(AX_PITCH, q2); F[leg + "_knee"] = (R, t)
            t = t + R @ np.array(P["wheel"]); F[leg + "_wheel"] = (R, t)
        return F

    # ---- 滤波 ----
    def keep_mask(self, p_base, now=None):
        p = np.asarray(p_base)
        n = len(p)
        keep = np.ones(n, bool)
        self.n_calls += 1
        self.n_in += n
        if self.q is None:
            self.n_nojoint += 1
            return keep
        m = self.margin
        if self.t_q is not None and ((time.monotonic() if now is None else now) - self.t_q) > self.stale_s:
            m += self.stale_extra
            self.n_stale += 1
        cand = (np.abs(p[:, 0]) < 0.70) & (np.abs(p[:, 1]) < 0.50) & (p[:, 2] < 0.30) & (p[:, 2] > -0.85)
        idx = np.nonzero(cand)[0]
        if len(idx) == 0:
            return keep
        pc = p[idx].astype(np.float64)
        hit = np.zeros(len(idx), bool)
        F = self.frames()
        cache = {}
        for link, kind, c, Rp, sz in self._prims:
            if link not in cache:
                R, t = F[link]
                cache[link] = (pc - t) @ R
            qq = (cache[link] - c) @ Rp
            if kind == "box":
                inside = np.all(np.abs(qq) <= sz + m, axis=1)
            else:
                inside = (qq[:, 0] ** 2 + qq[:, 1] ** 2 <= (sz[0] + m) ** 2) & (np.abs(qq[:, 2]) <= sz[1] + m)
            hit |= inside
        keep[idx[hit]] = False
        self.n_self += int(hit.sum())
        return keep

    def stats(self):
        return dict(calls=self.n_calls, pts=self.n_in, removed=self.n_self, no_joint=self.n_nojoint, stale=self.n_stale)


def _selftest():
    sf = SelfFilter()
    err = 0.0
    for raw, ref in REF:
        sf.set_joints_raw(raw); F = sf.frames()
        for link, pos in ref.items():
            err = max(err, float(np.abs(F[link][1] - np.asarray(pos)).max()))
    print("① 正运动学对 MuJoCo 参考值：最大误差 %.2e m  %s" % (err, "通过" if err < 1e-6 else "不通过"))
    raw0 = np.asarray(REF[0][0], float)
    sf.set_joints_published((raw0 - JOINT_OFFSET) * JOINT_DIR); F = sf.frames()
    err2 = max(float(np.abs(F[l][1] - np.asarray(p)).max()) for l, p in REF[0][1].items())
    print("② 发布值换算（raw = pub·DIR + OFFSET）：最大误差 %.2e m  %s" % (err2, "通过" if err2 < 1e-6 else "不通过"))
    wc = F["fr_wheel"][1]
    test = np.array([wc + [0.0, 0.0, 0.05], wc + [0.0, 0.0, -0.081], [wc[0] + 0.13, wc[1], -0.43 + 0.33], [1.0, 0.0, -0.43]])
    keep = sf.keep_mask(test)
    ok = list(keep) == [False, False, True, True]
    print("③ 站姿下：右前轮上方点删、轮底接地点删、轮前 13 cm 处 33 cm 台阶顶保留、1 m 外地面保留 → %s  %s" % (list(keep), "通过" if ok else "不通过"))
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.uniform([-3, -3, -0.6], [3, 3, 1.5], (27000, 3)), rng.uniform([-0.4, -0.35, -0.45], [0.4, 0.35, 0.05], (3000, 3))]).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(20):
        sf.keep_mask(pts)
    print("④ 耗时 %.1f ms / 3 万点（本机）" % ((time.perf_counter() - t0) / 20 * 1e3))
    return err < 1e-6 and err2 < 1e-6 and ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
