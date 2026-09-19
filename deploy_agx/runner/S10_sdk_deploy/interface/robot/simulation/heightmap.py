"""点云 → 高程图：训练侧和部署侧之间那座桥。

策略观测的第 58~244 维是一张 17×11 的地面高度表。训练时这 187 个数由 IsaacLab 的
RayCaster 直接对地形网格打射线得到；部署时没有地形网格，只有雷达的几千个散点，
必须自己算出**格式完全一致**的 187 个数，否则策略的输入就是乱码。

规格逐条对齐 IsaacLab（版本：/root/dl/src/IsaacLab）：

  栅格      GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
            x = arange(-0.8, +0.8, 0.1) → 17 个（机体前后，+x 朝前）
            y = arange(-0.5, +0.5, 0.1) → 11 个（机体左右，+y 朝左）
            patterns.py:45-47，ordering="xy" → meshgrid 形状 (11, 17)，
            展平是 **y 为主序**：第 k 个 = (x[k % 17], y[k // 17])

  朝向      ray_alignment="yaw" —— 栅格只跟随机身偏航，不随 roll/pitch 倾斜
            ray_caster.py:278

  取值      height = base_z - hit_z - 0.5        observations.py:300
            那个 offset=(0,0,20) 只加在射线**起点**上（ray_caster.py:224），
            不进 pos_w，所以不会让结果饱和。

  裁剪      clip(-1.0, 1.0)                      velocity_env_cfg.py:190
  噪声      训练侧加 Unoise(-0.1, 0.1)；部署侧不加（真实误差由雷达自己带）

打不到的格子（雷达被遮挡、台阶背面、机身自遮挡）没有真值，必须填补 —— 填法直接
决定策略会不会被骗，见 fill_holes 的说明。
"""
from __future__ import annotations

import numpy as np

# ---- 与 IsaacLab 对齐的常量，别随手改 ----
RESOLUTION = 0.1
SIZE_X, SIZE_Y = 1.6, 1.0
import os as _os
# 09-08 迁移断裂排查：训练侧（M20 资产 root z 0.58）平地高程图 = +0.08，部署侧（S10 base 0.44）= -0.06，整张图差 0.14。
# 用 S10_HMAP_OFFSET 临时改基准做 A/B（0.36 = 让 S10 平地也读 +0.08）；默认 0.5 不变。
OFFSET = float(_os.environ.get("S10_HMAP_OFFSET", "0.5"))
CLIP = 1.0

# 只看机器狗**能踩到的高度带**：机身以上 CEIL_MARGIN 之内。
# 实测逼出来的，不是拍脑袋（t=350s 楼梯段）：赛道是多层结构，头顶约 6m 有顶棚。
# IsaacLab 的射线从 base_z+20m 往下打、取第一个命中，在程序化地形上没问题
# （没有顶棚），在赛道上会把天花板当地面 —— 报"脚下地面 6.17m"而机器狗在 3.55m。
#
# 阈值定 0.15 而不是 0.5，是按**训练时策略实际见过的取值范围**倒推的：
#   obs = base_z - hit_z - 0.5，训练地形 hit_z ∈ [地面, 地面+0.455]，
#   机身离地约 0.40m  →  obs 落在约 [-0.56, -0.10]，过坑时为正。
#   最高的可攀爬台阶 0.455m，顶面在 地面+0.455 = base_z+0.055，
#   所以 base_z+0.15 已经覆盖一切能踩的东西，再高的只可能是天花板/护栏。
# 之前取 0.5 的后果：楼梯段 11 格顶在 base_z+0.5 上，造出 +0.933m 的假障碍，
# 且 obs 饱和到 -1 —— 而训练中 -1 **从未出现**，是彻底的分布外输入。
CEIL_MARGIN = 0.15

GRID_X = np.arange(-SIZE_X / 2, SIZE_X / 2 + 1e-9, RESOLUTION)   # 17
GRID_Y = np.arange(-SIZE_Y / 2, SIZE_Y / 2 + 1e-9, RESOLUTION)   # 11
NX, NY = len(GRID_X), len(GRID_Y)
NUM_RAYS = NX * NY                                                # 187

# 展平顺序：k = iy * NX + ix   （meshgrid indexing="xy" 后 row-major）
_MX, _MY = np.meshgrid(GRID_X, GRID_Y, indexing="xy")             # 均为 (11, 17)
CELL_XY = np.stack([_MX.ravel(), _MY.ravel()], axis=1)            # (187, 2) 机体系


def _yaw_rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def cell_centers_world(base_xy: np.ndarray, yaw: float) -> np.ndarray:
    """187 个格心在世界系的 xy。只按 yaw 旋转，与 ray_alignment='yaw' 一致。"""
    return base_xy[None, :] + CELL_XY @ _yaw_rot(yaw).T


def encode(base_z: float, hit_z: np.ndarray) -> np.ndarray:
    """把地面绝对高度换成策略吃的那个数。hit_z 里的 NaN 原样传出，留给填补。"""
    return np.clip(base_z - hit_z - OFFSET, -CLIP, CLIP)


# ---------------------------------------------------------------- 真值（训练侧等价）
class TerrainProbe:
    """直接对 MuJoCo 地形网格打射线 —— 等价于训练时 IsaacLab 的做法。

    只用于**验证和标定**：部署时没有地形网格，不能用它。
    """

    def __init__(self, model, robot_root_body: str = "base_link"):
        import mujoco

        self._mj = mujoco
        self.model = model
        # 把机器狗自身的 geom 挪到组 5，射线只放行组 0（mj_ray 的 geomgroup 是
        # 6 位分组掩码，不是逐 geom 数组 —— 这里踩过一次）
        robot = set()
        stack = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_body)]
        while stack:
            b = stack.pop()
            if b < 0:
                continue
            robot.add(b)
            for i in range(model.nbody):
                if model.body_parentid[i] == b and i != b:
                    stack.append(i)
        for g in range(model.ngeom):
            model.geom_group[g] = 5 if model.geom_bodyid[g] in robot else 0
        self.mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    def hit_z(self, data, base_xy: np.ndarray, base_z: float, yaw: float,
              top: float = CEIL_MARGIN) -> np.ndarray:
        """187 个格子的地面绝对高度。打空返回 NaN。

        射线起点是 base_z + top，**不是** IsaacLab 那个 +20m。
        实测教训（t=350s 楼梯段）：赛道有多层结构，头顶约 6m 处有顶棚，
        从 +20m 往下打会先命中天花板 —— 报出"脚下地面在 6.17m"而机器狗
        自己在 3.55m。IsaacLab 用 +20 没事是因为程序化地形没有顶棚。
        """
        pts = cell_centers_world(base_xy, yaw)
        out = np.full(NUM_RAYS, np.nan)
        vec = np.array([0.0, 0.0, -1.0])
        gid = np.zeros(1, dtype=np.int32)
        z0 = base_z + top
        for k in range(NUM_RAYS):
            pnt = np.array([pts[k, 0], pts[k, 1], z0])
            d = self._mj.mj_ray(self.model, data, pnt, vec, self.mask, 1, -1, gid)
            if gid[0] >= 0:
                out[k] = z0 - d
        return out


# ---------------------------------------------------------------- 部署侧：从点云算
def hit_z_from_cloud(points_world: np.ndarray, base_xy: np.ndarray, yaw: float,
                     z_lo: float = -3.0, z_hi: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """把世界系点云落进 187 个格子，每格取**最高**点当地面。

    取最高而不是最低/平均：雷达偶尔会穿过缝隙打到下层，取最低会把台阶读成坑，
    机器狗一脚踩空。取最高在有噪声时偏保守（把地面读高一点 → 策略以为障碍更近），
    这个方向的错误比读成坑安全。

    返回 (hit_z, n_points)，没有点的格子 hit_z = NaN。
    """
    if points_world.size == 0:
        return np.full(NUM_RAYS, np.nan), np.zeros(NUM_RAYS, dtype=np.int32)

    # 转到"机体 yaw 对齐"的局部系，与栅格同一个坐标系
    rel = points_world[:, :2] - base_xy[None, :]
    loc = rel @ _yaw_rot(yaw)                    # 逆旋转 = 乘转置的转置
    ix = np.round((loc[:, 0] - GRID_X[0]) / RESOLUTION).astype(np.int64)
    iy = np.round((loc[:, 1] - GRID_Y[0]) / RESOLUTION).astype(np.int64)
    z = points_world[:, 2]
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (z > z_lo) & (z < z_hi)
    ix, iy, z = ix[ok], iy[ok], z[ok]

    k = iy * NX + ix
    hit = np.full(NUM_RAYS, -np.inf)
    np.maximum.at(hit, k, z)
    n = np.bincount(k, minlength=NUM_RAYS).astype(np.int32)
    hit[n == 0] = np.nan
    return hit, n


def fill_holes(hit_z: np.ndarray, base_z: float) -> tuple[np.ndarray, int]:
    """填补雷达打不到的格子。

    空洞不是随机分布的，它们集中在**台阶背面和机身正下方**——恰恰是最需要看清的
    地方。填法有三种取舍：
      填成平地   → 策略以为前面没障碍，一头撞上去
      填成很高   → 策略以为前面是墙，永远不敢走
      邻域外推   → 台阶背面被外推成台阶的延续，这是三者里最接近真相的

    这里用邻域外推：先用有值的邻格做距离加权，仍填不上的（大片空洞）退回
    "与机身同高的地面"（即 encode 后接近 -0.5 那一档的中性值）。
    返回 (填好的 hit_z, 空洞个数)。
    """
    out = hit_z.copy()
    miss = np.isnan(out)
    n_hole = int(miss.sum())
    if n_hole == 0 or n_hole == NUM_RAYS:
        return np.where(np.isnan(out), base_z - OFFSET, out), n_hole

    grid = out.reshape(NY, NX)
    m = np.isnan(grid)
    for _ in range(max(NX, NY)):            # 逐圈向外长，直到填满
        if not m.any():
            break
        pad = np.pad(grid, 1, constant_values=np.nan)
        nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(nb, axis=0)
        grid = np.where(m & ~np.isnan(avg), avg, grid)
        m = np.isnan(grid)
    grid = np.where(np.isnan(grid), base_z - OFFSET, grid)
    return grid.ravel(), n_hole


def from_cloud(points_world: np.ndarray, base_xy: np.ndarray, base_z: float,
               yaw: float) -> tuple[np.ndarray, dict]:
    """部署侧完整链路：点云 → 187 维观测。

    返回 (obs187, 诊断信息)。诊断里的空洞率/点数分布就是标定训练噪声模型要的东西。
    """
    hit, n = hit_z_from_cloud(points_world, base_xy, yaw)
    filled, n_hole = fill_holes(hit, base_z)
    return encode(base_z, filled), {
        "n_hole": n_hole,
        "hole_rate": n_hole / NUM_RAYS,
        "pts_per_cell_median": float(np.median(n[n > 0])) if (n > 0).any() else 0.0,
        "cells_hit": int((n > 0).sum()),
    }
