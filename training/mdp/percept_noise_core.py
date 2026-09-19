# Copyright (c) 2026 —— GOAI 赛道四 S10 巡逻
# SPDX-License-Identifier: BSD 3-Clause

"""高程图感知缺陷噪声的**纯 torch 内核**。

单独成文件是为了能脱离 IsaacLab 单测 —— @configclass 依赖 isaaclab，
而验证这段数学不需要它。见 s10_dev/noisechk.py。
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# 与 GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) 一致。
# patterns.py:45-53 的 ordering="xy" → flat 索引 = iy*NX + ix
NX, NY = 17, 11

# 下游 ObsTerm 对 height_scan 的 clip 区间，消毒时用同一个值，
# 这样"射线打空"被当成"读数饱和"，与不加噪声那条路径的结果一致。
CLIP = 1.0

# S10_NOISE_DIAG=1 时统计送进来多少个非有限值。默认关：热路径里
# 每步做一次 .item() 会强制 GPU 同步，代价不小。
# 打开时也只在 GPU 上累加，不同步；用 diag_hits() 在需要时读一次。
_DIAG = os.environ.get("S10_NOISE_DIAG") == "1"
_diag_hits: dict = {}          # device -> 标量计数（按设备分开，避免跨设备加法报错）


def _diag_add(data: torch.Tensor) -> None:
    d = data.device
    if d not in _diag_hits:
        _diag_hits[d] = torch.zeros((), dtype=torch.long, device=d)
    _diag_hits[d] += (~torch.isfinite(data)).sum()


def diag_hits() -> int:
    """累计见到的非有限观测格数（S10_NOISE_DIAG=1 才计）。会同步一次 GPU。"""
    return sum(int(v) for v in _diag_hits.values())


def _patch_mask(n: int, p: float, blob: int, device, dtype) -> torch.Tensor:
    """(n,1,NY,NX) 的 0/1 遮挡掩膜，**成片**而不是散点。

    做法：在 blob 倍降采样的粗网格上抛硬币，再最近邻上采样。
    这样一片遮挡的典型尺寸就是 blob×blob 格，与边相交后
    形成长 3~7 格的条带 —— 与实测的连通域尺寸（中位 2、均值 3.2）对得上。

    逐格独立抛硬币是错的：那样出来 60% 是单格散点，而实测只有 8.5%，
    真实缺陷是"一整条边同时被遮挡"。形状恰恰是不能用白噪声的全部理由。
    """
    cy = (NY + blob - 1) // blob
    cx = (NX + blob - 1) // blob
    coarse = (torch.rand(n, 1, cy, cx, device=device, dtype=dtype) < p).to(dtype)
    up = F.interpolate(coarse, scale_factor=blob, mode="nearest")
    return up[:, :, :NY, :NX]


def corrupt(data: torch.Tensor, p_dilate: float, p_erode: float,
            sigma: float, blob: int = 3,
            scale_range: tuple[float, float] | None = None) -> torch.Tensor:
    """给 obs 空间的高程图加感知缺陷。data: (N, 187)，返回同形状。

    obs = base_z - hit_z - 0.5，**地面读高 = obs 读小**。
    所以"障碍朝机身膨胀一格"（实测的主要缺陷）在 obs 空间是 min-pool。
    平地做 min-pool 不变 —— 模型自带"只作用在突变边上"的性质，
    不需要显式检测边缘。
    """
    n = data.shape[0]

    # ---- 必须先消毒，否则训练会崩（#45，v5noisy 与 v6 两轮长跑各丢一次）----
    # 射线打空时 IsaacLab 给 ray_hits_w = ±inf，而 height_scan
    # (isaaclab/envs/mdp/observations.py:300) 是裸减，不做任何过滤；
    # ray_caster 里那句 "remove possible inf values" 在 _debug_vis_callback,
    # **只是画可视化标记用的**，真实数据从没被洗过。
    # 又因为 ObsTerm 的求值顺序是 noise → clip → scale（观测管理器实测），
    # 下游 ±1 的 clip 在我们**之后**，挡不住送进来的 inf。
    # 不消毒的话下面的混合会算出 inf*0 = NaN，而 clip 能修 inf、修不了 NaN
    # —— 这正是"不带噪声的 v5 跑得好好的，带噪声的两版都崩"的原因。
    # 替换值取 ±CLIP，就是下游 clip 本来会给的结果，语义一致。
    if _DIAG:
        _diag_add(data)
    data = torch.nan_to_num(data, nan=0.0, posinf=CLIP, neginf=-CLIP)
    g = data.view(n, 1, NY, NX)

    # max_pool2d 的 padding 用 -inf 填充，所以 -max_pool2d(-g) 是在
    # **有效邻域内**取 min，边界不会被假值污染。
    lo = -F.max_pool2d(-g, kernel_size=3, stride=1, padding=1)
    hi = F.max_pool2d(g, kernel_size=3, stride=1, padding=1)

    # scale_range：**每个环境**抽一个噪声强度 s，三个参数同乘 s。
    # 为什么要这个：训练侧一直固定 p_dilate=0.44，而**部署侧是零噪声**
    # （heightmap.py 注释明说不加噪声）。也就是说训练分布根本没覆盖过
    # 部署时会遇到的那种干净、锐利的高程图。取 lo=0 就把零噪声这一端包进来。
    # 不传 scale_range 时走原路径，逐位不变（见 s10_dev/noisescalechk.py）。
    # **别把这两个变量命名成 lo/hi** —— 上面 lo/hi 已经是最小/最大池化的结果，
    # 撞名会把两个张量覆盖成标量 0.0/1.0，于是被遮挡的格子写成 0.0、
    # 侵蚀的格子写成 1.0（饱和），高程图彻底变形。
    # 这个 bug 我真的写出来过：坏格率从应有的 3.7% 变成 20.9%，
    # 而验收脚本还报"全过"（A 项只测不传 scale_range 的路径，天然测不到它）。
    if scale_range is not None:
        s_lo, s_hi = scale_range
        s = torch.rand(n, 1, 1, 1, device=g.device, dtype=g.dtype) * (s_hi - s_lo) + s_lo
        p_dilate, p_erode, sigma = p_dilate * s, p_erode * s, sigma * s

    # p 传张量 (n,1,1,1) 时 _patch_mask 里的 `rand < p` 会自动广播，无需改它
    m_dil = _patch_mask(n, p_dilate, blob, g.device, g.dtype)
    # 侵蚀的斑块要和膨胀的错开，否则同一片地方两种缺陷互相抵消
    m_ero = _patch_mask(n, p_erode, blob, g.device, g.dtype) * (1.0 - m_dil)

    # 用 where 而不是 `g*(1-m) + lo*m`：加权混合在任一侧出现 inf 时
    # 会算 inf*0 = NaN。消毒之后其实已经不会有 inf，这里是第二道闸
    # —— 混合项一旦再引入非有限值（例如以后有人改上游），where 不会放大。
    out = torch.where(m_dil > 0.5, lo, g)
    out = torch.where(m_ero > 0.5, hi, out)

    # sigma 现在可能是标量也可能是 (n,1,1,1) 张量，别再用 `sigma > 0` 判真假
    if torch.is_tensor(sigma) or sigma > 0.0:
        out = out + sigma * torch.randn_like(out)
    return out.view(n, -1)
