# Copyright (c) 2026 —— GOAI 赛道四 S10 巡逻（决赛训练 v8）
# SPDX-License-Identifier: BSD 3-Clause

"""把**部署侧的高程图裁剪**原样搬进训练观测。

部署侧（s10_ws/.../elevation_map.py，冻结配置 release/20260814-2040）：
    S10_HMAP_FWD_CLIP=0.6   机体系 x > 0.6 的格子 → 填成近场(x<=0.1)格子的中位值
    S10_HMAP_LAT_CLIP=0.30  机体系 |y| > 0.3 的格子 → 同上
初赛 v6 的远场过敏（台阶在 0.8m 外就冻住）靠这两条裁剪救回，
但 v6 训练时看到的是**没裁的**世界。这里把裁剪写进训练观测，
让训练与部署看同一个世界——把「绕过」升格为「契约」。

接线：作为 NoiseCfg.func 挂在 height_scan 上，先做 v6 同款实测遮挡噪声
（percept_noise.corrupt_height_scan），再裁剪；ObsTerm 之后还有 clip(-1,1)。
部署顺序是 编码→clip(±1)→裁剪，这里是 噪声→裁剪→clip(±1)；
近场中位值在 ±1 内，两种顺序数值等价。

格子索引约定（与 percept_noise_core.py 一致，patterns.py ordering="xy"）：
    flat = iy*NX + ix,  x = -0.8 + 0.1*ix (17),  y = -0.5 + 0.1*iy (11)
部署侧 _clipprobe.sh 实测：FWD 0.6 抹平 22/187，近场 110/187。
V8 环境配置在 __post_init__ 用 grid_pattern 现算一遍索引集合做断言，
掩码与传感器真实光线顺序对不上就不许开训。
"""
from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .percept_noise import HeightScanOcclusionCfg, corrupt_height_scan
from .percept_noise_core import NX, NY

X_MIN, Y_MIN, RES = -0.8, -0.5, 0.1
NEAR_X = 0.1          # 部署侧 near = CELL_XY[:,0] <= 0.1


def cell_xy() -> tuple[torch.Tensor, torch.Tensor]:
    """187 个格子的机体系 (x, y)，按 flat = iy*NX + ix。"""
    ix = torch.arange(NX, dtype=torch.float32)
    iy = torch.arange(NY, dtype=torch.float32)
    x = (X_MIN + RES * ix)[None, :].expand(NY, NX).reshape(-1)
    y = (Y_MIN + RES * iy)[:, None].expand(NY, NX).reshape(-1)
    return x, y


def masks(fwd_clip: float | None, lat_clip: float | None):
    x, y = cell_xy()
    eps = 1e-6
    near = x <= NEAR_X + eps
    dead = torch.zeros_like(near)
    if fwd_clip is not None and fwd_clip > 0:
        dead |= x > fwd_clip + eps
    if lat_clip is not None and lat_clip > 0:
        dead |= y.abs() > lat_clip + eps
    return near, dead


_cache: dict = {}


def apply_clip(data: torch.Tensor, fwd_clip, lat_clip) -> torch.Tensor:
    """data (N,187) → 抹平区填近场中位值（逐环境）。"""
    key = (data.device, fwd_clip, lat_clip)
    if key not in _cache:
        near, dead = masks(fwd_clip, lat_clip)
        _cache[key] = (near.to(data.device), dead.to(data.device))
    near, dead = _cache[key]
    if not bool(dead.any()):
        return data
    fill = data[:, near].median(dim=1, keepdim=True).values      # (N,1)
    return torch.where(dead[None, :], fill.expand_as(data), data)


def corrupt_then_clip(data: torch.Tensor, cfg: "HeightScanOcclusionClipCfg") -> torch.Tensor:
    out = corrupt_height_scan(data, cfg)
    return apply_clip(out, cfg.fwd_clip, cfg.lat_clip)


@configclass
class HeightScanOcclusionClipCfg(HeightScanOcclusionCfg):
    """v6 同款遮挡噪声 + 部署裁剪。噪声参数全部继承，不动。"""

    func = corrupt_then_clip

    fwd_clip: float | None = 0.6
    """机体系 x 超过它的格子抹平。冻结部署值 0.6（S10_HMAP_FWD_CLIP）。"""

    lat_clip: float | None = 0.30
    """机体系 |y| 超过它的格子抹平。冻结部署值 0.30（S10_HMAP_LAT_CLIP）。"""
