# Copyright (c) 2026 —— GOAI 赛道四 S10 巡逻
# SPDX-License-Identifier: BSD 3-Clause

"""高程图的**感知缺陷**噪声模型 —— 替换官方那个 Unoise(-0.1, 0.1)。

为什么不能用均匀白噪声：我们把仿真雷达累积出的高程图和地形真值逐格比过
（s10_dev/emapchk.py，全程 4600 帧），真实缺陷完全不是白噪声：

    坏格(>0.1) 只占 4.7%，但 **107/117 恰好压在地形高度突变边上**
    边缘格坏率 25.7%，非边缘格 1.4%（18 倍差）
    平地格坏率 0.8% —— 平地几乎不出错
    形状是**宽 1 格、长 4~7 格的条带**，不是散点（37 个连通域装 117 格）
    幅度**等于该处台阶/墙的高度本身**（|误差|/局部起伏 中位 = 1.00）
    符号 90% 是"读高"（假障碍），10% 是"读低"
    结构检验：|地图 − 真值做一次 3×3 max 膨胀| 中位仅 0.010，79/117 命中

机制：雷达只看得到台阶**立面**，看不到台阶背面的顶面，
于是障碍在高程图上朝机身方向**膨胀了一格**。

数学内核在 percept_noise_core.py（纯 torch，可脱离 IsaacLab 单测）。
标定验证见 s10_dev/noisechk.py —— 拿赛道真实高程图跑一遍，
比对生成的缺陷统计和实测雷达缺陷统计是否吻合。

**接线注意**（源码读出来的坑）：
- `ObsTerm.noise` 只接受 NoiseCfg / NoiseModelCfg。塞裸函数会被
  observation_manager.py:400/402 的两个 isinstance 都跳过 ——
  **不报错、不加噪声**，静默失效。
- 施加顺序 func → modifiers → noise → clip(-1,1) → scale，
  噪声在 clip 之前。
- 只有 policy 组 enable_corruption=True，critic 组默认 False，
  所以 critic 拿到的高程图始终干净 —— asymmetric actor-critic 成立。
"""
from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from .percept_noise_core import corrupt


def corrupt_height_scan(data: torch.Tensor, cfg: "HeightScanOcclusionCfg") -> torch.Tensor:
    """输入输出形状必须相同（NoiseCfg 的约定，noise_cfg.py:24-28）。"""
    out = corrupt(data, cfg.p_dilate, cfg.p_erode, cfg.sigma, cfg.blob,
                  getattr(cfg, "scale_range", None))
    # 09-14 S1/T1：LIO z 漂造成的两种假图（21:24 真机侧翻：地图整幅低 0.09、前方 0.4～0.8 m 整幅 −0.33）
    p_off = float(getattr(cfg, "p_offset", 0.0)); p_tr = float(getattr(cfg, "p_trench", 0.0))
    if (p_off > 0 or p_tr > 0) and out.dim() == 2 and out.shape[1] == 187:
        n = out.shape[0]
        if p_off > 0:
            m = torch.rand(n, device=out.device) < p_off
            off = (torch.rand(n, device=out.device) * 2 - 1) * float(getattr(cfg, "offset_max", 0.10))
            out = out + (off * m.float()).unsqueeze(1)
        if p_tr > 0:
            m = torch.rand(n, device=out.device) < p_tr
            g = out.view(n, 11, 17).clone()
            g[:, :, 12:] = torch.where(m.view(n, 1, 1), torch.full_like(g[:, :, 12:], float(getattr(cfg, "trench_val", -0.30))), g[:, :, 12:])
            out = g.view(n, 187)
    # 09-19 S16b：真机 09-18 21:34 楼梯脚下的假图——单帧 46% 格子出现 0.3~0.6 m"幻墙"（整列/整带 −0.45~−0.65），整图偏移 +0.09。
    # p_wall：每局抛硬币，命中则随机挑 wall_cols 列宽的一带（任意 x 位置）整带置 wall_val。默认 0 = 关，不影响其它配置。
    p_wall = float(getattr(cfg, "p_wall", 0.0))
    if p_wall > 0 and out.dim() == 2 and out.shape[1] == 187:
        n = out.shape[0]; m = torch.rand(n, device=out.device) < p_wall
        if m.any():
            g = out.view(n, 11, 17).clone(); wcmax = int(getattr(cfg, "wall_cols", 3)); wv = float(getattr(cfg, "wall_val", -0.55))
            wc = torch.randint(2, wcmax + 1, (n,), device=out.device)                      # 带宽 2..wall_cols 列随机
            c0 = (torch.rand(n, device=out.device) * (17 - wc + 1).float()).long(); col = torch.arange(17, device=out.device).view(1, 1, 17)
            band = (col >= c0.view(n, 1, 1)) & (col < (c0 + wc).view(n, 1, 1)) & m.view(n, 1, 1)
            g = torch.where(band, torch.full_like(g, wv), g); out = g.view(n, 187)
    # 09-19 AGX 量化：真机专家段最差帧 86% 格子是"墙"、帧间跳变 p95 0.52 m（LIO 位姿抖）。p_chaos：该帧按 chaos_frac 比例的格子
    # 随机替换成 U(chaos_lo, chaos_hi)（−0.6 墙 ~ +0.05 坑），逼策略在单帧图不可信时靠本体。默认 0 = 关。
    p_ch = float(getattr(cfg, "p_chaos", 0.0))
    if p_ch > 0 and out.dim() == 2 and out.shape[1] == 187:
        n = out.shape[0]; m = (torch.rand(n, device=out.device) < p_ch).view(n, 1)
        if m.any():
            frac = float(getattr(cfg, "chaos_frac", 0.7)); lo = float(getattr(cfg, "chaos_lo", -0.60)); hi = float(getattr(cfg, "chaos_hi", 0.05))
            cell = (torch.rand_like(out) < frac) & m; junk = torch.rand_like(out) * (hi - lo) + lo
            out = torch.where(cell, junk, out)
    return out


@configclass
class HeightScanOcclusionCfg(NoiseCfg):
    """按实测标定。改参数之前先跑 s10_dev/noisechk.py 看验证输出。"""

    func = corrupt_height_scan

    p_dilate: float = 0.44
    """遮挡斑块的抛硬币概率（在 3 倍降采样的粗网格上抛，再上采样成片）。
    平地做 min-pool 不变，所以它实际只作用在突变边上。
    0.44 是标定出来的：产出的**边缘格坏率 25.3%**，实测雷达 25.7%。
    （逐格抛 0.26 也能凑到坏格率，但那样 60% 是单格散点，形状全错。）"""

    p_erode: float = 0.05
    """反向斑块：抬高面顶部被读低。实测坏格里约 7% 是这一类。"""

    sigma: float = 0.006
    """基础测距噪声。实测中位误差 0.003m、90 分位 0.009m；
    高斯 90 分位 = 1.645σ → σ ≈ 0.0055，取 0.006。"""

    scale_range: tuple[float, float] | None = None
    """**每个环境**抽一个强度 s∈U(lo,hi)，上面三个参数同乘 s。None = 关（原行为）。

    为什么需要它：上面的参数是按**真机雷达**标定的，而部署侧（仿真评测栈）
    是**零噪声**的干净高程图 —— heightmap.py 里明说不加噪声。
    固定 0.44 训出来的策略，从没在训练里见过部署时那种干净锐利的读数，
    这是一处已核实的训练/部署分布错配。取 lo=0 就把零噪声那一端包进训练分布。

    v7 用 (0.0, 1.0)。注意这不是"把噪声调小"，是**把噪声当成随机化维度**，
    高噪声那一端仍然按原强度出现。"""

    blob: int = 3
    """遮挡斑块的粗网格倍率。3 → 斑块约 3×3 格，与边相交后形成
    长 3~7 格的条带，连通域均值 4.2（实测 3.2）。"""
