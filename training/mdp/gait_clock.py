# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
"""步态相位钟：把 sin/cos 两维放进观测（244 → 246），作者 09-17 拍板「进观测」这条路。

## 为什么必须进观测

09-15 交付清单记录，压步频这一项是**关掉的**，原因写得很清楚：

> 「步频对齐官方 1.10 Hz — 关闭。**相位钟在观测里不可见 → 奖励被相位平均、无梯度**。
> 　正规解要改 244 维接口（约 1 天 + 真机风险），作者拍板不改」

步频机制本身**早就齐了**（`step_rate_penalty` / `sg_swing_time` / `sg_diag_sync` /
`knee_amp_penalty` / `lr_antiphase_reward`），缺的不是奖励，是策略**看不见自己处在周期哪一相**：
同一个观测下，"该抬腿"和"该落腿"无法区分，任何频率类奖励在相位上一平均就没了梯度。

加上 sin/cos 两维后，这些既有项立刻获得梯度 —— **这就是「两个新变量」的含义**。

## 差距（09-13 实测）

    官方   步频 **恒 1.10 Hz**（0.5~1.9 m/s 全速段不变）  摆动 **恒 0.35 s**  靠**步幅**提速
    我方   H7: 2.4/3.2/3.9 Hz   摆动 0.11~0.15 s   步幅只有官方 **40%**
           09-15 交付态 3.2~3.8 Hz（≈真机 2.3），**是官方的 2~3 倍** —— 作者看到的「踏得快、不稳」

## 约定

相位 φ ∈ [0,1)，按固定频率 `freq` 前进：`φ += freq · dt`，到 1 回绕。
观测给 `sin(2πφ)`、`cos(2πφ)` 两维，**追加在 244 维之后**（不插中间），
这样部署侧 runner 只需在末尾补两维，前 244 维的排布完全不动。

**部署侧必须用同一个钟**：runner 按自己的控制周期以同样的 `freq` 推进 φ。
钟是**开环**的（不依赖接触反馈），两边只要频率一致就同相 —— 这是它能跨 sim2real 的前提。

真机重验仍需人工，本文件只负责训练侧与仿真侧。
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

GAIT_FREQ_HZ = [1.10]      # 官方实测恒定步频；由 cfg 覆盖


USE_TROT_CLOCK = [True]    # 09-18：观测跟随奖励用的那口钟（见下）


def _phase(env: "ManagerBasedRLEnv", freq: float) -> torch.Tensor:
    """每步推进一次的相位 φ∈[0,1)。

    **09-18 关键修正：观测必须读「奖励正在用的那口钟」。**

    原版自己维护 `env._gait_phase`（每回合归零），而这条线的主相位奖励
    `st_trot_ref`（`TrotRefTrack`，权重 6.0、读数 1.39）用的是 **`env._trot_phase`**，
    后者在 `spawn_trot_ref` 里是 **`torch.rand()` 每回合随机起相**。
    → **两口独立的钟、每回合随机相位差，观测对 `st_trot_ref` 携带的信息量为零。**
    这就是「加了观测、600 点步频仍不动」的原因（实测 gp_swing 钉在 −23 无趋势）。

    改为：`_trot_phase` 存在就用它（`USE_TROT_CLOCK`），否则退回自有钟。
    IsaacLab 的 `step()` 里 **reward_manager 先于 obs_manager**，
    所以本步 `_trot_phase` 已被 `TrotRefTrack` 推进过，观测读到的是当步值，不滞后。

    **部署侧的好处**：`_trot_phase` 本来就随机起相 → 策略对**绝对相位**鲁棒，
    runner 从 φ=0 起即可，「同频同相」里的**同相那一半风险消失，只剩同频**。
    参考钟频率表 `[1.1, 1.1, 1.1, 1.1, 1.54] Hz`（对应指令 0.5/1.0/1.4/1.9/2.5 m/s），
    runner 需按同样的速度分档取频率。
    """
    if USE_TROT_CLOCK[0]:
        tp = getattr(env, "_trot_phase", None)
        if tp is not None and tp.shape[0] == env.num_envs:
            return tp
    n = env.num_envs
    ph = getattr(env, "_gait_phase", None)
    if ph is None or ph.shape[0] != n:
        ph = torch.zeros(n, device=env.device)
        env._gait_phase = ph
        env._gait_phase_step = -1
    if env._gait_phase_step != env.common_step_counter:
        env._gait_phase_step = env.common_step_counter
        # 回合刚重置的 env：episode_length_buf 很小 → 相位归零，保证每回合从同一相起步
        fresh = env.episode_length_buf <= 1
        ph = torch.where(fresh, torch.zeros_like(ph), ph + freq * env.step_dt)
        ph = ph - torch.floor(ph)
        env._gait_phase = ph
    return env._gait_phase


def gait_phase_obs(env: "ManagerBasedRLEnv", freq: float | None = None) -> torch.Tensor:
    """观测项：返回 (N,2) 的 [sin(2πφ), cos(2πφ)]。**这就是新增的两个变量。**"""
    f = GAIT_FREQ_HZ[0] if freq is None else freq
    ph = _phase(env, f)
    a = 2.0 * math.pi * ph
    return torch.stack([torch.sin(a), torch.cos(a)], dim=1)


def phase_swing_track(env: "ManagerBasedRLEnv", freq: float | None = None,
                      h_min: float = 0.04, v_min: float = 0.3,
                      command_name: str = "base_velocity",
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """**对角腿摆动与相位钟对齐**（惩罚）。

    钟只是"看得见"，还得有东西把动作**绑到**钟上，否则策略没有理由跟着它走。

    约定：φ∈[0,0.5) 时对角组 A（fl+hr）应当在摆动、组 B（fr+hl）支撑；φ∈[0.5,1) 反之。
    值 = 两组"该摆却没摆 + 不该摆却摆了"的偏差之和。

    摆动判定复用 `step_gait._swing`（轮底高于该轮自己的触地高 `h_min`），
    与既有步态项**同口径**，不另立标准 —— 避免同一现象两套判据。

    门：指令速度 ≥ `v_min`（站立不罚）。

    **为什么不可伪造**：它是惩罚且**双向**——不抬腿（步频 0）在"该摆"的半周期里照样被罚，
    治的正是 [[project-s10-sim2sim-gait]] 记录的那个退化终态（四腿锁死、纯轮子滚，
    使 `sg_air_frac` / `air_frac_lr_penalty` 读数全为 0 而管不到）。
    """
    from .step_gait import _swing, _gate
    f = GAIT_FREQ_HZ[0] if freq is None else freq
    ph = _phase(env, f)
    sw, _ = _swing(env, h_min=h_min)                           # `_swing` 返回 (摆动布尔(N,4), 触地高)
    sw = sw.float()                                            # 1=摆动
    # 09-18：原版写成 `_swing(...).float()`，把二元组当张量用 → AttributeError。
    # 本文件 09-17 写好后**从未训练过**，所以这个 bug 一直没被触发。
    # 轮序 fl,fr,hl,hr → 对角 A=(fl,hr)=(0,3)，B=(fr,hl)=(1,2)
    a = 0.5 * (sw[:, 0] + sw[:, 3])
    b = 0.5 * (sw[:, 1] + sw[:, 2])
    want_a = (ph < 0.5).float()
    err = (a - want_a).abs() + (b - (1.0 - want_a)).abs()
    # **门必须复用 `step_gait._gate`**（09-18 修）。原版自己写 `norm(cmd[:, :2]) >= v_min`，
    # 用**速度模长**门控 → **倒退时照样要求前进方向的摆腿相位**，把倒退学坏。
    # 这正是 `step_gait.FORWARD_ONLY` 注释里记过的那个坑：
    #   「H5 起：只按前向指令 vx 门控（H4 倒退 −0.5 时步态项照样按速度模长触发，
    #     倒退学坏了：实际 +0.10 m/s 还打转）」
    # 踏步线里 `sg.FORWARD_ONLY[0] = True` 本来是开着的，自己写门等于绕过它。
    # 实测症状：指令 −0.5 → 实际 **+0.37**（方向反）、指令 +1.8 → **−0.84**。
    return err * _gate(env, v_min, 99.0, command_name)


def big_action_penalty(env: "ManagerBasedRLEnv", lim: float = 3.0) -> torch.Tensor:
    r"""**原始动作幅值超过 lim 的部分（铰链平方，逐维求和）**（09-19，治 T10 真机自激）。

    真机事故：T10_15197 平地 roll 一歪，|动作| 0.9 → 5 → 44 → 1e14（1 s），根因是观测里的「上一动作」
    在训练侧 clip ±100、runner 不钳位，正反馈环没有闸。离线自激复现：T10 roll 20° 不动点 11.4，
    246-lr8 12.5，都越过打限位线 10.9（= 2.7227 rad ÷ 腿缩放 0.25）；V1H/R13/S16 在 2~6。

    本项让策略**自己**学会不进入大动作区：真机常态 |a| 0.9、峰 1.7，四个健康策略 roll 0° 自激点 1.8~3.3，
    取 lim=3.0 → 正常运行**恒为 0**（不可伪造：纯惩罚、只在越界时非零），越界越远罚越重。
    配套：观测 `actions` 项 clip 收到 ±4（训练里环有闸）、动作 cfg clip（仿真执行也有闸）、
    runner action clamp ±4（部署闸，待作者拍板）。三道闸 + 本项 = 策略在闸内学习且不靠闸。
    """
    a = env.action_manager.action                    # 原始策略输出（未缩放、未钳位）—— 与观测里的上一动作同源
    ex = torch.clamp(a.abs() - lim, min=0.0)
    return (ex * ex).sum(dim=1)
