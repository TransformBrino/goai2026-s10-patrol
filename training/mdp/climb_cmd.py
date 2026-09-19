# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
"""墙前速度指令剖面（块3，09-16）。

**作者 09-16 原话**：「墙前慢慢走动作不变形停下来　后退对称蓄力　这些呢？」

## 为什么以前训不出「停」

训练档 `lin_vel_x=(0.3, 0.6)`、`resampling_time_range=(10,10)` —— 整个回合恒定前进指令，
**训练侧从来没有「停」这个指令存在过**。部署侧 `V_NEAR=0.1` 不起作用，因为策略压根没学过
「指令 0.1 就走 0.1」，实测墙前 0.3~0.5 m 处指令 0.10 而实际 0.82 m/s。

## 官方真机数据（ref_climb34.npz，三段，按离墙分箱的机身 vx 中位数）

    离墙        0.60-0.45   0.45-0.35   0.35-0.25   0.25-0.15
    seg0         +0.017      +1.077      +0.290      +0.693
    seg1         +0.035      +1.078      +0.326      +0.532
    seg2         +0.006      +0.229      +0.413      +0.330

**官方在离墙约 0.5 m 处真的停住了**（0.006~0.035 m/s = 零），停完才爆发。
后退也是真的：seg0 最长连续后退 0.405 s、最低 −0.347 m/s、总后退 0.039 m。

我方（C36b/块1/块2，墙前 0.6~0.0 m 的最小 |vx|）：0.721 / 0.511 / 0.827，停住时长 **0.000 s**。
连带后果：蓄力窗（离墙 0.6→0.15）时长 我方 0.41~0.47 s vs 官方 1.80~2.48 s，**差 5 倍** —— 冲过去
就没有蓄力的时间了。

## 改法：只改指令，不加奖励

墙前把 vx 指令按离墙距离改写，让**已有的最强奖励** `track_lin_vel_xy_exp`（读数 +6.20，
全表第一）自己去要求策略停下来。不新增弱奖励项（新项典型读数 0.01~0.3，比主项弱 20~600 倍）。

两条既有机制正好配合，已核实：
  · `track_lin_vel_xy_exp_climb_aware` 只在 `dist_wall ≤ 0.30` 关闭 → 停止带 0.45~0.62 m 里它是**开着的**；
  · `stall_penalty_encounter_free` 整个遭遇段豁免停滞惩罚 → 指令停车**不会**被停滞项反罚；
  · `approach_wall`（遭遇段奖励前进速度，会与停车打架）在本任务族**未注册**，已 grep 确认 0 次。

## 剖面（sd = signed_dist，正 = 机身在墙面前方）

    sd > 0.75          不改写（沿用抽样 0.3~0.6）      慢速接近
    0.75 ≥ sd > 0.62   线性降到 0                      减速
    0.62 ≥ sd > 0.54   vx = back_v（默认 −0.15）       后退
    0.54 ≥ sd > 0.45   vx = 0                          停住蓄力
    sd ≤ 0.45          不改写                          释放（官方 0.45 起爆发 +1.08）

## 两个必须挡掉的坑（都已在实现里处理）

1. **非遭遇 env 的 signed_dist 是垃圾**：`climb_rewards._state` 第 181 行在 `~encounter` 时把
   `wall_pt` 清零，于是 `signed_dist = −(机身位置·前向)`，与墙无关。**必须用 `encounter` 门住**，
   否则平地 env 会随机停车。
2. **刚重置的 env 读到上一回合的墙距**：IsaacLab 6 的 step 顺序是
   234 `common_step_counter += 1` → 252 `reward_manager.compute` → 267 `_reset_idx` → 278
   `command_manager.compute`，四者同属一个 step 计数。`_state` 按 step 计数缓存，所以命令项拿到的
   是 **reset 之前**建的缓存。本类在 `reset()` 里标记这些 env，当步跳过改写。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.utils.configclass import configclass

from .commands import UniformThresholdVelocityCommand, UniformThresholdVelocityCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def profile_vx(base_vx: torch.Tensor, sd: torch.Tensor, gate: torch.Tensor, c) -> torch.Tensor:
    """按离墙距离改写 vx 的**纯函数**（无 env 依赖，便于离线测试）。

    Args:
        base_vx: (N,) 抽样得到的原始 vx —— 每步都要从它出发，不能在改写后的值上再改写。
        sd:      (N,) signed_dist，正 = 机身在墙面前方。
        gate:    (N,) bool，是否允许改写（遭遇中 & 非站立抽样 & 非刚重置）。
        c:       带 d_slow_hi / d_back_hi / d_stop_hi / d_release / back_v 的配置对象。

    Returns:
        (N,) 改写后的 vx。gate 为 False 或在剖面之外的 env 原样返回 base_vx。
    """
    span = max(c.d_slow_hi - c.d_back_hi, 1e-3)
    ramp = base_vx * torch.clamp((sd - c.d_back_hi) / span, 0.0, 1.0)
    vx = base_vx.clone()
    vx = torch.where(gate & (sd <= c.d_slow_hi) & (sd > c.d_back_hi), ramp, vx)
    vx = torch.where(gate & (sd <= c.d_back_hi) & (sd > c.d_stop_hi),
                     torch.full_like(vx, c.back_v), vx)
    vx = torch.where(gate & (sd <= c.d_stop_hi) & (sd > c.d_release),
                     torch.zeros_like(vx), vx)
    return vx


# ---- 状态机版剖面（块3c，09-16 18:4x）----
# 为什么弃用按距离分带：实测两个死结，都不是调参能解决的。
#   ① **停死后出不来**：释放条件是"到达更近的距离"，但它一停住就再也到不了 → 命令永远 0，
#      卡死约 100 s 直到超时（探针 probe_stopfar 实录）。
#   ② **后退带排在停住带外侧**：机器人在减速带就停住了，永远进不到后退带 → 后退 0.000 s
#      （probe_hold 实录，三条全部在 0.91 m 释放）。
# 状态机按**事件**推进，与驱动侧 switch_stair_driver.py 的 _wall_prof 逐条对应：
#   0 接近 → 进入剖面后指令线性降到 0
#   1 停住 → |vx| < v_stop 连续 hold_s 秒（或 timeout_s 超时）
#   2 后退 → 指令 back_v 持续 back_s 秒（官方 seg0 连续后退 0.405 s）
#   3 释放 → 单向，本次遭遇内不再回到停住带（否则上墙途中会被反复叫停）
ST_APPROACH, ST_STOP, ST_BACK, ST_DONE = 0, 1, 2, 3


def step_profile(state, hold, tmr, base_vx, sd, vx, gate, c, dt):
    """**纯函数**，推进一步剖面状态机。返回 (新state, 新hold, 新tmr, 指令vx)。

    state/hold/tmr/base_vx/sd/vx/gate 均为 (N,) 张量；gate=False 的 env 原样返回 base_vx
    且状态不推进。所有分支用张量算子，无 Python 分支，便于 8192 env 并行。
    """
    inprof = gate & (sd <= c.d_slow_hi) & (sd > 0.0) & (state != ST_DONE)
    still = inprof & (vx.abs() < c.v_stop)
    hold = torch.where(still, hold + dt, torch.where(inprof, torch.zeros_like(hold), hold))
    tmr = torch.where(inprof, tmr + dt, torch.zeros_like(tmr))

    # 接近 → 停住：进入剖面即算，指令已在降
    state = torch.where(inprof & (state == ST_APPROACH), torch.full_like(state, ST_STOP), state)
    # 停住 → 后退。**creep_v > 0 时改为按距离结束**（块10，09-16 21:5x）：
    # 官方蓄力段不是停死，而是以 +0.022~+0.102 m/s **蹭近 0.24~0.25 m**，从 0.48~0.59 一路
    # 走到 **0.24~0.34** 才起动作 —— 到那时前轮才够得着墙面，机身才仰得起来（官方 +36~+42°）。
    # 块8 用「停够就放行」，整段位置往外偏 0.12~0.20 m（0.69→0.46），前轮够不着墙，
    # 俯仰只有 −16°，正是作者说的「对的不够近」。
    # 用 hold 做结束条件在有蹭近指令时也不成立：指令 0.08 < v_stop 0.12，hold 会立刻攒满而提前放行。
    creep = getattr(c, "creep_v", 0.0)
    if creep > 0.0:
        to_back = inprof & (state == ST_STOP) & ((sd <= c.d_creep_end) | (tmr >= c.timeout_s))
    else:
        to_back = inprof & (state == ST_STOP) & ((hold >= c.hold_s) | (tmr >= c.timeout_s))
    state = torch.where(to_back, torch.full_like(state, ST_BACK), state)
    tmr = torch.where(to_back, torch.zeros_like(tmr), tmr)
    # 后退 → 释放
    to_done = inprof & (state == ST_BACK) & (tmr >= c.back_s)
    state = torch.where(to_done, torch.full_like(state, ST_DONE), state)

    span = max(c.d_slow_hi - c.d_stop_hi, 1e-3)
    ramp = base_vx * torch.clamp((sd - c.d_stop_hi) / span, 0.0, 1.0)
    out = base_vx.clone()
    # 停住段：creep_v>0 时发慢速蹭近指令（官方形态），否则沿用旧的「降到 0 并保持」
    stop_cmd = torch.full_like(out, creep) if creep > 0.0 else ramp
    out = torch.where(inprof & (state == ST_STOP), stop_cmd, out)
    out = torch.where(inprof & (state == ST_BACK), torch.full_like(out, c.back_v), out)
    return state, hold, tmr, out


class ClimbApproachVelocityCommand(UniformThresholdVelocityCommand):
    """在 `UniformThresholdVelocityCommand` 之上叠加「墙前减速→后退→停住→释放」的 vx 剖面。

    只改 vx（第 0 列），vy / wz / heading 一概不动。站立抽样 env（`is_standing_env`）不改写。
    """

    cfg: ClimbApproachVelocityCommandCfg

    def __init__(self, cfg: ClimbApproachVelocityCommandCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # 抽样得到的原始 vx，每步从它出发重算，绝不在改写后的值上再改写
        self._base_vx = torch.zeros(self.num_envs, device=self.device)
        # 本步刚重置的 env（其 _climb_cache 是 reset 之前的，见模块文档坑 2）
        self._fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prof_dbg = 0
        self._st = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hold = torch.zeros(self.num_envs, device=self.device)
        self._tmr = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        out = super().reset(env_ids)
        self._fresh[:] = False
        if env_ids is None:
            self._fresh[:] = True
        elif len(env_ids) > 0:
            self._fresh[env_ids] = True
        idx = slice(None) if env_ids is None else env_ids
        self._st[idx] = ST_APPROACH; self._hold[idx] = 0.0; self._tmr[idx] = 0.0
        return out

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) > 0:
            self._base_vx[env_ids] = self.vel_command_b[env_ids, 0]

    def _update_command(self):
        super()._update_command()
        if not self.cfg.enable_profile:
            return
        try:
            from .climb_rewards import _state
            s = _state(self._env)
        except Exception:
            return                                   # 没有墙/传感器的任务族：保持原指令
        sd = s["signed_dist"]
        enc = s["encounter"]
        c = self.cfg

        # 门：遭遇中 & 非站立抽样 & 非刚重置
        gate = enc & (~self.is_standing_env) & (~self._fresh)

        # **不再因遭遇瞬断而清状态**（09-16 20:4x 修）。
        # 原来写的是 `torch.where(enc, st, 0)`：`encounter` 只要掉一帧，状态机与计时器
        # 就一起归零 → 永远走不到 ST_DONE，相位门的 4 s 超时也永远攒不满 → 相位再不开。
        # 证据：`cs_all12` 随相位门收紧单调塌到 **0.0000**（C36b −0.1177 → 块3 −0.0893 →
        # 块3c −0.0208 → 块5 −0.0051 → 块6/8 **0**），而 `wall_ahead_frac` 全程不变 0.0002，
        # 说明不是遭遇变少，是门把参考跟踪掐死了。部署侧驱动用 time.time() 与单向 released
        # 标志，不受抖动影响，所以评测 6/6 正常、训练侧却学不到官方动作 —— 两边行为分叉的根因。
        # 状态只在 reset() 里清（回合边界），遭遇瞬断不清。
        bvx = s["vx"]
        self._st, self._hold, self._tmr, vx = step_profile(
            self._st, self._hold, self._tmr, self._base_vx, sd, bvx, gate, c, self._env.step_dt)
        # 把剖面状态发布到 env，供 climb_seq._stop_ready 的「已释放」门读取。
        # 时序：IsaacLab 6 的 step 里 reward_manager.compute(252) 在 command_manager.compute(278)
        # **之前**，所以奖励侧读到的是上一步的状态，滞后 1 步（20 ms）。这点滞后无害
        # （门只关心"是否已完成停住+后退"这个单向事件），但必须写明，不能让人以为是同步的。
        self._env._climb_cmd_state = self._st
        # 只写 gate 为真的 env。**绝不能**整列赋值：基类 _update_command() 刚把
        # rel_standing_envs=0.3 的站立抽样 env 整行清零，整列写回去会用 _base_vx 把它们的
        # vx 复原成 0.3~0.6，于是「指令≈0 要站住」这条训练（cc_zero_hold，v_cmd_max=0.05）
        # 对 30% 的 env 直接失效。09-16 首轮块3 就是这么被污染的：cc_zero_hold 读数
        # 从基线 −0.5123 塌到 −0.005（60 倍），是该 bug 的指纹。见 test_climb_cmd_profile.py。
        self.vel_command_b[gate, 0] = vx[gate]

        self._fresh[:] = False

        if c.debug_every > 0 and (self._env.common_step_counter % c.debug_every == 0):
            n = float(self.num_envs)
            f = lambda m: 100.0 * m.float().sum().item() / n
            inb = gate & (sd <= c.d_slow_hi) & (sd > 0.0)
            print(f"[blk3-cmd] step={self._env.common_step_counter} "
                  f"遭遇={f(enc):.1f}% 剖面内={f(inb):.2f}% "
                  f"停住段={f(inb & (self._st == ST_STOP)):.2f}% "
                  f"后退段={f(inb & (self._st == ST_BACK)):.2f}% "
                  f"已释放={f(enc & (self._st == ST_DONE)):.2f}% "
                  f"停住累计中位={self._hold[inb].median().item() if inb.any() else float('nan'):.2f}s "
                  f"停住段|vx|中位={bvx[self._st == ST_STOP].abs().median().item() if (self._st == ST_STOP).any() else float('nan'):.3f} "
                  f"停住段|vx|最小={bvx[self._st == ST_STOP].abs().min().item() if (self._st == ST_STOP).any() else float('nan'):.3f} "
                  f"(阈值 v_stop={c.v_stop:g}, 需连续 {c.hold_s:g}s)",
                  flush=True)


@configclass
class ClimbApproachVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """`ClimbApproachVelocityCommand` 的配置。默认值按官方 ref_climb34 三段实测定（见模块文档）。"""

    class_type: type = ClimbApproachVelocityCommand

    enable_profile: bool = True
    """总开关。False 时行为与 `UniformThresholdVelocityCommand` 完全一致。"""

    d_slow_hi: float = 0.75
    """减速带上沿（m，signed_dist）。大于它不改写。"""

    d_back_hi: float = 0.62
    """后退带上沿（m）。减速带在此降到 0。"""

    d_stop_hi: float = 0.54
    """停住带上沿（m）＝后退带下沿。"""

    d_release: float = 0.45
    """释放距离（m）。小于它不改写，交回原有翻越行为（官方 0.45 起爆发 +1.08 m/s）。"""

    back_v: float = -0.15
    """后退带指令 vx（m/s，负 = 后退）。官方 seg0 实测最低 −0.347、连续 0.405 s、总后退 0.039 m。"""

    creep_v: float = 0.0
    """>0 时停住段发这个慢速前进指令（m/s），并改为按距离结束该段。官方蓄力段 vx 中位 0.022~0.102。"""

    d_creep_end: float = 0.32
    """蹭近段的结束距离（m）。官方最深蓄力处离墙 0.243~0.341。"""

    v_stop: float = 0.10
    """判定「停住」的 |vx| 阈值（m/s）。"""

    hold_s: float = 0.4
    """需要连续停住多少秒才进入后退段。"""

    back_s: float = 0.4
    """后退段时长（s）。官方 seg0 最长连续后退 0.405 s。"""

    timeout_s: float = 4.0
    """进入剖面后多少秒仍没停住也放行（防死锁，保住登顶）。"""

    debug_every: int = 0
    """>0 时每 N 步打印一次剖面占比自检；0 关闭。"""
