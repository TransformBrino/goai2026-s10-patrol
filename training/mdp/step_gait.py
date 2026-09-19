"""几何净空版踏步奖励（09-13 下午）。为什么要它：trot_gait.py 里的 air_time / no_step / 按轮罚 / diag 全靠接触传感器 1 N 阈值判"离地"，
策略学会瞬间卸载轮子就能"离地"（MuJoCo 里抬轮 1～2 mm）——V1H-Trot、Trot-V、Trot-V2、连 09-10 的 TrotE 都是这样，从没真踏过步。
这里"摆动" = 轮底比四轮最低点高 ≥ h_min（默认 4 cm），必须真抬。四项：对角同步、抬高目标（官方 9～10 cm）、摆动时长目标（官方 0.35 s→0.24 s 随速度）、
按轮"该踏不踏"罚（任一轮超过 t_max 没摆动过）。都只在速度指令 v_min～v_max 内算。"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .climb_rewards import _state
from .trot_gait import _cmd_speed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


OWN_BASE = [False]   # T1 起：摆动高度按"该轮自己最近触地高"算（相对四轮最低点在坡 >5° 时两前轮永久算摆动、后轮永久没抬）


def _swing(env, h_min: float = 0.04):
    s = _state(env)
    z = s["z"]                                              # (N,4) 轮底高（相对出生地面）
    if OWN_BASE[0]:
        last = getattr(env, "_sg_last_z", None)
        if last is None or last.shape != z.shape: last = z.clone()
        last = torch.where(s["contact"], z, last); env._sg_last_z = last
        h = z - last
    else:
        h = z - z.min(dim=1, keepdim=True).values           # 相对四轮最低点
    return h > h_min, h


FORWARD_ONLY = [False]
EMA_TAU = 1.0            # 摆动占比滑动平均的时间常数（s）   # H5 起：只按前向指令 vx 门控（H4 倒退 −0.5 时步态项照样按速度模长触发，倒退学坏了：实际 +0.10 m/s 还打转）


def _gate(env, v_min, v_max, command_name):
    c = env.command_manager.get_command(command_name)
    sp = c[:, 0] if FORWARD_ONLY[0] else torch.linalg.norm(c[:, :2], dim=1)
    return ((sp > v_min) & (sp < v_max)).float()


def _bufs(env):
    b = getattr(env, "_sg", None)
    if b is None or b["last"].shape[0] != env.num_envs:
        z = torch.zeros(env.num_envs, 4, device=env.device)
        b = {"last": z.clone(), "start": z.clone(), "prev": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device), "step": -1, "ema": z.clone()}
        env._sg = b
    return b


def _update(env, sw):
    """每步一次：记每轮最近一次摆动的时刻、当前摆动的起点。时间用 episode_length_buf×dt（reset 后自动归零）。"""
    b = _bufs(env)
    if b["step"] == env.common_step_counter:
        return b
    b["step"] = env.common_step_counter
    t = (env.episode_length_buf.float() * env.step_dt).unsqueeze(1).expand(-1, 4)
    fresh = t < b["last"] - 1e-6                            # 刚 reset 的 env：时间倒退 → 清零
    b["last"] = torch.where(fresh, torch.zeros_like(t), b["last"]); b["start"] = torch.where(fresh, torch.zeros_like(t), b["start"])
    b["prev"] = torch.where(fresh, torch.zeros_like(b["prev"]), b["prev"])
    b["ema"] = torch.where(fresh, torch.zeros_like(t), b["ema"])
    b["ema"] = b["ema"] + (sw.float() - b["ema"]) * (env.step_dt / EMA_TAU)      # 每轮"摆动中"的 1 s 滑动占比（H6 起）
    rising = sw & ~b["prev"]
    if "rate" not in b: b["rate"] = torch.zeros_like(t)
    b["rate"] = torch.where(fresh, torch.zeros_like(t), b["rate"])
    b["rate"] = b["rate"] + (rising.float() / env.step_dt - b["rate"]) * (env.step_dt / 2.0)   # T2：每轮起抬频率的 2 s 滑动平均（Hz）
    b["start"] = torch.where(rising, t, b["start"])
    b["last"] = torch.where(sw, t, b["last"])
    b["ended"] = (~sw) & b["prev"]                          # 落地瞬间
    b["dur"] = t - b["start"]
    b["t"] = t
    b["prev"] = sw.clone()
    return b


def sg_diag_sync(env: "ManagerBasedRLEnv", v_min: float = 0.3, v_max: float = 2.4, h_min: float = 0.04, command_name: str = "base_velocity") -> torch.Tensor:
    sw, _ = _swing(env, h_min); c = sw.float()
    same_diag = 0.5 * ((c[:, 0] == c[:, 3]).float() + (c[:, 1] == c[:, 2]).float())
    same_side = 0.5 * ((c[:, 0] == c[:, 1]).float() + (c[:, 2] == c[:, 3]).float())
    return (same_diag - same_side) * sw.any(dim=1).float() * _gate(env, v_min, v_max, command_name)


def sg_clearance(env: "ManagerBasedRLEnv", h_target: float = 0.09, sigma: float = 0.04, v_min: float = 0.3, v_max: float = 2.4, h_min: float = 0.04,
                 command_name: str = "base_velocity", max_dur: float = 99.0) -> torch.Tensor:
    """max_dur：摆动超过这个时长就不再给抬高分（Trot-H 500 步学成"把一对对角腿举着不放"，抬高分是按步连续给的，举着=白拿）。"""
    sw, h = _swing(env, h_min)
    if max_dur < 50:
        b = _update(env, sw); sw = sw & (b["dur"] < max_dur)
    r = torch.exp(-((h - h_target) ** 2) / (sigma ** 2)) * sw.float()
    n = sw.float().sum(dim=1).clamp(min=1.0)
    return (r.sum(dim=1) / n) * sw.any(dim=1).float() * _gate(env, v_min, v_max, command_name)


def sg_swing_time(env: "ManagerBasedRLEnv", v_min: float = 0.3, v_max: float = 2.4, sigma: float = 0.08, h_min: float = 0.04,
                  command_name: str = "base_velocity", T_fixed: float = 0.0,
                  kernel: str = "gauss", width: float = 0.35, time_norm: bool = False) -> torch.Tensor:
    """落地瞬间按这次摆动时长给分：目标 T(v)=clip(0.45−0.13v, 0.22, 0.45)（官方 0.35 s @≤1 m/s → 0.24 s @1.8）。

    **kernel="tent"（T11，09-15 07:5x）：高斯核在当前误差处数值为零、没有梯度。**
      T10_15197 实测摆动 **0.12~0.13 s**，目标 T_fixed=0.35，误差 0.22，而 σ=0.08 →
      `exp(−(0.22/0.08)²) = e^−7.56 = 5.2e−4`，乘权重 3.0 = **0.0016**；
      训练日志里 `sg_swing` 读数 **+0.0015** —— **数值对得上，这一项是死的**。
      于是踏步线只有"别踏快"的罚（`sg_rate` −5.0/f_max 1.3，读数 −3.27，一直在狠罚），
      **没有"每步摆 0.35 s"的可用梯度**。而这条线早有定论：**光靠罚建立不起新行为**
      （C13 用 400 迭代证过；C14 换成正向塑形才把差速方向学会）。
      同一毛病此前出现过两次：限速罚 −250 失效、`wheel_diff_track_reward` σ=4 在误差 9~18 处无梯度（当时否掉了它）。
    tent 核：`clamp(1 − |dur − T| / width, 0)`，width 0.35 → 在 dur=0.13 处给 **0.37**（原来 5e-4，700 倍）。

    **但 T11 实训证明 tent 核单用是错的（09-15 08:41，步频 3.23 Hz 一动不动）。推导：**
      本项是**落地事件触发**的，每秒总收益 = f × r(dur)。设占空比 duty = dur·f 固定，
      tent 在 dur<T 时 r = dur/width，于是
        **每秒总收益 = (duty/dur) × (dur/0.35) = duty/0.35 —— 与 dur、f 完全无关，梯度精确为 0。**
      实算（duty 0.42）：dur 0.13/0.20/0.26/0.30/0.35 → 每秒收益 **全是 1.200**。
      而原高斯核的每秒收益是 0.002/0.062/0.456/0.947/1.200 —— **有梯度，只是起点量级太小**。
      → 我当初"高斯是死的"只对了一半：**单次值** 5e−4 确实可忽略，但**每秒总量的形状是对的**；
        tent 把量级提上来的同时**把梯度抹平了**，比原来更糟。

    **time_norm=True（T12）：单次奖励再乘以 `dur`。**
      每秒总收益 = (duty/dur) × r(dur) × dur = **duty × r(dur)** —— 只随 r(dur) 变，
      在 dur=T 处取最大，**与频率无关、方向正确**。
      实算（tent×dur，duty 0.42）：dur 0.13/0.20/0.26/0.30/0.35 → 0.156/0.240/0.312/0.360/**0.420**，单调。
    **注**：`sg_clearance` 不需要这个处理——它是**摆动期间逐步**给分（`sw.float()`），
    每秒总量 ∝ 悬空时间占比，本来就与频率无关（T11 记录里我一度写错成"落地事件触发"，已订正）。
    """
    sw, _ = _swing(env, h_min); b = _update(env, sw)
    v = _cmd_speed(env, command_name); T = torch.clamp(0.45 - 0.13 * v, 0.22, 0.45).unsqueeze(1)
    if T_fixed > 0: T = torch.full_like(T, T_fixed)          # T1：官方 0.5～1.9 m/s 摆动恒 0.35 s
    if kernel == "tent":
        k = torch.clamp(1.0 - (b["dur"] - T).abs() / width, min=0.0)
    else:
        k = torch.exp(-((b["dur"] - T) ** 2) / (sigma ** 2))
    if time_norm:
        k = k * b["dur"]          # T12：按时间归一，见下方推导
    r = k * b["ended"].float()
    return r.sum(dim=1) * _gate(env, v_min, v_max, command_name)


def sg_no_swing(env: "ManagerBasedRLEnv", t_max: float = 0.9, v_min: float = 0.3, v_max: float = 2.4, h_min: float = 0.04,
                command_name: str = "base_velocity") -> torch.Tensor:
    """有速度指令时，任一轮超过 t_max 没真抬过（净空 ≥ h_min）→ 罚 1/轮。"""
    sw, _ = _swing(env, h_min); b = _update(env, sw)
    idle = (b["t"] - b["last"]) > t_max
    return idle.float().sum(dim=1) * _gate(env, v_min, v_max, command_name)


def sg_hold_penalty(env: "ManagerBasedRLEnv", max_dur: float = 0.6, v_min: float = 0.3, v_max: float = 2.4, h_min: float = 0.04,
                    command_name: str = "base_velocity") -> torch.Tensor:
    """摆动（净空 ≥ h_min）持续超过 max_dur 还不落地 → 罚 1/轮（官方摆动 0.24～0.4 s，举着不放不是踏步）。"""
    sw, _ = _swing(env, h_min); b = _update(env, sw)
    return (sw & (b["dur"] > max_dur)).float().sum(dim=1) * _gate(env, v_min, v_max, command_name)


def sg_air_frac_penalty(env: "ManagerBasedRLEnv", frac_max: float = 0.6, v_min: float = 0.3, v_max: float = 2.4, h_min: float = 0.04,
                        command_name: str = "base_velocity") -> torch.Tensor:
    """某只轮子最近 1 s 里"摆动中"的占比超过 frac_max → 罚超出量/轮（H6 起）。
    针对 H4 在 0.5 m/s 的"三轮走、一只前轮点地即抬"（右前悬空 92%）：举着罚按单次摆动时长算，点一下地就归零，管不住；这里按占比算，点地躲不掉。
    正常对角小跑每轮悬空 35～50%，0.6 有余量。"""
    sw, _ = _swing(env, h_min); b = _update(env, sw)
    return torch.clamp(b["ema"] - frac_max, min=0.0).sum(dim=1) * _gate(env, v_min, v_max, command_name)


def step_rate_penalty(env: "ManagerBasedRLEnv", f_max: float = 1.3, v_min: float = 0.3, v_max: float = 2.4, h_min: float = 0.04,
                      command_name: str = "base_velocity") -> torch.Tensor:
    """每轮起抬频率（2 s 滑动平均）超过 f_max 的部分之和（Hz）。官方 0.5～1.9 m/s 步频恒 1.10 Hz，我方 T1 在 1.8 m/s 仍 3.6 Hz（T2）。"""
    sw, _ = _swing(env, h_min); b = _update(env, sw)
    return torch.clamp(b["rate"] - f_max, min=0.0).sum(dim=1) * _gate(env, v_min, v_max, command_name)


def air_frac_lr_penalty(env: "ManagerBasedRLEnv", v_min: float = 0.3, v_max: float = 2.2, h_min: float = 0.04,
                        command_name: str = "base_velocity") -> torch.Tensor:
    """左右悬空占比不对称罚（T6）：|左前−右前| + |左后−右后| 的 1 s 滑动平均之差。治真机上的"举一只轮、三轮滚"。"""
    sw, _ = _swing(env, h_min); b = _update(env, sw); e = b["ema"]
    return ((e[:, 0] - e[:, 1]).abs() + (e[:, 2] - e[:, 3]).abs()) * _gate(env, v_min, v_max, command_name)


_KNEE_IDS = {}


def knee_amp_penalty(env: "ManagerBasedRLEnv", amp_min: float = 0.06, tau: float = 0.5, v_min: float = 0.3,
                     v_max: float = 2.2, command_name: str = "base_velocity", joint_re: str = ".*_knee_joint") -> torch.Tensor:
    """每条腿膝关节摆幅（0.5 s 滑动平均绝对偏差）低于 amp_min 的缺口之和，有速度指令时才算。

    为什么要这一项（09-14 19:5x，Isaac↔MuJoCo 逐列对拍的结论）：
      T6_13999 在 MuJoCo 平地 0.5 m/s 下**四条腿完全锁死**——关节位置 σ 0.0008~0.0011、关节速度 σ 精确为 0、
      动作 σ 0.003，纯靠轮子滚；接触统计报的"右前悬空 91%"其实是姿势歪了那只轮碰不到地，不是在举腿。
      它靠"不抬腿"把起抬频率罚 sg_rate 从 3.22 清到 0.65，而所有基于接触判定的反作弊项
      （sg_air_frac 悬空占比、sg_air_sym 左右不对称）读数都是 0 —— 因为压根没有摆动可言，这些项管不到。
      拦它的只有 sg_no_swing（权重 −3），挡不住。
    本项不依赖接触判定，直接量关节动没动，Isaac 与 MuJoCo 同口径：
      T3_13400（两边都好）Isaac 膝 σ 0.065~0.138 / MuJoCo 0.032~0.071；T6_13999（坏）0.024~0.043 / 0.0008。
    正弦波下 MAD ≈ 0.9σ，amp_min 0.06 对应 Isaac 侧 σ≈0.067，按两边约 0.5 的衰减比可保 MuJoCo σ ≥ 0.03。
    """
    asset = env.scene["robot"]
    key = (id(env), joint_re)
    ids = _KNEE_IDS.get(key)
    if ids is None:
        ids = asset.find_joints([joint_re])[0]
        _KNEE_IDS[key] = ids
    q = asset.data.joint_pos
    q = (q.torch if hasattr(q, "torch") else q)[:, ids]                  # (N,4)
    b = getattr(env, "_sg_knee", None)
    if b is None or b["m"].shape != q.shape:
        b = {"m": q.clone(), "d": torch.zeros_like(q), "t": torch.zeros(env.num_envs, device=q.device), "step": -1}
        env._sg_knee = b
    if b["step"] != env.common_step_counter:
        b["step"] = env.common_step_counter
        t = env.episode_length_buf.float() * env.step_dt
        fresh = (t < b["t"] - 1e-6).unsqueeze(1)                          # 刚 reset：均值重置到当前值，缺口先归零
        b["m"] = torch.where(fresh, q, b["m"]); b["d"] = torch.where(fresh, torch.full_like(q, amp_min), b["d"])
        b["t"] = t
        k = env.step_dt / tau
        b["m"] = b["m"] + (q - b["m"]) * k
        b["d"] = b["d"] + ((q - b["m"]).abs() - b["d"]) * k
    return torch.clamp(amp_min - b["d"], min=0.0).sum(dim=1) * _gate(env, v_min, v_max, command_name)


_LR_IDS = {}


def lr_antiphase_reward(env: "ManagerBasedRLEnv", tau: float = 1.0, amp0: float = 0.05, v_min: float = 0.3,
                        v_max: float = 2.2, command_name: str = "base_velocity") -> torch.Tensor:
    """左右腿反相奖励（T9，09-15 00:1x）：前对 + 后对，各自"左右 hipy 偏离慢均值后符号相反"时给分。

    为什么加这一项 —— 09-14 夜用不依赖接触判定的相位判据比三方：
      官方 gftb1 / gftbzx1 平地踏步：左右反相 −0.53 / −0.75，hipy σ 0.172~0.197
      真机 T3_13400（作者现场"踏步走不稳"）：**+0.05 / +0.23**，hipy σ 0.065~0.120
      MuJoCo T3 / T6：−0.11 / +0.51、+0.88 / +0.67
    也就是**我们的"踏步"从来不是官方那种交替迈腿**，是四条腿同相小幅抖 + 轮子驱动。
    以前的判据全漏了：接触统计的"对角同步"按触地算（轮子贴地时怎么抖都算同步）；膝 σ 只量幅度不量相位；
    Isaac 侧 sg_diag 读数只有 +0.076、权重 2.0，从头到尾没真正拉动过相位。

    防刷分：用**归一化**的符号积 p = (d_l·d_r)/(|d_l||d_r|)，不动腿时 p→0 拿不到分（而不是白拿满分）；
    再乘一个幅度门 min(|d_l||d_r|/amp0², 1)，必须"既在动又反相"才给。与 sg_knee_amp（治锁死）配套使用。
    """
    asset = env.scene["robot"]
    key = id(env)
    ids = _LR_IDS.get(key)
    if ids is None:
        ids = asset.find_joints([".*_hipy_joint"])[0]          # 顺序由 find_joints 决定，下面按名字重排
        names = [asset.data.joint_names[i] for i in ids]
        order = [names.index(n) for n in ("fl_hipy_joint", "fr_hipy_joint", "hl_hipy_joint", "hr_hipy_joint")]
        ids = [ids[i] for i in order]
        _LR_IDS[key] = ids
    q = asset.data.joint_pos
    q = (q.torch if hasattr(q, "torch") else q)[:, ids]        # (N,4) fl fr hl hr
    b = getattr(env, "_sg_lr", None)
    if b is None or b["m"].shape != q.shape:
        b = {"m": q.clone(), "t": torch.zeros(env.num_envs, device=q.device), "step": -1}
        env._sg_lr = b
    if b["step"] != env.common_step_counter:
        b["step"] = env.common_step_counter
        t = env.episode_length_buf.float() * env.step_dt
        fresh = (t < b["t"] - 1e-6).unsqueeze(1)
        b["m"] = torch.where(fresh, q, b["m"]); b["t"] = t
        b["m"] = b["m"] + (q - b["m"]) * (env.step_dt / tau)
    d = q - b["m"]
    out = torch.zeros(env.num_envs, device=q.device)
    for a, c in ((0, 1), (2, 3)):                              # 前对、后对
        prod = d[:, a] * d[:, c]
        norm = d[:, a].abs() * d[:, c].abs()
        p = prod / (norm + 1e-6)                               # ∈[-1,1]，反相为负
        gate = torch.clamp(norm / (amp0 * amp0), 0.0, 1.0)      # 幅度门：不动就没分
        out = out + (-p) * gate
    return out * _gate(env, v_min, v_max, command_name)
