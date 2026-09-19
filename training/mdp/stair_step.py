"""楼梯"抬腿踩上去"步态项（09-12 晚）。作者："左右前轮轮流往上一个台阶踩，然后两个后轮跟上，这也是官方的""就是抬腿踩上去会不会比较好"。
真机 1139/1140 录像：右前上了台阶、左前没上 → 侧倾 32° 往左翻。仿真回放按帧算接触：StairN-C 9300 是"两只前轮一起跳上一级"
（前轮抬升过程中离地 58～68%，两前轮同时离地占 38%，顶着踢面滚的只 7～10%；后轮是滚上去的，离地 23～26%）——
两前轮腾空时只剩两只后轮撑着，一只落上一只没落上就没有第三点支撑。厂家包 s10_hmap_check_1 是左右交替抬腿、约 1.9 s 一级。
三项（后轮不管，滚上去或跟着踩都行）：
  ① front_pair_flight_penalty：两只前轮同时离地 → 罚 1（有速度指令时）；
  ② wheel_riser_push_penalty：轮子顶在立面上（接触力水平分量 > 0.5 × 垂直分量，climb_rewards 的 pressed）→ 每只罚 1，逼它抬而不是顶；
  ③ front_single_swing_reward：恰好一只前轮离地、且它比另一只前轮的触地点高 h_target 附近 → 奖励；只在翻越相位（轮高差 >12 cm 或前方有立面）算。
触地/高度/相位都用 climb_rewards._state（每步一次缓存），不加观测、不改部署维度。"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .climb_rewards import _state
from .trot_gait import _cmd_speed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def front_pair_flight_penalty(env: "ManagerBasedRLEnv", command_name: str = "base_velocity", v_min: float = 0.1) -> torch.Tensor:
    """两只前轮同时离地 → 1。"""
    s = _state(env)
    c = s["contact"]                                   # (N,4) fl fr hl hr
    both_air = (~c[:, 0]) & (~c[:, 1])
    return both_air.float() * (_cmd_speed(env, command_name) > v_min).float()


def wheel_riser_push_penalty(env: "ManagerBasedRLEnv", phase_only: bool = False, wheels=None) -> torch.Tensor:
    """顶在立面上的轮子数（0～4）。S1 起可只在翻越相位算（平地打滑/横推/转向时水平力大不该罚）。
    C4 起可只罚指定轮（wheels=(2, 3) 只罚后轮）——前轮蹬立面是正常的上墙动作，后轮顶立面就是"贴着拖上来"。"""
    s = _state(env)
    p = s["pressed"]
    r = (p[:, list(wheels)] if wheels is not None else p).float().sum(dim=1)
    return r * s["phase"].float() if phase_only else r


def front_single_swing_reward(env: "ManagerBasedRLEnv", h_target: float = 0.18, sigma: float = 0.08, phase_only: bool = True) -> torch.Tensor:
    """恰好一只前轮在空中，且它的触地点高度比另一只前轮高 h_target（一级 0.15 + 5 cm 余量）附近 → exp 奖励。"""
    s = _state(env)
    c = s["contact"]
    z = s["z"]                                         # (N,4) 轮子触地点高度
    one = c[:, 0] ^ c[:, 1]
    h = torch.where(~c[:, 0], z[:, 0] - z[:, 1], z[:, 1] - z[:, 0])
    r = torch.exp(-((h - h_target) ** 2) / (sigma ** 2)) * one.float()
    if phase_only:
        r = r * s["phase"].float()
    return r


def overspeed_penalty(env: "ManagerBasedRLEnv", margin: float = 0.10, command_name: str = "base_velocity",
                      asset_cfg=None, tau: float = 0.0) -> torch.Tensor:
    """机体前向速度超过指令 + margin 的部分（m/s）。9300 在楼梯上不听速度指令（仿真指令 0.2/0.3/0.5 实际都 0.50～0.54 m/s），
    作者要"上台阶再慢一点、像官方那样"，这项让速度旋钮真的管用。"""
    asset = env.scene["robot"]
    v = asset.data.root_lin_vel_b[:, 0]
    v = v.torch if hasattr(v, "torch") else v
    if tau > 0:                                              # S7：按 1 s 滑动平均算超速，允许迈步瞬间冲一下
        ema = getattr(env, "_ss_os_ema", None)
        if ema is None or ema.shape != v.shape: ema = v.clone()
        ema = ema + (v - ema) * (env.step_dt / tau); env._ss_os_ema = ema; v = ema
    cmd = env.command_manager.get_command(command_name)[:, 0]
    return torch.clamp(v.abs() - cmd.abs() - margin, min=0.0)   # E7 起按模长：指令 0 时任何 |v|>margin 都罚；前进指令下与原来相同


def wheel_torque_excess_penalty(env: "ManagerBasedRLEnv", tau_max: float = 5.0) -> torch.Tensor:
    """四个轮子 |施加力矩| 超过 tau_max 的部分之和（N·m）。官方上楼梯轮力矩 p95 只有 2.5 N·m（腿抬、轮辅助）；9300 真机左前轮顶台沿到 14 N·m 上限。"""
    from .climb_rewards import _T
    asset = env.scene["robot"]
    ids = getattr(env, "_ss_wheel_joint_ids", None)
    if ids is None:                                   # 关节下标（12~15），不是 body 下标——第一版用了 body 下标越界
        ids, names = asset.find_joints(".*_wheel_joint")
        env._ss_wheel_joint_ids = ids
        print(f"[stair_step] 轮子关节下标 {ids} {names}", flush=True)
    tau = _T(asset.data.applied_torque)[:, ids]
    return torch.clamp(tau.abs() - tau_max, min=0.0).sum(dim=1)


def multi_flight_penalty(env: "ManagerBasedRLEnv", command_name: str = "base_velocity", v_min: float = 0.1) -> torch.Tensor:
    """离地轮子数超过 1 的部分（官方上楼梯视频：任何时刻 ≥3 轮着地，两腿同时腾空 0 次）。有速度指令时算。"""
    s = _state(env)
    n_air = (~s["contact"]).float().sum(dim=1)
    return torch.clamp(n_air - 1.0, min=0.0) * (_cmd_speed(env, command_name) > v_min).float()


def front_lift_each_reward(env: "ManagerBasedRLEnv", h_target: float = 0.18, sigma: float = 0.06, phase_only: bool = True, wheels=(0, 1),
                           v_min: float = 0.0, command_name: str = "base_velocity", kernel: str = "gauss") -> torch.Tensor:
    """每只前轮各自算：腾空时相对**自己起抬点**（最近一次触地时的轮底高）抬高接近 h_target → 奖励，两只前轮分别累加。
    E2_10300 逐级核对：左前过台沿悬空 80%、净空 3～4 cm（真抬），右前只有 30%、净空≈0（贴着台沿滚上去）——
    因为 front_single_swing_reward 的高度是"相对另一只前轮"，第二只上的腿永远够不到目标，只有先抬的那只拿到奖励。"""
    s = _state(env)
    c = s["contact"]; z = s["z"]
    last = getattr(env, "_ss_last_contact_z", None)
    if last is None or last.shape[0] != z.shape[0]:
        last = z.clone(); env._ss_last_contact_z = last
    last = torch.where(c, z, last); env._ss_last_contact_z = last
    h = z - last                                        # (N,4) 相对起抬点
    if kernel == "linear":                              # S1：从 0 起就有梯度（高斯 0.18/σ0.06 在 0～2 cm 处没有梯度，E6 后轮因此学不到）
        r = torch.clamp(h / h_target, 0.0, 1.0) * (~c).float()
    else:
        r = torch.exp(-((h - h_target) ** 2) / (sigma ** 2)) * (~c).float()
    r = r[:, list(wheels)].sum(dim=1)                  # E6 起可选后轮 (2, 3)；默认前两轮，E3/E5 行为不变
    if phase_only:
        r = r * s["phase"].float()
    if v_min > 0:                                       # E7：没指令不奖励抬腿
        r = r * (_cmd_speed(env, command_name) > v_min).float()
    return r


def zero_cmd_motion_penalty(env: "ManagerBasedRLEnv", v_cmd_max: float = 0.05, w_ang: float = 0.5, command_name: str = "base_velocity") -> torch.Tensor:
    """指令为 0（xy 模长与 |wz| 都 < v_cmd_max）时，机体 |v_xy| + w_ang·|wz| → 罚（E7：作者要楼梯专家"给 0 能原地站住"，楼梯上不溜不转）。"""
    c = env.command_manager.get_command(command_name)
    zero = (torch.linalg.norm(c[:, :2], dim=1) < v_cmd_max) & (c[:, 2].abs() < v_cmd_max)
    asset = env.scene["robot"]
    v = asset.data.root_lin_vel_b; v = v.torch if hasattr(v, "torch") else v
    w = asset.data.root_ang_vel_b; w = w.torch if hasattr(w, "torch") else w
    return (torch.linalg.norm(v[:, :2], dim=1) + w_ang * w[:, 2].abs()) * zero.float()


def air_wheel_spin_penalty(env: "ManagerBasedRLEnv", w_max: float = 9.0) -> torch.Tensor:
    """腾空轮的 |轮速| 超过 w_max 的部分之和（rad/s）。官方楼梯摆动中轮子只前转 6～9 rad/s，我方 16～19 是"甩腿"观感来源（S1）。"""
    s = _state(env)
    asset = env.scene["robot"]
    ids = getattr(env, "_ss_wheel_joint_ids", None)
    if ids is None:
        ids, _ = asset.find_joints(".*_wheel_joint"); env._ss_wheel_joint_ids = ids
    jv = asset.data.joint_vel; jv = (jv.torch if hasattr(jv, "torch") else jv)[:, ids]
    return (torch.clamp(jv.abs() - w_max, min=0.0) * (~s["contact"]).float()).sum(dim=1)


def roll_penalty(env: "ManagerBasedRLEnv", phase_only: bool = False) -> torch.Tensor:
    """只罚侧倾（重力投影 y 分量平方），不罚俯仰——楼梯上机身要跟坡低头 15°，flat_orientation 会把它一起罚掉（S1）。
    C4 起可只在翻越相位算：C3 全程 −15 的实测读数只有 0.004（翻越那 1 s 被 30 s 平地稀释掉了），
    相位内加大权重才压得住翻台阶时的 20° 侧倾；平地侧倾本来就有 flat_orientation_l2 管。"""
    asset = env.scene["robot"]
    g = asset.data.projected_gravity_b; g = g.torch if hasattr(g, "torch") else g
    r = torch.square(g[:, 1])
    return r * _state(env)["phase"].float() if phase_only else r


def vy_track_reward(env: "ManagerBasedRLEnv", sigma: float = 0.15, vy_min: float = 0.1, command_name: str = "base_velocity") -> torch.Tensor:
    """横移专项：|cmd_vy| > vy_min 时按机体 vy 与指令的误差给 exp 奖励（S2：横移在 E7/E8/S1 共 3000 步里一直是 0，σ0.3 的合成跟踪给不出足够梯度）。"""
    c = env.command_manager.get_command(command_name)
    asset = env.scene["robot"]; v = asset.data.root_lin_vel_b; v = v.torch if hasattr(v, "torch") else v
    r = torch.exp(-((v[:, 1] - c[:, 1]) ** 2) / (sigma ** 2))
    return r * (c[:, 1].abs() > vy_min).float()


def lift_lr_symmetry_penalty(env: "ManagerBasedRLEnv", command_name: str = "base_velocity", v_min: float = 0.1, deadband: float = 0.0) -> torch.Tensor:
    """左右不对称罚（S3）：同一对轮子（前对、后对）在翻越相位里各自"最近一次腾空的最大抬高"之差。D 两前都抬，听指令线右前退化成贴沿滚（2 cm）而左前 8 cm。"""
    s = _state(env); c = s["contact"]; z = s["z"]
    last = getattr(env, "_ss_last_contact_z", None)
    if last is None or last.shape[0] != z.shape[0]:
        last = z.clone(); env._ss_last_contact_z = last
    h = torch.clamp(z - last, min=0.0) * (~c).float()                      # 当前腾空高度（相对起抬点）
    pk = getattr(env, "_ss_lift_peak", None)
    if pk is None or pk.shape != h.shape: pk = torch.zeros_like(h)
    pk = torch.where(c, pk, torch.maximum(pk, h))                            # 腾空期间累计峰值
    ended = c & (pk > 0)                                                     # 落地瞬间：把这次峰值存为"最近一次抬高"
    lastpk = getattr(env, "_ss_last_peak", None)
    if lastpk is None or lastpk.shape != h.shape: lastpk = torch.zeros_like(h)
    lastpk = torch.where(ended, pk, lastpk); pk = torch.where(c, torch.zeros_like(pk), pk)
    env._ss_lift_peak = pk; env._ss_last_peak = lastpk
    d = torch.clamp((lastpk[:, 0] - lastpk[:, 1]).abs() - deadband, min=0.0) + torch.clamp((lastpk[:, 2] - lastpk[:, 3]).abs() - deadband, min=0.0)   # S10：差值小于 deadband 不罚
    return d * s["phase"].float() * (_cmd_speed(env, command_name) > v_min).float()


def lift_pair_min_reward(env: "ManagerBasedRLEnv", h_cap: float = 0.15, command_name: str = "base_velocity", v_min: float = 0.1) -> torch.Tensor:
    """按对取"较低那只轮最近一次抬高"给分（前对 + 后对，各封顶 h_cap）：只奖励把低的一侧抬起来，不把高的拉下来（S5；S4 的绝对差对称罚把左前从 11 拉到 4、左后从 7 拉到 1）。"""
    s = _state(env)
    lastpk = getattr(env, "_ss_last_peak", None)
    if lastpk is None: return torch.zeros(env.num_envs, device=env.device)
    lo = torch.minimum(lastpk[:, 0], lastpk[:, 1]).clamp(max=h_cap) / h_cap + torch.minimum(lastpk[:, 2], lastpk[:, 3]).clamp(max=h_cap) / h_cap
    return lo * s["phase"].float() * (_cmd_speed(env, command_name) > v_min).float()


def underspeed_penalty(env: "ManagerBasedRLEnv", margin: float = 0.05, cmd_min: float = 0.1, command_name: str = "base_velocity") -> torch.Tensor:
    """欠速罚（S6）：前向指令 > cmd_min 时，指令 − 机体 vx − margin 的正部分。治 0.2 档"蠕行"（0.024 m/s）——停滞罚只对完全不动生效，跟踪 σ0.3 又太平。"""
    asset = env.scene["robot"]; v = asset.data.root_lin_vel_b; v = v.torch if hasattr(v, "torch") else v
    cmd = env.command_manager.get_command(command_name)[:, 0]
    return torch.clamp(cmd - v[:, 0] - margin, min=0.0) * (cmd > cmd_min).float()


def roll_rate_penalty(env: "ManagerBasedRLEnv", phase_only: bool = False) -> torch.Tensor:
    """机体 roll 角速度平方（楼梯上的晃动，S13：作者 09-14 "楼梯上要稳"）。phase_only 时只在翻越相位算。"""
    asset = env.scene["robot"]; w = asset.data.root_ang_vel_b; w = w.torch if hasattr(w, "torch") else w
    r = torch.square(w[:, 0])
    return r * _state(env)["phase"].float() if phase_only else r


def yaw_track_stairs_reward(env: "ManagerBasedRLEnv", sigma: float = 0.25, command_name: str = "base_velocity", phase_only: bool = True) -> torch.Tensor:
    """楼梯上的转向跟踪专项（S13）：|cmd_wz| > 0.05 时按机体 wz 与指令误差给 exp 奖励，只在翻越相位（平地转向由通用项管）。"""
    c = env.command_manager.get_command(command_name)
    asset = env.scene["robot"]; w = asset.data.root_ang_vel_b; w = w.torch if hasattr(w, "torch") else w
    r = torch.exp(-((w[:, 2] - c[:, 2]) ** 2) / (sigma ** 2)) * (c[:, 2].abs() > 0.05).float()
    return r * _state(env)["phase"].float() if phase_only else r


def edge_clearance_reward(env: "ManagerBasedRLEnv", wheels=(2, 3), cap: float = 0.08, phase_only: bool = True) -> torch.Tensor:
    """越沿净空（C4）：翻越相位内，腾空轮的触地点高度超过「记住的台沿高度 wall_mem」的部分 / cap，封顶 1，逐轮累加。

    为什么不用 front_lift_each_reward（相对自己起抬点抬高）：后轮贴着立面往上蹭时会不断轻触立面，
    起抬点被一路刷新，抬高量永远≈0，这一项读不出区别（C3 实测原始值 0.037）。官方的动作是
    「两条后腿折叠跨过沿」——轮子的绝对抬高不大，关键是**相对台沿**的净空，所以基准取 wall_mem。
    MuJoCo 里对应的量就是逐级核对表里的"越沿净空"（C3 后轮 0.001～0.064 m，前轮 0.21～0.28 m）。

    **09-14 23:1x 补丁（C7 事故）**：原版只要求"后轮腾空 + 高过台沿 + 处于翻越相位"，
    机器人**登顶之后**这三条同时满足——站在台面上抬后腿，z 天然高过 wall_mem，encounter 因轮高差未退出，
    于是每步白拿 2 分。C7 里该项从 +0.54 一路刷到 **+92.5**（170 倍），
    同期真·爬高进展回落到起点、速度跟踪下降、侧倾恶化，最终 10200 点登顶 0/3、平地零指令冲出 3.85 m。
    修法：加 `crossing` 条件——这只轮的**起抬点必须低于台沿 5 cm**，站在台面上抬腿不给分。
    """
    s = _state(env)
    c = s["contact"]; z = s["z"]
    last = getattr(env, "_ss_last_contact_z", None)          # 每只轮"最近一次触地时的轮底高"
    if last is None or last.shape != z.shape:
        last = z.clone()
    last = torch.where(c, z, last); env._ss_last_contact_z = last
    wm = s["wall_mem"].unsqueeze(1)
    crossing = last < (wm - 0.05)                            # C7 补丁①：只认"从台沿以下起抬"的轮
    h = (z - wm) / cap
    band = (h > 0.0) & (h < 1.5)                             # C7 补丁②：只在台沿上方 1.5×cap 以内给分
    #   ①单独不够：轮子若一直不落地，"起抬点"会永远停在爬台前那个低值，crossing 恒真 ——
    #   补丁①后刷分点仍读到 24.9（诚实点 0.92）。加上高度带之后，站在台面上把腿举得老高也拿不到分。
    r = (torch.clamp(h, 0.0, 1.0) * (~c).float() * crossing.float() * band.float())[:, list(wheels)].sum(dim=1)
    return r * s["encounter"].float() if phase_only else r


def climb_rate_penalty(env: "ManagerBasedRLEnv", v_max: float = 0.6, phase_only: bool = True) -> torch.Tensor:
    """翻越相位内机身**世界系垂直上升速度**超过 v_max 的部分（平方）。作者 09-14 21:3x："限制翻越速度，按官方 1 秒来。"

    官方翻高台 0.33 m 用 1.0 s（0.9~1.2），|pitch|>30° 持续 ≤0.45 s，隐含平均上升 ≈0.33 m/s。
    C4_9599 同口径（起抬→落平，|pitch| 回到 5° 以内为界）只用 **0.48 s**，vz 峰 1.39~1.57 m/s、均 0.55~0.58 m/s，
    |pitch|>30° 只持续 0.08 s —— 是"冲上去"而不是"折腿跨上去"。
    这正是后轮越沿净空一直上不去的机理：抬头尖峰只有 0.08 s，后腿来不及折叠跨沿，只能贴着立面蹭。
    v_max 0.6 = 允许瞬时冲一下（官方均值 0.33 的约 2 倍），但把 1.4~1.6 m/s 的猛冲罚掉。
    """
    s = _state(env)
    v = env.scene["robot"].data.root_lin_vel_w
    v = v.torch if hasattr(v, "torch") else v
    r = torch.clamp(v[:, 2] - v_max, min=0.0) ** 2
    return r * s["encounter"].float() if phase_only else r


def pitch_hold_reward(env: "ManagerBasedRLEnv", lo_deg: float = 25.0, hi_deg: float = 45.0,
                      vz_min: float = 0.05, phase_only: bool = True,
                      kernel: str = "band", tgt_deg: float = 0.0, fade_deg: float = 5.0,
                      budget_s: float = 0.0, gate: str = "encounter") -> torch.Tensor:
    """翻越相位内，机身抬头角落在 [lo, hi] 度**且机身在上升**的每一步给 1（C8，09-14 22:5x）。

    为什么要正向塑形而不是继续加限速罚（C7 实测 −250 完全无效，用时 0.48 → 0.39~0.51 反而更快）：
    算一笔账——一次翻越，冲刺的限速罚约 25 分（(1.4−0.4)² × 0.1 s × 250），而爬高进展给 130+ 分；
    罚金轻是一方面，更根本的是"慢慢翻"需要折腿→蹬伸→跨沿一整套新控制策略，冲上去是现成的局部最优，
    **罚"冲得快"不等于教会"慢慢翻"，只会让它更不想靠近台子**。
    官方六次事件 |pitch|>30° 能保持到 0.45 s，我方只有 0.07~0.09 s —— 抬头是个尖峰而不是平台，
    后腿因此来不及折叠跨沿。本项直接奖励"抬头姿态的保持时长"，把尖峰拉成平台。
    要求 vz > vz_min 是防止它停在台子前一直仰着不动骗分。
    """
    s = _state(env)
    asset = env.scene["robot"]
    g = asset.data.projected_gravity_b; g = g.torch if hasattr(g, "torch") else g
    pitch = torch.atan2(g[:, 0], -g[:, 2]).abs() * (180.0 / 3.141592653589793)
    v = asset.data.root_lin_vel_w; v = v.torch if hasattr(v, "torch") else v
    rising = (v[:, 2] > vz_min).float()
    if kernel == "ramp":
        # S15（09-15 11:1x）：**硬档判据在档外没有梯度**。
        #   楼梯线 `ss_pitch_slope` 用 band [8°,20°]，而爬楼段实际抬头只有 5°（官方 15.4°）→ 在档外，
        #   读数 0.147 ÷ 权重 10 = 原始值 **0.015（只有 1.5% 的步在档内）**，对比 track_lin_vel +10.7 差 70 倍。
        #   计划原先写的"权重 10 太弱，下一轮加码"是错的：**值恒为 0 时权重乘多少还是 0**。
        #   （与 T11 tent 核、T12 时间归一同一类教训：先看当前工作点有没有梯度，再谈权重。）
        # ramp：lo→tgt 线性 0→1，tgt→hi 保持 1，hi 之后在 fade_deg 内线性退到 0（不鼓励过度抬头）。
        up = torch.clamp((pitch - lo_deg) / max(tgt_deg - lo_deg, 1e-3), 0.0, 1.0)
        down = torch.clamp((hi_deg + fade_deg - pitch) / max(fade_deg, 1e-3), 0.0, 1.0)
        r = torch.minimum(up, down) * rising
    else:
        r = ((pitch > lo_deg) & (pitch < hi_deg)).float() * rising
    if budget_s > 0:
        # C23（09-15 11:4x，作者"按官方的来"）：**按步给分 = 赖得越久拿得越多**。
        # 官方录包（ref_climb34_50hz，以 pitch 首次>10° 为 t=0）：起抬后 0.6~1.0 s 时 pitch 已落到 **+4.5°**；
        # 我方同期还挂在 **+29.3°**（峰 43.9）。对应验收量：|pitch|>30° 持续 官方 ≤0.45 s / 我方 0.66~0.81 s，
        # 落平过冲 官方 ≤+12° / 我方 +16~20°。
        # 改法：每次遭遇只给前 budget_s 秒的在档时间发分，超出不再发（**不加对抗性惩罚**，只是不再奖励拖延）。
        n = r.shape[0]
        st = getattr(env, "_pitch_budget", None)
        if st is None or st.shape[0] != n:
            st = torch.zeros(n, device=r.device); env._pitch_budget = st
        step = getattr(env, "_pitch_budget_step", -1)
        if step != env.common_step_counter:
            env._pitch_budget_step = env.common_step_counter
            enc = s["encounter"]
            st = torch.where(enc, st + r * env.step_dt, torch.zeros_like(st))   # 离开遭遇即清零
            env._pitch_budget = st
        r = r * (env._pitch_budget < budget_s).float()
    if gate == "after_press":
        # C30（09-15 17:1x）：原闸门是 `encounter`，而 encounter = tall_wall | (split & mem>0) —— **纯地形量，看见墙就为真**。
        # 于是抬头奖励在**接近途中**就开始付钱，策略学会"还没到墙就仰起来"。
        # 实测（官方 seg0 对照 C29_13999，同一 t 轴）：压墙瞬间官方 −3.1°、我方 −32.2°；
        # 抬头峰值我方早 0.64 s；扫时间平移最优 −0.68~−0.70 s（RMS 0.781→0.626，降 20%）。
        # 改成：**前轮真的顶在立面上（pressed = 接触且水平分力 > 0.5×垂直分力）之后才开闸**，
        # 并锁存到本次遭遇结束 —— 锁存必须有，官方抬头峰值在压墙后 +0.78 s，纯 pressed 会太早关。
        n = r.shape[0]
        lat = getattr(env, "_pitch_gate", None)
        if lat is None or lat.shape[0] != n:
            lat = torch.zeros(n, dtype=torch.bool, device=r.device); env._pitch_gate = lat
        if getattr(env, "_pitch_gate_step", -1) != env.common_step_counter:
            env._pitch_gate_step = env.common_step_counter
            lat = (lat | s["pressed"][:, :2].any(dim=1)) & s["encounter"]
            env._pitch_gate = lat
        return r * env._pitch_gate.float()
    return r * s["encounter"].float() if phase_only else r


def wheel_slip_penalty(env: "ManagerBasedRLEnv", tol: float = 0.30, radius: float = 0.081) -> torch.Tensor:
    """触地轮的"轮面速度 vs 机身速度"之差超过 tol 的部分之和（m/s）。治"轮子空转打滑、既不推进也不转向"。

    诊断依据（09-15 02:5x，C12_10799 高台专家平地探针）：
      转 +0.5 → 实际 wz **+0.046**，左轮均 −1.79 / 右轮均 −32.62，**左右差 +30.83 rad/s**；
      零指令 → wz +0.001，右轮仍以 **−14.21 rad/s** 空转（轮面速度 1.15 m/s），而机身只走 0.1 m/s。
      差速转向的几何：轮距 0.362 / 轮半径 0.081 → 转 0.5 rad/s **只需要左右差 2.2 rad/s**。
      即差速已是需要量的 6~14 倍却一点不转 —— **信号早就到了（track_ang_vel_z_exp 读数 +5），执行端废掉了**。
      这种情况下继续加转向奖励权重毫无意义。
    对照：楼梯专家 F 有 `wheel_drive_balance −5` 与 `ss_air_spin −0.3`，转 ±0.5 能给出 +0.31/−0.56；
      高台专家这两项**一个都没有**，于是学出了这个无效空转。
    `air_wheel_spin_penalty` 只罚**腾空**轮，治不了**贴地打滑**，所以另写本项。
    轮速符号按 MJCF 约定"轮正转 = 后退"，故前向轮面速度 = −ω·R。
    """
    s = _state(env)
    asset = env.scene["robot"]
    ids = getattr(env, "_ss_wheel_joint_ids", None)
    if ids is None:
        ids, _ = asset.find_joints(".*_wheel_joint"); env._ss_wheel_joint_ids = ids
    jv = asset.data.joint_vel; jv = (jv.torch if hasattr(jv, "torch") else jv)[:, ids]
    v = asset.data.root_lin_vel_b; v = (v.torch if hasattr(v, "torch") else v)[:, 0:1]
    slip = ((-jv * radius) - v).abs()
    return (torch.clamp(slip - tol, min=0.0) * s["contact"].float()).sum(dim=1)


def wheel_diff_sign_reward(env: "ManagerBasedRLEnv", wz_min: float = 0.05, radius: float = 0.081,
                           track_w: float = 0.362, amp0: float = 1.0,
                           contact_only: bool = False,
                           command_name: str = "base_velocity") -> torch.Tensor:
    """左右轮差速**方向**与指令 wz 一致时给分（C14，09-15 03:3x；C17 补 contact_only）。

    为什么必须用正向塑形而不是继续加惩罚（C13 用 400 迭代证明了）：
      差速转向几何：轮距 0.362 / 轮半径 0.081 → 转 0.5 rad/s 只需左右差 2.2 rad/s。
      C13 起点/中点/末点的实测左右差（左 fl,hl 均 − 右 fr,hr 均）：
        零指令  +13.81 → −2.27 → −3.22   （**无效差速治好了**，三个轮子惩罚有效）
        转 +0.5 +30.83 → +25.95 → +18.08 （符号对、大小错 8 倍，wz 只到 +0.078）
        转 −0.5 +13.99 → +8.27  → **+8.28**（**符号自始至终是反的**）
      → 策略根本没有"反转差速"这个概念，只会把固定的右侧空转调大调小。
      查证：高台专家从 C4 起转向指令范围就是 (0,0)，**直到 C5 才第一次见到转向指令**——
      这条线从来没学过差速转向，是从零建立新模态，光靠罚无效动作建立不起来。
    本项只教"方向"，大小交给已有的 `track_ang_vel_z_exp` 和 `wheel_drive_balance`。
    轮速符号按 MJCF 约定"轮正转 = 后退"：左转(wz>0) 需要右侧前进更快 → ω_L − ω_R > 0。

    **contact_only（C17，09-15 05:4x 加）——第三次刷分的堵口**：
      C13/C15/C16 三代实测四轮转速（转 ±0.5，单位 rad/s）：
        C13_11198 转+0.5  fl −0.52  fr −1.05  hl −0.74  **hr −36.37**
        C15_12000 转−0.5  fl −13.49 fr +0.89  hl −2.89  **hr +14.59**
        C16_12200 转+0.5  fl +0.27  fr −3.00  hl −1.98  **hr −18.82**
      整个"差速"由 hr 一个轮子贡献，另外三个 |ω|<3。同期关节角：转向时 hr hipy +1.55 / 膝 −1.90
      （零指令 +0.57 / −1.27）→ **策略把右后腿折起来、让悬空的轮子空转**来满足本项的符号与幅度门。
      实测代价比：farming 收益 +4.2（weight 6.0）vs `cc_air_spin` −0.37（weight −0.05）= 8 倍，必然被刷。
      对照组：前进 0.5 时四轮 10.1/9.9/9.7/9.7 完全均匀 —— **直行是健康的，只有转向指令触发退化解**。
      堵法与 cc_edge_clr(crossing/band)、cc_pitch_hold(相位门) 同源：**改定义，不是改权重**。
        ① 左右均值只对**触地**轮取（悬空轮贡献 0，也不进分母）；
        ② 整项乘触地轮比例 `contact.mean()`，抬腿换分直接打折。
    """
    asset = env.scene["robot"]
    ids = getattr(env, "_ss_wheel_joint_ids", None)
    if ids is None:
        ids, _ = asset.find_joints(".*_wheel_joint"); env._ss_wheel_joint_ids = ids
    jv = asset.data.joint_vel; jv = (jv.torch if hasattr(jv, "torch") else jv)[:, ids]   # fl fr hl hr
    if contact_only:
        ct = _state(env)["contact"].float()                        # (N,4) fl fr hl hr
        nL = (ct[:, 0] + ct[:, 2]).clamp(min=1.0)
        nR = (ct[:, 1] + ct[:, 3]).clamp(min=1.0)
        d = (jv[:, 0] * ct[:, 0] + jv[:, 2] * ct[:, 2]) / nL - (jv[:, 1] * ct[:, 1] + jv[:, 3] * ct[:, 3]) / nR
        grounded = ct.mean(dim=1)                                  # 抬腿换分打折
    else:
        d = 0.5 * (jv[:, 0] + jv[:, 2]) - 0.5 * (jv[:, 1] + jv[:, 3])
        grounded = torch.ones_like(d)
    c = env.command_manager.get_command(command_name)
    d_star = c[:, 2] * track_w / radius
    p = (d * d_star) / (d.abs() * d_star.abs() + 1e-6)            # ∈[-1,1]，同号为正
    gate = torch.clamp(d.abs() / amp0, 0.0, 1.0)                  # 不动差速拿不到分
    return p * gate * grounded * (c[:, 2].abs() > wz_min).float()

def wheel_diff_track_reward(env: "ManagerBasedRLEnv", sigma: float = 4.0, wz_min: float = 0.05,
                            radius: float = 0.081, track_w: float = 0.362,
                            command_name: str = "base_velocity") -> torch.Tensor:
    """左右轮差速**跟踪目标值** d* = wz·轮距/轮半径 的 exp 奖励（C15，09-15 04:2x）。

    接续 C14：`wheel_diff_sign_reward` 只教方向，C14 实测把右转差速从 +8.28 翻到 −11.21（方向学会了），
    但**大小仍错 5~7 倍**（±10~20 vs 需要 ±2.2），差速过大时轮子打滑，实际 wz 只到 +0.198/−0.102。
    方向既已建立，就有基础直接跟踪大小——σ 取 4.0 rad/s：当前误差约 8~18 rad/s 时仍有可用梯度，
    收到 ±2.2 附近时奖励接近满分。
    轮速符号按 MJCF 约定"轮正转 = 后退"：左转(wz>0) 需 ω_L − ω_R > 0。
    """
    asset = env.scene["robot"]
    ids = getattr(env, "_ss_wheel_joint_ids", None)
    if ids is None:
        ids, _ = asset.find_joints(".*_wheel_joint"); env._ss_wheel_joint_ids = ids
    jv = asset.data.joint_vel; jv = (jv.torch if hasattr(jv, "torch") else jv)[:, ids]
    d = 0.5 * (jv[:, 0] + jv[:, 2]) - 0.5 * (jv[:, 1] + jv[:, 3])
    c = env.command_manager.get_command(command_name)
    d_star = c[:, 2] * track_w / radius
    return torch.exp(-((d - d_star) ** 2) / (sigma ** 2)) * (c[:, 2].abs() > wz_min).float()


def wheel_diff_excess_penalty(env: "ManagerBasedRLEnv", tol: float = 2.0, radius: float = 0.081,
                              track_w: float = 0.362, contact_only: bool = False, skip_climb: bool = False,
                              command_name: str = "base_velocity") -> torch.Tensor:
    """左右轮差速**超过指令所需**的部分（线性，处处有梯度）。

    C15/C16 曾用它收差速大小，**两代无效**——当时量的是"左右**均值**之差"，被**一个悬空空转的轮子**
    完全主导（hr 单轮 −36 rad/s、另三轮 <3），罚的是个假量。C17 因此删掉了它。

    **C21 重新启用，且必须 contact_only=True**，治的是另一个问题：台面直行绕圈。
    台面上指令 vx=0.4、**wz=0** 时左右轮持续不对称，实测：
      C19_12400  fl −4.67 fr −4.24 hl −4.44 **hr −1.53** → 左 −4.56 右 −2.88 差 **−1.67** → wz −0.040
      C20_12800  fl −7.01 fr −2.82 hl −4.04 **hr −1.03** → 左 −5.52 右 −1.93 差 **−3.60** → wz **−0.245**
    C20_12800 在台面上 **27 s 绕一圈**（yaw +6.8→−159→+166→…，横移 −1.78 m），平地直行 0.5 m/s 偏 41°，
    末点 12999 更糟（+61°）。偏置从 C17 就有（−0.045），C20 放大了 5 倍。
    先例：C13 用同类惩罚把**零指令**无效差速从 +13.81 治到 −3.22 ——
    对"抑制不该有的差速"是有效的（与"光靠罚建立不起新行为"不矛盾：直行本来就会，要去掉的是偏置）。
    """
    asset = env.scene["robot"]
    ids = getattr(env, "_ss_wheel_joint_ids", None)
    if ids is None:
        ids, _ = asset.find_joints(".*_wheel_joint"); env._ss_wheel_joint_ids = ids
    jv = asset.data.joint_vel; jv = (jv.torch if hasattr(jv, "torch") else jv)[:, ids]   # fl fr hl hr
    if contact_only:
        ct = _state(env)["contact"].float()
        nL = (ct[:, 0] + ct[:, 2]).clamp(min=1.0); nR = (ct[:, 1] + ct[:, 3]).clamp(min=1.0)
        d = (jv[:, 0] * ct[:, 0] + jv[:, 2] * ct[:, 2]) / nL - (jv[:, 1] * ct[:, 1] + jv[:, 3] * ct[:, 3]) / nR
    else:
        d = 0.5 * (jv[:, 0] + jv[:, 2]) - 0.5 * (jv[:, 1] + jv[:, 3])
    c = env.command_manager.get_command(command_name)
    need = (c[:, 2] * track_w / radius).abs()
    r = torch.clamp(d.abs() - need - tol, min=0.0)
    if skip_climb:
        # C22：翻越那 1 s 左右轮**本来就该不对称**（一侧蹬立面、一侧在地面），
        # C21 不分相位地罚，登顶直接掉到 0/3（两个检查点都是），守卫已自动停训。
        # 用纯地形触发 + 限时的 climb_exempt_mask 关掉它——该掩码不含任何机器人姿态量，
        # 不会被"轮速差大"这个被罚量自己满足（纪律 20）。
        from .climb_rewards import climb_exempt_mask
        r = r * (~climb_exempt_mask(env)).float()
    return r
