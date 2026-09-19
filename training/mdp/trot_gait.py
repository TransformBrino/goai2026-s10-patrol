"""踏步模式（厂家"踢踏舞"步态）奖励，无相位观测、无需改部署观测维度。目标数字来自真机 09-09 四段踏步（s10_dev 分析）：
  对角小跑（fl-hr 同步、fl-fr 反相）；抬腿 9–11 cm；站姿腿长 0.24–0.28（机身 0.34–0.36）；步频 1.1 Hz(≤1 m/s) → 2.07 Hz(1.8 m/s)，
  即单腿腾空时长 T ≈ 0.45 s → 0.24 s；平地推满 2.2 m/s；轮力矩常规 <4 N·m，尖峰 19.7。"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .climb_rewards import _state, _ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd_speed(env, command_name="base_velocity"):
    c = env.command_manager.get_command(command_name)
    return torch.linalg.norm(c[:, :2], dim=1)


ROUGH_NAMES = ("random_rough", "hf_random", "slope", "boxes", "real_patch", "stepping", "discrete")


def _rough_mask(env) -> torch.Tensor:
    """本 env 所在子地形是否"粗糙"（非 flat）。列→名字映射与课程模块共用；terrain_types 每个 env 固定。"""
    m = getattr(env, "_trot_rough_mask", None)
    if m is None or env.common_step_counter % 500 == 0:
        from .curriculums import _build_col2name
        terrain = env.scene.terrain
        names = getattr(env, "_climb_col2name", None)
        if names is None:
            names = _build_col2name(terrain); env._climb_col2name = names
        ty = terrain.terrain_types
        ty = ty.torch if hasattr(ty, "torch") else ty
        rough_cols = torch.tensor([c for c, nm in names.items() if any(k in nm.lower() for k in ROUGH_NAMES)], device=ty.device, dtype=ty.dtype)
        m = torch.isin(ty, rough_cols) if rough_cols.numel() else torch.zeros_like(ty, dtype=torch.bool)
        env._trot_rough_mask = m
    return m


def _gate(env, rough_only: bool) -> torch.Tensor:
    return _rough_mask(env).float() if rough_only else torch.ones(env.num_envs, device=env.device)


def diag_sync_reward(env: "ManagerBasedRLEnv", v_min: float = 0.3, command_name: str = "base_velocity", rough_only: bool = False, v_max: float = 99.0) -> torch.Tensor:
    """对角步态：fl 与 hr、fr 与 hl 触地状态一致得分，同侧一致扣分；四轮全触地或全离地时为 0。只在有速度指令时算。"""
    s = _state(env)
    c = s["contact"].float()                     # (N,4) fl fr hl hr
    same_diag = 0.5 * ((c[:, 0] == c[:, 3]).float() + (c[:, 1] == c[:, 2]).float())
    same_side = 0.5 * ((c[:, 0] == c[:, 1]).float() + (c[:, 2] == c[:, 3]).float())
    sp = _cmd_speed(env, command_name); moving = (sp > v_min) & (sp < v_max)
    return (same_diag - same_side) * moving.float() * _gate(env, rough_only)


def swing_clearance_reward(env: "ManagerBasedRLEnv", h_target: float = 0.09, sigma: float = 0.04, v_min: float = 0.3, command_name: str = "base_velocity", rough_only: bool = False, v_max: float = 99.0) -> torch.Tensor:
    """腾空轮离地高度接近 h_target（真机抬腿 9–11 cm）。高度 = 轮心 z − 最低轮心 z。"""
    s = _state(env)
    z = s["z"]                                    # (N,4) 轮心世界 z
    h = z - z.min(dim=1, keepdim=True).values
    swing = ~s["contact"]
    r = torch.exp(-((h - h_target) ** 2) / (sigma ** 2)) * swing.float()
    n = swing.float().sum(dim=1).clamp(min=1.0)
    sp = _cmd_speed(env, command_name); moving = (sp > v_min) & (sp < v_max)
    return (r.sum(dim=1) / n) * (swing.any(dim=1) & moving).float() * _gate(env, rough_only)


def air_time_target_reward(env: "ManagerBasedRLEnv", v_min: float = 0.3, sigma: float = 0.08, command_name: str = "base_velocity",
                           sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"), rough_only: bool = False, v_max: float = 99.0) -> torch.Tensor:
    """落地瞬间按"上一次腾空时长"给分：目标 T(v) = clip(0.45 − 0.13·v, 0.22, 0.45)（真机步频 1.1→2.07 Hz 的半周期）。"""
    cs = env.scene.sensors[sensor_cfg.name]
    ids = _ids(env)["c"]
    first = cs.compute_first_contact(env.step_dt)[:, ids]
    last_air = cs.data.last_air_time
    last_air = (last_air.torch if hasattr(last_air, "torch") else last_air)[:, ids]
    v = _cmd_speed(env, command_name)
    T = torch.clamp(0.45 - 0.13 * v, 0.22, 0.45).unsqueeze(1)
    r = torch.exp(-((last_air - T) ** 2) / (sigma ** 2)) * first.float()
    return r.sum(dim=1) * ((v > v_min) & (v < v_max)).float() * _gate(env, rough_only)


def no_step_penalty(env: "ManagerBasedRLEnv", v_min: float = 0.5, t_max: float = 1.0, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                    command_name: str = "base_velocity", rough_only: bool = False, v_max: float = 99.0) -> torch.Tensor:
    """有速度指令却四轮持续触地超过 t_max（纯滚不踏步）→ 罚 1。"""
    cs = env.scene.sensors[sensor_cfg.name]
    ids = _ids(env)["c"]
    ct = cs.data.current_contact_time
    ct = (ct.torch if hasattr(ct, "torch") else ct)[:, ids]
    rolling = (ct > t_max).all(dim=1); sp = _cmd_speed(env, command_name)
    return (rolling & (sp > v_min) & (sp < v_max)).float() * _gate(env, rough_only)


def wheel_no_step_penalty(env: "ManagerBasedRLEnv", v_min: float = 0.4, t_max: float = 0.8, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                          command_name: str = "base_velocity", v_max: float = 99.0) -> torch.Tensor:
    """按轮罚：有速度指令时，任一轮持续触地超过 t_max 就罚 1/轮（no_step_penalty 只看"四轮全触地"，策略学会只抬一只轮子躲过去：
    Trot-V 10699 在 MuJoCo 里右前轮离地 19～35%、其余 0～1%，不是小跑）。"""
    cs = env.scene.sensors[sensor_cfg.name]
    ids = _ids(env)["c"]
    ct = cs.data.current_contact_time
    ct = (ct.torch if hasattr(ct, "torch") else ct)[:, ids]
    sp = _cmd_speed(env, command_name)
    return (ct > t_max).float().sum(dim=1) * ((sp > v_min) & (sp < v_max)).float()
