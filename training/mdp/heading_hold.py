"""航向保持奖励（09-10，作者要求"走歪了给惩罚"）。

为什么有效、边界在哪：
  策略观测里没有 yaw，所以它无法"发现自己偏了再拐回来"。但航向漂移全部是它自己积出来的 ——
  偏航角速度在观测里（base_ang_vel[2]），只要 wz 指令为 0 时把角速度压到 0，航向就不会漂。
  本项直接惩罚"当前航向与指令航向目标之差"，等价于对角速度误差做积分惩罚，逼它不制造偏差。
  它治不了外力造成的偏差（被推、压到石头、上侧坡），那必须靠部署侧把航向外环补回来
  （训练里 heading_command=True 时 wz = heading_control_stiffness × 航向误差，真机上操作手给的 wz 不含这一项）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _T(x):
    return x.torch if hasattr(x, "torch") else x


def heading_error(env: "ManagerBasedRLEnv", command_name: str = "base_velocity",
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """当前航向与指令航向目标之差，wrap 到 ±π。命令项没开 heading 模式时返回 0。"""
    term = env.command_manager.get_term(command_name)
    tgt = getattr(term, "heading_target", None)
    if tgt is None or not getattr(term.cfg, "heading_command", False):
        return torch.zeros(env.num_envs, device=env.device)
    hw = _T(env.scene[asset_cfg.name].data.heading_w)
    err = wrap_to_pi(_T(tgt) - hw)
    keep = getattr(term, "is_heading_env", None)
    if keep is not None:
        err = err * _T(keep).float()
    return err


def heading_hold_reward(env: "ManagerBasedRLEnv", command_name: str = "base_velocity", sigma: float = 0.4,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """exp(−误差²/σ²)。对齐得越准分越高；刚重采样出大角度时分低，逼它尽快转到位再保持。"""
    e = heading_error(env, command_name, asset_cfg)
    return torch.exp(-(e ** 2) / (sigma ** 2))


def heading_drift_penalty(env: "ManagerBasedRLEnv", command_name: str = "base_velocity", dead: float = 0.15,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """死区外的航向误差平方惩罚。死区避免和正常转向指令打架，只罚"该直走却走歪"。"""
    e = heading_error(env, command_name, asset_cfg).abs()
    return torch.clamp(e - dead, min=0.0) ** 2


def yaw_ref_penalty(env: "ManagerBasedRLEnv", dead_deg: float = 6.0, max_deg: float = 25.0, command_name: str = "base_velocity",
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """参考航向按**指令角速度积分**，罚死区外的 (yaw − yaw_ref)²。不依赖 heading_command，指令模式下也能用。

    为什么需要（T9/S14，09-15 00:2x）：
      T7 的配置是 `heading_command: false`，于是本文件上面那两项（heading_hold_reward / heading_drift_penalty）
      **恒返回 0，从来没生效过**；踏步线只有 track_ang_vel_z_exp 罚**瞬时**角速度误差，没有任何一项
      罚**累计**航向偏差。后果（MuJoCo 实测 T7_13999）：15° 坡 1.0 m/s 偏航 −26°、越野 +17~+20°，
      正是作者现场说的"上斜坡或越野不够稳定、踏步会乱"。
      楼梯线同理：F 起步偏 −8° 时爬楼中漂到 +20.8°。
    做法：ref 每步按 cmd_wz·dt 积分（所以主动转向不会被罚），reset 时对齐当前航向。
    """
    asset = env.scene[asset_cfg.name]
    hw = _T(asset.data.heading_w)
    c = env.command_manager.get_command(command_name)
    b = getattr(env, "_yaw_ref", None)
    if b is None or b["ref"].shape[0] != env.num_envs:
        b = {"ref": hw.clone(), "t": torch.zeros(env.num_envs, device=hw.device), "step": -1}
        env._yaw_ref = b
    if b["step"] != env.common_step_counter:
        b["step"] = env.common_step_counter
        t = env.episode_length_buf.float() * env.step_dt
        fresh = t < b["t"] - 1e-6
        b["ref"] = torch.where(fresh, hw, b["ref"]); b["t"] = t
        b["ref"] = wrap_to_pi(b["ref"] + c[:, 2] * env.step_dt)
    e = wrap_to_pi(b["ref"] - hw).abs()
    # 09-15 00:4x 修：原来不封顶，弧度平方在大偏差下爆炸 —— C10 实测偏航 72° 时该项读到 −12.7/s，
    #   比速度跟踪(5.3)还大一倍、比真·爬高进展(0.56)大 23 倍，整个奖励被它主导，策略放弃爬台去修航向，
    #   结果登顶 0/3 而平地照样漂 5.0 m（两头没做好）。封顶到 max_deg，保证它只是"一项"不是"全部"。
    d = torch.clamp(e - dead_deg * 3.141592653589793 / 180.0, min=0.0)
    m = max_deg * 3.141592653589793 / 180.0
    # Huber：m 以内二次（小偏差细调），超出转线性（大偏差不爆炸但**仍有梯度**）。
    # 硬封顶不行——40° 和 72° 会读到同一个值，策略没有往回修的动力。
    return torch.where(d <= m, d * d, m * m + 2.0 * m * (d - m))
