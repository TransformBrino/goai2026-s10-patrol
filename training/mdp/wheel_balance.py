"""四轮驱动均衡惩罚（2026-09-12，真机反馈后加）。

起因（真机 + 仿真一致）：
  · 主策略 V1Down 直行时前轮出力、后轮几乎不转（仿真 0.3 m/s：前轮目标 −7 / +10，后轮 0.7～0.8 rad/s），作者要求前后轮一起发力；
  · 楼梯专家 StairN-C 9300 切入后左前轮动作 −13、其余三轮 −3（×5 = −65 rad/s 目标），真机石材上左前轮空转到 −57 rad/s，牵引不对称侧翻。
奖励里原来只有 joint_torques_l2（总量），没有"四轮分担"这一项，策略就学成了单轮/前轮驱动。

做法：按轮子关节的施加力矩 |τ|（applied_torque，轮子 kp=0、kd=0.6，力矩就是驱动量）算两项不均衡，除以 (平均|τ| + eps) 归一：
  前后：| mean(|τ_fl|, |τ_fr|) − mean(|τ_hl|, |τ_hr|) |
  左右：| mean(|τ_fl|, |τ_hl|) − mean(|τ_fr|, |τ_hr|) |
只在有前进指令（|vx 指令| > cmd_threshold）时算；lr_only=True 只算左右（上楼梯时前后本来就该不同）。
"""
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from .rewards import _T

_WHEELS = ("fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint")


def wheel_drive_balance(env, command_name: str = "base_velocity", cmd_threshold: float = 0.15, eps: float = 1.0,
                        lr_only: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), wz_max: float = 99.0) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = getattr(env, "_wheel_balance_ids", None)
    if ids is None:
        names = list(asset.joint_names)
        ids = [names.index(n) for n in _WHEELS]
        env._wheel_balance_ids = ids
        print("[wheel_balance] 轮子关节下标 fl/fr/hl/hr =", ids, "lr_only =", lr_only)
    tau = torch.abs(_T(asset.data.applied_torque)[:, ids])          # (N, 4)
    front = 0.5 * (tau[:, 0] + tau[:, 1]); rear = 0.5 * (tau[:, 2] + tau[:, 3])
    left = 0.5 * (tau[:, 0] + tau[:, 2]); right = 0.5 * (tau[:, 1] + tau[:, 3])
    mean = tau.mean(dim=1)
    imb = torch.abs(left - right)
    if not lr_only:
        imb = imb + torch.abs(front - rear)
    imb = imb / (mean + eps)
    cmd = env.command_manager.get_command(command_name)
    active = ((torch.abs(cmd[:, 0]) > cmd_threshold) & (torch.abs(cmd[:, 2]) < wz_max)).float()   # E8：转向指令时左右不均衡是应该的，不罚
    return imb * active


def rear_wheel_share(env, command_name: str = "base_velocity", cmd_threshold: float = 0.15,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """日志用：后轮 |τ| 占四轮 |τ| 的比例（0.5 = 前后均衡）。权重给 0，只为在训练曲线里看得见。"""
    asset = env.scene[asset_cfg.name]
    ids = getattr(env, "_wheel_balance_ids", None)
    if ids is None:
        names = list(asset.joint_names); ids = [names.index(n) for n in _WHEELS]; env._wheel_balance_ids = ids
    tau = torch.abs(_T(asset.data.applied_torque)[:, ids])
    share = (tau[:, 2] + tau[:, 3]) / (tau.sum(dim=1) + 1e-3)
    cmd = env.command_manager.get_command(command_name)
    return share * (torch.abs(cmd[:, 0]) > cmd_threshold).float()


def wheel_action_lr_imbalance(env, command_name: str = "base_velocity", cmd_threshold: float = 0.15, wz_max: float = 99.0) -> torch.Tensor:
    """左右轮**动作**差（不看力矩，直接罚策略输出）：|a_fl − a_fr| + |a_hl − a_hr|，有前进指令才算。
    09-12 StairN-B 用力矩不均衡 −0.5 训 1000 步没改动左前轮独大（−12.7 vs −2.5），力矩受触地状态影响、信号弱；动作差直接、可微。"""
    am = env.action_manager
    a = am.action   # (N, 16)，训练关节序：hipx×4、hipy×4、knee×4、wheel×4（fl fr hl hr）
    imb = torch.abs(a[:, 12] - a[:, 13]) + torch.abs(a[:, 14] - a[:, 15])
    cmd = env.command_manager.get_command(command_name)
    return imb * ((torch.abs(cmd[:, 0]) > cmd_threshold) & (torch.abs(cmd[:, 2]) < wz_max)).float()
