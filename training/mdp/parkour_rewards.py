# 借鉴 CMU Extreme Parkour（BSD）/ Robot Parkour Learning（MIT）的奖励项，按 IsaacLab 管理器写法移植。
# 2026-09-05：只移 feet_stumble；feet_edge 需地形边缘掩码，IsaacLab 无现成，暂不移。
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


STUMBLE_RATIO = 4.0   # Extreme Parkour 原式的 4；管理器按签名校验参数，所以不做成可选参数


def feet_stumble(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """足端（轮）撞立面惩罚：水平接触力 > 4 × 竖直接触力 的轮子个数。

    Extreme Parkour 原式：any(|F_xy| > 4·|F_z|)，我们改成计数（多轮同时撞罚更重）。
    无接触时 F=0，0 > 0 为假，不罚。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]           # (N, B, 3)
    stumble = f[..., :2].norm(dim=-1) > STUMBLE_RATIO * f[..., 2].abs()
    return stumble.float().sum(dim=1)


def flat_orientation_l2_terrain_aware(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_height_threshold: float = 0.06,
    high_terrain_penalty_scale: float = 0.1,
    phase_exempt: bool = False,
) -> torch.Tensor:
    """姿态惩罚的高地形豁免版：与 rewards.joint_pos_penalty_except_turn_side_cmd 用同一套地形门控。

    为什么：爬 33cm 单级立面要把前身扬到约 −40° 俯仰，flat_orientation_l2（权重 −15）在那一两秒里
    每步罚约 6，比速度跟踪的软奖励重得多——V9/V10 都是热启动后先出峰值再退化成「顶墙不试」，
    与这项惩罚把爬升姿态优化掉的假说一致。平地上惩罚原样保留（别乱晃），高地形上降到 1/10。
    """
    asset = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    height_sensor = env.scene[sensor_cfg.name]
    ray_hits = height_sensor.data.ray_hits_w[..., 2]
    valid_hits = torch.isfinite(ray_hits) & (torch.abs(ray_hits) <= 1e6)
    valid_count = torch.sum(valid_hits, dim=1)
    safe_hits = torch.where(valid_hits, ray_hits, torch.zeros_like(ray_hits))
    terrain_height = torch.sum(safe_hits, dim=1) / torch.clamp(valid_count, min=1) - env.scene.env_origins[:, 2]
    high_terrain = (terrain_height > terrain_height_threshold) & (valid_count > 0)
    if phase_exempt:
        from .climb_rewards import climb_exempt_mask       # C18，理由见该函数
        high_terrain = climb_exempt_mask(env)
    scale = torch.where(high_terrain, torch.full_like(reward, high_terrain_penalty_scale), torch.ones_like(reward))
    return reward * scale


def stall_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_min: float = 0.3,
    vel_max: float = 0.1,
) -> torch.Tensor:
    """停滞惩罚：指令要求前进（cmd_vx > cmd_min）而机身几乎不动（|vx| < vel_max）时罚 1。

    速度跟踪是 exp 型软奖励，停在墙根只损失一点点；这里把「顶墙不动 / 看见墙冻住」变成硬代价，
    与摔倒的机会成本抗衡。转向、侧移指令不罚（只看前进指令）。
    """
    cmd = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_b[:, 0]
    return ((cmd[:, 0] > cmd_min) & (vx.abs() < vel_max)).float()


def flat_orientation_normal_aware(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_slope_deg: float = 20.0,
    max_residual: float = 0.03,
    terrain_height_threshold: float = 0.06,
    high_terrain_penalty_scale: float = 0.1,
) -> torch.Tensor:
    """姿态惩罚按**局部地面法向**算，而不是按重力（T10，09-15 06:0x）。

    为什么（09-15 05:3x 用 T9 已有探针数据查出，未占 GPU）：
      作者现场原话"踏步模式上斜坡或者越野不够稳定，踏步会乱"。T9 四检查点在 10° 坡 1.0 m/s：
        14000 偏航 −3°  / |roll| p95 11.2°      14200 偏航 **+30°** / |roll| p95 5.4°
        14400 偏航 +4°  / |roll| p95 14.8°      14598 偏航 **+27°** / |roll| p95 6.1°
      **强反相关**：侧倾大 → 偏航不漂；侧倾小 → 偏航漂 27~30°。不是随机双峰，是两种姿态模式。
      机理：坡有横向分量时，机身跟坡侧倾 → 两侧轮子都吃上载荷 → 抓得住；
            机身保持水平 → 下坡侧载荷不足 → 打滑 → 整车往下坡方向偏转。
      根因：`flat_orientation_l2_terrain_aware` 名为地形感知，实际只按**地形高度**整体打折，
      惩罚本身仍是 `projected_gravity_b[:,:2]` 的平方 = "机身相对**重力**保持水平"，
      **完全不看坡面朝向**；10° 坡的高度差多数区域在 0.06 m 门槛之下 → 坡上这项全额生效，
      一直把机身往水平拽 → 坡上不敢侧倾 → 打滑 → 偏航漂。

    改法：用高程扫描点最小二乘拟合局部平面 z = a·x + b·y + c，法向 n_w = (−a, −b, 1)/‖·‖，
    转到机体系后罚 n_b 的水平分量 —— 即"机身 z 轴对齐**地面法向**"。
    平地上地面法向 = 重力反向，本项与原式**逐位等价**，平地行为不变。

    **台阶/墙面必须退化成原行为**（否则会把爬台弄坏）：
      · 残差门：平面拟合 RMS > `max_residual`（**0.03 m**）说明这不是个平面（台阶、墙、碎石）
        → 坡度直接归零，退回"对齐重力"，与改动前完全一致。
        **门槛必须是 0.03，不能是 0.05**：15 cm/25 cm 台阶在 ±0.4 m 扫描窗内拟合 RMS = **0.0404**，
        0.05 的门放它过去 → 会把楼梯的平均斜坡当成"坡面"、要求机身跟着倾（实测罚 0.117），
        这是离线八情形自检抓出来的（scratchpad/t10_test.py）。
      · 坡度夹限：|坡度| 夹到 `max_slope_deg`（默认 20°），防个别异常点把法向甩飞。
      · 高地形折扣沿用原式（阈值 0.06 / 0.1 倍），翻台阶豁免逻辑一个字没动。

    **离线自检八情形**（`scratchpad/t10_test.py`，纯 torch 不占 GPU；法向罚 vs 现行重力罚）：
      ① 平地+水平        0.0000 / 0.0000  ← 逐位等价
      ② 平地+侧倾10°     0.0302 / 0.0302  ← 逐位等价，平地行为不变
      ③ 10°横坡+水平     0.0302 / 0.0000  ← **新行为**：坡上保持水平要挨罚
      ④ 10°横坡+跟坡倾   0.0000 / 0.0302  ← **新行为**：跟坡侧倾不再挨罚（这就是打滑的解）
      ④b 10°横坡+反向倾  0.1170 / 0.0302  ← 反着倾重罚
      ⑤ 15cm 台阶+水平   0.0000 / 0.0000  ← 残差 0.0404 > 0.03，退化，逐位相同
      ⑥ 15cm 台阶+俯仰30 0.2500 / 0.2500  ← 逐位相同
      ⑦ 33cm 墙+俯仰30°  0.2500 / 0.2500  ← 残差 0.0882，退化，**爬台线完全不受影响**
      ⑧ 30°陡坡+水平     0.1170 / 0.0000  ← 坡度夹到 20°，部分生效
    **未在 GPU 上跑过，属"离线通过"，不等于训练有效**（版本口径）。
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w                                        # (N, R, 3) 世界系
    hits = hits.torch if hasattr(hits, "torch") else hits
    valid = torch.isfinite(hits).all(dim=-1) & (hits[..., 2].abs() <= 1e6)
    cnt = valid.sum(dim=1).clamp(min=1).float()
    w = valid.float()
    p = torch.where(valid.unsqueeze(-1), hits, torch.zeros_like(hits))
    mean = (p * w.unsqueeze(-1)).sum(dim=1) / cnt.unsqueeze(-1)          # (N,3)
    d = (p - mean.unsqueeze(1)) * w.unsqueeze(-1)                        # 去心，无效点为 0
    Sxx = (d[..., 0] * d[..., 0]).sum(1); Sxy = (d[..., 0] * d[..., 1]).sum(1)
    Syy = (d[..., 1] * d[..., 1]).sum(1)
    Sxz = (d[..., 0] * d[..., 2]).sum(1); Syz = (d[..., 1] * d[..., 2]).sum(1)
    det = Sxx * Syy - Sxy * Sxy
    ok = det.abs() > 1e-8
    det_s = torch.where(ok, det, torch.ones_like(det))
    a = torch.where(ok, (Syy * Sxz - Sxy * Syz) / det_s, torch.zeros_like(det))
    b = torch.where(ok, (Sxx * Syz - Sxy * Sxz) / det_s, torch.zeros_like(det))
    # 残差门：不是平面就不信这个法向
    res = d[..., 2] - (a.unsqueeze(1) * d[..., 0] + b.unsqueeze(1) * d[..., 1])
    rms = torch.sqrt((res * res * w).sum(1) / cnt)
    trust = (rms < max_residual) & ok & (cnt > 8)
    s_max = math.tan(math.radians(max_slope_deg))
    a = torch.where(trust, a.clamp(-s_max, s_max), torch.zeros_like(a))
    b = torch.where(trust, b.clamp(-s_max, s_max), torch.zeros_like(b))
    n_w = torch.stack([-a, -b, torch.ones_like(a)], dim=-1)
    n_w = n_w / n_w.norm(dim=-1, keepdim=True)
    q = asset.data.root_quat_w
    q = q.torch if hasattr(q, "torch") else q          # IsaacLab 6 里这些字段是 ProxyArray，不是 Tensor
    n_b = quat_rotate_inverse(q, n_w)                                    # 机体系地面法向
    reward = torch.sum(torch.square(n_b[:, :2]), dim=1)
    # 高地形折扣：与原式同一套门控，逐位保留
    ray_z = hits[..., 2]
    vz = torch.isfinite(ray_z) & (ray_z.abs() <= 1e6)
    vc = vz.sum(dim=1)
    safe = torch.where(vz, ray_z, torch.zeros_like(ray_z))
    terrain_height = safe.sum(dim=1) / torch.clamp(vc, min=1) - env.scene.env_origins[:, 2]
    high = (terrain_height > terrain_height_threshold) & (vc > 0)
    scale = torch.where(high, torch.full_like(reward, high_terrain_penalty_scale), torch.ones_like(reward))
    return reward * scale
