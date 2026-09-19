# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import Camera


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


# ──────────────────────────────────────────────────────────────
# Phase 4-B: 深度相机观测 (感知策略)
# 来源: MDP §4.2
# ──────────────────────────────────────────────────────────────

def depth_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    target_height: int = 58,
    target_width: int = 87,
    far_clip: float = 5.0,
    normalize: bool = True,
    flatten: bool = True,
) -> torch.Tensor:
    """Process depth image from camera sensor into a normalized observation tensor.

    Pipeline (参考 extreme-parkour):
      1. 读取深度图 (N, H, W, 1) → squeeze → (N, H, W)
      2. 裁剪深度值到 [0, far_clip]
      3. 缩放到目标分辨率 (58×87, 与 DepthOnlyFCBackbone58x87 匹配)
      4. 归一化到 [-0.5, 0.5]
      5. 可选: flatten 到 (N, H*W) 用于 1D MLP 输入

    Args:
        env: The environment.
        sensor_cfg: The sensor configuration for the depth camera.
        target_height: Target image height after resize (default 58).
        target_width: Target image width after resize (default 87).
        far_clip: Maximum depth value (m). Values beyond are clipped.
        normalize: Whether to normalize to [-0.5, 0.5] range.
        flatten: Whether to flatten to 1D (for MLP) or keep 2D (for CNN).

    Returns:
        Processed depth image tensor.
        If flatten=True: shape (num_envs, target_height * target_width).
        If flatten=False: shape (num_envs, target_height, target_width).
    """
    # extract the camera sensor
    camera: Camera = env.scene[sensor_cfg.name]
    # get depth data: shape (N, H, W, 1) or (N, H, W)
    depth_data = camera.data.output["depth"].torch
    if depth_data.ndim == 4:
        depth_data = depth_data.squeeze(-1)  # (N, H, W)
    # clip to valid range (handle inf/nan from far distances)
    depth_data = torch.clamp(depth_data, min=0.0, max=far_clip)
    # replace inf/nan with far_clip
    depth_data = torch.nan_to_num(depth_data, nan=far_clip, posinf=far_clip, neginf=0.0)
    # resize to target resolution
    # NCHW format for interpolate: (N, 1, H, W)
    depth_4d = depth_data.unsqueeze(1)  # (N, 1, H, W)
    depth_resized = F.interpolate(depth_4d, size=(target_height, target_width), mode="bilinear", align_corners=False)
    depth_resized = depth_resized.squeeze(1)  # (N, target_H, target_W)
    if normalize:
        # normalize to [-0.5, 0.5]
        depth_resized = depth_resized / far_clip - 0.5
    if flatten:
        depth_resized = depth_resized.flatten(1)  # (N, H*W)
    return depth_resized

