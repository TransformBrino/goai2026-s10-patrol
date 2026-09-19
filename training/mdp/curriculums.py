# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terrain curriculum with synchronized global gait_level update."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    mean_level = torch.mean(terrain.terrain_levels.float())
    # Local import avoids module-load circular dependency.
    from .rewards import update_gait_level_from_terrain_mean

    update_gait_level_from_terrain_mean(mean_level)

    _log_levels_by_subterrain(env, terrain, move_up, move_down, distance, command[env_ids])
    return mean_level


# ---------------------------------------------------------------------------
# 按子地形分别统计等级 —— 只有均值是不够的。
#
# 实测教训（2026-08-08）：训练的 terrain_levels 均值卡在 4.69 不动，
# 而这个均值既可能是"所有子地形都卡在 4.7"，也可能是"某一种卡在 1、
# 其余早就满级"。两种情况的对策完全相反，光看均值分不出来，
# 我为此推理了半天也没结论。缺的不是分析，是这个观测量。
#
# terrain_levels 是行（难度），terrain_types 是列（子地形种类），
# 两个都是 per-env 的张量，凑一起就能出分布。
_LOG_EVERY = 50
_log_calls = 0
# 累计统计。**不要报瞬时批次** —— 每次 reset 只涉及少数几个 env，
# 瞬时值在 0%/50%/100% 之间乱跳（实测就是这样），完全看不出趋势。
_cum = {"up": 0, "down": 0, "n": 0, "dist": []}
# 列 → 子地形名的映射。地形是 num_cols=20 列，而子地形定义只有 7 种，
# **列不等于子地形序号** —— 第一版直接拿列号去索引名字表，
# 结果 13 列显示成 type7..type19。映射规则见 IsaacLab
# terrain_generator.py:243-246：按归一化后的 proportion 累积分配。
_col2name = None


def _build_col2name(terrain):
    import numpy as np
    gen = getattr(terrain.cfg, "terrain_generator", None)
    if gen is None or not gen.sub_terrains:
        return {}
    names = list(gen.sub_terrains)
    props = np.array([c.proportion for c in gen.sub_terrains.values()], dtype=float)
    props = props / props.sum()
    cum = np.cumsum(props)
    ncol = int(getattr(gen, "num_cols", 20))
    out = {}
    for col in range(ncol):
        idx = int(np.min(np.where(col / ncol + 0.001 < cum)[0]))
        out[col] = names[idx]
    return out


def _log_levels_by_subterrain(env, terrain, move_up, move_down,
                              distance=None, command=None) -> None:
    global _log_calls, _col2name
    _log_calls += 1
    try:
        if move_up.numel():
            _cum["up"] += int(move_up.sum().item())
            _cum["down"] += int(move_down.sum().item())
            _cum["n"] += int(move_up.numel())
        if distance is not None and distance.numel():
            _cum["dist"].append(float(distance.mean().item()))
    except Exception:
        pass
    if (_log_calls % _LOG_EVERY) != 1:
        return
    try:
        import numpy as np
        if _col2name is None:
            _col2name = _build_col2name(terrain)
        lv = terrain.terrain_levels.float()
        ty = terrain.terrain_types
        # 同名的列合并统计 —— 关心的是"哪种子地形落后"，不是"哪一列"
        agg = {}
        for t in range(int(ty.max().item()) + 1):
            sel = ty == t
            n = int(sel.sum().item())
            if n == 0:
                continue
            nm = _col2name.get(t, f"col{t}")
            s, c = agg.get(nm, (0.0, 0))
            agg[nm] = (s + float(lv[sel].sum().item()), c + n)
        parts = [f"{nm}={s/c:.2f}({c})" for nm, (s, c) in
                 sorted(agg.items(), key=lambda kv: kv[1][0] / kv[1][1])]
        n = max(_cum["n"], 1)
        d = np.mean(_cum["dist"][-200:]) if _cum["dist"] else float("nan")
        print(f"[课程] {'  '.join(parts)}"
              f"   ‖累计 升级 {_cum['up']/n*100:.0f}% 降级 {_cum['down']/n*100:.0f}%"
              f"（n={n}）  平均离出生点 {d:.2f}m / 升级线 "
              f"{terrain.cfg.terrain_generator.size[0]/2:.1f}m", flush=True)
    except Exception as e:      # 日志不能拖垮训练
        if _log_calls < 200:
            print(f"[课程] 统计失败: {type(e).__name__}: {e}", flush=True)


def gait_level_curve(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Return current global gait_level for logging in curriculum curves."""
    from .rewards import gait_level

    return torch.tensor(gait_level, device=env.device)


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


# ---------------------------------------------------------------------------
# 诚实晋级（S10-V1，2026-09-08 深夜）
# approach_riser 是「车道 + 两侧立面」：沿车道走 4 m 也算"走远"，没爬墙就晋级 —— 课程等级虚高。
# climb_up 是坑底四面墙，不出坑走不远，本来就诚实。这里统一加一条：凡是带立面的子地形，
# 晋级还要求机身相对出生地面升高 > z_gain_min（上了台面才算）。其它子地形规则不变。
RISER_TERRAINS = ("approach_riser", "climb_up")


DOWN_FRAC = [0.4]   # 降级线 = 该比例 × 升级线（地形尺寸/2）。1.0 等价于旧行为里的棘轮。


def terrain_levels_climb_honest(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_gain_min: float = 0.03,
    z_gain_frac: float = 0.6,
) -> torch.Tensor:
    """terrain_levels_vel 的诚实版：带立面的子地形上，走远 且 升高（≥ 该行墙高的 z_gain_frac，且 ≥ z_gain_min）才晋级。"""
    global _col2name
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # 立面地形：还要升高
    if _col2name is None:
        _col2name = _build_col2name(terrain)
    names = _col2name
    ty = terrain.terrain_types[env_ids]
    riser_cols = torch.tensor([c for c, n in names.items() if n in RISER_TERRAINS], device=ty.device, dtype=ty.dtype)
    on_riser = torch.isin(ty, riser_cols) if riser_cols.numel() else torch.zeros_like(move_up)
    wid = getattr(env, "_curr_wheel_ids", None)
    if wid is None:
        wid = asset.find_bodies(".*_wheel")[0]; env._curr_wheel_ids = wid
    bp = asset.data.body_pos_w
    bp = bp.torch if hasattr(bp, "torch") else bp
    z_gain = (bp[env_ids][:, wid, 2] - 0.081 - env.scene.env_origins[env_ids, 2:3]).min(dim=1).values   # 15:35：改用四轮触地点最低高度（原来用机身 z，含站高 0.43，永远晋级）
    # 门槛按该行的墙高算：第 k 行最低墙高 = lo + k/rows × (hi−lo)，要求升高 ≥ z_gain_frac × 它（且 ≥ z_gain_min）。
    # 否则 6 cm 起步的低行永远达不到固定的 0.10 m 门槛，课程卡死在第 0 行（09-08 深夜发现）。
    gen = terrain.cfg.terrain_generator
    rng = None
    for nm in RISER_TERRAINS:
        sub = gen.sub_terrains.get(nm) if gen is not None and gen.sub_terrains else None
        if sub is not None:
            rng = getattr(sub, "riser_height_range", None) or getattr(sub, "step_height_range", None)
            if rng is not None:
                break
    if rng is not None:
        lo, hi = float(rng[0]), float(rng[1])
        lvl = terrain.terrain_levels[env_ids].float()
        h_row = lo + lvl / float(gen.num_rows) * (hi - lo)
        need = torch.clamp(z_gain_frac * h_row, min=z_gain_min)
    else:
        need = torch.full_like(z_gain, z_gain_min)
    move_up = move_up & (~on_riser | (z_gain > need))
    # 09-17 22:2x：**降级线与升级线重合，课程成了单向棘轮。**
    #   升级：走出 > 地形尺寸/2 = **4.0 m**  且  四轮最低点升高 ≥ 0.6×该行墙高
    #   降级：走出 < |指令| × 局时长 × 0.5 = 0.8 × 10 × 0.5 = **4.0 m**   ← 同一个数
    # 只要爬高没达标就必降；实测平均走 3.98 m（差 2 cm），每轮 99~100% 全降，等级钉死第 0 行。
    # 更糟的是**任何让机器人变慢的改动都会重新触发塌陷**：B2p 加撞墙罚（让它慢下来蹬地）之后，
    # 课程从 4.75 掉回 **0.32**（升级 1%/降级 97%）。而"慢下来"正是"先蹲实再蹬地"的前提 ——
    # **这个公式在和训练目标对着干。**
    # 改为：降级线 = DOWN_FRAC × 升级线，明显低于升级线，不再是棘轮。
    up_line = terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < DOWN_FRAC[0] * up_line
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    mean_level = torch.mean(terrain.terrain_levels.float())
    from .rewards import update_gait_level_from_terrain_mean
    update_gait_level_from_terrain_mean(mean_level)
    _log_levels_by_subterrain(env, terrain, move_up, move_down, distance, command[env_ids])
    return mean_level


def terrain_levels_climb_expert(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    up_frac: float = 0.6, down_frac: float = 0.15,
) -> torch.Tensor:
    """上墙专家课程（09-09）：只看升高。升高 ≥ up_frac×本行墙高 → 晋级；升高 < down_frac×本行墙高 → 降级。不看走远。"""
    global _col2name
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    gen = terrain.cfg.terrain_generator
    sub = gen.sub_terrains.get("approach_riser") if gen is not None and gen.sub_terrains else None
    lo, hi = (float(sub.riser_height_range[0]), float(sub.riser_height_range[1])) if sub is not None else (0.06, 0.34)
    lvl = terrain.terrain_levels[env_ids].float()
    h_row = lo + lvl / float(gen.num_rows) * (hi - lo)
    wid = getattr(env, "_curr_wheel_ids", None)
    if wid is None:
        wid = asset.find_bodies(".*_wheel")[0]; env._curr_wheel_ids = wid
    bp = asset.data.body_pos_w
    bp = bp.torch if hasattr(bp, "torch") else bp
    wz = bp[env_ids][:, wid, 2] - 0.081 - env.scene.env_origins[env_ids, 2:3]      # (n,4) 四轮触地点高度
    # 15:05：按"本局是否上去过"判（上去后会继续开到台面尽头掉下去，局末高度不可信）
    done = getattr(env, "_climb_ep_done", None); front = getattr(env, "_climb_ep_front", None)
    if done is not None:
        move_up = done[env_ids]
        move_down = ~front[env_ids] & ~move_up
    else:
        move_up = wz.min(dim=1).values > up_frac * h_row          # 四轮都上去了
        move_down = (wz.max(dim=1).values < down_frac * h_row) & ~move_up   # 一只轮子都没上
    terrain.update_env_origins(env_ids, move_up, move_down)
    mean_level = torch.mean(terrain.terrain_levels.float())
    from .rewards import update_gait_level_from_terrain_mean
    update_gait_level_from_terrain_mean(mean_level)
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    _log_levels_by_subterrain(env, terrain, move_up, move_down, distance, command[env_ids])
    return mean_level
