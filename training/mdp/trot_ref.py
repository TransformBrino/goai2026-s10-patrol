"""真机踏步参考（ref_trot.npz：按速度档的相位归一化周期，50 点 × 16 关节 articulation 顺序）：
  ① 参考态出生：随机速度档、随机相位，写 12 个腿关节角/速度（轮子只写速度），机身站高 0.35；
  ② 周期跟踪奖励 TrotRefTrack：相位钟按指令速度所在档的频率推进，跟踪 12 个腿关节角；只在有速度指令时算。
无相位观测，靠 RSI 让策略从自身状态里辨相位（上墙线已验证的做法）。"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REF_PATH = os.environ.get("S10_TROT_REF", os.path.expanduser("~/s10_logs/ref/ref_trot.npz"))
FORWARD_ONLY = [False]
FREQ_K = [0.0]
FREQ_MIN = [0.0]   # S2：钟频下限（Hz）   # S1 起 >0：参考钟频率 = FREQ_K × 指令速度（官方楼梯 0.5 m/s ↔ 1.10 Hz → 2.2），旋钮慢下来步频也慢   # H5 起由 cfg 置 True：只在前向指令时跟踪/门控
_REF = {}


def load_ref(device):
    key = str(device)
    if key not in _REF:
        d = np.load(REF_PATH, allow_pickle=True)
        bins = [str(b) for b in d["bins"]]
        q = np.stack([d[f"{b}_q"] for b in bins]); qd = np.stack([d[f"{b}_qd"] for b in bins])
        freq = np.array([float(d[f"{b}_freq"]) for b in bins]); v = np.array([float(d[f"{b}_v"]) for b in bins])
        clear = np.stack([d[f"{b}_clear"] for b in bins]) if all(f"{b}_clear" in d.files for b in bins) else None   # T4：各档各相位的轮子净空（官方），做触地相位模板
        r = dict(q=torch.as_tensor(q, dtype=torch.float32, device=device), qd=torch.as_tensor(qd, dtype=torch.float32, device=device),
                 freq=torch.as_tensor(freq, dtype=torch.float32, device=device), v=torch.as_tensor(v, dtype=torch.float32, device=device),
                 n_bins=len(bins), n_ph=q.shape[1], bins=bins,
                 air=torch.as_tensor(clear > 0.02, dtype=torch.bool, device=device) if clear is not None else None)
        _REF[key] = r
        print(f"[trot-ref] 载入 {REF_PATH}: 档 {bins} 频率 {np.round(freq, 2)} 速度 {np.round(v, 2)} 相位点 {q.shape[1]}", flush=True)
    return _REF[key]


def _bin_for_speed(ref, v_cmd):
    return torch.argmin((ref["v"].unsqueeze(0) - v_cmd.unsqueeze(1)).abs(), dim=1)


def spawn_trot_ref(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ref_prob: float = 0.5, z: float = 0.36) -> None:
    """参考态出生：随机档 + 随机相位，写腿关节角/速度与轮速，机身放平在原点上方 z。其余 env 不动。"""
    asset = env.scene[asset_cfg.name]
    ref = load_ref(env_ids.device)
    n = len(env_ids)
    pick = torch.rand(n, device=env_ids.device) < ref_prob
    if getattr(env, "_trot_phase", None) is None:
        env._trot_phase = torch.zeros(env.num_envs, device=env_ids.device)
    env._trot_phase[env_ids] = torch.rand(n, device=env_ids.device)
    if not pick.any():
        return
    ids = env_ids[pick]; m = len(ids)
    b = torch.randint(0, ref["n_bins"], (m,), device=ids.device)
    k = torch.randint(0, ref["n_ph"], (m,), device=ids.device)
    env._trot_phase[ids] = k.float() / ref["n_ph"]
    q = ref["q"][b, k].clone(); qd = ref["qd"][b, k].clone()
    q[:, 12:] = 0.0                                              # 轮子累积角无意义
    jp = asset.data.default_joint_pos.torch[ids].clone() if hasattr(asset.data.default_joint_pos, "torch") else asset.data.default_joint_pos[ids].clone()
    jp[:, :12] = q[:, :12]
    asset.write_joint_position_to_sim_index(position=jp, joint_ids=slice(None), env_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=qd, joint_ids=slice(None), env_ids=ids)
    origins = env.scene.env_origins
    origins = origins.torch if hasattr(origins, "torch") else origins
    pos = origins[ids].clone(); pos[:, 2] += z
    quat = torch.zeros(m, 4, device=ids.device); quat[:, 3] = 1.0          # xyzw 单位
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids)
    vel = torch.zeros(m, 6, device=ids.device); vel[:, 0] = ref["v"][b]
    asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)


class TrotRefTrack(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.ref = load_ref(env.device)
        if getattr(env, "_trot_phase", None) is None:
            env._trot_phase = torch.zeros(env.num_envs, device=env.device)
        self._last = -1

    def reset(self, env_ids=None):
        return {}

    def __call__(self, env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma_q: float = 0.3, v_min: float = 0.3,
                 command_name: str = "base_velocity", v_max: float = 99.0, legs: str = "all") -> torch.Tensor:
        cmd = env.command_manager.get_command(command_name)
        v = torch.linalg.norm(cmd[:, :2], dim=1)
        if FORWARD_ONLY[0]:
            v = cmd[:, 0]          # 倒退（vx<0）不跟踪参考（参考都是前进步态）
        b = _bin_for_speed(self.ref, v.clamp(min=0.0))
        if env.common_step_counter != self._last:
            self._last = env.common_step_counter
            fq = self.ref["freq"][b] if FREQ_K[0] <= 0 else (FREQ_K[0] * v.clamp(min=0.05)).clamp(min=FREQ_MIN[0])
            env._trot_phase = (env._trot_phase + fq * env.step_dt) % 1.0
        k = torch.round(env._trot_phase * self.ref["n_ph"]).long() % self.ref["n_ph"]
        q_ref = self.ref["q"][b, k][:, :12]
        jq = self.asset.data.joint_pos; jq = (jq.torch if hasattr(jq, "torch") else jq)[:, :12]
        if legs == "front":                                   # E6 起：后腿不受参考束缚（官方参考的后轮是贴踢面滚上，跟它就学不会抬后轮）
            idx = [0, 1, 4, 5, 8, 9]
            jq = jq[:, idx]; q_ref = q_ref[:, idx]
        r = torch.exp(-((jq - q_ref) ** 2).mean(dim=1) / (sigma_q ** 2))
        return r * ((v > v_min) & (v < v_max)).float()


def phase_contact_reward(env, v_min: float = 0.3, v_max: float = 2.2, h_min: float = 0.04, command_name: str = "base_velocity") -> torch.Tensor:
    """触地相位跟钟（T4）：按参考钟当前相位查官方模板"这只轮此刻该腾空还是该触地"，四轮各自符合 +0.25。
    直接约束节拍（官方步频恒 1.10 Hz 的本质），不再靠关节角相似度间接约束。"""
    from .step_gait import _swing
    ref = load_ref(env.device)
    if ref.get("air") is None or getattr(env, "_trot_phase", None) is None:
        return torch.zeros(env.num_envs, device=env.device)
    cmd = env.command_manager.get_command(command_name); v = cmd[:, 0]
    b = _bin_for_speed(ref, v.clamp(min=0.0)); k = torch.round(env._trot_phase * ref["n_ph"]).long() % ref["n_ph"]
    should_air = ref["air"][b, k]                                   # (N,4)
    sw, _ = _swing(env, h_min)
    ok = (sw == should_air).float().mean(dim=1)
    return ok * ((v > v_min) & (v < v_max)).float()
