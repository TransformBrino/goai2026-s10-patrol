"""真机上墙参考轨迹（09-09 石笼 34 cm，climb33_1/2/3，压墙→落平约 1 s）用于训练：
  ① 参考态出生（RSI）：从真机帧采样关节角/关节速度/机身高度/俯仰/离墙距离，直接落在上墙过程中的任一相位；
  ② 相位跟踪奖励 RefTrack：压墙事件（或参考态出生）后按相位时钟跟踪真机 12 个腿关节角与抬头角。
参考文件：~/s10_logs/ref/ref_climb34_50hz.npz（s10_dev/make_ref50.py 生成，关节为 articulation 顺序 hipx×4,hipy×4,knee×4,wheel×4）。"""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
import isaaclab.utils.math as _mu

from .climb_rewards import _state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REF_PATH = os.environ.get("S10_CLIMB_REF", os.path.expanduser("~/s10_logs/ref/ref_climb34_50hz.npz"))
_REF = {}


def load_ref(device) -> dict:
    key = str(device)
    if key not in _REF:
        d = np.load(REF_PATH)
        n_segs = int(d["n_segs"]); t = d["t"]; per = len(t) // n_segs
        r = {k: torch.as_tensor(np.asarray(d[k]), dtype=torch.float32, device=device) for k in ("t", "q", "qd", "pitch", "pitch_rate", "z", "x", "vx", "vz", "roll")}
        r["seg"] = torch.as_tensor(np.asarray(d["seg"]), dtype=torch.long, device=device)
        r["n_segs"] = n_segs; r["per"] = per; r["t0"] = float(t[0]); r["dt"] = float(t[1] - t[0]); r["wall_h"] = float(d["wall_h"]); r["t_end"] = float(t[per - 1])
        _REF[key] = r
        print(f"[climb-ref] 载入 {REF_PATH}: {n_segs} 段 × {per} 帧, t {r['t0']:.2f}~{r['t_end']:.2f} s, 墙高 {r['wall_h']:.2f}", flush=True)
    return _REF[key]


def _frame_index(ref: dict, seg: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    k = torch.round((phase - ref["t0"]) / ref["dt"]).long().clamp(0, ref["per"] - 1)
    return seg * ref["per"] + k


def apply_ref_spawn(env, asset, ids: torch.Tensor, origins: torch.Tensor, wall_x: float, h: torch.Tensor, yaw: torch.Tensor, jp: torch.Tensor, jv: torch.Tensor, rows: torch.Tensor,
                    t_lo: float = -0.3, t_hi: float = 1.15) -> None:
    """对 ids（子集）写参考态：根位姿/速度、关节角/速度（写进 jp/jv 的 rows 行），并设相位与段号。"""
    ref = load_ref(ids.device)
    n = len(ids)
    seg = torch.randint(0, ref["n_segs"], (n,), device=ids.device)
    phase = torch.empty(n, device=ids.device).uniform_(t_lo, t_hi)
    idx = _frame_index(ref, seg, phase)
    q = ref["q"][idx]; qd = ref["qd"][idx]
    pitch = ref["pitch"][idx]; z = ref["z"][idx]; x = ref["x"][idx]; vx = ref["vx"][idx]; vz = ref["vz"][idx]; pr = ref["pitch_rate"][idx]
    # 墙高不同：机身升高部分按比例缩放（0.37 → 0.70 对应 0.34 墙）
    frac = ((z - 0.37) / (0.70 - 0.37)).clamp(0.0, 1.0)
    z_adj = z - (ref["wall_h"] - h) * frac
    pos = origins.clone(); pos[:, 0] = origins[:, 0] + wall_x + x; pos[:, 2] = origins[:, 2] + z_adj
    quat = _mu.quat_from_euler_xyz(torch.zeros(n, device=ids.device), -pitch, yaw)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids)
    vel = torch.zeros(n, 6, device=ids.device); vel[:, 0] = vx; vel[:, 2] = vz; vel[:, 4] = -pr
    asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
    jp[rows] = q; jv[rows] = qd
    if getattr(env, "_climb_ref_phase", None) is None:
        env._climb_ref_phase = torch.full((env.num_envs,), -1.0, device=ids.device)
        env._climb_ref_seg = torch.zeros(env.num_envs, dtype=torch.long, device=ids.device)
    env._climb_ref_phase[ids] = phase; env._climb_ref_seg[ids] = seg
    try:
        env._climb_wallh_mem[ids] = h; env._climb_flexed[ids] = True
    except Exception:
        pass


def mark_no_ref(env, ids: torch.Tensor) -> None:
    if getattr(env, "_climb_ref_phase", None) is None:
        env._climb_ref_phase = torch.full((env.num_envs,), -1.0, device=ids.device)
        env._climb_ref_seg = torch.zeros(env.num_envs, dtype=torch.long, device=ids.device)
    env._climb_ref_phase[ids] = -1.0


class RefTrack(ManagerTermBase):
    """相位跟踪奖励：phase<0 未激活；压墙事件（遭遇 & 离墙 <0.35 m & 任一前轮前向轮速 >5 rad/s）置 0；参考态出生直接带相位。
    激活期间每步 += dt，超过参考末尾即失活。奖励 = exp(−mean(Δq²)/σq²)·exp(−Δpitch²/σp²)，Δq 为 12 个腿关节。"""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.sigma_q = float(cfg.params.get("sigma_q", 0.35)); self.sigma_p = float(cfg.params.get("sigma_p", 0.35))
        self.press_dist = float(cfg.params.get("press_dist", 0.35)); self.press_w = float(cfg.params.get("press_w", 5.0))
        self.ref = load_ref(env.device)
        if getattr(env, "_climb_ref_phase", None) is None:
            env._climb_ref_phase = torch.full((env.num_envs,), -1.0, device=env.device)
            env._climb_ref_seg = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._last_step = -1
        # ===== C33（09-15 18:0x）：距离→帧 查表，供位置驱动相位使用 =====
        # 作者规格：「学会在 0.58 米起步 走过去 上墙」。
        # 原来相位是**时间驱动**且锚在我方压墙事件（dist<0.35），而参考的 t=0 在 **dist 0.56**
        # —— 等于把参考的"起步那一刻"对齐到我方"已经快贴墙"的时刻，**整条参考被平移了 0.21 m**。
        # 位置驱动：走到哪个距离，就比官方在那个距离上的姿态。天然免疫速度差
        # （我方到那儿是 0.9 m/s 在走，官方是从静止起步）。
        # 只用 seg0/seg1：作者看回放认可这两段"像官方运控"，且实测两段在位置域上高度一致
        # （0.55 m: −3.3/−4.2°；0.35 m: −20.1/−18.2°；0.25 m: −38.1/−32.8°），seg2 明显不同。
        per = self.ref["per"]; xs = self.ref["x"].view(-1, per)      # (n_segs, per) 的 x_rel
        ts = self.ref["t"].view(-1, per)
        self.DL, self.DH, self.NB = -0.35, 0.60, 190                 # 有符号离墙距离查表区间
        tbl = torch.zeros((xs.shape[0], self.NB), dtype=torch.long, device=env.device)
        edges = torch.linspace(self.DH, self.DL, self.NB, device=env.device)   # 由远及近
        for sgi in range(xs.shape[0]):
            m = ts[sgi] >= 0.0                                        # 只取 t≥0（t<0 是停着的那段）
            idxs = torch.nonzero(m).flatten()
            dist = (-xs[sgi][m])                                      # 离墙距离，随帧递减
            dist = torch.cummin(dist, dim=0).values                   # 消掉 ≤0.007 m 的微小回退
            for b in range(self.NB):
                j = torch.searchsorted(-dist, -edges[b].reshape(1)).clamp(0, dist.shape[0] - 1)
                tbl[sgi, b] = idxs[j.item()]
        self._dist_tbl = tbl

    def reset(self, env_ids=None):
        return {}

    def __call__(self, env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma_q: float = 0.35, sigma_p: float = 0.35,
                 press_dist: float = 0.35, press_w: float = 5.0,
                 pre_dist: float = 0.0, pre_max_s: float = 0.5,
                 phase_mode: str = "time", anchor_dist: float = 0.58,
                 segs: tuple = (0, 1)) -> torch.Tensor:
        s = _state(env)
        ph = env._climb_ref_phase; seg = env._climb_ref_seg
        jv = self.asset.data.joint_vel; jv = jv.torch if hasattr(jv, "torch") else jv
        jq = self.asset.data.joint_pos; jq = jq.torch if hasattr(jq, "torch") else jq
        if env.common_step_counter != self._last_step:          # 每步只推进一次相位
            self._last_step = env.common_step_counter
            front_fwd = (-jv[:, 12:14]).max(dim=1).values
            # ===== C32（09-15 17:4x）：打开**预压墙窗口** =====
            # 原来 ph 只在 press 时置 0，`active = ph >= 0` —— 参考文件 t=−0.50~0 那 0.5 s
            # **从来没有被跟踪过**。而实测我方与官方最大的偏差恰好全在这个窗口里：
            #   离墙 0.60 m：官方 −3.3° / 我方 −27.8°；官方那 0.5 s 只走了 0.009 m，
            #   是【停在离墙 0.56 m、身子放平】的**静止保持**，姿态恒定 → 很好跟。
            # 这也解释了 C26/C27 加权重为什么没用：加的是压墙**之后**那段的权重，
            # 而那段本来就没差那么多。
            if pre_dist > 0.0:
                used = getattr(env, "_ref_pre_used", None)
                if used is None or used.shape[0] != ph.shape[0]:
                    used = torch.zeros_like(ph, dtype=torch.bool); env._ref_pre_used = used
                pt = getattr(env, "_ref_pre_t", None)
                if pt is None or pt.shape[0] != ph.shape[0]:
                    pt = torch.zeros_like(ph); env._ref_pre_t = pt
                enc = s["encounter"]
                pre = (ph <= -0.99) & enc & (s["dist_wall"] < pre_dist) & (~used)
                ph[pre] = -0.5; pt[pre] = 0.0
            press = (ph < 0) & s["encounter"] & (s["dist_wall"] < self.press_dist) & (front_fwd > self.press_w)
            ph[press] = 0.0
            active = ph > -0.99
            ph[active] = ph[active] + env.step_dt
            if pre_dist > 0.0:
                # 预窗口内不许自行走到 0（目标姿态恒定，钳住等真正压墙）
                hold = (ph < 0.0) & (ph > -0.99)
                ph[hold] = torch.clamp(ph[hold], max=-0.02)
                # **反刷分**：预窗口限时 pre_max_s（= 官方自己那段的长度 0.5 s），
                # 超时即失活并标记 used，遭遇结束才清 —— 否则策略停在墙前不压墙就能无限领平姿的钱
                # （对应"门的条件不会结束"那类刷分）。
                env._ref_pre_t = torch.where(hold, env._ref_pre_t + env.step_dt, env._ref_pre_t)
                over = hold & (env._ref_pre_t > pre_max_s)
                ph[over] = -1.0
                env._ref_pre_used = (env._ref_pre_used | over) & s["encounter"]
            ph[ph > self.ref["t_end"]] = -1.0
        active = ph > -0.99
        if phase_mode == "dist":
            # ===== C33：位置驱动相位 =====
            # `signed_dist`：正 = 机身在墙面前方（= 官方的 −x_rel）；非遭遇/无墙点时为 1.0。
            # 走到哪个距离，就比官方在那个距离的姿态。锚点 anchor_dist=0.58
            # （作者规格"0.58 米起步"，也正是参考 t=0 所在的位置 x_rel −0.561）。
            sd_ = s["signed_dist"]
            active = s["encounter"] & (sd_ < anchor_dist) & (sd_ > self.DL)
            b = ((self.DH - sd_) / (self.DH - self.DL) * (self.NB - 1)).long().clamp(0, self.NB - 1)
            seg_use = torch.as_tensor(segs, device=seg.device)[seg % len(segs)]   # 只用 seg0/seg1
            idx = self._dist_tbl[seg_use, b]
        else:
            idx = _frame_index(self.ref, seg, ph.clamp(min=self.ref["t0"]))
        q_ref = self.ref["q"][idx][:, :12]; p_ref = self.ref["pitch"][idx]
        dq = (jq[:, :12] - q_ref).pow(2).mean(dim=1)
        dp = (s["pitch_up"] - p_ref).pow(2)
        if pre_dist > 0.0 and phase_mode != "dist":
            # C32b（09-15 17:5x，作者问"能不能把官方那段放长/走过去"后查出来的）：
            # **预压墙窗口里只跟 pitch，不跟关节角。**
            # 理由：官方在 t=−0.5~0 是**站着不动**的（1.5 s 只挪 0.018 m），
            # 拿它的"站立关节角"去要求一台**正在往墙走**的机器人，自相矛盾，
            # 而且会直接奖励"停在墙前不动"（刷分）。
            # 走路是周期动作、**没有相位锚点**（翻越靠压墙事件锚住才跟得了），
            # 逐关节比对会因无法控制的步态相位差产生巨大误差。
            # "平着走过去"量化下来就是**机身俯仰**——相位无关，且正是实测的真实差异
            # （离墙 0.60 m：官方 −3.3° vs 我方 −27.8°）。压墙后恢复关节角+pitch 全跟。
            dq = torch.where(ph < 0.0, torch.zeros_like(dq), dq)
        r = torch.exp(-dq / (self.sigma_q ** 2)) * torch.exp(-dp / (self.sigma_p ** 2))
        return r * active.float()
