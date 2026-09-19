# 上台面（33 cm 立面）专用奖励 / 终止项 —— S10-V1（2026-09-08 深夜）
#
# 依据：决赛地形/厂家高台动作拆解与训练计划-20260908.md §一、§二（厂家高台模式慢动作逐帧拆解）。
# 真实动作：蓄力 → 右后猛蹬把前端顶上墙、右前轮沿立面滚到顶沿、左前跟上 → 机身 45° 保持、前轮拉后轮推到墙根
#          → 右后腿收腿跨沿 → 左后腿伸直贴立面被拉上来 → 落平。全程至少三轮触地、无腾空，真实约 4.5 s。
#
# 由此定的原则（只加不改，V0 的其它项一律沿用）：
#   1. 里程碑按「每个轮子的触地点相对出生地面的高度」给，势函数差分（只在轮子真上去了才有分，掉下来扣回去）；
#      只算触地的轮子 —— 抬着不落地不给分，想靠"举着轮子"骗分行不通。
#   2. 翻越相位内豁免：停滞惩罚、轮子撞立面（feet_stumble）、对角镜像；俯仰终止线从 44° 放到 60°（横滚不放）。
#   3. 禁腾空（四轮同时离地），翻越相位内少于三轮触地软罚。
#   4. 不规定哪只脚先动、不规定最后一只后轮是跨沿还是贴面拉上。
#
# 「翻越相位」= 四轮触地点高差 > 12 cm  或  前方 0.15～0.75 m 内地形比脚下高 > 15 cm（高程图射线）。
# 每步只算一次，缓存在 env 上，奖励与终止共用。
from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch
import os
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .parkour_rewards import feet_stumble as _feet_stumble_raw
from .parkour_rewards import stall_penalty as _stall_raw
from .rewards import _T
from .rewards import joint_mirror as _joint_mirror_raw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

WHEEL_RADIUS = 0.081      # S10 轮半径（URDF）
CLIMB_SPLIT = 0.12        # 四轮触地点高差 > 12 cm → 正在翻越
WALL_AHEAD_H = 0.15       # 前方地形比脚下高 > 15 cm → 面前有立面
TALL_WALL_H = 0.18        # ≥ 18 cm 才算「要用厂家那套动作上」的高墙（低坎直接滚过去，不给相位奖励）
BODY_WEIGHT_N = 19.0 * 9.81
WALL_AHEAD_X = (0.15, 0.75)   # 「前方」的纵向窗口（机体偏航系，m）
WALL_AHEAD_Y = 0.30           # 「前方」的横向半宽
CONTACT_N = 1.0           # 触地判定的接触力阈值（N）


def _ids(env: "ManagerBasedRLEnv") -> dict:
    c = getattr(env, "_climb_ids", None)
    if c is None:
        asset = env.scene["robot"]
        cs = env.scene.sensors["contact_forces"]
        a_ids, a_names = asset.find_bodies(".*_wheel")
        c_ids, c_names = cs.find_bodies(".*_wheel")
        if len(a_ids) != 4 or len(c_ids) != 4:
            raise RuntimeError(f"[climb] 轮子数不是 4：asset={a_names} contact={c_names}")
        k_ids, k_names = asset.find_joints(["hl_knee_joint", "hr_knee_joint"], preserve_order=True)
        if list(a_names) != ["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"]:
            raise RuntimeError(f"[climb] 轮序不是 fl,fr,hl,hr：{a_names}；front_leg_support/hind_leg_push 按 [:2]=前 [2:]=后 取列，顺序变了会取错腿")
        c = {"a": a_ids, "c": c_ids, "knee_h": k_ids}
        env._climb_ids = c
        print(f"[climb] 后膝关节：{k_names}", flush=True)
        print(f"[climb] 轮子 body：{a_names}（触地判定阈值 {CONTACT_N} N，翻越相位 高差>{CLIMB_SPLIT} 或 前方立面>{WALL_AHEAD_H}）", flush=True)
    return c


CMD_GATE_V = [0.0]
FLEX_GATE = [0.25]        # 「已蓄力」的门槛：后膝屈曲量（= −膝角 − 0.65）超过它才算蓄力到位。
#   0.25 → 膝角 −0.90。**实测这个门太松**（09-17 15:1x，MuJoCo 真模型逐帧，n=6）：
#     蓄力开始 → 抬前腿    官方 0.94~1.78 s，我方 **0.02~0.04 s**
#     抬腿那一刻的后膝      官方 −1.42~−1.50，我方 **−1.11**（刚过 −0.90 的门就抬）
#     该段推进              官方 +0.038~0.148 m，我方 **−0.014 m（反而退了）**
#   作者目视："没有走到靠近台阶，在靠近台阶之前就开始做动作" —— 就是这个门放行太早。
#   注意它是**闩锁**（flexed_prev | ...），一旦为真整个遭遇段保持，所以门槛只在"第一次"起作用。
#   被 6 个门共用（front_lift / hind_extend / push_posture / level_drag 等），提高它等于要求整条序列先蹲实。
SPEED_MATCH_TOL = [0.0]   # S1 起：|机体 vx − 指令 vx| 超过它时爬高进展/纪录不给分（速度旋钮真正管用）
HOLD_LAST_CONTACT = [False]
TOP_WIN_STEPS = [75]
LIFT_WIN_STEPS = [25]   # 抬前轮后罚「蓄力太短」的窗口（0.5 s @50Hz）      # 「刚翻上台」窗口长度（策略步）。decimation 4 × sim.dt 0.005 = 0.02 s/步 → 75 步 = 1.5 s。
ENCOUNTER_STALL_S = [0.0]   # >0 时：翻越相位在"最近这么多秒没有继续爬高"后自动退出（C9，09-14 23:5x）。
#   今晚两轮高台都栽在同一个机理：encounter 靠「墙高记忆 + 四轮高差>12cm」维持，机器人在墙前反复仰起就能长期满足，
#   于是任何挂在该相位上的**正向**奖励都能白拿 —— C7 的 cc_edge_clr 刷到 170 倍、C8 的 cc_pitch_hold 刷到 1000 倍，
#   两次都是同期真·爬高进展回落、扫描点登顶 0/3。打单项补丁只压住那一项，下一项照旧被刷，所以改门本身。
PROGRESS_SKIP_PRESSED = [False]   # C4（09-14 19:4x）：True 时爬高进展/纪录只算「触地且没顶在立面上」的轮。
#   C3 在 MuJoCo 里后轮越沿净空 0.001~0.06 m、顶踢面 11~19%，是「贴着立面拖上来」；官方是「两条后腿折叠跨过沿」。
#   根因就在这里：势函数 Φ 只看触地点高度，轮子顶着立面往上蹭一路都在涨 Φ，w=100 一路给分，正好奖励了官方不做的动作。
SPEED_MATCH_UPPER_ONLY = [False]   # S4：速度贴合门控只对超速生效
SPEED_EMA_TAU = [0.0]   # S7：>0 时门控用滑动平均速度（s）   # S1 起：爬高势函数用"最后触地高"，抬腿瞬间不再被扣分   # E7 起由 cfg 置 >0：指令速度（xy 模长或 |wz|）低于它时，翻越相位/遇墙/爬高进展全部关掉——给 0 要能站住


def _state(env: "ManagerBasedRLEnv") -> dict:
    """每步一次：四轮触地点高度（相对出生地面）、触地布尔、触地数、翻越相位。"""
    cache = getattr(env, "_climb_cache", None)
    if cache is not None and cache["step"] == env.common_step_counter:
        return cache
    ids = _ids(env)
    asset = env.scene["robot"]
    cs = env.scene.sensors["contact_forces"]
    hs = env.scene.sensors["height_scanner"]
    # 轮子触地点高度（轮心 − 半径），相对本 env 出生点地面
    z = _T(asset.data.body_pos_w)[:, ids["a"], 2] - WHEEL_RADIUS - env.scene.env_origins[:, 2:3]   # (N,4)
    fw = _T(cs.data.net_forces_w)[:, ids["c"], :]                                                   # (N,4,3)
    f = fw.norm(dim=-1)                                                                            # (N,4)
    f_xy = fw[..., :2].norm(dim=-1)
    f_z = fw[..., 2].abs()
    contact = f > CONTACT_N
    pressed = contact & (f_xy > 0.5 * f_z)          # 顶在立面上：水平分量大
    n_contact = contact.sum(dim=1)
    split = (z.max(dim=1).values - z.min(dim=1).values) > CLIMB_SPLIT
    # 前方有没有立面：高程图射线命中点转到传感器偏航系，取前窗最高点 − 脚下均值
    hits = _T(hs.data.ray_hits_w)                     # (N,R,3) 世界系
    pos = _T(hs.data.pos_w)                           # (N,3)
    quat = _T(hs.data.quat_w)                         # (N,4)
    rel = hits - pos.unsqueeze(1)
    valid = torch.isfinite(rel).all(dim=-1) & (rel.abs() < 1e5).all(dim=-1)
    rel = torch.where(valid.unsqueeze(-1), rel, torch.zeros_like(rel))
    n, r = rel.shape[0], rel.shape[1]
    local = quat_apply_inverse(quat.unsqueeze(1).expand(n, r, 4).reshape(-1, 4), rel.reshape(-1, 3)).reshape(n, r, 3)
    lx, ly = local[..., 0], local[..., 1]
    hz = torch.where(valid, hits[..., 2], torch.full_like(hits[..., 2], -1e6))
    ahead = valid & (lx > WALL_AHEAD_X[0]) & (lx < WALL_AHEAD_X[1]) & (ly.abs() < WALL_AHEAD_Y)
    under = valid & (lx.abs() < 0.12) & (ly.abs() < 0.12)
    ahead_z = torch.where(ahead, hz, torch.full_like(hz, -1e6)).max(dim=1).values
    under_cnt = under.sum(dim=1).clamp(min=1)
    under_z = torch.where(under, hits[..., 2], torch.zeros_like(hz)).sum(dim=1) / under_cnt
    wall_h = torch.clamp(ahead_z - under_z, 0.0, 0.6)
    wall_h = torch.where(ahead.any(dim=1) & under.any(dim=1), wall_h, torch.zeros_like(wall_h))
    wall_ahead = wall_h > WALL_AHEAD_H
    tall_wall = wall_h > TALL_WALL_H
    phase = split | wall_ahead
    # 遭遇记忆：一次「上墙」从看见高墙开始，到轮子高差消失且面前无墙结束；期间记住墙高（前轮上去后前方扫描就看不到墙了）
    mem = getattr(env, "_climb_wallh_mem", None)
    if mem is None or mem.shape[0] != n:
        mem = torch.zeros(n, device=z.device)
    mem = torch.where(tall_wall, torch.maximum(mem, wall_h), mem)
    encounter = tall_wall | (split & (mem > 0))
    if ENCOUNTER_STALL_S[0] > 0:                      # C9：爬高停滞就退出相位（见文件头 ENCOUNTER_STALL_S 说明）
        zb = z.max(dim=1).values
        st = getattr(env, "_climb_stall", None)
        t_now = env.episode_length_buf.float() * env.step_dt
        if st is None or st["zb"].shape[0] != n:
            st = {"zb": zb.clone(), "t": torch.zeros(n, device=z.device), "last": t_now.clone(), "step": -1}
            env._climb_stall = st
        if st["step"] != env.common_step_counter:
            st["step"] = env.common_step_counter
            fresh = t_now < st["last"] - 1e-6                      # 刚 reset：清零
            st["zb"] = torch.where(fresh, zb, st["zb"]); st["t"] = torch.where(fresh, torch.zeros_like(st["t"]), st["t"])
            st["last"] = t_now
            up = zb > st["zb"] + 0.01                              # 爬高又涨了 1 cm 以上 → 重新计时
            st["zb"] = torch.where(up, zb, st["zb"])
            st["t"] = torch.where(up, torch.zeros_like(st["t"]), st["t"] + env.step_dt)
        encounter = encounter & (st["t"] < ENCOUNTER_STALL_S[0])
        phase = phase & (st["t"] < ENCOUNTER_STALL_S[0])
    go = torch.ones(n, dtype=torch.bool, device=z.device)
    if CMD_GATE_V[0] > 0:
        cmdv = env.command_manager.get_command("base_velocity")
        go = (torch.linalg.norm(cmdv[:, :2], dim=1) > CMD_GATE_V[0]) | (cmdv[:, 2].abs() > CMD_GATE_V[0])
        phase = phase & go; encounter = encounter & go
    go_v = go
    if SPEED_MATCH_TOL[0] > 0:
        vb = _T(asset.data.root_lin_vel_b)[:, 0]; cmdv = env.command_manager.get_command("base_velocity")
        if SPEED_EMA_TAU[0] > 0:                                   # S7：用 1 s 滑动平均速度做门控——慢档迈上一级要瞬时冲一下，不该因此丢爬高分
            ema = getattr(env, "_cr_v_ema", None)
            if ema is None or ema.shape != vb.shape: ema = vb.clone()
            ema = ema + (vb - ema) * (env.step_dt / SPEED_EMA_TAU[0]); env._cr_v_ema = ema; vb = ema
        go_v = go & (((vb - cmdv[:, 0]).abs() < SPEED_MATCH_TOL[0]) if not SPEED_MATCH_UPPER_ONLY[0] else ((vb - cmdv[:, 0]) < SPEED_MATCH_TOL[0]))   # S4：只卡超速一侧，慢于指令也给爬高分（0.2 档停滞的根源）
    # ---- 「刚翻上台」窗口（B2i，09-17 16:0x）----
    # **不能用 `z > 常数` 判「已上台」**：z 是相对 **env 出生点** 算的，多级课程地形上
    # 「爬过任何东西之后永久为真」—— B2g 就是栽在这里（登顶 6/6 → 2/6，惩罚落到抬前腿上）。
    # 这里用**状态转变**：encounter 由真转假、转假前记过墙高、且四轮都已高过那个旧墙高 → 起计时。
    tp = getattr(env, "_climb_top_prev", None)
    if tp is None or tp["enc"].shape[0] != n:
        tp = {"enc": torch.zeros(n, dtype=torch.bool, device=z.device),
              "mem": torch.zeros(n, device=z.device),
              "cnt": torch.zeros(n, device=z.device)}
    just_topped = tp["enc"] & (~encounter) & (tp["mem"] > 0) & (z > 0.6 * tp["mem"].unsqueeze(1)).all(dim=1)
    cnt = torch.where(just_topped, torch.full_like(tp["cnt"], float(TOP_WIN_STEPS[0])),
                      torch.clamp(tp["cnt"] - 1.0, min=0.0))
    env._climb_top_prev = {"enc": encounter.clone(), "mem": mem.clone(), "cnt": cnt}
    top_win = cnt > 0
    # ---- 落台那一刻的腿长快照（B2z，09-18 08:3x，作者点名「左前轮上墙后没支撑」）----
    # 腿长 = 髋心→轮心距离；**只取决于膝角**（hipx/hipy 只把腿绕髋转，不改变该距离，已实测）。
    # 用 0.18/0.18 两连杆解析式 + 常数项，避免逐帧取 body 位置：
    #   L = sqrt(a² + b² + 2ab·cos q + c²)，c² = 0.015324（轮心相对膝轴的横向/纵向偏置）
    # **与 MuJoCo 真模型逐点核对：膝 0.0~2.7 全程误差 0.00000 m**（不加 c² 时差 0.021~0.068 m）。
    jq_all = _T(asset.data.joint_pos)
    ki_all = [asset.joint_names.index("%s_knee_joint" % l) for l in ("fl", "fr", "hl", "hr")]
    leg_len = torch.sqrt(torch.clamp(0.0648 + 0.0648 * torch.cos(jq_all[:, ki_all]) + 0.015324, min=1e-6))
    lg = getattr(env, "_climb_leg0", None)
    if lg is None or lg.shape[0] != n:
        lg = leg_len.clone()
    lg = torch.where(just_topped.unsqueeze(1), leg_len, lg)      # 刚进窗口 → 记下当时的腿长
    env._climb_leg0 = lg
    leg_drop = torch.clamp(lg - leg_len, min=0.0)                # 相对落台那刻缩短了多少（只记缩短）
    # ---- 推起段快照（B3a，09-18 08:5x，作者：「蹬腿那个是不是一起做，不然身子起不来、左前腿也伸不直」）----
    # 推起段 = 两前轮已上台 & 后轮还没全上 & 遭遇中。官方在这一段后腿**伸长** +0.041（蹬地）；
    # 我方右后腿**塌 0.173 m**（官方 0.049）—— 不是在蹬，是被压垮。
    # 用「相对该段起点的缩短量」做判据，**不依赖 `slow` 门**（昨晚六种办法都卡在那道门上，它只开 0~8%）。
    # 「已上台」用与 just_topped 同一把尺：轮子高过本次遭遇记下的墙高的 0.6 倍。
    # 轮列顺序 fl,fr,hl,hr 由 _ids 断言保证 → [:, :2] 前轮、[:, 2:] 后轮。
    _above = z > 0.6 * mem.unsqueeze(1)
    push_win = encounter & (mem > 0) & _above[:, :2].all(dim=1) & (~_above[:, 2:].all(dim=1))
    pp = getattr(env, "_climb_push_prev", None)
    if pp is None or pp["w"].shape[0] != n:
        pp = {"w": torch.zeros(n, dtype=torch.bool, device=z.device), "l": leg_len.clone()}
    just_push = push_win & (~pp["w"])                            # 推起段的上升沿
    pl = torch.where(just_push.unsqueeze(1), leg_len, pp["l"])
    env._climb_push_prev = {"w": push_win.clone(), "l": pl}
    hind_drop = torch.clamp(pl - leg_len, min=0.0)               # 相对推起段起点的缩短量
    mem = torch.where(encounter, mem, torch.zeros_like(mem))
    env._climb_wallh_mem = mem

    # 姿态与关节
    g = _T(asset.data.projected_gravity_b)
    pitch_up = -g[:, 0]                              # 抬头为正；45° ≈ 0.71
    q = _T(asset.data.joint_pos)[:, ids["knee_h"]]   # (N,2) 后膝角，默认 -0.65；更负 = 更屈
    hind_flex = torch.clamp(-q - 0.65, min=0.0)      # 屈膝量
    hind_ext = torch.clamp(0.65 - q.abs(), min=0.0)  # 伸直量
    root_z = _T(asset.data.root_pos_w)[:, 2] - torch.where(under.any(dim=1), under_z, env.scene.env_origins[:, 2])
    vx = _T(asset.data.root_lin_vel_b)[:, 0]
    # 后轮相对后髋的纵向位置（机体系）：正 = 收到髋前/正下方，负 = 拖在髋后面。后髋 x = −0.2277（URDF）
    bp = _T(asset.data.root_pos_w); bq = _T(asset.data.root_quat_w)
    wp = _T(asset.data.body_pos_w)[:, ids["a"], :]                                                # (N,4,3)
    wb = quat_apply_inverse(bq.unsqueeze(1).expand(n, 4, 4).reshape(-1, 4), (wp - bp.unsqueeze(1)).reshape(-1, 3)).reshape(n, 4, 3)
    hind_tuck = wb[:, 2:, 0] - (-0.2277)                                                            # (N,2)
    # 机身中心到立面的水平距离（偏航系，m）：前窗里比脚下高 >0.5×记忆墙高 的最近格子；没有就 1.0（机身已过沿或无墙）
    wall_cells = valid & (lx > 0.0) & (ly.abs() < WALL_AHEAD_Y) & ((hits[..., 2] - under_z.unsqueeze(1)) > 0.5 * torch.clamp(mem, min=TALL_WALL_H).unsqueeze(1))
    dist_wall = torch.where(wall_cells, lx, torch.full_like(lx, 1.0)).min(dim=1).values
    # 墙面位置记忆（16:35）：看见高墙时把"机身沿偏航方向到墙面的距离"换算成世界坐标存下来，越过墙面线后仍能算有符号距离
    bp_w = _T(asset.data.root_pos_w)
    bq_w = _T(asset.data.root_quat_w)
    from isaaclab.utils.math import quat_apply as _qa
    xaxis = _qa(bq_w, torch.tensor([1.0, 0.0, 0.0], device=bp_w.device).expand(n, 3))
    xaxis_xy = xaxis[:, :2] / xaxis[:, :2].norm(dim=1, keepdim=True).clamp(min=1e-6)
    wall_pt = getattr(env, "_climb_wall_pt", None)
    if wall_pt is None or wall_pt.shape[0] != n:
        wall_pt = torch.zeros(n, 2, device=bp_w.device)
    seen = tall_wall & (dist_wall < 0.99)
    wall_pt = torch.where(seen.unsqueeze(1), bp_w[:, :2] + xaxis_xy * dist_wall.unsqueeze(1), wall_pt)
    wall_pt = torch.where(encounter.unsqueeze(1), wall_pt, torch.zeros_like(wall_pt))
    env._climb_wall_pt = wall_pt
    signed_dist = ((wall_pt - bp_w[:, :2]) * xaxis_xy).sum(dim=1)          # 正 = 机身在墙面前方
    signed_dist = torch.where(encounter & (wall_pt.abs().sum(dim=1) > 0), signed_dist, torch.full_like(signed_dist, 1.0))
    # 本次遭遇是否已蓄力：后膝屈曲 ≥0.25 rad 且后轮着地 出现过（遭遇结束清零）
    flexed_prev = getattr(env, "_climb_flexed", None)
    if flexed_prev is None or flexed_prev.shape[0] != n:
        flexed_prev = torch.zeros(n, dtype=torch.bool, device=z.device)
    hind_flex_now = torch.clamp(-_T(asset.data.joint_pos)[:, ids["knee_h"]] - 0.65, min=0.0)
    flexed = (flexed_prev | ((hind_flex_now.max(dim=1).values > FLEX_GATE[0]) & contact[:, 2:].all(dim=1))) & encounter
    env._climb_flexed = flexed
    front_up = z[:, :2] > 0.6 * mem.unsqueeze(1)     # (N,2) 前轮已到台面高度（按记忆墙高）
    hind_up = z[:, 2:] > 0.6 * mem.unsqueeze(1)
    # ---- 蓄力时长（B3h，09-18 11:5x）----
    # 判据表里最大的缺口：**蓄力→抬腿 官方 1.135~1.870 s，我方 0.312 s（4~6 倍）**。
    # `st_hind_flex` 是 StageRecord —— 只奖励**蹲得多深**，**从没有任何项管过蹲了多久**。
    # 计时：遭遇中 & 已蓄力 & 两前轮都还没抬起 → 累加；离开遭遇 → 清零。
    # 抬前轮那一刻把时长锁存，供 crouch_time_penalty 在其后一小段窗口内持续罚。
    _fr_up = front_up.any(dim=1)
    cs = getattr(env, "_climb_crouch", None)
    if cs is None or cs["dur"].shape[0] != n:
        cs = {"dur": torch.zeros(n, device=z.device), "up": torch.zeros(n, dtype=torch.bool, device=z.device),
              "d0": torch.zeros(n, device=z.device), "cnt": torch.zeros(n, device=z.device)}
    holding = encounter & flexed & (~_fr_up)
    cdur = torch.where(holding, cs["dur"] + env.step_dt, torch.where(encounter, cs["dur"], torch.zeros_like(cs["dur"])))
    just_lift = encounter & _fr_up & (~cs["up"])                  # 抬前轮的上升沿
    d0 = torch.where(just_lift, cdur, cs["d0"])
    lcnt = torch.where(just_lift, torch.full_like(cs["cnt"], float(LIFT_WIN_STEPS[0])),
                       torch.clamp(cs["cnt"] - 1.0, min=0.0))
    env._climb_crouch = {"dur": cdur, "up": _fr_up.clone(), "d0": d0, "cnt": lcnt}
    crouch_dur0, lift_win = d0, lcnt > 0
    cache = {"go": go, "go_v": go_v, "step": env.common_step_counter, "z": z, "contact": contact, "n_contact": n_contact, "pressed": pressed,
             "f_z": f_z, "split": split, "wall_ahead": wall_ahead, "tall_wall": tall_wall, "wall_h": wall_h, "wall_mem": mem,
             "encounter": encounter, "phase": phase, "top_win": top_win, "leg_drop": leg_drop, "leg_len": leg_len, "crouch_dur0": crouch_dur0, "lift_win": lift_win, "push_win": push_win, "hind_drop": hind_drop, "under_z": under_z, "under_ok": under.any(dim=1),
             "pitch_up": pitch_up, "hind_flex": hind_flex, "hind_ext": hind_ext, "root_z": root_z, "vx": vx,
             "front_up": front_up, "hind_up": hind_up, "hind_tuck": hind_tuck, "dist_wall": dist_wall, "flexed": flexed, "signed_dist": signed_dist}
    # 注：轮底高 z（轮心 − 半径 − 出生地面）已在上面的 cache 里，块13 的 lead_front_onto_top 直接用
    env._climb_cache = cache
    return cache


class WheelHeightProgress(ManagerTermBase):
    """里程碑（势函数差分）：Φ = Σ_轮 触地 × clamp(触地点高度, 0, max_rise)，r = Φ_t − Φ_{t−1}。

    轮子沿立面贴着滚上去时一路有分（起步奖励）；跨沿放到台面时一次性得 0.33；掉回去扣回去；
    抬着不放不得分。episode 开始那一步不算差分。
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.phi_prev = torch.zeros(env.num_envs, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.valid[env_ids] = False
        if getattr(self, "zc_hold", None) is not None: self.zc_hold[env_ids] = 0.0

    def __call__(self, env: "ManagerBasedRLEnv", max_rise: float = 0.5) -> torch.Tensor:
        s = _state(env)
        zc = torch.clamp(s["z"], 0.0, max_rise)
        ok = s["contact"] & (~s["pressed"]) if PROGRESS_SKIP_PRESSED[0] else s["contact"]   # C4：顶着立面往上蹭不算进展
        if HOLD_LAST_CONTACT[0]:                                   # S1：腾空轮保留最后触地高，抬腿不掉势能
            prev = getattr(self, "zc_hold", None)
            if prev is None or prev.shape != zc.shape: prev = zc.clone()
            zc = torch.where(ok, zc, prev); self.zc_hold = zc
            phi = zc.sum(dim=1)
        else:
            phi = (zc * ok.float()).sum(dim=1)
        r = torch.where(self.valid, phi - self.phi_prev, torch.zeros_like(phi)) * s["go_v"].float()   # E7：没指令不给；S1：速度不贴指令也不给
        self.phi_prev = phi
        self.valid[:] = True
        return r


class WheelHeightRecord(ManagerTermBase):
    """里程碑（首达纪录）：每局每只轮子的「触地点最高纪录」每被刷新一次，给 新纪录 − 旧纪录；掉下来不扣。

    与势函数差分互补：差分项让「上去并待住」和「掉回去」有正负，这项让「多爬高一点」永远是正的 ——
    课程直接从 31 cm 起步时，部分进展就靠它给梯度。只算触地的轮子；每局上限 max_rise/轮。
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.zmax = torch.zeros(env.num_envs, 4, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.zmax[env_ids] = 0.0

    def __call__(self, env: "ManagerBasedRLEnv", max_rise: float = 0.5) -> torch.Tensor:
        s = _state(env)
        _ok = s["contact"] & (~s["pressed"]) if PROGRESS_SKIP_PRESSED[0] else s["contact"]   # C4：同 Progress，贴立面蹭上去不算纪录
        zc = torch.clamp(s["z"], 0.0, max_rise) * _ok.float()
        gain = torch.clamp(zc - self.zmax, min=0.0)
        self.zmax = torch.maximum(self.zmax, zc)
        return gain.sum(dim=1) * s["go_v"].float()


def base_height_flat_l2(env: "ManagerBasedRLEnv", target_height: float = 0.43, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                        target_still: float = 0.0, command_name: str = "base_velocity") -> torch.Tensor:
    """站高惩罚（翻越相位外）：(机身 z − 脚下地形 z − target)²。面前有墙或轮子已分层时不罚（蓄力下沉、45° 抬头都是正常动作）。

    为什么加：09-08 V1 首跑 200 迭代后策略塌成趴低劈腿爬行（机身 0.23 m、0.15 m/s），V16 配方里 base_height 权重是 0，
    没有任何项把机身托回站高；盲策略原生站高 0.43，这里就钉在 0.43。
    """
    s = _state(env)
    z = _T(env.scene[asset_cfg.name].data.root_pos_w)[:, 2]
    ref = torch.where(s["under_ok"], s["under_z"], env.scene.env_origins[:, 2])
    tgt = torch.full_like(z, target_height)
    if target_still > 0:                                         # T1：零指令时站高目标另给（官方静止 0.376 vs 踏步 0.345）
        c = env.command_manager.get_command(command_name)
        still = (torch.linalg.norm(c[:, :2], dim=1) < 0.05) & (c[:, 2].abs() < 0.05)
        tgt = torch.where(still, torch.full_like(z, target_still), tgt)
    err = z - ref - tgt
    return torch.square(err) * (~s["phase"]).float()


class StageRecord(ManagerTermBase):
    """厂家上墙动作各相位的「首达纪录」奖励（作者 09-08 深夜确认的八段拆解）。

    每个 env 每次遭遇高墙（≥18 cm）记一次纪录：量每被刷新一次给 新纪录 − 旧纪录，封顶 cap；遭遇结束（轮子高差消失且面前无墙）
    或 episode 重置时清零。只给一次、不奖励"保持"，所以在墙前一直翘着/一直蹲着拿不到更多分。
    quantity：
      crouch      蓄力·下沉：前轮顶墙、四轮着地、前轮未上 → 机身低于站高 0.43 的量（cap 0.05）
      hind_flex   蓄力·后腿屈膝：同上条件 → 后膝屈曲量（cap 0.6 rad）
      front_lift  右前搭/左前跟·前腿主动抬轮：高墙前、两后轮着地 → 每只前轮离地高度（空中也算，cap 墙高+0.05）
      pitch_up    抬头：高墙遭遇中、两后轮着地 → 抬头角（cap 0.71 ≈ 45°）
      hind_push   右后蹬·蹬地力：高墙遭遇中、前轮未全上或刚上 → 单只后轮竖直接触力/体重（cap 1.0）
      hind_extend 右后蹬·伸直：两前轮已上、后轮着地 → 后膝伸直量（cap 0.6 rad）
      hind_lift   右后跨·收腿抬轮：两前轮已上 → 每只后轮离地高度（空中也算，cap 墙高+0.05）
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.q = cfg.params["quantity"]
        self.per_wheel = self.q in ("front_lift", "hind_lift")
        shape = (env.num_envs, 2) if self.per_wheel else (env.num_envs,)
        self.rec = torch.zeros(shape, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.rec[env_ids] = 0.0

    def __call__(self, env: "ManagerBasedRLEnv", quantity: str, cap: float = 1.0) -> torch.Tensor:
        s = _state(env)
        hind_down = s["contact"][:, 2:].all(dim=1)
        front_any_pressed = s["pressed"][:, :2].any(dim=1)
        front_all_up = s["front_up"].all(dim=1)
        front_none_up = ~s["front_up"].any(dim=1)
        front_part_up = (s["z"][:, :2] > 0.15 * s["wall_mem"].unsqueeze(1)).any(dim=1) & (s["wall_mem"] > 0)
        slow = s["vx"].abs() < 0.5
        enc = s["encounter"]
        capv = torch.full_like(s["pitch_up"], cap)
        if self.q == "crouch":
            tucked = (s["hind_tuck"] > -0.10).all(dim=1)          # 12:25：后轮收到髋下才算蓄力，向后劈开压低不算
            # 15:10（GPT 拆解 + 作者）：屈腿预备与右前轮抬起是重叠的，不要求前轮都没抬；后轮着地、前轮顶墙或正在抬即可
            gate = enc & hind_down & (front_any_pressed | s["front_up"].any(dim=1) | (s["n_contact"] < 4)) & ~front_all_up & tucked
            val = torch.clamp(0.43 - s["root_z"], 0.0, cap)
        elif self.q == "hind_flex":
            gate = enc & hind_down & ~front_all_up
            val = torch.clamp(s["hind_flex"].max(dim=1).values, 0.0, cap)
        elif self.q == "pitch_up":
            # 作者 09-09 00:20：不要整个身子架上去。抬头封顶按阶段：前轮没搭上 20°(0.34)，一只搭上一部分 30°(0.5)，两只都上了才 45°(cap)
            gate = enc & hind_down & (front_any_pressed | front_part_up) & slow
            stage_cap = torch.where(front_all_up, capv, torch.where(front_part_up, torch.full_like(capv, 0.55), torch.full_like(capv, 0.42)))   # 01:00：前轮未搭上 24°，搭上一部分 32°
            val = torch.minimum(torch.clamp(s["pitch_up"], min=0.0), stage_cap)
        elif self.q == "hind_push":
            # 右后猛蹬（作者 13:20）：必须接在蓄力之后；一只前轮刚搭上一点、两后轮着地、机身还留在后面（离墙>20 cm）或已抬到 30°
            room = (s["dist_wall"] > 0.20) | (s["pitch_up"] > 0.5)
            gate = enc & hind_down & (front_any_pressed | front_part_up) & ~front_all_up & slow & s["flexed"] & room
            val = torch.clamp(s["f_z"][:, 2:].max(dim=1).values / BODY_WEIGHT_N, 0.0, cap)
        elif self.q == "hind_extend":
            room = (s["dist_wall"] > 0.20) | (s["pitch_up"] > 0.5)
            gate = enc & (front_part_up | front_all_up) & s["contact"][:, 2:].any(dim=1) & slow & s["flexed"] & room
            val = torch.clamp(s["hind_ext"].max(dim=1).values, 0.0, cap)
        elif self.q == "front_lift":
            # 一只一只上：某只前轮抬起只在另一只前轮仍触地（地面或墙面）时计分；两只一起抬不给分
            # 15:45：并且必须已蓄力（作者顺序：后腿蓄力 → 右前搭）。从地面起步时策略是后腿伸直着把前轮架上去，之后没有蹬地行程
            other_down = s["contact"][:, :2].flip(dims=[1])            # (N,2)：对侧前轮是否触地
            gate = enc & hind_down & s["flexed"]
            capw = torch.clamp(s["wall_mem"] + 0.05, max=cap).unsqueeze(1)
            val = torch.clamp(s["z"][:, :2], min=0.0)
            val = torch.minimum(val, capw) * other_down.float()
        elif self.q == "body_rise":
            # **给「身子上去」付钱**（B2l，09-17 17:4x）。作者：「你先完成能翻墙站稳」。
            # 实测：官方翻越时把机身从 0.435 抬到 **0.715**（≈台上站高，+0.28 m）；
            # 我方只抬到 **0.576**（比台上应有站高 0.75 低 **0.18 m**），过去之后再花 2~3 s 撑起来。
            # 根因：奖励付的是「轮子上去」（wheel_height_progress 300 + record 200），
            # **没有任何一项付「身子上去」**——`base_height_flat` 在翻越相位被 `(~phase)` 整个豁免。
            # 策略照做：把四个轮子送上台面，身子留在低处爬过去。
            # 量 = 机身比平地站高（0.43）高出多少，相对**出生地面**（不是脚下地形，避免过墙沿时跳变）。
            # 门 = 两前轮都已在台面：必须真爬上去才进得来，墙前抬头翘身子刷不到。
            asset = env.scene["robot"]
            z_abs = _T(asset.data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
            gate = enc & s["front_up"].all(dim=1)
            val = torch.clamp(z_abs - 0.43, torch.zeros_like(z_abs), torch.clamp(s["wall_mem"], max=cap))
        elif self.q == "hind_lift":
            gate = enc & front_all_up
            capw = torch.clamp(s["wall_mem"] + 0.05, max=cap).unsqueeze(1)
            val = torch.clamp(s["z"][:, 2:], min=0.0)
            val = torch.minimum(val, capw)
        else:
            raise ValueError(f"[climb] 未知 quantity {self.q}")
        g = gate.unsqueeze(1) if self.per_wheel else gate
        val = torch.where(g, val, torch.zeros_like(val))
        gain = torch.clamp(val - self.rec, min=0.0)
        self.rec = torch.maximum(self.rec, val)
        # 遭遇结束清零，下一堵墙重新记
        self.rec = torch.where(enc.unsqueeze(1) if self.per_wheel else enc, self.rec, torch.zeros_like(self.rec))
        return gain.sum(dim=1) if self.per_wheel else gain


def approach_speed_penalty(env: "ManagerBasedRLEnv", v_max: float = 0.6) -> torch.Tensor:
    """提·慢速抵上：前方 0.75 m 内有 ≥18 cm 高墙时，机身前进速度超过 v_max 的部分（作者：0.3~0.5 m/s 抵上；
    首版策略学成 1.3 m/s 撞墙靠惯性抬头）。"""
    s = _state(env)
    return s["tall_wall"].float() * torch.clamp(s["vx"] - v_max, min=0.0)


def hoist_penalty(env: "ManagerBasedRLEnv", z_max: float = 0.46) -> torch.Tensor:
    """架空惩罚（作者 09-09 00:20）：高墙遭遇中、前轮还没搭上一部分（<15% 墙高）时，机身高于 z_max 的量。
    整个身子架上去后腿就伸直了没法蹬；正确的是右前轮刚搭上一部分、机身还低、右后腿再猛蹬。"""
    s = _state(env)
    front_part_up = (s["z"][:, :2] > 0.15 * s["wall_mem"].unsqueeze(1)).any(dim=1) & (s["wall_mem"] > 0)
    return (s["encounter"] & ~front_part_up).float() * torch.clamp(s["root_z"] - z_max, min=0.0)


def push_drive(env: "ManagerBasedRLEnv", v_max: float = 0.5) -> torch.Tensor:
    """右后蹬·前推（每步，11:55 加）：一只或两只前轮已在台面、两后轮着地时，按机身前进速度给分（clamp 0~v_max）。
    model_1300 的稳态是"前轮搭上后原地不动"，推进姿态项要求在前进但静止不亏，所以要有一项直接奖励向前推。"""
    s = _state(env)
    ok = s["encounter"] & s["front_up"].any(dim=1) & s["contact"][:, 2:].all(dim=1) & ~s["hind_up"].any(dim=1) & s["flexed"]
    return ok.float() * torch.clamp(s["vx"], 0.0, v_max)


def hind_splay_penalty(env: "ManagerBasedRLEnv", back_max: float = 0.12) -> torch.Tensor:
    """后腿后摊惩罚（12:25）：上墙遭遇中、后轮还没上去时，后轮拖在后髋后方超过 back_max 的量之和。
    model_1600 把"蓄力"学成了后腿向后完全伸直压低身子，没有蹬地行程；视频里是后轮收到髋正下方、膝屈曲。"""
    s = _state(env)
    gate = s["encounter"] & ~s["hind_up"].any(dim=1)
    return gate.float() * torch.clamp(-s["hind_tuck"] - back_max, min=0.0).sum(dim=1)


def yaw_keep_penalty(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), always: bool = False) -> torch.Tensor:
    """偏航惩罚：正对墙才能上，转 90° 沿墙跑是漏洞（专家环境只朝 +x 出生，偏航 0 = 正对墙）。值 = (1−cos yaw)，always=False 时只在遭遇段算。
    23:40 直测：策略在离墙 1 m（遭遇门槛之外）就开始转，所以专家环境用 always=True 全程算。"""
    s = _state(env)
    asset = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    q = q.torch if hasattr(q, "torch") else q
    x_, y_, z_, w_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]                    # IsaacLab 6：xyzw
    yaw = torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ ** 2 + z_ ** 2))
    gate = torch.ones_like(yaw) if always else s["encounter"].float()
    return (1.0 - torch.cos(yaw)) * gate


def wall_approach_reward(env: "ManagerBasedRLEnv", v_max: float = 0.5, d_touch: float = 0.30) -> torch.Tensor:
    """遭遇段内向墙前进奖励（2500 直测：策略停在遭遇段边界 0.8 m 处不靠近——段内速度跟踪已关、靠近只有惩罚）。
    门：遭遇 & 机身中心离墙 > d_touch（前轮尚未触墙）& 前轮未抬；值 = clip(机体前向速度, 0, v_max)。"""
    s = _state(env)
    gate = s["encounter"] & (s["dist_wall"] > d_touch) & ~s["front_up"].any(dim=1)
    return gate.float() * torch.clamp(s["vx"], 0.0, v_max)


def too_close_penalty(env: "ManagerBasedRLEnv", d_min: float = 0.22, pitch_free: float = 0.5) -> torch.Tensor:
    """趴墙惩罚（作者 13:20：右前轮只搭一点、不要整个身子趴上去）：后轮还在地上、机身没抬到 pitch_free 时，
    机身中心离立面近于 d_min 的量。留出后腿发力空间。"""
    s = _state(env)
    gate = s["encounter"] & ~s["hind_up"].any(dim=1) & (s["pitch_up"] < pitch_free)
    return gate.float() * torch.clamp(d_min - s["signed_dist"], min=0.0)     # 16:35：有符号距离，越过墙面线也罚


def launch_reward(env: "ManagerBasedRLEnv", vz_max: float = 0.8) -> torch.Tensor:
    """蹬起奖励（作者 13:20：右后腿发力蹬起来、整个机身上墙）：已蓄力、一只前轮刚搭上一点、两后轮着地时，
    机身竖直速度（clamp 0~vz_max）按步给分。奖励的是"整机被蹬起来"这一下。"""
    s = _state(env)
    front_part_up = (s["z"][:, :2] > 0.15 * s["wall_mem"].unsqueeze(1)).any(dim=1) & (s["wall_mem"] > 0)
    gate = s["encounter"] & s["flexed"] & front_part_up & s["contact"][:, 2:].all(dim=1) & ~s["hind_up"].any(dim=1)
    vz = _T(env.scene["robot"].data.root_lin_vel_w)[:, 2]
    return gate.float() * torch.clamp(vz, 0.0, vz_max)


def flexed_frac(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """只做日志：遭遇中已蓄力的占比。"""
    return _state(env)["flexed"].float()


def track_lin_vel_xy_exp_climb_aware(env: "ManagerBasedRLEnv", command_name: str = "base_velocity", std: float = 0.7071,
                                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), d_touch: float = 0.30) -> torch.Tensor:
    """速度跟踪奖励：只在"遭遇且前轮已触墙（机身中心离墙 ≤ d_touch）"时关掉。
    09-10 03:00 探针：原来整个遭遇段关掉 → 一进遭遇段少 +2~4.7/步，策略停在遭遇边界 0.8 m 处来回蹭，永远不靠墙。"""
    from .rewards import track_lin_vel_xy_exp as _track
    r = _track(env, command_name=command_name, std=std, asset_cfg=asset_cfg)
    s = _state(env)
    off = s["encounter"] & (s["dist_wall"] <= d_touch)
    return r * (~off).float()


def stall_penalty_encounter_free(env: "ManagerBasedRLEnv", command_name: str = "base_velocity",
                                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), cmd_min: float = 0.3, vel_max: float = 0.1) -> torch.Tensor:
    """停滞惩罚，整个高墙遭遇中豁免（17:05）：蓄力那 0.6 s 本来就是停着的。"""
    return _stall_raw(env, command_name, asset_cfg, cmd_min, vel_max) * (~_state(env)["encounter"]).float()


def push_posture(env: "ManagerBasedRLEnv", pitch_lo: float = 0.42, pitch_hi: float = 0.82, vx_min: float = 0.05) -> torch.Tensor:
    """右后蹬·推进姿态（每步）：两前轮已在台面、两后轮着地、机身抬头 25°~55°、机身在前进 → 1。原地翘着不算。"""
    s = _state(env)
    ok = s["encounter"] & s["front_up"].all(dim=1) & s["contact"][:, 2:].all(dim=1) & ~s["hind_up"].any(dim=1) & s["flexed"]
    ok &= (s["pitch_up"] > pitch_lo) & (s["pitch_up"] < pitch_hi) & (s["vx"] > vx_min)
    return ok.float()


def level_drag(env: "ManagerBasedRLEnv", vx_min: float = 0.05) -> torch.Tensor:
    """左后爬/右后跨·拉上落平（每步）：三轮已在台面、最后一只后轮还在下面且有接触、机身前进 → (1 − 俯仰/45°)。
    奖励"靠前进和落平把最后一只轮子拉上来"，不规定它贴面还是跨沿。"""
    s = _state(env)
    up = torch.cat([s["front_up"], s["hind_up"]], dim=1)
    three_up = up.sum(dim=1) == 3
    last_down_contact = ((~up) & s["contact"]).any(dim=1)
    ok = s["encounter"] & three_up & last_down_contact & (s["vx"] > vx_min)
    return ok.float() * torch.clamp(1.0 - s["pitch_up"].abs() / 0.71, min=0.0)


def lin_vel_z_l2_climb_aware(env: "ManagerBasedRLEnv", land_k: float = 1.0,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """竖直速度惩罚。**豁免只覆盖「往上推」，不覆盖「往下砸」**（09-17 14:0x 修）。

    原实现是 `v² * (~phase)` —— 整个翻越相位全关。注释写明豁免是为了**起步那 0.3 s 的猛推**，
    但**落地也在同一相位里**，于是被顺带豁免：**从起跳到落地整段没有任何项在罚竖直冲击**。

    实测代价（B2c，n=6，作者目视指出「落地不够稳，真实情况下容易摔」）：

        台面站高应 ≈0.75        机身最低掉到 **0.527~0.545**（砸下去 0.22 m）
        落地 vz                 **−0.41 ~ −0.74 m/s**
        |roll| 峰               **17.6~19.9°**（官方 9.4~14.5°）
        恢复用时                2.1~2.6 s

    对比官方：末端 z 0.705~0.718、末端 vz +0.007~+0.031（平稳停住），|roll| 峰 9.4~14.5°。
    而我方 |roll| **中位 1.0°、90 分位 3.0°，都比官方好** —— 说明不是「一直不稳」，
    是**落地那一下没缓冲**。所以不该全程压横滚（会拖慢动作），要治的就是竖直冲击本身。

    改法：相位内只罚 `vz < 0`（下落），`vz > 0`（上推）仍然免罚。

    `land_k`：**只放大相位内的下落罚**，非翻越段（走路/踏步）行为完全不变 ——
    直接调全局 weight 会连带影响平地步态，那不是本次要动的东西。
    实测该项全局 weight = −2.0，落地 vz≈−0.7 → 该刻每步罚 0.98，主项约 6.3/步，仅占 16%，太弱。

    ## 实测证伪（09-17 14:0x，**本改动已回退**）

    作者问：「你不是落地惩罚只给落地部分吗，为什么会影响上墙」。用 MuJoCo 真模型 FK 逐帧查：

    **一、`vz<0` 不等于落地。** B2c 翻越段 1148 帧里 372 帧 vz<0，按阶段拆 Σvz²·dt：

        蓄力下蹲（离墙 0.80→0.45）  304 帧  0.0113  **48%**   最负 vz −0.37
        起动上冲（0.45→0.15）         0 帧  0.0000
        前轮上台（0.15→−0.30）       35 帧  0.0114  **48%**   最负 vz −0.41
        落地稳住（过墙后）            33 帧  0.0008   **3.4%**

    **96.6% 的罚落在上墙过程里。** 蓄力本身就是往下的，land_k=4 等于把「别蹲」放大四倍。

    **二、落地那一刻 `phase` 本来就是假的，根本没被豁免过。** 砸得最猛那一帧（n=3）：

        vz −0.41 / −0.72 / −0.39      四轮高差 0.062 / 0.034 / 0.083   均 < CLIMB_SPLIT=0.12

    机器人已在台面上、前方无立面 → `split` 与 `wall_ahead` 双假 → `phase` 假
    → **落地一直在挨 1× 的标准罚（权重 −2.0）**。

    净效果：落地罚没变，蓄力和上台平白多挨 land_k 倍。**纯负作用、零收益**，
    f1（land_k=4）登顶 0/6 即此。land_k=2 同理，14:15 停训回退，不再验证。

    教训：**「豁免」写在代码里不等于它真的覆盖了那一段**。判定门是 phase，
    我却按时间直觉以为落地在 phase 内，从没验过。以后加门先量门在哪一段为真。
    """
    v = _T(env.scene[asset_cfg.name].data.root_lin_vel_b)[:, 2]
    ph = _state(env)["phase"]
    # 09-17 14:1x 回退：land_k 这条路已证伪，见 docstring 末尾「实测证伪」。
    # 相位内全豁免（原始行为）；land_k 保留只为配置兼容，**不再生效**。
    return torch.square(v) * (~ph).float()


def joint_torques_l2_climb_aware(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """力矩正则，翻越相位内关掉（允许单腿大力矩）。"""
    tau = _T(env.scene[asset_cfg.name].data.applied_torque)[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(tau), dim=1) * (~_state(env)["phase"]).float()


def encounter_frac(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """只做日志：处于高墙遭遇中的占比。"""
    return _state(env)["encounter"].float()


class ClimbCompleteBonus(ManagerTermBase):
    """上墙完成奖励：四轮都到台面高度（按记忆墙高 60%）第一次出现时给 1（每局一次）。"""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.done[env_ids] = False
        if getattr(self._env, "_climb_ep_done", None) is not None:
            self._env._climb_ep_done[env_ids] = False
            self._env._climb_ep_front[env_ids] = False

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        s = _state(env)
        allup = s["front_up"].all(dim=1) & s["hind_up"].all(dim=1) & (s["wall_mem"] > 0)
        first = allup & ~self.done
        self.done |= allup
        if getattr(env, "_climb_ep_done", None) is None:
            env._climb_ep_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            env._climb_ep_front = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._climb_ep_done |= allup
        env._climb_ep_front |= s["front_up"].any(dim=1) & (s["wall_mem"] > 0)
        hk = getattr(env, "_climb_spawn_hooked", None)
        env._climb_ground_first = first & (~hk if hk is not None else torch.ones_like(first))
        return first.float()


def ground_complete_frac(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """只做日志（w=0.001）：不是从"前轮已搭上"出生、而是从地面/墙前起步的局里，首次四轮上顶的事件。"""
    g = getattr(env, "_climb_ground_first", None)
    return g.float() if g is not None else torch.zeros(env.num_envs, device=env.device)


def airborne_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """四轮同时离地 → 1（禁腾空；真实动作全程有轮着地）。"""
    return (_state(env)["n_contact"] == 0).float()


def support_deficit_climb(env: "ManagerBasedRLEnv", min_contacts: int = 3) -> torch.Tensor:
    """轮子已分层（翻越中）时触地轮数少于 min_contacts 的差额（软约束：真实动作至少三轮支撑）。平地/转向步态不管。"""
    s = _state(env)
    return (s["split"] | s["encounter"]).float() * torch.clamp(min_contacts - s["n_contact"].float(), min=0.0)


def climb_phase_frac(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """只做日志：翻越相位占比（权重 0）。"""
    return _state(env)["phase"].float()


def wall_ahead_frac(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """只做日志：前方有立面的占比（权重 0）。"""
    return _state(env)["wall_ahead"].float()


def stall_penalty_climb_aware(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_min: float = 0.3,
    vel_max: float = 0.1,
) -> torch.Tensor:
    """停滞惩罚：只在轮子已经分层（真正在翻越）时豁免。面前有墙但还没动手仍然罚 —— 「顶墙不动」必须有代价。"""
    return _stall_raw(env, command_name, asset_cfg, cmd_min, vel_max) * (~_state(env)["split"]).float()


def feet_stumble_climb_aware(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """轮子撞立面惩罚，翻越相位内豁免（前轮顶着立面滚上去、后轮贴面被拉上来都是水平力远大于竖直力）。"""
    return _feet_stumble_raw(env, sensor_cfg) * (~_state(env)["phase"]).float()


def _climb_seq_state(env, asset_cfg, lo, hi):
    """C38 时序链的公共量：蓄力深度 load、前轮离地高度、窗口闸门。"""
    s = _state(env)
    asset = env.scene[asset_cfg.name]
    jq = asset.data.joint_pos; jq = jq.torch if hasattr(jq, "torch") else jq
    bp = asset.data.body_pos_w; bp = bp.torch if hasattr(bp, "torch") else bp
    jn = asset.joint_names; bn = asset.body_names
    hl = jq[:, jn.index("hl_hipy_joint")]; hr = jq[:, jn.index("hr_hipy_joint")]
    load = torch.minimum(hl, hr)                    # 两后腿都蹲够才算蓄力到位
    org = env.scene.env_origins[:, 2]
    fl = bp[:, bn.index("fl_wheel"), 2] - org - 0.081
    fr = bp[:, bn.index("fr_wheel"), 2] - org - 0.081
    sd = s["signed_dist"]
    gate = s["encounter"] & (sd >= lo) & (sd <= hi)
    return load, fl, fr, gate


def climb_load_depth(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     lo: float = 0.15, hi: float = 0.70, target: float = 1.0) -> torch.Tensor:
    """**蓄力深度罚**（C38-A，作者五条②"后腿弯曲蓄力"）。
    官方蓄力段后腿 hipy 峰 hl **+1.08** / hr **+1.01**（两腿都深蹲）；
    我方 C36b 切0.5m 只到 +0.70/+0.68（对称做到了、**深度只有七成**），切1.0m 更浅 +0.44/+0.28。
    罚 (target − load)² ，load = min(hl_hipy, hr_hipy) —— 取小的那条，防止"一条蹲够就交差"。"""
    load, _, _, gate = _climb_seq_state(env, asset_cfg, lo, hi)
    return (target - load).clamp(min=0.0).pow(2) * gate.float()


def climb_premature_front(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                          lo: float = 0.15, hi: float = 0.70, ready: float = 0.85,
                          thr: float = 0.05) -> torch.Tensor:
    """**蓄力未完成就抬前轮罚**（C38-B，作者"你可以不那么急，完成蓄力动作，右前轮再开始往上"）。
    放行条件 load > ready(0.85)；未放行时前轮高出 thr 的部分按平方罚。
    实测我方是"没蹲够就急着抬左前轮"：蓄力段 fl +175mm 而后腿只蹲到 +0.70。"""
    load, fl, fr, gate = _climb_seq_state(env, asset_cfg, lo, hi)
    not_ready = (load < ready) & gate
    ex = torch.maximum(fl, fr).sub(thr).clamp(min=0.0)
    return ex.pow(2) * not_ready.float()


def climb_front_order(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                      lo: float = 0.15, hi: float = 0.70, ready: float = 0.85) -> torch.Tensor:
    """**前轮顺序罚：右前先、左前跟**（C38-C，作者五条②③"右前轮伸出…右前轮上去后左前轮跟随"）。
    官方蓄力段 fr **+168mm** > fl **+91mm**（右前领先 77mm）；我方反过来 fl +175 > fr +123（左前领先 52mm）。
    只在**蓄力已完成**（load ≥ ready）时生效——蓄力阶段两轮都该在地上，顺序无意义。
    罚 (fl − fr) 的**正值部分**：左前高过右前才罚，右前领先不罚，不强求领先多少。"""
    load, fl, fr, gate = _climb_seq_state(env, asset_cfg, lo, hi)
    active = (load >= ready) & gate
    return (fl - fr).clamp(min=0.0).pow(2) * active.float()


def front_early_lift(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     lo: float = 0.45, hi: float = 0.70, thr: float = 0.05) -> torch.Tensor:
    """**近墙前抬前轮罚**（C37，09-15 21:1x，作者五条里 ③④ 的根）。

    实测（离墙 0.45~0.60 m，官方此时四轮仍全贴地）：
        官方 seg0  fl 离地占比 **0%**（均高 +28mm）   fr 7%（+26mm）
        C36_14798  fl 离地占比 **100%**（均高 **+334mm**）  fr 40%（+51mm）
        C35_14399  fl **100%**（+327mm）  fr 54%
    → **在离墙半米、官方四轮还全贴地时，我方左前轮已举在 33 cm 高空、全时不落地。**

    顺序也反了（0.15~0.60 m 全段均高）：官方 **fr 168mm > fl 91mm**（右前轮领先），
    我方 **fl 332mm > fr 282mm**（左前轮领先）。

    这解释了作者的 ③④：右前轮到 0.35 m 才动，只能在 0.25 s 内窜 522mm 去够墙沿（③）；
    四个支撑点只剩两后轮，推不起身子（④）。

    **不硬编"哪个轮先抬"** —— 只要求近墙前别抬（离墙 lo~hi 窗口内，罚前轮离地超 thr 的部分），
    剩下的左右顺序交给位置驱动的 `st_ref_track` 去塑形。硬编顺序等于又造一个手工先验，
    而 C34 已经证明手工俯仰项与官方参考打架（见 C35 注释）。
    """
    s = _state(env)
    asset = env.scene[asset_cfg.name]
    bp = asset.data.body_pos_w; bp = bp.torch if hasattr(bp, "torch") else bp
    names = asset.body_names
    fl = bp[:, names.index("fl_wheel"), 2]
    fr = bp[:, names.index("fr_wheel"), 2]
    # 地面高度：翻越前机器人在地面上，取环境原点 z
    org = env.scene.env_origins[:, 2]
    hfl = (fl - org - 0.081 - thr).clamp(min=0.0)
    hfr = (fr - org - 0.081 - thr).clamp(min=0.0)
    sd = s["signed_dist"]
    gate = s["encounter"] & (sd >= lo) & (sd <= hi)
    if os.environ.get("S10_DBG_FRONT") == "2" and env.common_step_counter % 100 == 0:
        # 按 signed_dist 分箱打 fl/fr 轮离地高度 —— 查训练里到底在哪个距离抬轮
        enc = s["encounter"]
        hfl_raw = (fl - org - 0.081); hfr_raw = (fr - org - 0.081)
        print(f"[dbg2] step={env.common_step_counter} encounter={enc.float().mean().item()*100:.1f}%", flush=True)
        for a, b in ((0.90,1.20),(0.70,0.90),(0.60,0.70),(0.45,0.60),(0.30,0.45),(0.15,0.30),(0.00,0.15)):
            mk = enc & (sd >= a) & (sd < b)
            if mk.sum() >= 3:
                print(f"       离墙 {a:.2f}~{b:.2f}m  n={int(mk.sum())}  "
                      f"fl {hfl_raw[mk].mean().item()*1000:+5.0f}mm(离地{ (hfl_raw[mk]>0.05).float().mean().item()*100:3.0f}%)  "
                      f"fr {hfr_raw[mk].mean().item()*1000:+5.0f}mm(离地{ (hfr_raw[mk]>0.05).float().mean().item()*100:3.0f}%)", flush=True)
    if os.environ.get("S10_DBG_FRONT") == "1" and env.common_step_counter % 50 == 0:
        g = gate.float().mean().item()
        enc = s["encounter"].float().mean().item()
        inw = ((sd >= lo) & (sd <= hi)).float().mean().item()
        if gate.any():
            print(f"[dbg-front] step={env.common_step_counter} 闸门开={g*100:.2f}% "
                  f"(encounter={enc*100:.1f}%, 在窗口={inw*100:.1f}%)  "
                  f"闸内 fl超出均={hfl[gate].mean().item()*1000:.0f}mm fr超出均={hfr[gate].mean().item()*1000:.0f}mm "
                  f"项值均={(hfl.pow(2)+hfr.pow(2))[gate].mean().item():.4f}", flush=True)
        else:
            print(f"[dbg-front] step={env.common_step_counter} 闸门开=0% "
                  f"(encounter={enc*100:.1f}%, 在窗口={inw*100:.1f}%)", flush=True)
    return (hfl.pow(2) + hfr.pow(2)) * gate.float()


def wall_load_pose(env: "ManagerBasedRLEnv", knee_target: float = -1.45,
                   use_max: bool = False,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """**停住蓄力段的后腿深蹲**（块9，09-16 20:3x，作者看回放后点名）。

    作者原话：「应该是正常平地姿态走到墙前 然后抬头 后退弯曲蓄力」「后退蓄力不对称」。

    ## 这个缺口是我自己造出来的

    块3c 把相位门改成「停住后才起动作」，块5 又改成「停住+后退完成才起动作」。
    于是**在停住蓄力那一整段里，七个 `cs_*` 参考跟踪项全部是关着的**（相位尚未起跑），
    **没有任何奖励在要求后腿弯曲或对称** —— 机器人只是站着等放行。

    官方恰恰相反：深蹲与抬头**就发生在停住段之内**。干净窗口（D 0.80 → 起动瞬间）实测：

        后膝深度   官方 hl −1.465/−1.463/−1.445  hr −1.428/−1.438/−1.397
                   我方 块7 hl −0.877~−0.909     hr −0.724~−0.729   ← 只弯一半
        机身俯仰   官方 1.2~2.7° 单调抬到 5.6~7.4°
                   我方 块7 −4.4° ~ −2.1°（全程低头，没抬过）
        |Δknee|均  官方 0.041~0.072   我方 0.153~0.181   ← 差 3~4 倍

    ## 本项定义

    门：遭遇 & 命令剖面处于**停住段或后退段** & **相位尚未起跑**（k<0）。
    值：两条后膝到 `knee_target` 的绝对误差之和（作为惩罚，weight<0）。

    **两条腿用同一个目标，所以对称是自动的** —— 不需要再单独加一项对称罚
    （`cc_rear_sym` 保持不动）。

    **为什么不可伪造**：
      · 它是惩罚，在门内待得越久只会罚得越多，不能靠"多停一会儿"刷分；
      · 门由命令状态机与相位共同决定，策略要躲开只能不靠近墙，那会丢掉所有其他奖励；
      · 目标 −1.45 在膝限位 ±2.7227 之内，可达，不是死核。
    """
    s = _state(env)
    asset = env.scene[asset_cfg.name]
    jq = asset.data.joint_pos; jq = jq.torch if hasattr(jq, "torch") else jq
    names = asset.joint_names
    st = getattr(env, "_climb_cmd_state", None)
    k = getattr(env, "_cseq_k", None)
    if st is None or k is None:
        return torch.zeros(env.num_envs, device=jq.device)
    # 1 = ST_STOP, 2 = ST_BACK（见 mdp/climb_cmd.py）
    gate = s["encounter"] & ((st == 1) | (st == 2)) & (k < 0)
    hl = jq[:, names.index("hl_knee_joint")]
    hr = jq[:, names.index("hr_knee_joint")]
    # use_max（块17）：**和**可以被「一条过 + 一条欠」以较低代价满足
    # （实测 −2.34 / −1.38，和 0.96 却不改），改 max 后只有把最差那条腿拉回来才能降罚。
    e1 = (hl - knee_target).abs(); e2 = (hr - knee_target).abs()
    err = torch.maximum(e1, e2) if use_max else (e1 + e2)
    return err * gate.float()


def rear_hipx_splay(env: "ManagerBasedRLEnv", lim: float = 0.15,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """**后腿侧向外张惩罚**（块12，09-16 22:3x，作者看回放图2 指出「右后腿的关节」）。

    ## 这一项此前完全不存在

    `hind_splay_penalty` 罚的是后轮**前后方向**拖在髋后面（用 `hind_tuck`，取 x 坐标），
    **侧向的 hipx 没有任何奖励在管**。于是 `hr_hipx` 一路顶在关节限位上没人拦。

    ## 实测（最深蓄力帧，n=6）

        官方   hl_hipx −0.047~−0.181   hr_hipx −0.038~−0.094   （量级 0.04~0.18）
        块8    hr_hipx **+0.643**
        块10   hr_hipx **+0.617**
        块11   hr_hipx **+0.618**      ← hipx 限位是 ±0.6109，**已顶到限位**

    差 3~13 倍，且是**限位饱和**，不是轻微偏差。作者一眼看出的「右后腿关节不对」就是它。

    ## 定义

    门：遭遇 & 后轮尚未上台（蓄力/蹬地期）。
    值：两条后腿 `|hipx|` 超出 `lim` 的部分之和，作为惩罚。

    **对两腿用同一个上限**，不编码镜像方向 —— 避免我在 G 判据上犯过的错
    （用 hl−hr 直接比，会把正确的镜像算成误差）。

    **为什么不可伪造**：惩罚项，门内待久只会罚更多；门由几何（遭遇＋后轮未上台）决定，
    躲开只能不靠近墙；`lim=0.15` 在限位 0.6109 之内且官方实测可达，不是死核。
    """
    s = _state(env)
    asset = env.scene[asset_cfg.name]
    jq = asset.data.joint_pos; jq = jq.torch if hasattr(jq, "torch") else jq
    names = asset.joint_names
    gate = s["encounter"] & ~s["hind_up"].any(dim=1)
    hl = jq[:, names.index("hl_hipx_joint")].abs()
    hr = jq[:, names.index("hr_hipx_joint")].abs()
    over = torch.clamp(hl - lim, min=0.0) + torch.clamp(hr - lim, min=0.0)
    return over * gate.float()


def lead_front_onto_top(env: "ManagerBasedRLEnv", margin: float = 0.02,
                        knee_lo: float = -1.2) -> torch.Tensor:
    """**领先前轮搭上台面**（块13，09-16 23:0x，作者点明的因果链）。

    > 作者：「官方后退是弯曲蓄力，前腿一只先抬起上去，这样整体就抬头了」

    与既有分析一致（官方上台顺序 fr → fr 领先 fl 0.400/0.375/0.505 s，领先腿是支点、
    支撑占比 82.7/80.5/85.6%）。本项训的就是**「一只前腿先上去」这个事件本身**。

    ## 官方实测：俯仰是被前轮撑起来的，不是先抬头再上去

        领先前轮过台面   t=0.53/0.38/0.25   离墙 0.316/0.322/0.259   该刻俯仰 +27.3/+26.3/+28.3°
        上台前 0.3 s 俯仰仅 +4.4/+4.3/+4.5°  →  上台瞬间 +27°  →  再 0.3 s 到 +34~+38°
        最深蓄力在领先轮上台**之后** 0.10~0.13 s

    ## 我方缺口（块12，最深蓄力帧，n=6）

        离墙 0.307 ✅  机身z 0.389 ✅（官方 0.391~0.409）  领先前腿膝 +2.046（官方 +2.37~+2.62）
        **俯仰 −10.8°** ← 前轮始终没真正搭上台面，机身撑不起来

    ## 定义

    门：遭遇 & 后腿已进入蓄力（两后膝均值 ≤ `knee_lo`）& 两只前轮都还没上台。
    值：**较高那只**前轮轮底距「台面高 + margin」还差多少（惩罚）。

    取 `max` 而非指定某条腿 → **镜像无关**，哪条腿先上都算数（作者已同意左前先抬）。
    台面高用 `_state` 的墙高记忆 `wall_h`，不写死 0.33。

    **为什么不可伪造**：惩罚项，门内待久只会罚更多；门要求后腿已蓄力且前轮未上台，
    一旦真的上去门就关、罚归零 —— 只有完成动作才能止损，不能靠姿势凑。
    轮底用 `_state` 里既有的 `z`（轮心 − 半径 − 出生地面），与登顶判据同源，不另立口径。
    """
    s = _state(env)
    z = s["z"] if "z" in s else None
    if z is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene["robot"]
    jq = asset.data.joint_pos; jq = jq.torch if hasattr(jq, "torch") else jq
    names = asset.joint_names
    kn = 0.5 * (jq[:, names.index("hl_knee_joint")] + jq[:, names.index("hr_knee_joint")])
    top = torch.clamp(s["wall_mem"], min=0.0) + margin
    front_hi = torch.maximum(z[:, 0], z[:, 1])                 # 较高那只前轮的轮底
    # 门与 cc_load_pose 用**同一套**（命令剖面停住/后退段 & 相位未起），而不是要求后膝已到 knee_lo。
    # 09-16 23:1x 实测：用 `kn <= -1.2` 做门，读数只有 −0.0001~−0.0003（w=1），比 cc_load_pose
    # 的 −0.09~−0.28 小 300~1000 倍 —— **门几乎不开**。原因是那个深蹲是**部署侧**行为
    # （驱动把指令压住才蹲到 −1.9）；训练里遭遇仅 ~1.3 s 且墙区只占 0.3~0.6%，后膝很少到 −1.2。
    # `knee_lo` 保留为软条件：只用来在两个前轮都没抬时区分「还没开始蓄力」，不再做硬门。
    st = getattr(env, "_climb_cmd_state", None)
    k = getattr(env, "_cseq_k", None)
    if st is None or k is None:
        gate = s["encounter"] & (kn <= knee_lo)
    else:
        gate = s["encounter"] & ((st == 1) | (st == 2)) & (k < 0)     # 1=ST_STOP, 2=ST_BACK
    # **去掉 `~front_up.any()`**（09-16 23:2x）：`front_up` 的门槛是 0.6×墙高 = **0.198 m**，
    # 机器人把前轮抬到 0.2 就被判为「已上台」、门随即关闭 —— 恰好在 **0.198→0.33 这段
    # 最需要罚的区间里失效**。实测加了它读数 −0.0013，比 cc_load_pose −0.1975 小 150 倍。
    # 该条件本来也多余：项值 clamp(台面 − 轮底, 0) 在轮子真正够到台面时自动归零，
    # 不需要额外的门来"提前收手"。
    return torch.clamp(top - front_hi, min=0.0) * gate.float()


def on_top_posture(env: "ManagerBasedRLEnv", knee_nom: float = -0.65, hipy_nom: float = 0.35,
                   w_sym: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """**台上后腿姿态**（块15，09-16 23:3x，作者看回放图2 点名）。

    > 作者：「翻上台阶后的右后腿的关节你觉得没问题吗」

    ## 实测（块7，机身 z>0.70 后 1~3 s 稳定段，n=3）

        hl  hipx −0.114  hipy **+0.318**  knee **−0.500**
        hr  hipx +0.016  hipy **+0.179**  knee **−0.842**      标称 hipy +0.35 / knee −0.65
        左右差  Δhipy +0.139   Δknee **+0.342**

    右后腿比左后多蹲 0.34 rad、髋却少转 0.14 rad —— 一条腿塌着、一条腿撑着。
    「站住」的二值判据 6/6 通过，但**姿态不对**，作者一眼看出来了。

    ## 定义

    门：四轮都已上台（`front_up` 与 `hind_up` 全真）& 指令接近零（站立/落平段）。
    值：两条后腿各自到标称站姿的偏差 + `w_sym` × 左右差。

    标称取部署 runner 的 `dof_default`（后腿 hipy +0.35 / knee −0.65），与真机一致，不另立口径。
    左右差项**不编码镜像**：台上是静态站姿，官方两条后腿本就该对称（不同于蓄力段）。

    **为什么不可伪造**：惩罚项；门要求四轮已上台且指令为零，
    这两条都不是策略能"提前满足"的——必须真的爬上去并停住才进入该状态。
    """
    s = _state(env)
    asset = env.scene[asset_cfg.name]
    jq = asset.data.joint_pos; jq = jq.torch if hasattr(jq, "torch") else jq
    names = asset.joint_names
    cmd = env.command_manager.get_command("base_velocity")
    still = (torch.linalg.norm(cmd[:, :2], dim=1) < 0.10) & (cmd[:, 2].abs() < 0.10)
    on_top = s["front_up"].all(dim=1) & s["hind_up"].all(dim=1)
    hlk = jq[:, names.index("hl_knee_joint")]; hrk = jq[:, names.index("hr_knee_joint")]
    hly = jq[:, names.index("hl_hipy_joint")]; hry = jq[:, names.index("hr_hipy_joint")]
    dev = ((hlk - knee_nom).abs() + (hrk - knee_nom).abs()
           + (hly - hipy_nom).abs() + (hry - hipy_nom).abs())
    sym = (hlk - hrk).abs() + (hly - hry).abs()
    return (dev + w_sym * sym) * (on_top & still).float()


def rear_wheels_planted(env: "ManagerBasedRLEnv", lim: float = 0.05) -> torch.Tensor:
    """**蓄力时两条后轮必须压在地上**（块20，09-17 09:0x，MuJoCo 真模型 FK 查出的根因）。

    ## 此前十几块一直没有这个约束，机器人就一路把后腿越翘越高

    用 MJCF 真模型做 FK 取四轮世界位置（`s10_dev/wheel_fk.py`），最深蓄力帧：

        台面 0.33          fl      fr      hl      hr    领先前轮  后轮最高   俯仰
        官方              0.16  **0.44**  0.02   0.000   0.38~0.44  0.00~0.03  **+36~+42**
        块8               0.275   0.001   0.001   0.162     0.275     0.162     −16.1
        块12              0.216   0.001   0.001   0.290     0.216     0.290     −10.8
        块15              0.162   0.003   0.001 **0.362**   0.162   **0.362**    −4.3
        块16              0.219   0.001   0.002   0.312     0.219     0.312      −7.0

    **官方抬的是右前轮（0.44），两后轮踩死（0.000~0.025）；我方抬的是右后轮，且越抬越高。**

    ### 连锁后果（全部由此解释）

    · 「两后腿蹲深差 0.64」**不是蹲的差异**，是一条着地、一条悬在 0.31 m 空中。
      块16/19 用对称权重逼「悬空腿匹配着地腿」，方向从根上错，两块白做。
    · `hr_hipx` 钉在限位 0.62 八块不动 —— 那条腿**悬空、无地面约束**，自然外甩到限位。
      我曾猜「被接触力推出去」，**方向反了，是没有接触**。
    · 俯仰上不去 —— 后腿抬起，后半身没支撑点；官方两后轮踩死、前轮搭台，身体才立 41°。
    · **最讽刺的一条**：后轮抬得越高，俯仰角看起来"越好"（0.162→−16.1°，0.362→−4.3°）。
      我一直以为「俯仰在改善、方向对」，**其实是机器人用翘后腿假装抬头**，
      而我的判据只看俯仰角，**正好奖励了这个错误动作**。

    ## 定义

    门：与 `cc_load_pose` **同源**（遭遇 & 命令剖面停住/后退段 & 相位未起）——
    只管蓄力段，**真正上墙时后轮该抬就抬，不受本项约束**。
    值：两条后轮轮底超出 `lim` 的部分之和（惩罚）。轮底用 `_state` 的 `z`，与登顶判据同源。

    **为什么不可伪造**：惩罚项，只有把后轮真放回地面才能降罚；门在相位起跑时关闭，
    不会妨碍上墙阶段抬后腿；`lim=0.05` 在官方实测 0.000~0.025 之上留了余量，可达。
    """
    s = _state(env)
    z = s["z"]
    st = getattr(env, "_climb_cmd_state", None)
    k = getattr(env, "_cseq_k", None)
    if st is None or k is None:
        gate = s["encounter"]
    else:
        gate = s["encounter"] & ((st == 1) | (st == 2)) & (k < 0)   # 1=ST_STOP, 2=ST_BACK
    over = torch.clamp(z[:, 2] - lim, min=0.0) + torch.clamp(z[:, 3] - lim, min=0.0)
    return over * gate.float()


def rear_lr_symmetry(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     lo: float = 0.15, hi: float = 0.70, w_knee: float = 0.5) -> torch.Tensor:
    """**蓄力段左右后腿对称罚**（C36，09-15 20:0x，作者逐帧对照后定位）。

    作者看 MuJoCo 回放逐帧提的五条里,①② 指向同一个根:
    「官方靠近墙时后腿弯曲蓄力、伸右前轮;我们后腿动作变形没蓄力、伸的还是左前轮」。
    查官方真机数据,**形态正好是反的**（|hl_hipy − hr_hipy| 逐帧均值）:

        蓄力段 0.15~0.70 m   官方 seg0 **0.098** / seg1 **0.206**   我方 C35 **0.609** C24 **0.665**
        蹬地段 <0.15 m       官方 seg0 **1.090** / seg1 **0.640**   我方 C35 0.247  C24 0.393

    → **官方"对称蓄力 → 不对称蹬地";我方"不对称蓄力 → 对称蹬地"。不对称用在了相反的阶段。**

    **为什么二十多轮没人拦住它**:现有 `joint_mirror` 配的是**对角**对（fl↔hr、fr↔hl),
    那是 trot 的对角同步;翻越时前腿够高、后腿蹬地本来就该不同（官方 t=0.52 fl −0.82 vs hr +1.01),
    所以"翻越相位豁免对角镜像"这个豁免**本身是对的**,收窄它只会把前后腿强行拉平、更糟。
    **左右对称（hl vs hr）这个约束从来就不存在** —— 不是被豁免掉,是压根没有。
    （我第一版改法就是想去收窄那个豁免,量完官方数据才发现方向错了,已作废。）

    本项只在**蓄力段**（离墙 lo~hi）生效,蹬地段（<lo）不罚 —— 那里官方自己就不对称。
    量 hipy 为主、knee 次之（官方 knee 的左右差本来就比 hipy 大:均 0.18/0.32）。
    """
    s = _state(env)
    asset = env.scene[asset_cfg.name]
    jq = asset.data.joint_pos; jq = jq.torch if hasattr(jq, "torch") else jq
    names = asset.joint_names
    def _j(n):
        return names.index(n)
    dy = jq[:, _j("hl_hipy_joint")] - jq[:, _j("hr_hipy_joint")]
    dk = jq[:, _j("hl_knee_joint")] - jq[:, _j("hr_knee_joint")]
    sd = s["signed_dist"]
    gate = s["encounter"] & (sd >= lo) & (sd <= hi)
    return (dy.pow(2) + w_knee * dk.pow(2)) * gate.float()


def joint_mirror_climb_aware(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """对角镜像惩罚，翻越相位内豁免（右后猛蹬、左后不出力，本来就不对称）。"""
    return _joint_mirror_raw(env, asset_cfg, mirror_joints) * (~_state(env)["phase"]).float()


def _bad_orientation(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, limit: float, pitch_limit_climb: float) -> torch.Tensor:
    g = _T(env.scene[asset_cfg.name].data.projected_gravity_b)
    phase = _state(env)["phase"]
    pitch_lim = torch.where(phase, torch.full_like(g[:, 0], pitch_limit_climb), torch.full_like(g[:, 0], limit))
    return (g[:, 2] > 0) | (g[:, 1].abs() > limit) | (g[:, 0].abs() > pitch_lim)


def bad_orientation_climb_aware(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit: float = 0.7,
    pitch_limit_climb: float = 0.87,
) -> torch.Tensor:
    """终止：翻倒 / 横滚 > asin(limit) / 俯仰 > asin(limit)；翻越相位内俯仰线放宽到 asin(pitch_limit_climb)。
    0.7 → 44°（V0 原值，正好是厂家翻越的最大俯仰，会误杀）；0.87 → 60°。"""
    return _bad_orientation(env, asset_cfg, limit, pitch_limit_climb)


def bad_orientation_penalty_climb_aware(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit: float = 0.7,
    pitch_limit_climb: float = 0.87,
    encounter_scale: float = 1.0,
) -> torch.Tensor:
    """与 bad_orientation_climb_aware 同条件的惩罚版（配 −1000 权重）；上墙遭遇中按 encounter_scale 打折（11:55：0.3，让它在墙沿敢发力）。"""
    bad = _bad_orientation(env, asset_cfg, limit, pitch_limit_climb).float()
    enc = _state(env)["encounter"]
    return torch.where(enc, bad * encounter_scale, bad)


# ---------------------------------------------------------------------------
# 墙前出生（RSI 简化版，09-09 14:00）：把一部分立面地形的 episode 直接从"离墙 dist、后腿屈膝蓄力"开始。
# 原因：完整动作（走 2 m → 蓄力 → 搭 → 蹬 → 拉 → 落平）太长，随机探索几乎碰不到"搭上后猛蹬"这一步；
# 直接生在墙前，每个迭代有上千次机会练后半段。approach_riser / climb_up 的立面都在出生点 ±2.0 m（x 向）。
import isaaclab.utils.math as _mu
from collections.abc import Sequence as _Seq

LEG_L = 0.18          # 大腿 = 小腿 = 0.18 m（URDF）
HIP_X = 0.2277        # 前后髋相对机身中心的纵向距离


def _leg_ik(x: torch.Tensor, z: torch.Tensor, front: bool):
    """两连杆平面逆解（机体系：x 前，z 上；轮心相对髋的位置 (x, z)，z<0 在下方）→ (hipy, knee)。
    约定（09-09 运动学实验核实）：hipy 正 = 大腿前摆；knee 正 = 小腿相对大腿前摆；前腿默认 (−0.35, +0.65)，后腿 (+0.35, −0.65)。
    对称两连杆：轮心方向 α 是大腿与小腿方向的平分线 → hipy = α − knee/2。"""
    r = torch.sqrt(x * x + z * z).clamp(0.06, 2 * LEG_L - 0.01)
    beta = torch.acos((1.0 - r * r / (2 * LEG_L * LEG_L)).clamp(-1.0, 1.0))      # 膝内角
    mag = math.pi - beta                                                         # 关节角大小
    knee = mag if front else -mag
    alpha = torch.atan2(x, -z)                                                   # 从竖直向下转向 +x 的角
    hipy = alpha - knee / 2
    return hipy, knee


def spawn_at_wall(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  frac: float = 0.5, dist: float = 0.5, wall_x: float = 2.0, hind_knee: float = -1.05, hind_hipy: float = 0.55,
                  z: float = 0.40, dist_range: tuple | None = None, crouch_prob: float = 1.0, both_sides: bool = True,
                  hooked_prob: float = 0.0, pushed_prob: float = 0.0, ref_prob: float = 0.0,
                  categorical: bool = False, ref_t_range: tuple | None = None) -> None:
    from .curriculums import _build_col2name, RISER_TERRAINS
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    names = getattr(env, "_climb_col2name", None)
    if names is None:
        names = _build_col2name(terrain); env._climb_col2name = names
    ty = terrain.terrain_types[env_ids]
    riser_cols = torch.tensor([c for c, nm in names.items() if nm in RISER_TERRAINS], device=ty.device, dtype=ty.dtype)
    if riser_cols.numel() == 0:
        return
    on_riser = torch.isin(ty, riser_cols)
    pick = on_riser & (torch.rand(len(env_ids), device=ty.device) < frac)
    if not pick.any():
        return
    ids = env_ids[pick]
    n = len(ids)
    sign = torch.where(torch.rand(n, device=ids.device) < 0.5, -1.0, 1.0) if both_sides else torch.ones(n, device=ids.device)
    yaw = torch.where(sign > 0, torch.zeros(n, device=ids.device), torch.full((n,), math.pi, device=ids.device))
    origins = env.scene.env_origins[ids]
    pos = origins.clone()
    d = torch.empty(n, device=ids.device).uniform_(*dist_range) if dist_range else torch.full((n,), dist, device=ids.device) + torch.empty(n, device=ids.device).uniform_(-0.05, 0.05)
    pos[:, 0] += sign * (wall_x - d)
    pos[:, 1] += torch.empty(n, device=ids.device).uniform_(-0.6, 0.6)
    pos[:, 2] += z
    quat = _mu.quat_from_euler_xyz(torch.zeros(n, device=ids.device), torch.zeros(n, device=ids.device), yaw)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=ids.device), env_ids=ids)
    # 蓄力姿态：后膝屈、后髋前收，其余默认
    jp = asset.data.default_joint_pos.torch[ids].clone()
    jv = torch.zeros_like(jp)
    kid, _ = asset.find_joints(["hl_knee_joint", "hr_knee_joint"], preserve_order=True)
    hid, _ = asset.find_joints(["hl_hipy_joint", "hr_hipy_joint"], preserve_order=True)
    crouch = torch.rand(n, device=ids.device) < crouch_prob
    jp[:, kid] = torch.where(crouch.unsqueeze(1), torch.full_like(jp[:, kid], hind_knee), jp[:, kid])
    jp[:, hid] = torch.where(crouch.unsqueeze(1), torch.full_like(jp[:, hid], hind_hipy), jp[:, hid])
    # ---- 前轮已搭上、后腿蓄力、机身抬头的出生态（GPT 拆解建议的倒序课程：先练"前方支撑下的主蹬"）----
    # ===== 09-16 夜：一次**类别采样**（计划 5.2）=====
    # 旧法是逐层条件：hooked → pushed(~hooked) → ref(~hooked&~pushed)，
    # 于是"配置 ref_prob=0.7、实际参考态出生只有 0.7×(1−.15)×(1−.128)≈50%"，
    # 再叠 sign>0 约一半 → 实测相位起点深屈仅 **12.9%**，远低于配置意图。
    # 本版：一次抽类别，比例即配置比例（仍限 sign>0 一侧，该侧内比例准确）。
    if categorical:
        _p = torch.tensor([hooked_prob, pushed_prob, ref_prob], device=ids.device).clamp(min=0.0)
        _rest = (1.0 - _p.sum()).clamp(min=0.0)
        _w = torch.cat([_p, _rest.view(1)])
        _c = torch.multinomial(_w.expand(n, 4), 1).squeeze(1)      # 0=hooked 1=pushed 2=ref 3=普通
        hooked = (_c == 0) & (sign > 0)
        pushed = (_c == 1) & (sign > 0)
        refm_cat = (_c == 2) & (sign > 0)
    else:
        hooked = (torch.rand(n, device=ids.device) < hooked_prob) & (sign > 0)
        refm_cat = None
    cur_pos, cur_quat = pos, quat
    if hooked.any():
        hk = hooked
        gen = terrain.cfg.terrain_generator
        sub = gen.sub_terrains.get("approach_riser") if gen is not None and gen.sub_terrains else None
        lo, hi = (float(sub.riser_height_range[0]), float(sub.riser_height_range[1])) if sub is not None else (0.06, 0.34)
        lvl = terrain.terrain_levels[ids].float()
        h = (lo + (lvl + 0.5) / float(gen.num_rows) * (hi - lo))                            # 本行名义墙高（取行中值）
        pitch = 0.15 + 1.1 * h                                                               # 0.33 m → 约 29°
        bx = -(0.15 + 0.15 * torch.rand_like(h))                                             # 机身中心离墙面 0.15~0.30（墙面 x=0）
        pitch = pitch + 0.2 * (torch.rand_like(h) - 0.5)                                     # ±0.1 rad
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        bz = 0.20 + 0.55 * h                                                                 # 0.33 → 0.38
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        # 前髋/后髋世界位置（相对墙面与地面）
        fhx, fhz = bx + HIP_X * cp, bz + HIP_X * sp
        hhx, hhz = bx - HIP_X * cp, bz - HIP_X * sp
        # 目标轮心：前轮在台面上 10 cm、离顶面 1.5 cm；后轮在地面、髋略后
        fwx, fwz = torch.full_like(h, 0.10), h + WHEEL_RADIUS + 0.015
        hwx, hwz = torch.full_like(h, -0.35), torch.full_like(h, WHEEL_RADIUS)
        def to_base(dx, dz):
            return dx * cp + dz * sp, -dx * sp + dz * cp
        fx, fz = to_base(fwx - fhx, fwz - fhz)
        hx, hz = to_base(hwx - hhx, hwz - hhz)
        f_hipy, f_knee = _leg_ik(fx, fz, front=True)
        h_hipy, h_knee = _leg_ik(hx, hz, front=False)
        # 写机身位姿：x = 墙面(原点+wall_x) + bx，抬头 = 负俯仰
        pos2 = pos.clone(); pos2[:, 0] = origins[:, 0] + wall_x + bx; pos2[:, 2] = origins[:, 2] + bz
        quat2 = _mu.quat_from_euler_xyz(torch.zeros(n, device=ids.device), -pitch, yaw)
        pos_w = torch.where(hk.unsqueeze(1), pos2, pos); quat_w = torch.where(hk.unsqueeze(1), quat2, quat)
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos_w, quat_w], dim=-1), env_ids=ids)
        cur_pos, cur_quat = pos_w, quat_w
        fh, _ = asset.find_joints(["fl_hipy_joint", "fr_hipy_joint"], preserve_order=True)
        fk, _ = asset.find_joints(["fl_knee_joint", "fr_knee_joint"], preserve_order=True)
        for j in fh: jp[:, j] = torch.where(hk, f_hipy, jp[:, j])
        for j in fk: jp[:, j] = torch.where(hk, f_knee, jp[:, j])
        for j in hid: jp[:, j] = torch.where(hk, h_hipy, jp[:, j])
        for j in kid: jp[:, j] = torch.where(hk, h_knee, jp[:, j])
        # 记住墙高并标记已蓄力，让相位奖励从"前轮已上"状态接上
        try:
            env._climb_wallh_mem[ids[hk]] = h[hk]
            env._climb_flexed[ids[hk]] = True
        except Exception:
            pass
    # ---- 蹬完态（09-09 脚本控制器实测几何，MuJoCo/PhysX 一致）：前轮在台面沿口前 ~0.32、后轮顶在墙根、机身中心在墙面线、高 h+0.22、抬头约 50°、四轮全受力
    #      从这里往后是"单腿抬起 + 落平"的平衡问题，交给策略练 ----
    if not categorical:
        pushed = (torch.rand(n, device=ids.device) < pushed_prob) & (sign > 0) & ~hooked
    if pushed.any():
        pk = pushed
        gen = terrain.cfg.terrain_generator
        sub = gen.sub_terrains.get("approach_riser") if gen is not None and gen.sub_terrains else None
        lo, hi = (float(sub.riser_height_range[0]), float(sub.riser_height_range[1])) if sub is not None else (0.06, 0.34)
        lvl = terrain.terrain_levels[ids].float()
        h = (lo + (lvl + 0.5) / float(gen.num_rows) * (hi - lo))
        pitch = (0.15 + 2.2 * h) + 0.16 * (torch.rand_like(h) - 0.5)                    # 0.33 → 0.876 rad ≈ 50°
        bx = -0.01 + 0.08 * (torch.rand_like(h) - 0.5)
        bz = 0.216 + h
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        fhx, fhz = bx + HIP_X * cp, bz + HIP_X * sp
        hhx, hhz = bx - HIP_X * cp, bz - HIP_X * sp
        fwx, fwz = 0.12 + 0.6 * h + 0.06 * (torch.rand_like(h) - 0.5), h + WHEEL_RADIUS + 0.01
        hwx, hwz = torch.full_like(h, -WHEEL_RADIUS - 0.02), torch.full_like(h, WHEEL_RADIUS)
        def to_base2(dx, dz):
            return dx * cp + dz * sp, -dx * sp + dz * cp
        fx, fz = to_base2(fwx - fhx, fwz - fhz)
        hx, hz = to_base2(hwx - hhx, hwz - hhz)
        f_hipy, f_knee = _leg_ik(fx, fz, front=True)
        h_hipy, h_knee = _leg_ik(hx, hz, front=False)
        pos3 = pos.clone(); pos3[:, 0] = origins[:, 0] + wall_x + bx; pos3[:, 2] = origins[:, 2] + bz
        quat3 = _mu.quat_from_euler_xyz(torch.zeros(n, device=ids.device), -pitch, yaw)
        pos_w = torch.where(pk.unsqueeze(1), pos3, cur_pos); quat_w = torch.where(pk.unsqueeze(1), quat3, cur_quat)
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos_w, quat_w], dim=-1), env_ids=ids)
        fh, _ = asset.find_joints(["fl_hipy_joint", "fr_hipy_joint"], preserve_order=True)
        fk, _ = asset.find_joints(["fl_knee_joint", "fr_knee_joint"], preserve_order=True)
        for j in fh: jp[:, j] = torch.where(pk, f_hipy, jp[:, j])
        for j in fk: jp[:, j] = torch.where(pk, f_knee, jp[:, j])
        for j in hid: jp[:, j] = torch.where(pk, h_hipy, jp[:, j])
        for j in kid: jp[:, j] = torch.where(pk, h_knee, jp[:, j])
        try:
            env._climb_wallh_mem[ids[pk]] = h[pk]
            env._climb_flexed[ids[pk]] = True
        except Exception:
            pass
    # ---- 真机参考态（09-10）：从 climb33 真机帧采样，落在上墙过程任一相位 ----
    from . import climb_ref as _cr
    _cr.mark_no_ref(env, ids)
    refm = refm_cat if categorical else ((torch.rand(n, device=ids.device) < ref_prob)
                                         & (sign > 0) & ~hooked & ~pushed)
    if refm.any():
        gen = terrain.cfg.terrain_generator
        sub = gen.sub_terrains.get("approach_riser") if gen is not None and gen.sub_terrains else None
        lo, hi = (float(sub.riser_height_range[0]), float(sub.riser_height_range[1])) if sub is not None else (0.06, 0.34)
        lvl = terrain.terrain_levels[ids].float()
        h = (lo + (lvl + 0.5) / float(gen.num_rows) * (hi - lo))
        # 09-16 夜：ref_t_range 让课程按**事件**选关键区间（计划 5.2），不套绝对秒数。
        # 关键区间由官方数据定位：后腿共同最深 t=0.52/0.38、右前膝峰 t=0.56/0.38 → 两事件重合。
        if ref_t_range is not None:
            _cr.apply_ref_spawn(env, asset, ids[refm], origins[refm], wall_x, h[refm], yaw[refm], jp, jv, refm,
                                t_lo=float(ref_t_range[0]), t_hi=float(ref_t_range[1]))
        else:
            _cr.apply_ref_spawn(env, asset, ids[refm], origins[refm], wall_x, h[refm], yaw[refm], jp, jv, refm)
    if getattr(env, "_climb_spawn_hooked", None) is None:
        env._climb_spawn_hooked = torch.zeros(env.num_envs, dtype=torch.bool, device=ids.device)
    env._climb_spawn_hooked[ids] = (hooked | pushed | refm) if (hooked_prob > 0 or pushed_prob > 0 or ref_prob > 0) else False
    asset.write_joint_position_to_sim_index(position=jp, joint_ids=slice(None), env_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=jv, joint_ids=slice(None), env_ids=ids)


EXEMPT_TAIL_S = [2.0]     # C19：地形触发后维持多久（秒）。翻越 起抬→落平 实测 0.82~0.95 s，2.0 留足余量


def climb_exempt_mask(env) -> "torch.Tensor":
    """"翻越豁免"掩码 v2（C19，09-15 07:3x）。**只由地形触发 + 限时**，不含任何机器人自身姿态量。

    v1（C18）用 `s["phase"] | s["encounter"]` —— 600 迭代**完全无效**，台面姿态纹丝不动
    （pitch +24.4°、hr 膝 −2.69 顶限位 25%，与 C17 一模一样）。查出是**自我维持回路**：
      `phase = split | wall_ahead`，`split = 四轮触地高差 > 0.12 m`（CLIMB_SPLIT）。
      轴距 0.4554 m，**机身 pitch 24.4° 本身就产生 0.4554×sin(24.4°) = 0.188 m 的四轮高差**，
      已经超过 0.12；折起的右后腿再加一截。
      → **坏姿态本身满足"正在翻越"的判定，于是把自己豁免掉了。**
      C18 训练侧姿态罚确实涨了 2~3 倍（flat_orientation −0.41→−0.86、膝 −0.29→−0.86），
      但那是接近段和别的地形贡献的；**台面段依旧免罚**，所以姿态一步没动、罚值还在涨
      （策略宁可在别处挨罚，也不改这个在台面上不要钱的姿态）。

    v2 改法：触发条件**只看前方地形**（`wall_ahead | tall_wall`，来自高程扫描），
    触发后维持 `EXEMPT_TAIL_S` 秒。落平后墙不在前方了，计时一到豁免自动关，
    **机器人无论摆成什么姿势都续不上这个豁免**。
    """
    s = _state(env)
    trig = s["wall_ahead"] | s["tall_wall"]          # 纯地形量，机器人姿态无法伪造
    t_now = env.episode_length_buf.float() * env.step_dt
    st = getattr(env, "_climb_exempt", None)
    if st is None or st["t"].shape[0] != trig.shape[0]:
        st = {"t": torch.full_like(t_now, -1e9), "last": t_now.clone(), "step": -1}
        env._climb_exempt = st
    if st["step"] != env.common_step_counter:        # 一步只更新一次（本函数被 4 个奖励项各调一次）
        st["step"] = env.common_step_counter
        fresh = t_now < st["last"] - 1e-6            # 刚 reset
        st["t"] = torch.where(fresh, torch.full_like(st["t"], -1e9), st["t"])
        st["t"] = torch.where(trig, t_now, st["t"])
        st["last"] = t_now
    return (t_now - st["t"]) < EXEMPT_TAIL_S[0]


def early_rear_penalty(env: "ManagerBasedRLEnv", far_m: float = 0.30,
                       allow_deg: float = 10.0) -> torch.Tensor:
    """C31（09-15 17:2x）：**罚"离墙还远就仰着"**。

    实测（官方 seg0 vs C24_13800/C30_13999，按离墙距离对齐——距离没有歧义，比时间轴硬）：

    | 离墙 | 官方 | C24 | C30 |
    |------|------|-----|-----|
    | 0.60 m | **−3.3°** | −27.8° | −25.4° |
    | 0.25 m | **−38.1°（官方峰值）** | −33.4° | −40.4° |
    | 0.00 m（墙面）| **−0.6°（已放平）** | **−42.0°** | **−33.0°** |

    官方是"抬头—放平"的一次扑击：离墙 0.25 m 抬到峰值把前脚搭上去，**机身跨过墙面时已放平**，
    身体被带上去（对应 09-08 拆视频的 蓄力→前左搭→前右跟→后腿推）。
    我方是**全程仰着开过去**，最大抬头恰好发生在跨墙面那一刻。

    本项只管"远处别仰"，**不碰近处**（离墙 < far_m 完全不罚，官方峰值就在 0.25 m）。
    单边：allow_deg 以内零代价，超出按 sin 超量平方罚。

    反刷分检查（对照 [[feedback-reward-unfakeable]] 三条）：
      · 能不能靠"不结束"赚？——本项是**惩罚**不是奖励，拖长只会多罚。
      · 门会不会被被罚量自己满足？——门是 `dist_wall > far_m`（地形量）与 `encounter`，
        与 pitch 无关，仰起来不会让门自己关上。
      · 能不能靠单轮刷？——逐步累加，没有均值差可刷。
    唯一的规避路径是"冲得更快让远处停留更短"——那正是要的行为（官方就是平着走到墙根）。
    """
    s = _state(env)
    thr = math.sin(math.radians(allow_deg))
    excess = torch.clamp(s["pitch_up"] - thr, min=0.0)
    far = (s["dist_wall"] > far_m) & s["encounter"]
    # 返回**正量**，权重取负（IsaacLab 惯例；对照 joint_pos_limits 在权重 −1.0 下读数 −0.0672）。
    # 09-15 探针抓到过一次反号：函数返回负 × 权重负 = 正奖励，等于在奖励提前抬头。
    return (excess ** 2) * far.float()


def on_top_knee_limit(env: "ManagerBasedRLEnv", lim: float = 2.30, h_min: float = 0.18,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    r"""**台上膝关节不许压到限位**（B2g，09-17 14:4x）。作者：「差最多的是落地不够稳，容易摔倒」。

    ## 先说清楚前两次改落地都错在哪（都已实测证伪并回退）

    1. `lin_vel_z_l2_climb_aware(land_k)` —— 以为"落地没被罚"。实测砸得最猛那一帧四轮高差
       0.034~0.083 < `CLIMB_SPLIT`=0.12、前方无立面 → `phase` 为假 → **落地一直在挨 1× 的罚**。
       那个改动只把**蓄力**多罚了 land_k 倍（96.6% 的罚落在上墙过程里）。f1 登顶 0/6。
    2. 「前膝左右差 1.37 rad 导致横滚」—— 那是**单帧**读数。看分布：台上前膝差中位我方 0.13、
       官方 0.06~0.21；后膝差峰我方 1.06~1.24 **比官方 1.38~1.56 还小**。**对称性我方不差**，已撤回。

    ## 站得住的差距（同口径，`s10_dev/land_check.py`，n=3）

        下陷（自身接近站高 − 台上最低站高）   官方 **0.042**   我方 **0.206**
        台上 |roll| 峰                        官方 **0.11°**   我方 **18.3°**
        恢复（回到 0.95×末端站高）            官方 **0.03 s**  我方 **1.45 s**
        台上最大下落 vz                       官方 −0.487      我方 −0.398   ← **我方反而更轻**
        **台上膝最大幅值**                    官方 **2.18**    我方 **2.70**（限位 2.7227）

    **官方掉得比我们还快，却 0.03 s 就接住了**；我们一路沉 0.21 m、1.45 s 才回来。
    差别不在竖直速度，在**落地那一刻腿还撑不撑得住** —— 右后膝压到 2.70 已经没有行程了。

    ## 为什么用"超限位量"而不是"追标称站姿"

    官方台上末端后膝是 **−1.35~−1.52**（不是标称 −0.65），它自己就是蹲着的；
    我方台上末端站高 0.408 还比官方 0.372 **高**。所以不能拿标称姿态当靶子去拉，
    只能卡住**不许压过头**这一条 —— 官方全程台上 ≤2.18，留 0.12 余量取 `lim=2.30`。

    ## 门：`四轮轮底都高于 h_min`（绝对高度，不用 front_up）

    **不能用 `front_up`/`hind_up`** —— 遭遇结束后 `wall_mem` 被清零，那两个门退化成 `z > 0`，
    **平地上也会成立**（块15 的 `on_top_posture` 就是踩了这个坑）。这里用轮底绝对高度
    `h_min=0.18`（= `TALL_WALL_H`，"值得上的墙"的定义），平地恒为假。

    ## 先验读数（B2c n=3，`lim=2.30`）

        触发 286~358 步（1.4~1.8 s，占台上 8.6%~10.6%）—— 正好覆盖下沉+恢复那一段
        峰值 0.163      主项 track_lin_vel 约 6.3/步
        w=10 → 峰值罚 1.6（占主项 26%）；w=30 → 4.8（76%，过大）
    取 **w=10**。

    **不可伪造**：纯惩罚；门要求四轮都真的上到 0.18 m 以上，只能靠真爬上去才进入，
    且策略无法通过"不爬"来规避（不爬就丢 `wheel_height_progress` w=300 与登顶，守卫直接抓）。
    """
    s = _state(env)
    asset = env.scene[asset_cfg.name]
    jq = _T(asset.data.joint_pos)
    names = asset.joint_names
    ki = [names.index("%s_knee_joint" % l) for l in ("fl", "fr", "hl", "hr")]
    on_top = s["z"].min(dim=1).values > h_min
    ex = torch.clamp(jq[:, ki].abs() - lim, min=0.0) ** 2
    return ex.sum(dim=1) * on_top.float()


def front_leg_hold(env: "ManagerBasedRLEnv", lim: float = 1.70,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    r"""**刚翻上台后前腿不许折塌**（B2i，09-17 16:0x）。作者看并排回放后点名：

    > 「同一个过墙之后，我们的左前腿应该做支撑，而不是塌下去，你先把这个问题解决，应该就稳一些了」

    ## 实测（MuJoCo 真模型，腿长 = 髋心→轮心距离，n=6 vs 官方 n=3）

    | 腿长(m) | fl | fr | hl | hr |
    |---|---|---|---|---|
    | 我方 四轮上台那刻 | 0.323 | 0.268 | 0.321 | 0.179 |
    | 我方 机身最低那刻 | **0.237** | 0.378 | 0.233 | 0.166 |
    | 我方 变化 | **−0.086** | +0.110 | −0.089 | −0.019 |
    | **官方 全程变化** | **0.000** | 0.000 | 0.000 | 0.000 |

    **左前腿承重时缩短 8.6 cm —— 被压塌的；右前腿伸长 11 cm 去接管。**
    官方那三个 0.000 还有一层意思：**它上台后机身最低点就是刚上台那一刻，之后只升不降。**

    **不是撑不动，是没在撑**（力矩实测，塌陷段 0.22~0.29 s）：

        左前膝 |τ| 峰 13.5 / 18.5 / 26.1  （上限 50，**饱和 0%**）
        右前膝 |τ| 峰 49.1 / 49.1 / 50.0  （**9% 时间打满**）

    整机重量被甩到右前一条腿上，它扛不住 → 机身下沉 0.21 m、1.45 s 才回来。

    ## 为什么写成膝角而不是腿长

    髋→轮距离**只取决于膝角**（hipx/hipy 只是把整条腿绕髋转，不改变该距离，已实测：
    hipx/hipy 任意变化时 L 恒为 0.3228）。对应关系：

        膝 0.00→L 0.381   1.19→0.323   1.70→0.268   1.96→0.236   2.74→0.143

    所以直接卡膝角，不必取 body 位置，更省更稳。

    ## 阈值 1.70 的由来（能把两边分开）

        官方台上前膝最大  1.42 / 1.53 / 1.59        我方 1.98 / 2.01 / 1.99

    取 1.70 落在中间。先验读数（门内 1.5 s）：**官方触发 0%**，我方触发 59%~82%、
    峰值 0.079~0.094（均 0.086）。w=20 → 峰值罚 1.72（主项 7.0/步的 25%），**官方恒为 0**。

    ## 门：`top_win`（「刚翻上台」窗口），不是高度阈值

    B2g 的教训：`s["z"]` 相对 **env 出生点**，课程地形上爬过任何东西后永久为真。
    这里用状态转变起计时（`encounter` 由真转假 + 转假前记过墙高 + 四轮都高过那个旧墙高），
    窗口 `TOP_WIN_STEPS`=75 步 = 1.5 s，正好覆盖实测的下沉+恢复段。

    **不可伪造**：纯惩罚；进窗口必须真的完成一次翻越（`encounter` 真→假且四轮过线），
    策略无法靠"不爬"规避（不爬就丢 `wheel_height_progress` w=300 与登顶，守卫直接抓）。
    """
    s = _state(env)
    asset = env.scene[asset_cfg.name]
    jq = _T(asset.data.joint_pos)
    names = asset.joint_names
    ki = [names.index("fl_knee_joint"), names.index("fr_knee_joint")]
    ex = torch.clamp(jq[:, ki].abs() - lim, min=0.0) ** 2
    return ex.sum(dim=1) * s["top_win"].float()


def top_low_posture(env: "ManagerBasedRLEnv", ratio: float = 0.90, nom: float = 0.43,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), pw: float = 2.0) -> torch.Tensor:
    r"""**翻上台之后不许一直趴着**（B2m，09-17 18:1x）。作者：「你先完成能翻墙站稳」。

    ## 为什么前四轮治落地全都无效

        B2c 基线 下陷 0.216 恢复 1.71 s
        B2f 罚竖直速度      → 登顶 0/6（罚错了段：96.6% 的罚落在蓄力上）
        B2i 罚前腿弯曲      → 下陷 0.204 恢复 1.91 s
        B2k 罚水平速度      → 下陷 0.273 恢复 2.94 s（能量换到腿上，腿功 88→131 J）
        B2l 给身子上去付钱  → 翻越最高机身 0.572→**0.579**，纹丝不动

    **因为「落地砸下去」这个病根本不存在**：最高→最低落差我方 0.038~0.042，官方 0.050 还更大。
    真相是**过墙时身子本来就低**（我方翻越最高 0.572，官方 0.715 ≈ 台上站高），
    过去之后再花 2~3 s 撑起来。

    B2l 失败的原因也清楚了：**付钱给一个从没到达过的状态，PPO 摸不到，没有梯度。**
    所有样本翻越最高都是 0.57~0.58，纪录奖励悬在 0.66 够不着的地方。

    ## 本项为什么有梯度

    机器人**最后是能站起来的**（末端 z 0.746 ≈ 应有 0.75），只是要 2~3 s。
    **这个行为它做得到、只是慢** —— 罚"停在低姿态的时长"，它撑起来的过程会连续经过每个高度，
    每快一点就少罚一点，梯度处处存在。

    ## 先验读数（门内，n=6 vs 官方 n=3，各按自身标称站高）

        低于 0.90×站高的时长   官方 **0.00 s**    我方 **1.50 s**
        亏空中位               官方 0            我方 0.106（绝对口径）

    本实现用 `root_z`（机身高 − 脚下地形高），阈值 `ratio × nom` = 0.90 × 0.43 = 0.387。
    台上 `under_z` = 墙高，故 root_z = z − 墙高；我方最低点 root_z ≈ 0.20 → 亏空 0.186 → 平方 0.035。
    取 **w=30** → 每步罚约 1.0（主项 7.0 的 15%），整段 1.5 s（75 步）合计约 39，
    与爬高奖 +99 相比不会把爬墙变成不划算。

    **不可伪造**：纯惩罚；门是 `top_win`（翻上台后 1.5 s 窗口，由 encounter 真→假 + 四轮过旧墙高
    这个**状态转变**起计时，不是高度阈值——B2g 用相对出生点的高度当门，在多级地形上永久为真，
    把惩罚落到了抬前腿上，登顶 6/6→2/6）。进窗口必须真的翻上去，不爬就丢 +500 的爬高奖与登顶。
    """
    s = _state(env)
    thr = ratio * nom
    ex = torch.clamp(thr - s["root_z"], min=0.0)
    # pw=1 线性（B3m 起，09-18 14:0x）：亏空量级只有 0.05~0.16，平方后剩 0.0025~0.026，
    # 与今晚 cc_hind_push / cc_front_total 同一个坑（平方把价格再砍一到两个数量级）。
    ex = ex ** pw if pw != 1.0 else ex
    return ex * s["top_win"].float()


def front_leg_support(env: "ManagerBasedRLEnv", lim: float = 0.04, w_hind: float = 0.0) -> torch.Tensor:
    r"""**上台之后前腿要当支撑柱，不许软下去**（B2z，09-18 08:3x）。作者看回放点名：

    > 「目前就是因为没有蹬腿以及左前轮没有上墙后支撑，导致会有个摔倒墙上的动作」

    ## 实测（交付模型 R13，n=12 vs 官方 n=3；上台后 1 s 内腿长缩短量）

        | | fl 左前 | fr 右前 | hl 左后 | hr 右后 | 合计 |
        | 我方 | **−0.094** | −0.029 | −0.134 | −0.052 | **0.309** |
        | 官方 | **−0.022** | −0.010 | −0.099 | **+0.000** | **0.131** |

    **官方有一条腿（右后）从头到尾一点不缩 —— 那是它的支撑柱；我们四条腿一起软，总量是官方 2.4 倍。**
    前腿这一项差 **4.3 倍**，正是作者说的「左前轮上墙后没支撑」。

    ## 与昨晚失败的 B2i 的区别

    B2i 罚的是**膝角绝对值**（`|knee| > 1.70`）—— **官方台上前膝也到 1.47~1.59，会被一起罚**，
    而且膝角大是**结果**不是原因。本项罚的是**相对落台那一刻缩短了多少**：

        阈值 0.04（官方前腿上限 0.022）→ 我方超出 **0.054**，**官方恰好 0**。干净分离。

    另一个被先验读数否掉的假设（记此防再犯）：「落台那刻腿太直所以撑不住」——
    实测踏台瞬间较直那条前膝 我方 0.80 / **官方 0.97**，差距只有 0.17，
    按阈值 1.2 去罚会把官方也罚进去。**不是病因。**

    ## 门

    `top_win` —— 翻上台后 1.5 s 窗口，由 `encounter` 真→假 + 四轮过旧墙高这个**状态转变**起计时
    （不是高度阈值：`s["z"]` 相对 env 出生点，多级地形上「爬过任何东西后永久为真」，B2g 栽在那）。
    昨晚 B2i 已验证该门能正常触发（其目标量 上台前腿长差 0.057→0.009 真的动了）。

    `w_hind > 0` 时把后腿也纳入（官方后腿本就会缩 0.099，阈值要另设，默认关闭）。

    **不可伪造**：纯惩罚；进窗口必须真的翻上台面，不爬就丢爬高奖与登顶，守卫直接抓。
    """
    s = _state(env)
    d = s["leg_drop"]
    ex = torch.clamp(d[:, :2] - lim, min=0.0) ** 2
    out = ex.sum(dim=1)
    if w_hind > 0.0:
        out = out + w_hind * (torch.clamp(d[:, 2:] - 0.11, min=0.0) ** 2).sum(dim=1)
    return out * s["top_win"].float()


def hind_leg_push(env: "ManagerBasedRLEnv", lim: float = 0.06, pw: float = 2.0) -> torch.Tensor:
    r"""**推起段后腿不许被压垮**（B3a，09-18 08:5x）。作者：

    > 「蹬腿那个是不是一起做，不然等不起来、左前腿也伸不直？」

    力学上确实耦合：**后腿不蹬 → 身子起不来 → 过墙那一刻整个重量压到前腿 → 前腿再有力也被压软。**
    只治前腿等于让它单独对抗物理。

    ## 实测（R13 n=12 vs 官方 n=3；推起段内相对该段起点的最大缩短量）

        | | hl 左后 | hr 右后 |
        | 我方 | 0.002 | **0.173** |
        | 官方 | 0.000 | **0.049** |

    **我们的右后腿在推起过程中塌了 17.3 cm** —— 不是在蹬，是被压垮。
    阈值 **0.06**（官方上限 0.049 之上）：我方超出 0.113 → 值 0.0128；**官方恰好 0**。

    ## 为什么这次可能行（昨晚六种办法失败的原因）

    昨晚治蹬腿的六种办法（撞墙罚 2→6、蹬地奖 5→60、RSI 两种窗口、低速指令、
    爬高奖 500→250→125、收腿降价 54→18）**全部卡在 `slow` 那道门上**——
    `st_hind_extend` 的门要求 `|vx| < 0.5`，而推起段实测 1.2~2.1 m/s，门只开 **0~8%**。

    **本项不依赖那道门**：门是「两前轮已上台 & 后轮还没全上 & 遭遇中」，实测 100% 命中。
    也不依赖「伸直到 |knee|<0.65」这种官方自己都未必满足的状态，只要求**别被压垮**。

    **不可伪造**：纯惩罚；门要求两前轮真的上了台面，不爬就进不去。
    """
    s = _state(env)
    # pw=1 线性（B3c 起）：量本身只有 0.0128，再平方就剩 0.000074 —— 每局只值 0.0105，
    # 而「奖励把后腿收上来」的 cc_hind_lift 每局 +2.26，**差 215 倍**，罚项被直接盖过。
    # 线性形同权重下价格涨 12 倍，且阈值附近梯度不消失（要把 0.17 压到 0.05，全程都需要梯度）。
    ex = torch.clamp(s["hind_drop"][:, 2:] - lim, min=0.0)
    ex = ex ** pw if pw != 1.0 else ex
    return ex.sum(dim=1) * s["push_win"].float()


def front_leg_support_total(env: "ManagerBasedRLEnv", lim: float = 0.06, pw: float = 2.0) -> torch.Tensor:
    r"""**上台后两前腿「合计」不许塌**（B3b，09-18 09:2x）。R20 证伪了按腿分开罚的写法。

    ## R20（`front_leg_support`，Σ clamp(d_i−0.04,0)²，w=300，n=12）的实测后果

        前腿塌陷 0.0775 → **0.0735**（纹丝不动，目标没达成）
        领先前轮 0.282  → **0.084**（塌了 70%）
        台上侧倾 15.4   → **20.9 / 最差 34.7**
        抬头     27.7   → 19.5 ；机身最高 0.741 → 0.688

    **机制**：逐腿平方和奖励「把塌陷摊到两条腿上」。一条腿塌 0.10 → (0.06)²=0.0036；
    两条各塌 0.05 → 2×(0.01)²=0.0002，**减免 36 倍**。策略于是让两前轮一起上墙来分摊，
    领先前轮因此消失，而总塌陷量根本没降 —— 它用改步态满足了罚项，没有真的把腿撑硬。

    ## 和形（本项）

    罚 `clamp(Σd_front − lim, 0)²`：**只看两前腿一共塌了多少，对怎么分摊完全中性**，
    堵死上面那条捷径。

    先验读数（上台后 1.5 s 窗口内 两前腿缩短量之和的峰值，与 climb_check 同口径）：

        | | 中位 | 最差 |
        | 官方 (n=3)    | 0.024 | **0.054** |
        | 我方 R13 (n=12)| **0.111** | 0.137 |

    阈值 **0.06**（官方最差 0.054 之上）→ 我方 0.00258、**官方 0.00000**。
    w=600 → 我方每步 1.55（主项约 7 的 22%），官方恒为 0。

    **不可伪造**：纯惩罚；门 `top_win` 要求真的翻上过台面，不爬就进不去。
    """
    s = _state(env)
    tot = s["leg_drop"][:, :2].sum(dim=1)
    ex = torch.clamp(tot - lim, min=0.0)
    ex = ex ** pw if pw != 1.0 else ex                       # pw=1 线性（B3c 起），理由同 hind_leg_push
    return ex * s["top_win"].float()


def front_leg_symmetry(env: "ManagerBasedRLEnv", lim: float = 0.03) -> torch.Tensor:
    r"""**上台后两条前腿必须等长**（B3g，09-18 11:3x）。作者：「左前轮要支撑」。

    ## 官方与我方的形态差异（先验读数，与 climb_check 同口径）

        |fl−fr| (m)         官方    R13    R24    R25
        前轮上台前 0.2 s     0.186  0.036  0.063  0.056    ← **官方远比我们不对称**
        两前轮上台那刻       0.132  0.028  0.033  0.044    ← 官方仍不对称
        **四轮全上那刻**     0.003  0.032  0.075  0.098    ← **官方几乎为 0，我方 16~30 倍**
        上台后 0.5 s 均      0.010  0.040  0.031  0.044

    **官方是「先极不对称、后极对称」**：上墙时一条腿远远领先（领先前轮 0.41~0.47），
    四轮全上的瞬间两腿收成几乎一样长（0.003）并保持。
    **我方正相反「先对称、后发散」**：上墙时比官方还对称，四轮全上时反而张到 0.075。

    这就是「左前轮没支撑」的物理形态：**官方两条前腿一起站成支柱，我方一条撑、一条吊着。**
    也解释了为什么罚「塌陷总量」三轮都没用 —— 总量不变，只要一长一短，支撑就是坏的。

    **门只在四轮全上之后（top_win）**：上墙过程中官方比我们还不对称，那一段不能罚。

    ## 标定（top_win 1.5 s 窗口）

        |fl−fr| 均值  官方 0.014 / R24 **0.011**（我方平均更对称！）
        |fl−fr| 峰值  官方 **0.028** / R24 **0.096**  ← 问题全在瞬时尖峰
        lim=0.03（官方峰值之上）→ 每局 官方 **0.00000**、R24 0.00063
        w=800 → 我方每局 −0.50，**官方恒为 0**

    ## 不可伪造

    纯惩罚；门要求真的四轮上台；**官方轨迹在本项上恒为 0**，不会误伤官方式动作。

    ## 诚实标注

    「两前腿不等长 → 摔倒」的**因果未被证实**：跨轮次相关 +0.79，但轮次内去均值后只有 +0.37，
    且各轮符号不一致（R13 −0.08 / R21 −0.35 / R24 +0.63 / R25 +0.42）。
    **本项是「对齐官方一个从未被管过的维度」，不是「已证实的摔倒病因」。**
    """
    s = _state(env)
    d = torch.abs(s["leg_len"][:, 0] - s["leg_len"][:, 1])
    return torch.clamp(d - lim, min=0.0) * s["top_win"].float()


def crouch_time_penalty(env: "ManagerBasedRLEnv", min_s: float = 1.0) -> torch.Tensor:
    r"""**蓄力太短就罚**（B3h，09-18 11:5x）。判据表里**最大的缺口**，且从未被任何项管过。

        蓄力→抬腿(s)   官方 **1.135~1.870**   我方 R26 **0.312**   ← 4~6 倍

    `st_hind_flex` 是 `StageRecord`：奖励**蹲得多深**（cap 0.9 = 膝角 −1.55），
    **不奖励蹲多久**。所以策略学成「蹲下去立刻弹起来」—— 0.3 秒就抬前轮。

    ## 为什么是这一项（R20/R22/R25/R26 四轮的教训）

    我先后罚过 逐腿塌陷 / 合计塌陷 / 两腿等长 三个量，形式和价格都按标定做到位，
    **三个量全都没被治动，落地反而一次比一次差**（侧倾 24.8 → 33.9 → 37.6）。
    ⇒ **落地时的腿部动作不是「选择」，是被动力学逼出来的**：
      机身以 38° 仰角冲到台沿再砸到前腿上，腿只能被动吸收。
      **要治的是「身子怎么到达的」，不是「腿到了之后做什么」。**

    ## 形式

    在抬前轮之后的 0.5 s 窗口内，持续罚 `clamp(min_s − 蓄力时长, 0)`。
    **官方蓄力 >=1.135 s，取 min_s=1.0 → 官方恒为 0。** 纯惩罚，不可刷分
    （门是「抬前轮」这个必须发生的事件，拖着不抬拿不到任何好处，还要照付撞墙/停滞罚）。
    """
    s = _state(env)
    return torch.clamp(min_s - s["crouch_dur0"], min=0.0) * s["lift_win"].float()


def top_roll_penalty(env: "ManagerBasedRLEnv", thr_deg: float = 10.0) -> torch.Tensor:
    r"""**落地那 1.5 秒的侧倾罚**（B3p，09-18 15:4x）。补一个**定价漏洞**，不是新机制。

    ## 漏洞

        cc_roll_ph   w=**−60**  门 = `phase`（翻越相位：轮高差>12cm **或** 前方立面>15cm）
        cc_roll      w=−15      **无门**（全程），被 30 s episode 稀释 → 每局仅 **−0.299**

    **四轮一上台面、前方没墙了，`phase` 立刻关掉，那个 −60 随之消失。**
    而台上侧倾峰实测出现在**上台后 0.4~0.6 s** —— 正好落在 `phase` 之后、`top_win` 之内。
    **落地窗口是整套动作里侧倾定价最弱的一段。**

    ## 标定（top_win 1.5 s 窗口，线性形，口径 = |g_y| ≈ sin(roll)）

        窗口内侧倾峰    官方 **6.9°**   R13 18.2°   R31 23.1°
        阈值 10°（sin 0.174）→ 每局值 官方 **0.00000**（三段最差也 0）/ R13 0.00099 / R31 0.00322
        w=400 → R31 约 **−1.29**/局、R13 −0.40、**官方恒 0**

    线性形（不用 `roll_penalty` 的平方）：今晚第四次同一个坑 —— 量只有 0.003，平方后没有价格。

    **不可伪造**：纯惩罚；门 `top_win` 要求真的四轮翻上台面；**官方在本项上结构性为 0**。
    """
    asset = env.scene["robot"]
    g = asset.data.projected_gravity_b
    g = g.torch if hasattr(g, "torch") else g
    ex = torch.clamp(g[:, 1].abs() - math.sin(math.radians(thr_deg)), min=0.0)
    return ex * _state(env)["top_win"].float()


def top_hind_knee_limit(env: "ManagerBasedRLEnv", lim: float = 2.30) -> torch.Tensor:
    r"""**台上后膝不许折到限位**（B3q，09-18 16:2x）。B2g 的想法，换上修好的门重做。

    ## 实测：我方把后腿折到了机械限位

    机身最低那一刻（n=12 vs 官方 n=3）：

        |          | 后膝角 | 后腿长 |
        | **官方** | **2.185** | **0.207 m** |
        | R13      | 2.690 | 0.148 |
        | R31      | **2.693** | **0.147** |

    **关节限位 2.7227 —— 我方离限位只剩 0.029 rad，等于顶死。后腿被折成最短的 0.147 m。**
    官方只折到 2.185，腿比我们长 **6 厘米**。

    **所以「机身沉下去」不是力矩撑不住，是策略主动把后腿折到机械限位。**
    这与「净空 vs 侧倾 轮次内相关 −0.83/−0.82/−0.88/−0.97（四轮）」接上：
    后腿折死 → 机身低 → 侧倾大。

    ## 与 B2g 的关系（必须说清）

    `on_top_knee_limit`（B2g，09-17）罚的就是这件事，lim 也是 2.30。**但它的门是
    「四轮轮底绝对高度 > 0.18」，而 z 是相对 env 出生点算的 —— 在多级课程地形上
    「爬过任何东西之后永久为真」，惩罚落到了抬前腿上，登顶 6/6 → 2/6。**

    **那是门的 bug，不是想法的失败。** 后来建的 `top_win`（由 encounter 真→假 +
    四轮过旧墙高这个**状态转变**起计时）修的正是这个问题。本项用 `top_win`，**从没被有效检验过**。

    ## 标定（top_win 1.5 s 窗口，线性形，两条后腿超限位量之和）

        后膝峰   官方 **2.185**  R13 2.716  R31 2.721
        lim=2.30（官方峰 2.223 之上留 0.08）→ 每局值 官方 **0.00000**（三段最差也 0）
                                              R13 0.01678 / R31 **0.01896**
        w=80 → R31 约 **−1.52**/局、**官方恒 0**

    **不可伪造**：纯惩罚；门要求真的四轮翻上台面；**官方结构性为 0**。
    """
    asset = env.scene["robot"]
    ids = _ids(env)
    kh = torch.abs(_T(asset.data.joint_pos)[:, ids["knee_h"]])
    return torch.clamp(kh - lim, min=0.0).sum(dim=1) * _state(env)["top_win"].float()
