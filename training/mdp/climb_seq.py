"""翻墙时序相位 + 分组跟踪 v3（09-15，Astra 直接修复共享推进与RSI对应）。

v3：七个奖励实例共享逐环境步缓存，一步只推进一次；距离轴按各环境参考段选择。
RSI首次进入保持出生相位，出生种子每回合只消费一次。集成测试见 tests/test_climb_seq.py。

v1 的五处错误（已逐条自验属实）：
  P1-1 **帧号混用**：`j_t` 是相对 t=0 的帧号，`j_d` 却加了 `t0i=25`（含 t<0 预备段的完整下标），
       两套坐标系直接限幅。ph=0 时 j_t=0 → 限幅区间 [0,1]，而 j_d≥25 被硬压到 1。
       实测最大右前膝误差 1.302 rad，不是自检报的 0.000。
  P1-1b **自检没调用生产函数**：`phase2.py` 是另一份实现（先裁掉 t<0 段，帧号天然一致），
       它报 0.000 与生产代码无关。**本版把推进逻辑写成纯函数 `advance_phase`，生产与自检共用。**
  P1-2 **缺生命周期**：无 reset 清理、跨回合继承、忽略 RSI 出生相位、超时后从头重触发。
  P1-4 离线探针取错列（Q[:,2] 是 hl_hipx 不是 fl_knee）—— 本版一律**按关节名取列**。
  P2   领先用峰值大小判（不含先后时间）—— 归验收器修，不在本文件。

坐标约定：**内部一律用"相对 t=0 的前向帧号 k"**（0 .. n_fwd-1），
只在最后索引参考数组时加 `t0i`。参考的 t<0 段（停在墙前）不参与本机制。
"""
from __future__ import annotations
import torch
import os
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.managers import RewardTermCfg
from .climb_ref import load_ref, _state


def _stop_ready(env, s, fresh):
    """块3c：相位进入前必须「已停住」。返回 fresh 子集上的 bool，或 None（关闭时）。

    环境变量（默认 0 = 关，行为与加门之前完全一致）：
      S10_SEQ_STOP_HOLD_S   需要连续停住多少秒才放行（0 = 关门）
      S10_SEQ_STOP_V        判定"停住"的 |vx| 阈值，默认 0.10 m/s
      S10_SEQ_STOP_TIMEOUT  遭遇后多少秒仍没停住也放行（防死锁，保住登顶），默认 4.0 s

    状态存在 env 上（与 _cseq_* 同一套生命周期）：遭遇中累计停住时长与遭遇时长，
    离开遭遇即清零。**不能存在实例上** —— 七个 cs_* 项共用同一份相位（P1-1b 教训）。
    """
    # 模式 released（块5，09-16 18:5x）：门 = 命令剖面已走完「停住 → 后退」并释放。
    # 为什么从「已停住」改成「已释放」：块3c 实测停住 0.21 s、墙前最小 |vx| 0.056（判据①达标），
    # 但**后退仍是 0.010 s**。原因是停够 0.4 s 后相位门立刻开，上墙序列一启动机器人就往前冲，
    # −0.15 的后退指令被同一个「见墙就冲」反射压掉。官方次序是 **停 → 退 → 才起动作**
    # （ref_climb34 seg0 连续后退 0.405 s，发生在 vx≈+0.02 的停住段之内）。
    # 改成等剖面释放，等于把后退段整个放在相位起跑之前，同时把蓄力窗再拉长 back_s 秒。
    if os.environ.get("S10_SEQ_GATE", "") == "released":
        st = getattr(env, "_climb_cmd_state", None)
        if st is None:
            return None                      # 命令项还没发布过状态（首步）：不挡
        t_out = float(os.environ.get("S10_SEQ_STOP_TIMEOUT", "4.0"))
        n = env.num_envs
        if getattr(env, "_cseq_enct", None) is None or env._cseq_enct.shape[0] != n:
            env._cseq_enct = torch.zeros(n, device=env.device)
        enc = s["encounter"]
        adv = torch.zeros(n, dtype=torch.bool, device=env.device); adv[fresh] = True
        # 同上：只累加，**不因遭遇瞬断清零**（清零改由回合 reset 负责）。
        env._cseq_enct = torch.where(adv & enc, env._cseq_enct + env.step_dt, env._cseq_enct)
        ready = (st == 3) | (env._cseq_enct >= t_out)
        ev = int(os.environ.get("S10_SEQ_DBG", "0") or 0)
        if ev > 0 and (env.common_step_counter % ev == 0):
            n = float(env.num_envs); f = lambda m: 100.0 * m.float().sum().item() / n
            k_ = env._cseq_k
            print(f"[seq-gate] step={env.common_step_counter} 遭遇={f(enc):.2f}% "
                  f"剖面ST_DONE={f(st == 3):.2f}% 超时={f(env._cseq_enct >= t_out):.2f}% "
                  f"ready={f(ready):.2f}% 相位已激活={f(k_ >= 0):.2f}% "
                  f"待进(k<0&遭遇&sd<=0.70)={f((k_ < 0) & enc & (s['signed_dist'] <= 0.70)):.2f}% "
                  f"enct中位={env._cseq_enct[enc].median().item() if enc.any() else float('nan'):.2f}s "
                  f"st分布={[int((st == i).sum().item()) for i in range(4)]}", flush=True)
            # 按离墙分箱看 encounter 保持率：验证「越近越丢遭遇」这一假设
            sdv = s["signed_dist"]
            bands = [(1.00, 0.75), (0.75, 0.60), (0.60, 0.50), (0.50, 0.40), (0.40, 0.30), (0.30, 0.15)]
            parts = []
            for a, b in bands:
                inb = (sdv <= a) & (sdv > b)
                tot = int(inb.sum().item())
                if tot == 0:
                    parts.append(f"{a:.2f}-{b:.2f}:—")
                else:
                    parts.append(f"{a:.2f}-{b:.2f}:{100.0*float((inb & enc).sum().item())/tot:.0f}%({tot})")
            print("[enc-band] 遭遇保持率 " + "  ".join(parts), flush=True)
        return ready[fresh]    # 3 = climb_cmd.ST_DONE

    hold_s = float(os.environ.get("S10_SEQ_STOP_HOLD_S", "0") or 0)
    if hold_s <= 0:
        return None
    v_thr = float(os.environ.get("S10_SEQ_STOP_V", "0.10"))
    t_out = float(os.environ.get("S10_SEQ_STOP_TIMEOUT", "4.0"))
    n = env.num_envs
    if getattr(env, "_cseq_stophold", None) is None or env._cseq_stophold.shape[0] != n:
        env._cseq_stophold = torch.zeros(n, device=env.device)
        env._cseq_enct = torch.zeros(n, device=env.device)
    enc = s["encounter"]
    vx = s["vx"].abs()
    dt = env.step_dt
    # 只在本步 fresh 的 env 上推进，避免同一步被七个奖励项累加七次
    adv = torch.zeros(n, dtype=torch.bool, device=env.device)
    adv[fresh] = True
    still = adv & enc & (vx < v_thr)
    moving = adv & enc & (vx >= v_thr)
    env._cseq_stophold = torch.where(still, env._cseq_stophold + dt, env._cseq_stophold)
    env._cseq_stophold = torch.where(moving, torch.zeros_like(env._cseq_stophold), env._cseq_stophold)
    env._cseq_enct = torch.where(adv & enc, env._cseq_enct + dt, env._cseq_enct)
    env._cseq_stophold = torch.where(enc, env._cseq_stophold, torch.zeros_like(env._cseq_stophold))
    env._cseq_enct = torch.where(enc, env._cseq_enct, torch.zeros_like(env._cseq_enct))
    return ((env._cseq_stophold >= hold_s) | (env._cseq_enct >= t_out))[fresh]


def advance_phase(k, done, ph, sd, enc, x_fwd, *, dt_env, dt_ref, n_fwd,
                  enter_dist, corr=0.30, spawn_ph=None, ready=None):
    """**纯函数**：推进一步相位。生产与自检共用同一份实现（P1-1b 教训）。

    k     (n,) long  相对 t=0 的前向帧号；-1 = 未激活
    done  (n,) bool  本次遭遇已完成/超时，不再重触发（治"一步失活、下一步从头重来"）
    ph    (n,) float 相对 t=0 的秒数
    sd    (n,) float signed_dist，正 = 机身在墙面前方
    enc   (n,) bool  遭遇
    x_fwd (m,) 或 (n,m) float，参考 t>=0 段的 x_rel；批量形式每行对应本环境的参考段
    spawn_ph (n,) float 或 None：本回合首次调用的RSI相位；负值表示没有前向RSI种子

    返回 (k, done, ph, active)
    """
    # 遭遇结束 → 本次生命周期结束，清空（可再次进入新遭遇）
    k = torch.where(enc, k, torch.full_like(k, -1))
    done = done & enc
    ph = torch.where(enc, ph, torch.zeros_like(ph))

    # 进入：未激活 & 本次遭遇未完成 & 已到触发距离 &（可选）**已经停住**
    # ready 门（块3c，09-16）：作者原话「慢速开到墙前 **停下来** 然后开始做动作」。
    # 官方 ref_climb34 三段在离墙 0.60~0.45 m 处 vx 只有 +0.006~+0.035，即先停住再起动作。
    # 我方原来一到 0.70 m 就起相位时钟，机身仍有 0.87 m/s，于是"边冲边做动作"。
    # ready=None 时行为与加门之前逐位一致。
    enter = (k < 0) & (~done) & enc & (sd <= enter_dist)
    if ready is not None:
        enter = enter & ready
    seeded = torch.zeros_like(enter)
    ph = torch.where(enter, torch.zeros_like(ph), ph)
    if spawn_ph is not None:
        # 出生相位是已知真值，进入首帧不让距离噪声/非单调距离覆盖它。
        seeded = enter & (spawn_ph >= 0)
        ph = torch.where(seeded, spawn_ph, ph)
    k = torch.where(enter, torch.round(ph / dt_ref).long(), k)

    active = k >= 0
    # 时间推进
    ph = torch.where(active & ~seeded, ph + dt_env, ph)
    k_t = torch.round(ph / dt_ref).long()

    # 距离建议的帧号（**同一坐标系**：x_fwd 就是 t>=0 段，下标即 k）
    axis = x_fwd.unsqueeze(0) if x_fwd.ndim == 1 else x_fwd
    k_d = torch.argmin((axis + sd.unsqueeze(1)).abs(), dim=1)

    # ±corr 限幅（对 k_t，同坐标系）+ 绝不回退
    span = (k_t.float() * corr).long() + 1
    k_new = torch.max(torch.min(k_d, k_t + span), k_t - span)
    k_new = torch.maximum(k_new, k)                      # 只向前
    k_new = torch.where(seeded, k, k_new)
    k = torch.where(active, k_new, k)

    # 完成 / 超时
    over = active & (k >= n_fwd - 1)
    done = done | over
    k = torch.where(over, torch.full_like(k, -1), k)
    active = k >= 0
    return k, done, ph, active


class _Seq(ManagerTermBase):
    """共享时序相位状态（每 env）。所有分组跟踪项从它取相位；每步只推进一次。"""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.ref = load_ref(env.device)
        self.dt_ref = float(self.ref["dt"]); self.per = int(self.ref["per"])
        self.t0i = int(round((0.0 - float(self.ref["t0"])) / self.dt_ref))
        self.n_fwd = self.per - self.t0i
        xs = self.ref["x"].view(-1, self.per)
        self.x_fwd = xs[:, self.t0i:].contiguous()
        n = env.num_envs; dev = env.device
        if getattr(env, "_cseq_k", None) is None or env._cseq_k.shape[0] != n:
            env._cseq_k = torch.full((n,), -1, dtype=torch.long, device=dev)
            env._cseq_done = torch.zeros(n, dtype=torch.bool, device=dev)
            env._cseq_ph = torch.zeros(n, device=dev)
            env._cseq_ready = torch.zeros(n, dtype=torch.bool, device=dev)
            env._cseq_last_step = torch.full((n,), -1, dtype=torch.long, device=dev)
            env._cseq_refseg = torch.zeros(n, dtype=torch.long, device=dev)
            env._cseq_spawn_pending = torch.ones(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None):
        """**回合重置**（P1-2）：清空相位、完成标志与准备锁存，防止跨回合继承。"""
        e = self._env
        ids = slice(None) if env_ids is None else env_ids
        e._cseq_k[ids] = -1; e._cseq_done[ids] = False
        e._cseq_ph[ids] = 0.0; e._cseq_ready[ids] = False
        e._cseq_last_step[ids] = -1
        # 诊断统计随回合**结算并清零**（Astra 交接后核查 问题一）：
        # 原先 `_cseq_dur`/`_cseq_bins` 未接入 reset，会把两个回合拼起来统计
        # （复现：旧回合 0.10s，reset 后仍 0.10s，新回合首步变 0.12s）。
        if getattr(e, "_cseq_dur", None) is not None:
            d_ = e._cseq_dur[ids]
            keep = d_ > 0
            if bool(keep.any()):
                nb_ = e._cseq_bins[ids][keep].sum(1).tolist()
                for dd_, bb_ in zip(d_[keep].tolist(), nb_):
                    e._cseq_hist.append((dd_, int(bb_), "reset"))     # 结束原因：回合重置/中断
                e._cseq_hist[:] = e._cseq_hist[-2000:]
            e._cseq_dur[ids] = 0.0
            e._cseq_bins[ids] = False
        e._cseq_refseg[ids] = 0
        e._cseq_spawn_pending[ids] = True
        if getattr(e, "_cseq_enct", None) is not None:
            e._cseq_enct[ids] = 0.0            # 相位门的遭遇计时：只在回合边界清
        return {}

    def step(self, env, enter_dist, corr):
        # 步标记必须与相位一起放在env上；实例各自缓存会把同一步推进七次。
        # 逐环境标记也允许部分reset：仅重置的环境重算，其他环境仍读本步快照。
        fresh = env._cseq_last_step != env.common_step_counter
        if not fresh.any():
            return env._cseq_k >= 0
        s = _state(env)
        seg = getattr(env, "_climb_ref_seg", None)
        if seg is None:
            seg = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._cseq_refseg[fresh] = seg[fresh]
        spawn = getattr(env, "_climb_ref_phase", None)
        if spawn is not None:
            spawn = torch.where(env._cseq_spawn_pending[fresh], spawn[fresh],
                                torch.full_like(spawn[fresh], -1.0))
        k, done, ph, _ = advance_phase(
            env._cseq_k[fresh], env._cseq_done[fresh], env._cseq_ph[fresh],
            s["signed_dist"][fresh], s["encounter"][fresh], self.x_fwd[env._cseq_refseg[fresh]],
            dt_env=env.step_dt, dt_ref=self.dt_ref, n_fwd=self.n_fwd,
            enter_dist=enter_dist, corr=corr, spawn_ph=spawn,
            ready=_stop_ready(env, s, fresh))
        env._cseq_k[fresh] = k; env._cseq_done[fresh] = done; env._cseq_ph[fresh] = ph
        env._cseq_ready[fresh] &= k >= 0
        env._cseq_spawn_pending[fresh] = False
        env._cseq_last_step[fresh] = env.common_step_counter
        active = env._cseq_k >= 0
        # ===== 每条轨迹的真实相位时长与累计阶段访问（Astra 交接第 2 条）=====
        # 横截面的"覆盖 N/61"只是某一瞬间所有 env 的帧号分布，**不是完整动作链覆盖率**。
        # 这里按 env 累计：本次相位持续了多少步、访问过哪些帧段（分 6 档）。
        if os.environ.get("S10_DBG_SEQ") == "1":
            n_ = env.num_envs
            if getattr(env, "_cseq_dur", None) is None or env._cseq_dur.shape[0] != n_:
                env._cseq_dur = torch.zeros(n_, device=env.device)
                env._cseq_bins = torch.zeros((n_, 6), dtype=torch.bool, device=env.device)
                env._cseq_hist = []            # 已结束轨迹的 (时长秒, 访问档数)
            # **只在本步真正推进过的 env 上累计**（fresh 掩码），避免 7 个实例重复累加
            a_ = (env._cseq_k >= 0) & fresh
            env._cseq_dur = torch.where(a_, env._cseq_dur + env.step_dt, env._cseq_dur)
            b_ = (env._cseq_k.clamp(min=0).float() / max(self.n_fwd, 1) * 6).long().clamp(0, 5)
            env._cseq_bins[torch.arange(n_, device=env.device)[a_], b_[a_]] = True
            fin = (env._cseq_k < 0) & (env._cseq_dur > 0) & fresh
            if fin.any():
                for d_, nb_ in zip(env._cseq_dur[fin].tolist(), env._cseq_bins[fin].sum(1).tolist()):
                    env._cseq_hist.append((d_, int(nb_), "finish"))   # 结束原因：相位走完/遭遇结束
                env._cseq_hist[:] = env._cseq_hist[-2000:]
                env._cseq_dur[fin] = 0.0; env._cseq_bins[fin] = False
        if os.environ.get("S10_DBG_SEQ") == "1" and env.common_step_counter % 100 == 0:
            k = env._cseq_k; a = k >= 0
            enc = s["encounter"]
            msg = (f"[dbg-seq] step={env.common_step_counter} encounter={enc.float().mean()*100:.1f}% "
                   f"激活={a.float().mean()*100:.2f}% done={env._cseq_done.float().mean()*100:.2f}% "
                   f"ready={env._cseq_ready.float().mean()*100:.2f}%")
            if a.any():
                ka = k[a].float()
                msg += (f" | 帧k 均{ka.mean():.1f} 中位{ka.median():.0f} 范围{int(ka.min())}~{int(ka.max())}"
                        f" 覆盖{len(torch.unique(k[a]))}/{self.n_fwd}")
                lc = getattr(env, "_cseq_lastcost", None)
                if lc is not None:
                    msg += " | 闸内单步代价 " + " ".join(f"{g}{v:.3f}" for g, v in lc.items())
            h_ = getattr(env, "_cseq_hist", None)
            if h_:
                du = [x[0] for x in h_]; nb = [x[1] for x in h_]
                full = sum(1 for x in nb if x >= 6)
                nres = sum(1 for x in h_ if len(x) > 2 and x[2] == "reset")
                # **口径**（Astra）：六档是把参考帧号均分六段，这是**参考相位区间覆盖率**，
                # **不是**右前搭接/后腿推进/稳定落地的动作链成功率。
                msg += (f" || 轨迹 {len(h_)}（其中中断 {nres}）  相位时长 均{sum(du)/len(du):.2f}s "
                        f"中位{sorted(du)[len(du)//2]:.2f}s  区间访问均{sum(nb)/len(nb):.1f}/6  "
                        f"参考相位区间覆盖率 {full}/{len(h_)} = {100*full/len(h_):.0f}%")
            print(msg, flush=True)
        return active

    def frame(self, env):
        k = env._cseq_k.clamp(0, self.n_fwd - 1)
        return env._cseq_refseg * self.per + (k + self.t0i)


def _huber(x, d=0.5):
    ax = x.abs()
    return torch.where(ax < d, 0.5 * ax.pow(2), d * (ax - 0.5 * d))


# 参考为 articulation 序（按类分块）：hipx 0-3, hipy 4-7, knee 8-11, wheel 12-15；每块顺序 fl,fr,hl,hr
# **一律按关节名取列**（P1-4 教训：v1 离线探针把 Q[:,2]=hl_hipx 当成了 fl_knee）
_BLK = {"hipx": 0, "hipy": 4, "knee": 8, "wheel": 12}
_LEG = {"fl": 0, "fr": 1, "hl": 2, "hr": 3}
def ref_col(name: str) -> int:
    """'fr_knee_joint' → 参考数组列号。按名字算，不靠下标记忆。"""
    leg, blk = name.split("_")[0], name.split("_")[1]
    return _BLK[blk] + _LEG[leg]


_ALL12 = [f"{L}_{J}_joint" for L in ("fl", "fr", "hl", "hr") for J in ("hipx", "hipy", "knee")]
_GROUPS = {
    "rear":      ["hl_hipy_joint", "hr_hipy_joint"],
    "rear_knee": ["hl_knee_joint", "hr_knee_joint"],
    "fr_knee":   ["fr_knee_joint"],
    "fl_knee":   ["fl_knee_joint"],
    # 全身 12 腿关节 —— 取代旧 `st_ref_track`（它用的是有相位倒退 bug 的距离查表，
    # 且把 12 关节平均、把单关节误差摊薄成 1/12）。本组共享**修正后的相位**。
    "all12":     _ALL12,
    # 09-16 块2：按**角色**分组（作者拍板镜像：领先=左前、对角发力=右后、支撑=左后）。
    # 旧的 "rear"/"rear_knee" 把两条后腿同权跟踪，分不出发力腿与支撑腿 ——
    # 实测我方是「左前+左后同侧」而非「左前+右后对角」，同权项无法纠正。
    "drive_rear": ["hr_hipy_joint", "hr_knee_joint"],   # 对角发力腿（官方左后的镜像）
    "supp_rear":  ["hl_hipy_joint", "hl_knee_joint"],   # 支撑腿（官方右后的镜像）
    "lead_front": ["fl_hipy_joint", "fl_knee_joint"],   # 领先前腿（官方右前的镜像）
    "foll_front": ["fr_hipy_joint", "fr_knee_joint"],   # 跟随前腿
}

# 09-16：镜像取列 —— 作者拍板「完整左右镜像」，参考按 fl<->fr / hl<->hr 取，hipx 变号。
_MIRROR_LEG = {"fl": "fr", "fr": "fl", "hl": "hr", "hr": "hl"}


def ref_col_mirror(name: str):
    """返回 (列号, 符号)。镜像后 hipx 变号，hipy/knee/wheel 同号。"""
    leg, blk = name.split("_")[0], name.split("_")[1]
    return _BLK[blk] + _LEG[_MIRROR_LEG[leg]], (-1.0 if blk == "hipx" else 1.0)


class SeqTrack(_Seq):
    """分组跟踪代价（**返回正代价，配负权重**）。Huber 不用窄高斯
    （σ=0.5 在 2.58rad 处 = 2.7e-12，与 C26 的 σ=0.35 死核同错）。
    官方靶 seg0：后腿 hipy 峰 +1.07/+1.10；**fr膝峰 +2.72**；fl膝峰 +1.46（右前领先）。
    我方 C36b：后腿 +0.69/+0.71；fr膝 +1.19；fl膝 +2.23（左前领先，反了）。"""

    def __call__(self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 group: str = "rear", enter_dist: float = 0.70,
                 corr: float = 0.30, huber_d: float = 0.5,
                 mirror: bool = False) -> torch.Tensor:
        active = self.step(env, enter_dist, corr)
        fidx = self.frame(env)
        asset = env.scene[asset_cfg.name]
        jq = asset.data.joint_pos; jq = jq.torch if hasattr(jq, "torch") else jq
        jn = asset.joint_names
        qr = self.ref["q"][fidx]
        c = torch.zeros(env.num_envs, device=env.device)
        for nm in _GROUPS[group]:
            if mirror:
                col, sgn = ref_col_mirror(nm)
                tgt = sgn * qr[:, col]
            else:
                tgt = qr[:, ref_col(nm)]
            c = c + _huber(jq[:, jn.index(nm)] - tgt, huber_d)
        if os.environ.get("S10_DBG_SEQ") == "1" and active.any():
            d = getattr(env, "_cseq_lastcost", None)
            if d is None: d = {}; env._cseq_lastcost = d
            d[group] = float(c[active].mean())          # **闸内**单步代价，不被回合长度稀释
        return c * active.float()


class SeqReadyLatch(_Seq):
    """**准备门锁存**版的「蓄力未完成就抬前轮」罚（治 Astra 3.3）。
    旧版每步重判 → 蹬地时后腿伸展、hipy 下降 → 又判「未准备好」，**会罚官方动作**
    （官方被罚 21/23 帧、罚值和 3.02/3.49）；锁存后 2/1 帧、罚值 0.0002/0.0000。
    锁存随**相位生命周期**清理（reset / 遭遇结束），不再挂在自造计时器上。"""

    def __call__(self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 enter_dist: float = 0.70, corr: float = 0.30,
                 ready: float = 0.85, thr: float = 0.05) -> torch.Tensor:
        active = self.step(env, enter_dist, corr)
        asset = env.scene[asset_cfg.name]
        jq = asset.data.joint_pos; jq = jq.torch if hasattr(jq, "torch") else jq
        bp = asset.data.body_pos_w; bp = bp.torch if hasattr(bp, "torch") else bp
        jn = asset.joint_names; bn = asset.body_names
        load = torch.minimum(jq[:, jn.index("hl_hipy_joint")], jq[:, jn.index("hr_hipy_joint")])
        env._cseq_ready = (env._cseq_ready | (load >= ready)) & active
        org = env.scene.env_origins[:, 2]
        fh = torch.maximum(bp[:, bn.index("fl_wheel"), 2], bp[:, bn.index("fr_wheel"), 2]) - org - 0.081
        return (fh - thr).clamp(min=0.0).pow(2) * (active & ~env._cseq_ready).float()



class SeqPitch(_Seq):
    """机身俯仰跟踪，**统一量纲**（Astra 5.2 指出的口径错误）。
    旧 `RefTrack` 直接拿 `pitch_up = −projected_gravity_b[:,0]`（重力投影，≈sin 角）
    与参考的 **弧度角** 相减，42° 处差 9%，横滚大时更不成立。
    本项把投影量反解成角度：`asin(clamp(pitch_up,-1,1))`，再与参考弧度比。"""

    def __call__(self, env, enter_dist: float = 0.70, corr: float = 0.30,
                 huber_d: float = 0.20) -> torch.Tensor:
        active = self.step(env, enter_dist, corr)
        fidx = self.frame(env)
        s = _state(env)
        ang = torch.asin(s["pitch_up"].clamp(-1.0, 1.0))
        return _huber(ang - self.ref["pitch"][fidx], huber_d) * active.float()
