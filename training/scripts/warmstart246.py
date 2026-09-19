#!/usr/bin/env python3
"""244 维 checkpoint → 246 维（末尾补两列零），用于 Trot-Phase 从 T10_15197 续训。

与当年 57→244 的 `mkwarmstart.py` 同一做法：**新增维度的权重置零**，
于是第一步的输出与原策略**逐位相同**，相位信息从零开始被学出来，不会一上来就打乱已有步态。

actor  policy 组 244 → 246（mlp.0.weight  (H,244) → (H,246)）
critic critic 组 247 → 249（mlp.0.weight  (H,247) → (H,249)）
两组都是**末尾追加**，与 cfg 里 `observations.*.gait_phase` 的声明位置一致。

用法: warmstart246.py <in.pt> <out.pt>
"""
import sys, torch

if len(sys.argv) != 3:
    sys.exit(__doc__)
src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu", weights_only=False)

for grp, add in (("actor_state_dict", 2), ("critic_state_dict", 2)):
    sd = d[grp]
    W = sd["mlp.0.weight"]
    h, n = W.shape
    sd["mlp.0.weight"] = torch.cat([W, torch.zeros(h, add, dtype=W.dtype)], dim=1)
    print(f"  {grp}: mlp.0.weight ({h},{n}) -> ({h},{n + add})  新增列全零")

# 优化器动量（exp_avg / exp_avg_sq）与新形状对不上。
# **不能整个删掉** —— rsl_rl 的 `ppo.load()` 无条件读 `loaded_dict["optimizer_state_dict"]`，
# 删了会 KeyError（09-18 实测）。做法：保留 `param_groups`（参数个数不变），**清空 `state`**，
# Adam 在下一步自行初始化各参数的动量，形状不会冲突。
_opt = d.get("optimizer_state_dict")
if isinstance(_opt, dict):
    _n = len(_opt.get("state", {}))
    _opt["state"] = {}
    d["optimizer_state_dict"] = _opt
    print(f"  optimizer_state_dict：保留 param_groups，清空 state（原 {_n} 项）—— Adam 重新积累动量")
torch.save(d, dst)
print(f"已写出 {dst}（iter={d.get('iter')}）")
