#!/usr/bin/env python3
"""把带感知的 rsl_rl checkpoint 导成部署用的 **246** 维 ONNX，并当场验证。

= `export244.py` 只改 `OBS_DIM` 244→246（09-18，作者拍板"直接上乙"）。
246 = 57 + 187 高程图 + **2 维步态相位钟 sin/cos（末尾追加）**。
前 244 维排布与 244 版**逐位相同** —— 部署侧 runner 只需在末尾补两维。
runner 里是三个独立 `S10PolicyRunner` 实例、各自从自己 ONNX 读 `observation_dim`，
故 **246 维踏步策略可与 244 维爬台/楼梯专家同时装**，另两条线不用重训。

为什么现在就导（策略才练到 0.225m，翻不过 wp15 那 0.377m）：
验的不是策略，是**管道**。导出、维度、输入输出名、C++ 侧订阅与兜底 ——
这条链从头到尾一次都没通跑过。这些 bug 现在发现和 8.17 发现是两回事。

约束（都是从代码里读出来的，不是记忆）：
  输入名 obs / 输出名 actions   s10_policy_runner.hpp:153-154 写死
  observation_dim 244            = 57 + 187，运行时从 ONNX 首输入形状读
  empirical_normalization=False  rsl_rl_ppo_cfg.py:17 —— 部署侧不用复刻归一化
  [512,256,128] + ELU            同上 :22-24
  权重必须内嵌                    Ort::Session 只吃单文件，外置权重会静默出错

导完立刻对拍 ONNX vs torch。导出静默改变数值是这类事故的常见形态，
而策略输出错了不会报错，只会表现为"机器狗变笨了"。
"""
import re
import sys

import numpy as np
import torch
import torch.nn as nn

CKPT = sys.argv[1] if len(sys.argv) > 1 else None
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/s10_logs/policy_percept.onnx"
OBS_DIM, ACT_DIM = 246, 16
HIDDEN = [512, 256, 128]

if not CKPT:
    sys.exit("用法: export244.py <model_xxx.pt> [out.onnx]")

sd_all = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"checkpoint: {CKPT}")
print(f"顶层键: {sorted(sd_all.keys())}")
print(f"iter: {sd_all.get('iter')}")

# rsl_rl 版本之间布局不一样：老版是单个 model_state_dict 里 actor.0.weight，
# 新版把 actor/critic 拆成两个 dict、层名是 0.weight。两种都认。
if "actor_state_dict" in sd_all:
    sd = sd_all["actor_state_dict"]
    prefix = ""
elif "model_state_dict" in sd_all:
    sd = sd_all["model_state_dict"]
    prefix = "actor."
else:
    sys.exit(f"!! 不认识的 checkpoint 布局: {sorted(sd_all.keys())}")

# 先打印再筛选 —— 上一版把打印放在排序之后，一报错反而看不到键名
print(f"\nactor 侧全部键: {sorted(sd.keys())}")

# 只要 Linear 的权重：这里还可能有 std / log_std 之类
cand = [k for k in sd
        if k.startswith(prefix) and k.endswith(".weight") and sd[k].ndim == 2]


def layer_no(k: str) -> int:
    """层号在键里的位置随版本变（actor.0.weight / mlp.0.weight），
    取里面最后一个整数即可。不能用字典序 —— 10 会排到 2 前面。"""
    nums = re.findall(r"\d+", k)
    return int(nums[-1]) if nums else -1


actor_keys = sorted(cand, key=layer_no)
print(f"取作 Linear 层的: {actor_keys}")
for k in actor_keys:
    print(f"    {k:<22}{tuple(sd[k].shape)}")
if not actor_keys:
    sys.exit("!! 没找到二维权重，布局不对")

in_dim = sd[actor_keys[0]].shape[1]
out_dim = sd[actor_keys[-1]].shape[0]
print(f"\n实际输入维度 {in_dim}（应为 {OBS_DIM}）  输出维度 {out_dim}（应为 {ACT_DIM}）")
if in_dim != OBS_DIM or out_dim != ACT_DIM:
    sys.exit(f"!! 维度不对，别导。in={in_dim} out={out_dim}")

# 按 rsl_rl 的顺序重建：Linear, ELU, Linear, ELU, Linear, ELU, Linear
layers = []
dims = [OBS_DIM] + HIDDEN + [ACT_DIM]
for i in range(len(dims) - 1):
    layers.append(nn.Linear(dims[i], dims[i + 1]))
    if i < len(dims) - 2:
        layers.append(nn.ELU())
actor = nn.Sequential(*layers).eval()

# 按 state_dict 里的层号搬权重（不假设编号连续）
idx_in_seq = [i for i, m in enumerate(actor) if isinstance(m, nn.Linear)]
assert len(idx_in_seq) == len(actor_keys), \
    f"层数对不上：重建 {len(idx_in_seq)} 层，checkpoint {len(actor_keys)} 层"
with torch.no_grad():
    for seq_i, k in zip(idx_in_seq, actor_keys):
        w = sd[k]
        b = sd[k[:-len("weight")] + "bias"]
        assert actor[seq_i].weight.shape == w.shape, \
            f"{k} 形状不符 {tuple(actor[seq_i].weight.shape)} vs {tuple(w.shape)}"
        actor[seq_i].weight.copy_(w)
        actor[seq_i].bias.copy_(b)
print("权重搬运完成（逐层校验过形状）")

torch.onnx.export(
    actor, torch.zeros(1, OBS_DIM), OUT,
    input_names=["obs"], output_names=["actions"],
    dynamic_axes=None, opset_version=13,
    dynamo=False,   # 09-08 新机：torch 2.11 默认 dynamo 导出会把权重外置成 .onnx.data，部署要单文件，走旧导出器
)
print(f"\n已导出 {OUT}")

# ---------------------------------------------------------------- 验证
import onnx                       # noqa: E402
import onnxruntime as ort         # noqa: E402

m = onnx.load(OUT, load_external_data=False)
ext = [t.name for t in m.graph.initializer
       if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL]
n_param = sum(int(np.prod(t.dims)) for t in m.graph.initializer)
print(f"权重内嵌: {'否 !! ' + str(ext) if ext else '是'}   参数量 {n_param:,}")

sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
i0 = sess.get_inputs()[0]
o0 = sess.get_outputs()[0]
print(f"ONNX 输入 name={i0.name!r} shape={i0.shape}   "
      f"输出 name={o0.name!r} shape={o0.shape}")
ok_name = (i0.name == "obs" and o0.name == "actions")
ok_shape = (list(i0.shape) == [1, OBS_DIM] and list(o0.shape) == [1, ACT_DIM])
print(f"  名字与 C++ 侧一致: {'是' if ok_name else '否 !!'}   "
      f"形状对: {'是' if ok_shape else '否 !!'}")

rng = np.random.default_rng(0)
worst = 0.0
for trial in range(200):
    if trial < 100:
        x = rng.normal(0, 1, (1, OBS_DIM)).astype(np.float32)
    else:
        # 也喂极端值：高程图那 187 维在真机上会出现 ±1 的钳位值和兜底值
        x = rng.normal(0, 1, (1, OBS_DIM)).astype(np.float32)
        x[0, 57:57 + 187] = rng.choice([-1.0, 1.0, -0.10, 0.0], 187).astype(np.float32)
        # 末尾 2 维是相位钟 sin/cos —— 真机上永远落在单位圆上，按真实分布喂
        _a = rng.uniform(0.0, 2.0 * np.pi)
        x[0, 244:246] = np.array([np.sin(_a), np.cos(_a)], dtype=np.float32)
    y_ort = sess.run(["actions"], {"obs": x})[0]
    with torch.no_grad():
        y_pt = actor(torch.from_numpy(x)).numpy()
    worst = max(worst, float(np.max(np.abs(y_ort - y_pt))))
print(f"\nONNX vs torch 最大绝对差（200 组，含极端输入）: {worst:.3e}")

# 阈值 1e-3 是量出来的，不是拍的。onnxdiff.py 用 float64 当裁判测过：
#   ONNX  vs fp64 真值   中位 1.5e-05  最大 1.1e-04
#   torch vs fp64 真值   中位 8.1e-06  最大 3.9e-05
# 比值 1.86 —— 同量级，ONNX 只是 GEMM 分块方式不同；误差均匀散在 16 个
# 关节维上（最大/最小 3.5，不是堆在某一两维），换成物理量是 0.0015°。
# 一开始我卡在 1e-5，那对四层 fp32 MLP 本来就太紧。
# 但别放宽到 1e-2：结构性错误（层序错、ELU alpha 不对、某层没搬）
# 会给出 1e-1 量级且集中在特定维的误差，这个阈值要能拦住它。
TOL = 1e-3

# 再看一眼输出量级是否合理 —— 全零观测下动作不该爆
with torch.no_grad():
    y0 = actor(torch.zeros(1, OBS_DIM)).numpy()
print(f"全零观测的动作范围: {y0.min():+.3f} ~ {y0.max():+.3f}")

good = ok_name and ok_shape and not ext and worst < TOL
print("\n" + ("== 导出可用 ==" if good else "== 有问题，别拿去部署 =="))
sys.exit(0 if good else 1)
