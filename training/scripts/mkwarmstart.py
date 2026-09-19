#!/usr/bin/env python3
"""用官方 policy.onnx 热启动带感知的新策略。

思路：新策略观测 244 = 原来的 57 + 新增 187 维高程图。
      把 actor 第一层 Linear(244,512) 的**前 57 列**填成官方权重、**后 187 列置零**，
      其余各层原样拷贝 —— 这样新策略开局的行为与官方策略逐位相同（高程图那部分
      乘 0 不产生任何贡献），却已经会走路，省掉几千 iter 的"学站立学行走"。
      零初始化不会锁死：那 187 列的梯度不为零，训练一开始就会长出来。

不手工拼 checkpoint 格式 —— 拿一个训练真实产出的 .pt 当模板，只换 actor 的权重，
这样 rsl_rl 版本怎么变都不会拼错键名。

用法：
    mkwarmstart.py <模板.pt> [输出.pt]
模板可以用点火检查（max_iterations=2）存下来的那个。
"""
import os
import sys

import numpy as np
import torch

P = os.environ.get("S10_PROJ", "/mnt/c/Users/86156/Desktop/机器狗项目")
# 09-08：热启动源可换（作者拍板：只用官方 09-04 版，层名 actor.*；7-25 版层名 mlp.*，脚本两种都认）
ONNX = os.environ.get("S10_WARM_ONNX", f"{P}/s10_ws/src/S10_sdk_deploy/policy/policy.onnx")
OLD_OBS = 57          # 官方盲策略的观测维度

if len(sys.argv) < 2:
    sys.exit(__doc__)
TPL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{P}/s10_dev/train/warmstart_percept.pt"


# ---------- 1. 读官方权重 ----------
import onnx                                  # noqa: E402
from onnx import numpy_helper                # noqa: E402

g = onnx.load(ONNX).graph
W = {i.name: numpy_helper.to_array(i) for i in g.initializer}
# Gemm 全部 transB=1，即权重是 [out, in]，与 torch.nn.Linear.weight 排布一致
layers = []
_prefix = next((pf for pf in ("mlp", "actor") if f"{pf}.0.weight" in W), None)
if _prefix is None:
    sys.exit(f"ONNX 里找不到 mlp.0.weight / actor.0.weight，initializer 名：{sorted(W)[:12]}")
print(f"官方 onnx: {ONNX}\n  层名前缀 {_prefix}.*")
for k in ("0", "2", "4", "6"):
    layers.append((W[f"{_prefix}.{k}.weight"], W[f"{_prefix}.{k}.bias"]))
print("官方 onnx 各层:")
for i, (w, b) in enumerate(layers):
    print(f"  层{i}  W{w.shape}  b{b.shape}")
assert layers[0][0].shape[1] == OLD_OBS, f"第一层输入不是 {OLD_OBS}"


# ---------- 2. 读模板 checkpoint，看新策略长什么样 ----------
ck = torch.load(TPL, map_location="cpu", weights_only=False)
print(f"\n模板 {os.path.relpath(TPL, P)}")
print(f"  顶层键: {list(ck)}")
asd = ck["actor_state_dict"]
print("  actor_state_dict:")
for k, v in asd.items():
    print(f"    {k:<28} {tuple(v.shape)}")

# 找出各层权重的键名（按参数张量的形状配对，不猜命名）
wkeys = [k for k, v in asd.items() if v.ndim == 2]
wkeys.sort(key=lambda k: list(asd).index(k))
if len(wkeys) != len(layers):
    sys.exit(f"层数对不上：模板 {len(wkeys)} 个权重矩阵，onnx {len(layers)} 个。"
             f"网络结构改过了？模板键：{wkeys}")

new_obs = asd[wkeys[0]].shape[1]
print(f"\n新策略观测维度 {new_obs}（官方 {OLD_OBS}，新增 {new_obs - OLD_OBS}）")
if new_obs < OLD_OBS:
    sys.exit("新观测比旧的还小，不该发生")

for i, (wk, (w, b)) in enumerate(zip(wkeys, layers)):
    tgt = asd[wk].shape
    src = w.shape
    ok = (tgt[0] == src[0]) and (tgt[1] == src[1] if i else True)
    print(f"  层{i}  模板{tuple(tgt)}  官方{src}  {'✓' if ok else '✗ 形状不匹配'}")
    if not ok:
        sys.exit("除第一层输入维度外，其余形状必须完全一致，否则不能热启动")


# ---------- 3. 拼新权重 ----------
bkeys = [k for k, v in asd.items() if v.ndim == 1 and "std" not in k.lower()]
bkeys.sort(key=lambda k: list(asd).index(k))
if len(bkeys) != len(layers):
    sys.exit(f"偏置数对不上：模板 {bkeys}")

for i, (wk, bk, (w, b)) in enumerate(zip(wkeys, bkeys, layers)):
    if i == 0:
        nw = np.zeros((w.shape[0], new_obs), dtype=np.float32)
        nw[:, :OLD_OBS] = w                        # 前 57 列 = 官方；其余保持 0
        asd[wk] = torch.from_numpy(nw)
    else:
        asd[wk] = torch.from_numpy(w.copy())
    asd[bk] = torch.from_numpy(b.copy())
ck["actor_state_dict"] = asd
ck["iter"] = 0                                     # 从头计数，别继承模板的迭代数


# ---------- 4. 验证：新策略在高程图=0 时，输出必须与官方逐位一致 ----------
def forward_np(obs, ls, dt=np.float32):
    """手写前向，不依赖 rsl_rl —— 独立于被验证的那套代码。"""
    x = obs.astype(dt)
    for i, (w, b) in enumerate(ls):
        x = x @ w.T.astype(dt) + b.astype(dt)
        if i < len(ls) - 1:                        # 最后一层没有激活
            x = np.where(x > 0, x, np.expm1(np.clip(x, -50, 0)))   # ELU
    return x


new_layers = [(asd[wk].numpy(), asd[bk].numpy()) for wk, bk in zip(wkeys, bkeys)]

# 判据一（决定性的）：新增那些列必须逐位为 0。
# 这一条是精确的整数级判断，不受浮点影响。
tail = new_layers[0][0][:, OLD_OBS:]
n_nonzero = int(np.count_nonzero(tail))
print(f"\n验证")
print(f"  a) 新增 {new_obs - OLD_OBS} 列里非零元素个数: {n_nonzero}   ← 必须是 0")
if n_nonzero:
    sys.exit("✗ 新增列没有真正置零")

# 判据二：数值等价。float32 下会有求和顺序造成的误差（244 项 vs 57 项的
# pairwise 求和分组树不同），所以同时用 float64 做对照 —— 若 float64 下
# 误差降到 1e-12 量级，就证明 float32 那点差异纯粹是累加噪声，不是权重错了。
rng = np.random.default_rng(0)
res = {}
for dt, tag in ((np.float32, "float32"), (np.float64, "float64")):
    w0 = w1 = 0.0
    for _ in range(200):
        o57 = rng.normal(0, 1.5, OLD_OBS)
        ref = forward_np(o57, layers, dt)
        o_zero = np.concatenate([o57, np.zeros(new_obs - OLD_OBS)])
        o_ex = np.concatenate([o57, rng.uniform(-20, 20, new_obs - OLD_OBS)])
        w0 = max(w0, float(np.abs(forward_np(o_zero, new_layers, dt) - ref).max()))
        w1 = max(w1, float(np.abs(forward_np(o_ex, new_layers, dt) - ref).max()))
    res[tag] = (w0, w1)
    print(f"  b) {tag}: 高程图=0 差异 {w0:.3e}   高程图=±20 差异 {w1:.3e}")

f32_0, f32_ex = res["float32"]
f64_0, f64_ex = res["float64"]
# ±20 与 0 的差异必须相同 —— 不同就说明那些列真的参与了运算
if abs(f32_ex - f32_0) > 1e-9:
    sys.exit("✗ 灌极端值改变了输出，新增列并未被真正忽略")
if f64_0 > 1e-10:
    sys.exit(f"✗ float64 下仍有 {f64_0:.3e} 的差异，不是累加噪声，权重拼错了")
print(f"  → float64 下降到 {f64_0:.1e}，证明 float32 那 {f32_0:.1e} 是求和顺序噪声，权重是对的")

# 顺带确认新增的 187 列不是"死"的：给它们一个梯度看看
wt = torch.tensor(new_layers[0][0], requires_grad=True)
o = torch.randn(64, new_obs)
loss = torch.nn.functional.elu(o @ wt.T).pow(2).mean()
loss.backward()
gn = float(wt.grad[:, OLD_OBS:].norm())
print(f"  新增 {new_obs - OLD_OBS} 列的梯度范数: {gn:.1f}   ← 非零说明训练时会长出来")
if gn < 1e-6:
    sys.exit("✗ 新增列梯度为零，会永远学不到东西")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(ck, OUT)
print(f"\n✓ 写出 {os.path.relpath(OUT, P)}")
print(f"  用法: train.py --task Percept-Deeprobotics-M20-v0 --resume True "
      f"--load_run <目录> --checkpoint {os.path.basename(OUT)}")
