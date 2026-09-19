# -*- coding: utf-8 -*-
"""S10 基础测试（09-11，作者拍板）。在 AGX 上跑，只经控制台 :8089 的接口操作，像操作员一样。
  t1：起立 → 保持 10 s → 趴下
  t2：起立 → 原地 RL（零速度、真实高程图）→ 10 s 后 runner 自己受控趴下
  t3：起立 → RL → 前进 0.3 m/s 3 s → 停 1 s → 原地左转 0.3 rad/s 2 s → 停 → 10 s 时 runner 自己趴下
中止（任何时刻）：发起立 1.2 s 内关节不动；倾角 >20°；runner 日志出现 "!! [STAND_GUARD]" / "!! [RL_TEST]" / 失控保护；
  → 阻尼、停 runner、释放接管。每个测试结束都停 runner、释放接管。
用法：python3 basic_test.py t1|t2|t3
"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

B = "http://127.0.0.1:8089"
LOG = "/home/robot/rl_deploy.log"
T = sys.argv[1] if len(sys.argv) > 1 else "t1"
S0 = os.path.getsize(LOG)
PID = "bt%d" % int(time.time())
SEQ = [0]
REC = {"t0": time.time(), "tilt": 0.0, "tau": 0.0, "wheel": 0.0, "yaw": []}


def get(p, t=5):
    # 09-12：控制台 night3 起只接受操作员的控制指令。每个请求都带本脚本的 PID（/api/state 带上它 = 心跳）
    if "pid=" not in p:
        p += ("&" if "?" in p else "?") + "pid=" + PID
    return json.load(urllib.request.urlopen(B + p, timeout=t))


def op(action):
    """认领 / 交还控制台操作权。旧控制台没有 /api/op/*（404）→ 当作不需要。"""
    try:
        return get("/api/op/" + action)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"ok": True, "msg": "控制台没有操作权机制（旧版），跳过"}
        raise


def st():
    s = get("/api/state")
    return s["state"], s["live"]


def joints(L):
    return {j["n"]: float(j["deg"]) for j in (L.get("joints") or []) if "wheel" not in j["n"]}


def newlog():
    with open(LOG, "rb") as f:
        f.seek(S0)
        return f.read().decode("utf-8", "replace")


def knees(L):
    J = joints(L)
    return [abs(J[l + "_knee"]) for l in ("fl", "fr", "hl", "hr")]


def say(*a):
    print("%5.1fs" % (time.time() - REC["t0"]), *a, flush=True)


def finish(ok, why=""):
    try:
        if not ok:
            am = REC.get("abort_mode", 2)
            say("!! 中止：%s → %s、停 runner、释放接管" % (why, "受控趴下" if am == 4 else "阻尼"))
            get("/api/mode?m=%d" % am)
            time.sleep(4.0 if am == 4 else 1.0)
        get("/api/runner/stop", 20)
        get("/api/arm?on=0")
        op("release")
    except Exception as e:
        say("收尾出错", e)
    lg = newlog()
    alarms = [l.strip()[:150] for l in lg.splitlines() if l.startswith("!!") or "STAND_GUARD" in l or "[RL_TEST]" in l]
    S, L = st()
    print("\n==== %s %s ====" % (T, "通过" if ok else "未通过：" + why))
    print("  峰值：倾角 %.1f°  力矩 %.1f N·m  轮速 %.2f rad/s" % (REC["tilt"], REC["tau"], REC["wheel"]))
    if REC["yaw"]:
        y0 = REC["yaw"][0][1]
        print("  偏航变化：结束时 %+.1f°（相对开始）" % (((REC["yaw"][-1][1] - y0 + 180) % 360) - 180))
    print("  runner 告警/测试行：", alarms[:8] or "无")
    print("  runner 状态切换：", [l.strip()[:40] for l in lg.splitlines() if "------------>" in l][:10])
    print("  [IMU] 统计行：", [l.strip()[:110] for l in lg.splitlines() if "[IMU] 近" in l][:2])
    print("  结束：armed", S["armed"], "runner", S["runner_pid"], "膝", [round(k) for k in knees(L)], "tau", L.get("tau_max"))
    sys.exit(0 if ok else 1)


def watch(L, phase):
    """每次采样都查中止条件。返回 None 或中止原因。"""
    im = L.get("imu") or {}
    r, p = abs(im.get("roll", 0) or 0), abs(im.get("pitch", 0) or 0)
    REC["tilt"] = max(REC["tilt"], r, p)
    REC["tau"] = max(REC["tau"], L.get("tau_max") or 0)
    REC["wheel"] = max(REC["wheel"], max([abs(x) for x in (L.get("wheel_rad_s") or [0])]))
    if im.get("yaw") is not None:
        REC["yaw"].append((time.time(), im["yaw"]))
    if r > 20 or p > 20:
        return "倾角超 20°（横滚 %.1f 俯仰 %.1f，%s）" % (r, p, phase)
    lg = newlog()
    for key in ("!! [STAND_GUARD]", "!! [RL_TEST]", "!! [失控保护]"):
        if key in lg:
            return "runner 防护触发：%s" % [l for l in lg.splitlines() if key in l][0][:120]
    # 09-11 晚（Astra 意见）：0911g 的 RL 停顿只告警，但平地验收里出现停顿就判失败；中止用受控趴下，不用阻尼
    if "!! [RL_STALL]" in lg:
        REC["abort_mode"] = 4
        return "RL 停顿告警（平地验收判失败）：%s" % [l for l in lg.splitlines() if "!! [RL_STALL]" in l][0][:120]
    return None


def cmd(vx=0.0, wz=0.0):
    SEQ[0] += 1
    return get("/api/cmd?vx=%.3f&vy=0&wz=%.3f&pid=%s&seq=%d" % (vx, wz, PID, SEQ[0]))


def mode(m, what):
    d = get("/api/mode?m=%d&dry=1" % m)
    if not (d.get("transition_ok") and d.get("would_publish")):
        finish(False, "%s 演练不通过：%s" % (what, d.get("msg")))
    r = get("/api/mode?m=%d" % m)
    say("发 %s：" % what, r.get("msg"))


def hold(secs, phase, send=None):
    t1 = time.time()
    while time.time() - t1 < secs:
        if send:
            send(time.time() - t1)
        S, L = st()
        why = watch(L, phase)
        if why:
            finish(False, why)
        time.sleep(0.1)


# ---------------------------------------------------------------- 前置
S, L = st()
say("前置：armed", S["armed"], "runner", S["runner_pid"], "c_hz", L.get("c_hz"), "imu", L.get("imu"), "hs_ok", L.get("hs_ok"),
    "膝", [round(k) for k in knees(L)])
if S["runner_pid"] or S["armed"] or (L.get("c_hz") or 0) > 5 or S["estop"]:
    sys.exit("前置不满足：已有 runner / 已接管 / 总线有人发指令 / 急停中")
# 09-11 21:11 教训：控制台留着别人选的 V1H，脚本没查就跑了。策略必须是指定的那个。
WANT = sys.argv[2] if len(sys.argv) > 2 else "V1Down_10695"
if S.get("policy") != WANT:
    sys.exit("前置不满足：控制台选中的策略是 %s，本测试要 %s（先在控制台选好，或第二个参数写明）" % (S.get("policy"), WANT))
if min(knees(L)) < 120:
    sys.exit("狗不在趴姿（膝 %s），不测" % [round(k) for k in knees(L)])
if T in ("t2", "t3") and not L.get("hs_ok"):
    sys.exit("高程图不健康（hs_ok=%s），不进 RL" % L.get("hs_ok"))
r = op("claim")
if not r.get("ok"):
    sys.exit("前置不满足：拿不到控制台操作权（%s）" % r.get("msg"))
say("操作权：", r.get("msg"))
say("接管：", get("/api/arm?on=1"))
say("起 runner：", get("/api/runner/start?yes=1", 20))
t0 = time.time()
while time.time() - t0 < 25 and "[IMU] 近" not in newlog():
    time.sleep(0.5)
if "[IMU] 近" not in newlog():
    finish(False, "25 s 内没等到 runner 的 IMU 统计行")
time.sleep(2.0)

# ---------------------------------------------------------------- 起立
S, L = st()
J0 = joints(L)
mode(1, "起立")
t1, moving = time.time(), False
while time.time() - t1 < 8:
    S, L = st()
    why = watch(L, "起立")
    if why:
        finish(False, why)
    J = joints(L)
    dmax = max(abs(J[k] - J0[k]) for k in J0)
    if not moving and dmax > 3.0:
        moving = True
        say("关节开始动（%.2f s）" % (time.time() - t1))
    if not moving and time.time() - t1 > 1.2:
        finish(False, "发起立 1.2 s 关节没动（电机没接受指令？先重切手柄 SDK 模式）")
    if moving and max(knees(L)) < 60 and time.time() - t1 > 3.5:
        break
    time.sleep(0.1)
S, L = st()
if max(knees(L)) >= 60:
    finish(False, "8 s 内没站起来（膝 %s）" % [round(k) for k in knees(L)])
say("站起来了：膝", [round(k) for k in knees(L)], "hipx", [round(joints(L)[l + "_hipx"], 1) for l in ("fl", "fr", "hl", "hr")],
    "倾角", L.get("imu"), "tau", L.get("tau_max"))

# ---------------------------------------------------------------- 各测试
if T == "t1":
    hold(10, "站立保持")
    say("保持 10 s 完成：倾角峰值 %.1f°，力矩峰值 %.1f" % (REC["tilt"], REC["tau"]))
    mode(4, "趴下")
    t2 = time.time()
    while time.time() - t2 < 10 and min(knees(st()[1])) < 120:
        S, L = st()
        why = watch(L, "趴下")
        if why:
            finish(False, why)
        time.sleep(0.1)
    S, L = st()
    if min(knees(L)) < 120:
        finish(False, "10 s 内没趴下（膝 %s）" % [round(k) for k in knees(L)])
    say("趴下了：膝", [round(k) for k in knees(L)], "tau", L.get("tau_max"))
    finish(True)

mode(6, "策略接管（RL）")
REC["yaw"] = []
if T == "t2":
    plan = None
else:
    def plan(t):
        if 1.0 <= t < 4.0:
            cmd(0.3, 0.0)
        elif 5.0 <= t < 7.0:
            cmd(0.0, 0.3)
        else:
            cmd(0.0, 0.0)
t3 = time.time()
lay = False
sent_lie = False
while time.time() - t3 < 16:
    el = time.time() - t3
    if plan and el < 9.0:
        plan(el)
    S, L = st()
    why = watch(L, "RL")
    if why:
        finish(False, why)
    k = int(el)
    if abs(el - k) < 0.1:
        im = L.get("imu") or {}
        od = L.get("odom") or {}
        say("RL %2ds 横滚%+5.1f 俯仰%+5.1f 偏航%+6.1f | 出指令 %s | 里程计 x %s y %s | tau %s | 轮速 %s"
            % (k, im.get("roll", 0), im.get("pitch", 0), im.get("yaw", 0), L.get("cmd_out"), od.get("x"), od.get("y"),
               L.get("tau_max"), L.get("wheel_rad_s")))
    rl_lie = "[RL_TEST] time limit reached" in newlog()
    if not rl_lie and not sent_lie and el >= 10.0:
        # 0911g 起没有 RL_TEST 钩子：脚本自己在 10 s 发趴下（等价于诊断版 runner 的到点受控趴下）
        mode(4, "趴下（脚本 10 s 到点）")
        sent_lie = True
    if (rl_lie or sent_lie) and min(knees(L)) > 120:
        lay = True
        break
    time.sleep(0.1)
S, L = st()
if not lay:
    finish(False, "16 s 内没趴下（膝 %s）" % [round(k) for k in knees(L)])
say("10 s 到点趴下了（%s）：膝" % ("runner 自己" if not sent_lie else "脚本发的"), [round(k) for k in knees(L)], "tau", L.get("tau_max"))
finish(True)
