# -*- coding: utf-8 -*-
"""离线测试控制台候选 night4f（… + F 段前方地形判读）。原说明：night3（假 rclpy：不连 AGX、不碰 ROS、不发任何真话题）。用法：python3 test_night3.py <s10_ctrl.night3.py>
覆盖 Astra 09-12 凌晨建议：
  H：① 切专家前重查高程图（缺失 / 过期 / 非法 / 身下覆盖不足都不切；途中变差只报警）；② 空洞率 /hmap_info 过期 = 未知，不是 0。
  O：③ 唯一操作员：旁观页面切走（失焦发零）、刷新（新 pid）、关闭，以及旧页面迟到的指令，都不会撤专家、解锁、改速度。"""
import importlib.abc, importlib.machinery, importlib.util, json, math, os, sys, threading, time, types
PATH = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nightreview", "hmap-v2", "agx_src"))   # AGX 同款 heightmap.py（格心 CELL_XY）
FAKE_ROOTS = ("rclpy", "std_msgs", "geometry_msgs", "nav_msgs", "sensor_msgs", "drdds", "rcl_interfaces",
              "builtin_interfaces", "rosidl_runtime_py")
class Msg:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        v = Msg(); object.__setattr__(self, n, v); return v
def _fake_module(name):
    m = types.ModuleType(name); m.__path__ = []
    m.__getattr__ = lambda attr: Msg if attr[:1].isupper() else (lambda *a, **k: None)
    return m
class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        return importlib.machinery.ModuleSpec(name, self, is_package=True) if name.split(".")[0] in FAKE_ROOTS else None
    def create_module(self, spec): return _fake_module(spec.name)
    def exec_module(self, m): pass
sys.meta_path.insert(0, Finder())
spec = importlib.util.spec_from_file_location("ctrl", PATH); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
assert C._HS_X is not None, "没取到格心定义（heightmap.CELL_XY）"
OUT, OLK = [], threading.Lock()
def num(x): return float(x) if isinstance(x, (int, float)) else 0.0
class Rec:
    def __init__(self, topic): self.topic = topic
    def publish(self, m):
        with OLK:
            if self.topic == "/cmd_vel": OUT.append(("cmd", round(num(m.linear.x), 3), round(num(m.angular.z), 3)))
            elif self.topic in ("/climb_mode", "/robot_mode"): OUT.append((self.topic[1:], int(m.data)))
            else: OUT.append((self.topic,))
SUBS = {}
class Node:
    def create_publisher(self, T, topic, qos): return Rec(topic)
    def create_subscription(self, T, topic, cb, qos, **k): SUBS[topic] = cb; return None
    def get_node_names_and_namespaces(self): return [("rl_deploy", "/"), ("s10_user_command", "/")]
RUN = {"on": True}
C.rclpy.init = lambda *a, **k: None
C.rclpy.create_node = lambda name: Node()
C.rclpy.ok = lambda: RUN["on"]
C.rclpy.spin_once = lambda n, timeout_sec=0.0: time.sleep(min(timeout_sec, 0.005)) if timeout_sec else None
C.rclpy.shutdown = lambda: None
C.QoSProfile = lambda **k: None
C.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1, RELIABLE=2)
C.DurabilityPolicy = types.SimpleNamespace(VOLATILE=1)
C.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
C.Twist = C.UInt8 = C.String = C.Float32MultiArray = Msg
C.rec_status = lambda force=False: {"running": False}
TMP = os.path.join(HERE, "fake_rl_deploy.log"); open(TMP, "w").close(); C.RLOG_PATH = TMP


def http(path, ip="127.0.0.9"):
    h = C.H.__new__(C.H); h.path = path; h.client_address = (ip, 0); res = {}
    h._s = lambda code, body, ct=None: res.update(code=code, body=body)
    h.do_GET()
    try:
        return json.loads(res["body"])
    except Exception:
        return {"_code": res.get("code")}


# ---- 假 hmap：50 Hz 发 /height_scan（补后）、/height_scan_raw（补前）、/hmap_info
X = C._HS_X
UB = [i for i in range(187) if abs(X[i]) <= 0.30 + 1e-6]
def raw_with_under(frac):
    v = [-0.08] * 187
    k = int(round(frac * len(UB)))
    for j, i in enumerate(UB):
        if j >= k:
            v[i] = 0.0
    return v
class FM:
    def __init__(self, d): self.data = d
GOOD_INFO = [0.30, 0.9, 100.0, 0.05, 0.0]
FEED = {"hs": [-0.08] * 187, "raw": raw_with_under(0.6), "info": list(GOOD_INFO), "hs_on": True, "raw_on": True, "info_on": True}
def good():
    FEED.update(hs=[-0.08] * 187, raw=raw_with_under(0.6), info=list(GOOD_INFO), hs_on=True, raw_on=True, info_on=True)
def feeder():
    while RUN["on"]:
        if "/height_scan_raw" in SUBS:
            if FEED["hs_on"]: SUBS["/height_scan"](FM(list(FEED["hs"])))
            if FEED["raw_on"]: SUBS["/height_scan_raw"](FM(list(FEED["raw"])))
            if FEED["info_on"]: SUBS["/hmap_info"](FM(list(FEED["info"])))
        time.sleep(0.02)

# ---- 页面：操作员页面每 150 ms 发 /api/cmd（按住 W）；各页面每 0.7 s 轮询 /api/state（带 pid = 心跳）
OPS = {"on": False, "seq": 0, "vx": 0.5, "wz": 0.0, "pid": "PAGEA"}
def op_page():
    while RUN["on"]:
        if OPS["on"]:
            OPS["seq"] += 1
            http("/api/cmd?vx=%.3f&vy=0&wz=%.3f&pid=%s&seq=%d" % (OPS["vx"], OPS.get("wz", 0.0), OPS["pid"], OPS["seq"]))
        time.sleep(0.15)
HB = {}
def heart():
    while RUN["on"]:
        for p, on in list(HB.items()):
            if on: http("/api/state?pid=" + p)
        time.sleep(0.7)
YAW = {"v": 0.0}
def imu():
    while RUN["on"]:
        C.IMU_NOW.update(yaw=YAW["v"], t=time.time()); time.sleep(0.05)
for f in (C.ros_thread, feeder, op_page, heart, imu, C.rlog_loop):
    threading.Thread(target=f, daemon=True).start()
while "climb" not in C.PUB or "/height_scan_raw" not in SUBS: time.sleep(0.01)
time.sleep(1.2)

def base():
    C.PROC["p"] = None
    C.STATE.update(armed=True, estop=False, runner_pid=os.getpid(), running_group="0.42", running_policy="V1Down_10695",
                   running_climb=C.EXPERT, running_pit=C.STAIR, pit_loaded=True, max_vx=0.5, max_wz=0.5, climb=0,
                   climb_t0=None, climb_cap=None, stair_yaw_ref=None, stair_k=2.0, mode=6, mode_src="runner 确认",
                   stair_heading=None, hscan=False)
def try_enter(pid):
    OPS.update(on=True, vx=0.5, wz=0.0, pid=pid); time.sleep(0.8)
    return http("/api/climb?on=2&src=test&pid=%s&seq=%d" % (pid, OPS["seq"] + 1))
def release():
    OPS["on"] = False; time.sleep(0.9)
def mark():
    with OLK: return len(OUT)
def since(m):
    with OLK: return list(OUT[m:])
def logs(): return list(C.STATE["log"])
FAIL = []
def check(name, cond, info=""):
    print(("通过 " if cond else "失败 ") + name + ("" if cond else "  ← %r" % (info,)), flush=True)
    if not cond: FAIL.append(name)

# ======================================================================= 先认领操作员（后面 H 段要用）
HB["PAGEA"] = True
r = http("/api/op/claim?pid=PAGEA")
check("O0 页面 A 认领操作权", r.get("ok"), r)

# ======================================================================= H：高程图质量
print("---- H：切专家前重查高程图 / 空洞率未知 ----")
base(); good(); time.sleep(0.8)
q = C.hmap_quality()
check("H1 数据齐全、新鲜：质量 ok", q == ("ok", ""), q)
ok, why = C.stair_gate(); check("H1 楼梯闸门放行", ok, why)
check("H1 页面数据：空洞 0.30、身下有效(含邻域) 0.60、hs_ok True",
      C.LIVE["hs"]["hole"] == 0.3 and abs(C.LIVE["hs"]["under_valid"] - 0.6) < 0.02 and C.LIVE["hs_ok"] is True, C.LIVE.get("hs"))

FEED.update(hs_on=False, raw_on=False, info_on=False); time.sleep(1.2); base()
ok, why = C.stair_gate()
check("H2 【Astra 复现 1】高程图完全缺失、其余条件满足：楼梯闸门不放行（night2 这里返回 True）", (not ok) and "质量未知" in why, why)
ok2, why2 = C.climb_gate()
check("H2 ClimbH 闸门同样不放行", (not ok2) and "质量未知" in why2, why2)
g = C.hscan_gate()
check("H2 起 runner / 进 RL 的 hscan_gate 语义不变：没有高程图时放行（runner 按平地兜底）", g == (True, ""), g)
good(); time.sleep(0.8); base()
check("H3 恢复后楼梯闸门放行", C.stair_gate()[0], C.stair_gate())

FEED["info"] = [0.85, 0.2, 100.0, 0.05, 0.0]; time.sleep(0.8)
check("H4 【Astra 复现 2】原始空洞率 85%、/hmap_info 新鲜：显示 85%、hs_ok False",
      C.LIVE["hs"]["hole"] == 0.85 and C.LIVE["hs_ok"] is False, C.LIVE.get("hs"))
g = C.hscan_gate(); check("H4 hscan_gate 拦下（85%）", (not g[0]) and "85%" in g[1], g)
ok, why = C.stair_gate(); check("H4 楼梯闸门拦下（不可信）", (not ok) and "不可信" in why, why)
FEED["info_on"] = False; time.sleep(1.0)
hr = C._hole_rate(C.hs_metrics(list(FEED["hs"])))
check("H5 【Astra 复现 2】/hmap_info 过期后：空洞率 = 未知（None），不是 0%（night2 这里返回 0）", hr[0] is None and "没更新" in hr[1], hr)
check("H5 页面数据 hole=None、hs_ok False", C.LIVE["hs"]["hole"] is None and C.LIVE["hs_ok"] is False, C.LIVE.get("hs"))
g = C.hscan_gate(); check("H5 hscan_gate 以「空洞率未知」拦下", (not g[0]) and "未知" in g[1], g)
ok, why = C.stair_gate(); check("H5 楼梯闸门以「质量未知」拦下", (not ok) and "质量未知" in why, why)
for bad, what in (([float("nan"), 0.9, 1.0, 0.05, 0.0], "NaN"), ([1.5, 0.9, 1.0, 0.05, 0.0], "1.5"), ([-0.1, 0.9, 1.0, 0.05, 0.0], "-0.1")):
    FEED.update(info=bad, info_on=True); time.sleep(0.4)
    hr = C._hole_rate(C.hs_metrics(list(FEED["hs"])))
    check("H6 /hmap_info 空洞率非法（%s）→ 未知" % what, hr[0] is None and "非法" in hr[1], hr)
FEED["info"] = [0.3]; time.sleep(0.4)
q = C.hmap_quality(); check("H6 /hmap_info 缺点云年龄字段 → 质量未知", q[0] == "unknown" and "点云年龄" in q[1], q)
FEED["info"] = [0.3, 0.9, 100.0, 0.40, 0.0]; time.sleep(0.4)
q = C.hmap_quality(); check("H7 点云年龄 0.40 s → 不可信", q[0] == "bad" and "点云年龄" in q[1], q)
good(); FEED["raw"] = raw_with_under(0.2); time.sleep(0.4)
q = C.hmap_quality(); check("H8 机身下方 77 格补洞前有效 20%（下限 30%）→ 不可信", q[0] == "bad" and "77 格" in q[1] and "含邻域补值" in q[1], q)
FEED["raw"] = raw_with_under(0.35); time.sleep(0.4)
check("H8 有效 35% → ok", C.hmap_quality()[0] == "ok", C.hmap_quality())
FEED["raw_on"] = False; time.sleep(0.8)
q = C.hmap_quality(); check("H8 /height_scan_raw 停了 → 身下有效率未知 → 质量未知", q[0] == "unknown" and "机身下方有效率未知" in q[1], q)
good(); FEED["hs_on"] = False; time.sleep(0.5)
q = C.hmap_quality(); check("H9 /height_scan 停了、/hmap_info 还在 → 质量未知", q[0] == "unknown" and "/height_scan 已" in q[1], q)
good(); time.sleep(0.4); C.STATE["hscan"] = True
q = C.hmap_quality(); check("H9 我方在发零填充 → 不可信", q[0] == "bad", q)
C.STATE["hscan"] = False

good(); time.sleep(0.8); base()
r = try_enter("PAGEA")
check("H10 地图正常时切入楼梯专家", r.get("ok") and C.STATE["climb"] == 2, r)
m = mark(); FEED["info_on"] = False; time.sleep(1.6)
check("H10 专家途中 /hmap_info 断了：仍在楼梯专家（不自动动作）", C.STATE["climb"] == 2, C.STATE["climb"])
check("H10 途中没有发 /climb_mode 0", ("climb_mode", 0) not in since(m), since(m)[-5:])
check("H10 途中记了一条「!! 楼梯专家途中高程图质量未知」", any("楼梯专家途中高程图质量未知" in l for l in logs()), logs()[:3])
FEED["info_on"] = True; time.sleep(1.2)
check("H10 恢复后记「专家途中高程图恢复正常」", any("专家途中高程图恢复正常" in l for l in logs()), logs()[:3])
release()
check("H10 松键后照旧退出专家（原有停刷语义）", C.STATE["climb"] == 0, C.STATE["climb"])

# ======================================================================= O：唯一操作员
print("---- O：唯一操作员 ----")
HB["PAGEB"] = True
r = http("/api/op/claim?pid=PAGEB")
check("O1 A 在线时 B 认领被拒", (not r.get("ok")) and "已有操作员" in r.get("msg", ""), r)
sa, sb = http("/api/state?pid=PAGEA")["op"], http("/api/state?pid=PAGEB")["op"]
check("O1 /api/state：A 是操作员、B 不是；不泄露操作员 pid", sa["you"] and (not sb["you"]) and sb["held"] and "PAGEA" not in json.dumps(sb), (sa, sb))

good(); time.sleep(0.5); base()
r = try_enter("PAGEA")
check("O2 A 切入楼梯专家", r.get("ok") and C.STATE["climb"] == 2, r)
st0 = dict(armed=C.STATE["armed"], max_vx=C.STATE["max_vx"], max_wz=C.STATE["max_wz"], estop=C.STATE["estop"],
           stair_k=C.STATE["stair_k"], runner_pid=C.STATE["runner_pid"], hscan=C.STATE["hscan"], policy=C.STATE["policy"],
           mode=C.STATE["mode"], stair_heading=C.STATE["stair_heading"])
m = mark()
SPECT = ["/api/cmd?vx=0.000&vy=0.000&wz=0.000&pid=PAGEB&seq=1",      # B 切走标签页（失焦）发的零
         "/api/cmd?vx=0.000&vy=0.000&wz=0.000&pid=PAGEB&seq=2",
         "/api/climb?on=0&src=%E9%94%AE%E7%9B%98X&pid=PAGEB&seq=3",     # B 按 X
         "/api/climb?on=2&src=x&pid=PAGEB&seq=4",
         "/api/arm?on=0&pid=PAGEB", "/api/arm?on=0&dry=1&pid=PAGEB",   # dry=1 不能拿来绕过
         "/api/lim?vx=0.1&wz=0.1&pid=PAGEB", "/api/estop?pid=PAGEB", "/api/mode?m=4&pid=PAGEB",
         "/api/key?k=G20_KEY_R2&pid=PAGEB", "/api/policy?p=V1Down_10695&pid=PAGEB", "/api/stairk?k=0&pid=PAGEB",
         "/api/stairhead?clear=1&pid=PAGEB", "/api/runner/stop?pid=PAGEB", "/api/clear?pid=PAGEB",
         "/api/setmode?m=0&pid=PAGEB", "/api/hscan?on=1&pid=PAGEB", "/api/rec/start?name=x1&pid=PAGEB",
         "/api/rec/stop?pid=PAGEB", "/api/arm?on=0", "/api/cmd?vx=0&vy=0&wz=0"]   # 最后两条不带 pid
res = [(u, http(u)) for u in SPECT]
bad = [(u, r) for u, r in res if not r.get("readonly")]
check("O2 旁观页面 B 的 %d 种请求（含失焦发零、按 X、释放接管、改档位、急停、dry 绕过、不带 pid）全部只读" % len(SPECT), not bad, bad[:3])
time.sleep(0.6)
st1 = dict(armed=C.STATE["armed"], max_vx=C.STATE["max_vx"], max_wz=C.STATE["max_wz"], estop=C.STATE["estop"],
           stair_k=C.STATE["stair_k"], runner_pid=C.STATE["runner_pid"], hscan=C.STATE["hscan"], policy=C.STATE["policy"],
           mode=C.STATE["mode"], stair_heading=C.STATE["stair_heading"])
check("O2 楼梯专家没被撤（climb 仍是 2）", C.STATE["climb"] == 2, C.STATE["climb"])
check("O2 接管 / 档位 / 急停 / 增益 / runner / 零填充 / 策略 / 状态 / 楼梯朝向都没变", st0 == st1, (st0, st1))
ev = since(m)
check("O2 这期间没发 /climb_mode 0、/robot_mode、/GAMEPAD_KEY", not [e for e in ev if e[0] in ("climb_mode", "robot_mode", "/GAMEPAD_KEY")],
      [e for e in ev if e[0] != "cmd"][:5])
cm = [e for e in ev if e[0] == "cmd"]
check("O2 /cmd_vel 一直是 A 的 0.5（没被 B 的零顶掉）", len(cm) >= 10 and all(e[1] >= 0.49 for e in cm), cm[:3] + cm[-3:])
r = http("/api/mode?m=4&dry=1&pid=PAGEB")
check("O2 旁观页面可以用 /api/mode 的 dry=1 演练（不发布）", r.get("dry") and ("robot_mode", 4) not in since(m), r)

HB["PAGEC"] = True
r = http("/api/op/claim?pid=PAGEC")
check("O3 B 刷新成新页面 C：A 在线时认领被拒", (not r.get("ok")) and "已有操作员" in r.get("msg", ""), r)
r = http("/api/climb?on=0&src=x&pid=PAGEC&seq=1")
check("O3 C 的指令同样只读，专家仍在", r.get("readonly") and C.STATE["climb"] == 2, (r, C.STATE["climb"]))

OPS["on"] = False; HB["PAGEA"] = False     # A 关闭
time.sleep(1.0)
check("O4 A 关闭后：原有停刷看门狗照旧撤专家（这是已批准的行为，不是新加的）", C.STATE["climb"] == 0 and any("停刷" in l for l in logs()[:6]), logs()[:4])
time.sleep(2.6)
r = http("/api/op/claim?pid=PAGEC", ip="10.18.9.9")
check("O4 A 心跳超过 3 s 后 C 认领成功", r.get("ok") and C.OP["pid"] == "PAGEC" and C.OP["who"] == "10.18.9.9", (r, C.OP))
t_before = C.TGT["t"]; vx_before = C.TGT["vx"]; m = mark()
late = [http("/api/cmd?vx=0.500&vy=0&wz=0&pid=PAGEA&seq=99999"),
        http("/api/climb?on=2&src=late&pid=PAGEA&seq=100000"),
        http("/api/arm?on=0&pid=PAGEA"), http("/api/lim?vx=1.0&wz=1.0&pid=PAGEA")]
time.sleep(0.3)
check("O4 A 迟到的指令（前进 / 切楼梯 / 释放接管 / 改档位）全部只读", all(x.get("readonly") for x in late), late)
check("O4 迟到指令没改目标速度、没切专家、没释放接管、没改档位",
      C.TGT["t"] == t_before and C.TGT["vx"] == vx_before and C.STATE["climb"] == 0 and C.STATE["armed"] and C.STATE["max_vx"] == 0.5,
      (C.TGT, C.STATE["climb"], C.STATE["armed"], C.STATE["max_vx"]))
check("O4 这期间没发 /climb_mode 2", ("climb_mode", 2) not in since(m), since(m)[-4:])
check("O4 日志记下了操作权转移和来源", any("操作权 → 页面 PAGE" in l and "10.18.9.9" in l for l in logs()), logs()[:3])

hb0 = C.OP["hb"]; time.sleep(0.05)
http("/api/state"); http("/api/state?pid=PAGEX")
check("O5 不带 pid / 别的 pid 轮询不算操作员心跳", C.OP["hb"] == hb0, (hb0, C.OP["hb"]))
r = http("/api/op/release?pid=PAGEB")
check("O5 非操作员不能交出", (not r.get("ok")) and C.OP["pid"] == "PAGEC", r)
r = http("/api/op/release?pid=PAGEC")
check("O5 C 交出操作权", r.get("ok") and C.OP["pid"] == "", r)
r = http("/api/lim?vx=0.3&wz=0.3&pid=PAGEC")
check("O5 交出后 C 的指令也只读，提示「现在没有操作员」", r.get("readonly") and "没有操作员" in r.get("msg", "") and C.STATE["max_vx"] == 0.5, r)
r = http("/api/op/claim?pid=PAGEB")
check("O5 没有操作员时 B 可以认领", r.get("ok") and C.OP["pid"] == "PAGEB", r)
for bad_pid in ("", "ab", "x" * 40, "a<b>"):
    r = http("/api/op/claim?pid=" + bad_pid)
    check("O6 非法 pid %r 认领被拒" % bad_pid, not r.get("ok"), r)
check("O6 只读请求计数在涨（/api/state 里可见）", http("/api/state")["op"]["refused"] >= len(SPECT), http("/api/state")["op"])

# ======================================================================= U：night4b 楼梯赛段
print("---- U：楼梯赛段（按住 W 自动切专家、松 W 交回主策略、S 倒退、切不进不前进、退出赛段）----")
HB["PAGEA"] = True; OPS["pid"] = "PAGEA"; OPS["on"] = False; time.sleep(1.0)
http("/api/op/release?pid=PAGEB"); r = http("/api/op/claim?pid=PAGEA"); check("U0 页面 A 认领操作权", r.get("ok"), r)
good(); time.sleep(0.6); base(); C.STATE["segment"] = "flat"; C.SEG.update(t_exit=0.0, t_try=0.0, t_log=0.0, deny=None); YAW["v"] = 0.0; time.sleep(0.9)
check("U0 默认平地赛段", C.STATE["segment"] == "flat" and C.LIVE.get("segment") == "flat", (C.STATE.get("segment"), C.LIVE.get("segment")))
pol = http("/api/state")["policies"]
check("U0 策略列表 = V1Down / V1H(备选) / ClimbH / StairN-C / StairN-E2(候选)", sorted(pol.keys()) == ["ClimbH_8797", "ClimbR13", "StairN-C_9300", "StairN-E2_10300", "StairN-E4_10600", "StairN-E4_11099", "StairN-E5_11200", "StairN-S3_13800", "StairN-S8_14800", "StairS16_15200", "T10_15197", "V1Down_10695", "V1HTrotH4_11500", "V1HTrotH6_12099", "V1HTrotH7_11800", "V1HTrotT3_13400", "V1H_9196"], sorted(pol.keys()))
r = http("/api/segment?s=stairs&pid=PAGEA"); check("U1 选楼梯赛段", r.get("ok") and C.STATE["segment"] == "stairs", (r, C.STATE.get("segment")))
check("U1 选赛段本身不切专家", C.STATE["climb"] == 0, C.STATE["climb"])
m = mark(); OPS.update(on=True, vx=0.5, wz=0.0); time.sleep(1.2)
ev = since(m)
check("U2 按住 W → 自动切楼梯专家（climb=2，发了 /climb_mode 2）", C.STATE["climb"] == 2 and ("climb_mode", 2) in ev, (C.STATE["climb"], [e for e in ev if e[0] != "cmd"][:4]))
cm = [e for e in ev if e[0] == "cmd"][-4:]
check("U2 专家里出指令 vx=0.3（档位）", cm and all(abs(e[1] - 0.3) < 1e-6 for e in cm), cm)
check("U2 航向目标 = 切入时偏航（0°），不记楼梯朝向", C.STATE.get("stair_yaw_ref") is not None and abs(C.STATE["stair_yaw_ref"]) < 1e-9 and C.STATE.get("stair_heading") is None, (C.STATE.get("stair_yaw_ref"), C.STATE.get("stair_heading")))
check("U2 日志有「赛段自动」切入", any("赛段自动" in l for l in logs()), logs()[:3])
m = mark(); OPS["on"] = False; time.sleep(1.2)
ev = since(m)
check("U3 松 W → 先 /climb_mode 0 再零速，climb=0，赛段仍是楼梯", C.STATE["climb"] == 0 and ("climb_mode", 0) in ev and C.STATE["segment"] == "stairs", (C.STATE["climb"], [e for e in ev if e[0] != "cmd"][:4], C.STATE["segment"]))
cm = [e for e in ev if e[0] == "cmd"][-3:]
check("U3 松 W 后出指令为零", cm and all(abs(e[1]) < 1e-6 for e in cm), cm)
m = mark(); OPS.update(on=True, vx=0.5); time.sleep(1.2)
check("U4 再按 W → 自动再切（climb=2）", C.STATE["climb"] == 2 and ("climb_mode", 2) in since(m), (C.STATE["climb"], [e for e in since(m) if e[0] != "cmd"][:4]))
OPS["on"] = False; time.sleep(1.0)
m = mark(); OPS.update(on=True, vx=-0.3); time.sleep(1.0)
ev = since(m); cm = [e for e in ev if e[0] == "cmd"][-3:]
check("U5 按 S（vx=-0.3）→ 主策略倒退，不切专家", C.STATE["climb"] == 0 and ("climb_mode", 2) not in ev and cm and all(abs(e[1] + 0.3) < 1e-6 for e in cm), (C.STATE["climb"], cm))
OPS["on"] = False; time.sleep(1.0)
FEED["info_on"] = False; time.sleep(0.8)          # /hmap_info 停更 → 地图质量未知 → 切不进
m = mark(); OPS.update(on=True, vx=0.5); time.sleep(1.5)
ev = since(m); cm = [e for e in ev if e[0] == "cmd"][-4:]
check("U6 地图质量未知时按 W：不切专家、主策略也不前进（vx=0）", C.STATE["climb"] == 0 and ("climb_mode", 2) not in ev and cm and all(abs(e[1]) < 1e-6 for e in cm), (C.STATE["climb"], cm))
time.sleep(0.8)
check("U6 /api/state 里 seg_deny 说明原因；日志记了「切不进楼梯专家 → 不前进」", bool(C.LIVE.get("seg_deny")) and any("切不进专家" in l for l in logs()), (C.LIVE.get("seg_deny"), logs()[:2]))
good(); time.sleep(1.5)
check("U6 地图恢复后自动切入（climb=2）", C.STATE["climb"] == 2, C.STATE["climb"])
r = http("/api/segment?s=flat&pid=PAGEA"); time.sleep(0.6)
check("U7 专家里选平地赛段 → 退出专家、赛段平地", r.get("ok") and C.STATE["climb"] == 0 and C.STATE["segment"] == "flat", (r, C.STATE["climb"], C.STATE["segment"]))
OPS["on"] = False; time.sleep(1.0); C.STATE["climb_cap"] = None
m = mark(); OPS.update(on=True, vx=0.5); time.sleep(1.0)
ev = since(m); cm = [e for e in ev if e[0] == "cmd"][-3:]
check("U7 平地赛段按 W → 主策略前进 0.5、不切专家", C.STATE["climb"] == 0 and ("climb_mode", 2) not in ev and cm and all(abs(e[1] - 0.5) < 1e-6 for e in cm), (C.STATE["climb"], cm))
OPS["on"] = False; time.sleep(0.8)
C.STATE["running_pit"] = None; r = http("/api/segment?s=stairs&pid=PAGEA")
check("U8 runner 没带楼梯槽时选楼梯赛段被拒", (not r.get("ok")) and C.STATE["segment"] == "flat", r)
C.STATE["running_pit"] = C.STAIR
C.STATE["segment"] = "stairs"; C._seg_reset("测试"); check("U8 runner 停/起 → 赛段回平地", C.STATE["segment"] == "flat", C.STATE["segment"])
r = http("/api/segment?s=stairs&pid=PAGEB"); check("U8 非操作员选赛段只读", r.get("readonly") and C.STATE["segment"] == "flat", r)
srcp = open(PATH, encoding="utf-8").read()
check("U9 页面有「楼梯赛段」「平地赛段」按钮和状态", "/api/segment?s=stairs" in srcp and "/api/segment?s=flat" in srcp and "id=segst" in srcp)
check("U9 代码里不再用楼梯朝向做闸门", "STAIR_HEAD_GATE_DEG:" not in srcp.split("def stair_gate")[1].split("def stair_enter")[0])
HB["PAGEA"] = False; time.sleep(0.3)
# ======================================================================= G：night4d 石笼赛段
print("---- G：石笼赛段（ClimbH，按住 W 自动 /climb_mode 1 以 0.4、松 W 交回主策略、S 倒退、换赛段）----")
HB["PAGEA"] = True; OPS["pid"] = "PAGEA"; OPS["on"] = False; time.sleep(1.0)
http("/api/op/release?pid=PAGEB"); r = http("/api/op/claim?pid=PAGEA"); check("G0 页面 A 认领操作权", r.get("ok"), r)
good(); time.sleep(0.6); base(); C.STATE["segment"] = "flat"; C.SEG.update(t_exit=0.0, t_try=0.0, t_log=0.0, deny=None); YAW["v"] = 0.0; time.sleep(0.9)
r = http("/api/segment?s=gabion&pid=PAGEA"); check("G1 选石笼赛段", r.get("ok") and C.STATE["segment"] == "gabion" and C.STATE["climb"] == 0, (r, C.STATE.get("segment"), C.STATE["climb"]))
m = mark(); OPS.update(on=True, vx=0.5, wz=0.0); time.sleep(1.2); ev = since(m)
check("G2 按住 W → 自动切 ClimbH（climb=1，发了 /climb_mode 1，没发 2）", C.STATE["climb"] == 1 and ("climb_mode", 1) in ev and ("climb_mode", 2) not in ev, (C.STATE["climb"], [e for e in ev if e[0] != "cmd"][:4]))
cm = [e for e in ev if e[0] == "cmd"][-4:]
check("G2 专家里出指令 vx=0.4（不是 0.5）", cm and all(abs(e[1] - 0.4) < 1e-6 for e in cm), cm)
m = mark(); OPS["on"] = False; time.sleep(1.2); ev = since(m)
check("G3 松 W → /climb_mode 0 + 零速，赛段仍是石笼", C.STATE["climb"] == 0 and ("climb_mode", 0) in ev and C.STATE["segment"] == "gabion", (C.STATE["climb"], C.STATE["segment"]))
m = mark(); OPS.update(on=True, vx=-0.3); time.sleep(1.0); ev = since(m); cm = [e for e in ev if e[0] == "cmd"][-3:]
check("G4 按 S → 主策略倒退，不切专家", C.STATE["climb"] == 0 and ("climb_mode", 1) not in ev and cm and all(abs(e[1] + 0.3) < 1e-6 for e in cm), (C.STATE["climb"], cm))
OPS["on"] = False; time.sleep(1.0)
m = mark(); OPS.update(on=True, vx=0.5); time.sleep(1.2)
check("G5 再按 W → 再切 ClimbH", C.STATE["climb"] == 1 and ("climb_mode", 1) in since(m), C.STATE["climb"])
r = http("/api/segment?s=stairs&pid=PAGEA"); time.sleep(0.5)
check("G6 ClimbH 里改选楼梯赛段 → 先退 ClimbH，赛段=楼梯", r.get("ok") and C.STATE["segment"] == "stairs" and C.STATE["climb"] != 1, (r, C.STATE["segment"], C.STATE["climb"]))
time.sleep(1.0); check("G6 按着 W 的情况下随后自动切进楼梯专家（climb=2）", C.STATE["climb"] == 2, C.STATE["climb"])
r = http("/api/segment?s=flat&pid=PAGEA"); time.sleep(0.6)
check("G7 选平地赛段 → 退出专家", r.get("ok") and C.STATE["climb"] == 0 and C.STATE["segment"] == "flat", (r, C.STATE["climb"], C.STATE["segment"]))
OPS["on"] = False; time.sleep(0.8)
C.STATE["running_climb"] = None; r = http("/api/segment?s=gabion&pid=PAGEA")
check("G8 runner 专家槽没装 ClimbH 时选石笼赛段被拒", (not r.get("ok")) and C.STATE["segment"] == "flat", r)
C.STATE["running_climb"] = C.EXPERT
srcp = open(PATH, encoding="utf-8").read()
check("G9 页面有三个赛段按钮和赛段表（下坎 ≤1.2 / 路缘 1.8 / 平台上先 Q/E）", "/api/segment?s=gabion" in srcp and "≤1.2" in srcp and "1.8" in srcp and "先 Q/E 对着下一段" in srcp)
HB["PAGEA"] = False; time.sleep(0.3)
# ======================================================================= E：night4e 选赛段自动调档位 / 顶栏
print("---- E：选赛段自动调档位、回平地恢复、顶栏 ----")
HB["PAGEA"] = True; OPS["pid"] = "PAGEA"; OPS["on"] = False; time.sleep(1.0)
http("/api/op/release?pid=PAGEB"); r = http("/api/op/claim?pid=PAGEA"); check("E0 页面 A 认领操作权", r.get("ok"), r)
good(); time.sleep(0.6); base(); C.STATE["segment"] = "flat"; C.SEG.update(t_exit=0.0, t_try=0.0, t_log=0.0, deny=None, flat_vx=None); C.STATE["max_vx"] = 0.3; time.sleep(0.5)
r = http("/api/segment?s=stairs&pid=PAGEA")
check("E1 选楼梯赛段：档位自动调到 0.3（官方速度）", r.get("ok") and abs(C.STATE["max_vx"] - 0.3) < 1e-9 and C.STATE["segment"] == "stairs", (r, C.STATE["max_vx"]))
check("E1 回复里写了档位", "0.3" in r.get("msg", ""), r.get("msg"))
r = http("/api/segment?s=gabion&pid=PAGEA")
check("E2 换石笼赛段：档位调到 0.4", r.get("ok") and abs(C.STATE["max_vx"] - 0.4) < 1e-9, (r, C.STATE["max_vx"]))
http("/api/lim?vx=0.6&wz=0.5&pid=PAGEA"); check("E2 赛段里滑块仍可调（0.6）", abs(C.STATE["max_vx"] - 0.6) < 1e-9, C.STATE["max_vx"])
r = http("/api/segment?s=flat&pid=PAGEA")
check("E3 回平地：恢复选赛段前的 0.3", r.get("ok") and abs(C.STATE["max_vx"] - 0.3) < 1e-9 and C.SEG.get("flat_vx") is None, (r, C.STATE["max_vx"], C.SEG.get("flat_vx")))
srcp = open(PATH, encoding="utf-8").read()
check("E4 速度控制在顶栏（急停条之后、grid 之前），方向控制卡里不再有滑块", srcp.index("id=speedbar") < srcp.index("<div class=grid>") and srcp.index('id=mvx ') < srcp.index("<div class=grid>") and "id=segtop" in srcp)
http("/api/op/release?pid=PAGEA"); HB["PAGEA"] = False; time.sleep(0.3)
# ======================================================================= F：night4f 前方地形判读（只显示）
print("---- F：前方地形判读 ----")
X = list(C._HS_X); Y = list(C._HS_Y)
def mk(riser_l=None, riser_r=None, h=0.12, holes=()):
    v = [-0.08] * 187
    for i in range(187):
        d = riser_l if Y[i] >= 0.15 - 1e-6 else (riser_r if Y[i] <= -0.15 + 1e-6 else (riser_l if riser_l is not None else riser_r))
        if d is not None and X[i] >= d - 1e-6: v[i] = -0.08 - h
    for i in holes: v[i] = 0.0
    return v
def feed(v):
    FEED["raw"] = v; FEED["raw_on"] = True; time.sleep(0.9); return C.LIVE.get("terrain")
good(); time.sleep(0.6)
t = feed(mk()); check("F1 平地 → kind 平地", t and t["kind"] == "平地" and t["rows"] == 0, t)
t = feed(mk(0.5, 0.5, 0.12)); check("F2 正对的 12 cm 台阶沿在 0.5 → 楼梯、沿 0.45、正对", t and t["kind"] == "楼梯" and abs(t["dist"] - 0.45) < 1e-6 and t["square"] is True and abs(t["h"] - 0.12) < 1e-6 and t["src"] == "raw", t)
t = feed(mk(0.4, 0.7, 0.12)); check("F3 左 0.4 / 右 0.7 → 歪了（square=False）", t and t["kind"] == "楼梯" and t["square"] is False and abs(t["dist_l"] - 0.35) < 1e-6 and abs(t["dist_r"] - 0.65) < 1e-6, t)
t = feed(mk(0.6, 0.6, 0.33)); check("F4 33 cm 抬高 → 台面", t and t["kind"] == "台面", t)
t = feed(mk(0.5, 0.5, 0.55)); check("F5 55 cm 抬高 → 障碍", t and t["kind"] == "障碍", t)
t = feed(mk(0.3, 0.3, 0.12)); check("F6 抬高从 0.3 起（TERR_X_MIN=0.25 以内的腿不算）→ 楼梯 沿 0.25", t and t["kind"] == "楼梯" and abs(t["dist"] - 0.25) < 1e-6, t)
hidx = [i for i in range(187) if X[i] >= 0.45 and Y[i] > 0.05][:60]
t = feed(mk(0.5, 0.5, 0.12, holes=hidx)); check("F7 左半边前方全是空洞 → 行数不够仍按剩余行判（右侧有值）", t is not None and (t["kind"] in ("楼梯", "平地")), t)
FEED["raw_on"] = False; FEED["hs"] = mk(0.5, 0.5, 0.12); time.sleep(1.8); t = C.LIVE.get("terrain")
check("F8 /height_scan_raw 停更 → 退回补洞后的图（src=filled）", t is not None and t.get("src") == "filled" and t.get("kind") == "楼梯", t)
FEED["hs"] = [-0.08] * 187; FEED["raw_on"] = True; FEED["raw"] = mk(); time.sleep(0.5); t = C.LIVE.get("terrain")
check("F8b 台阶消失后 1 s 内保持上次判读（held）", t is not None and t.get("held") is True and t.get("kind") == "楼梯", t)
time.sleep(1.2); t = C.LIVE.get("terrain"); check("F8c 超过 1 s 恢复平地", t is not None and t.get("kind") == "平地", t)
good(); time.sleep(0.6)
srcp = open(PATH, encoding="utf-8").read()
check("F9 页面顶栏有前方地形显示", "id=terr" in srcp and "Q/E 对正" in srcp)
# ======================================================================= T：night4a（只有孤儿 runner 判断）
print("---- T：孤儿 runner 只认可执行文件 ----")
HB["PAGEA"] = False; OPS["on"] = False; time.sleep(0.3)
import shutil, subprocess, tempfile
tmpd = tempfile.mkdtemp(prefix="n4_", dir=HERE)
exe = os.path.join(tmpd, "rl_deploy"); shutil.copy("/bin/sleep", exe); os.chmod(exe, 0o755)
old_exe = C.EXE_PATH; C.EXE_PATH = exe; C.STATE["runner_pid"] = None
real = subprocess.Popen([exe, "30"]); decoy = subprocess.Popen(["bash", "-c", "sleep 30; : " + exe]); decoy2 = subprocess.Popen(["python3", "-c", "import time; time.sleep(30)", exe])
time.sleep(0.5)
got = C.stray_runners()
check("T6 stray_runners 只找到真 runner %d，不算命令行里带路径的诱饵 %d/%d（得到 %s）" % (real.pid, decoy.pid, decoy2.pid, got), got == [real.pid], got)
pg = subprocess.run(["pgrep", "-f", exe], capture_output=True, text=True).stdout.split()
check("T6 对照：night3b 的 pgrep -f 会把诱饵也算进去", str(decoy.pid) in pg and str(decoy2.pid) in pg, pg)
C.STATE["runner_pid"] = real.pid; check("T6 记账里的 runner 不算孤儿", C.stray_runners() == [], C.stray_runners())
C.STATE["runner_pid"] = None
for pr in (real, decoy, decoy2):
    try: pr.kill(); pr.wait(timeout=2)
    except Exception: pass
time.sleep(0.2)
check("T6 进程退出后不再报孤儿", C.stray_runners() == [], C.stray_runners())
C.EXE_PATH = old_exe; shutil.rmtree(tmpd, ignore_errors=True)
HB["PAGEA"] = False; time.sleep(0.2)
src = open(PATH, encoding="utf-8").read()
check("T7 代码里不再用 pgrep -f 找 runner", "pgrep\", \"-f\", EXE_PATH" not in src and "/proc/%d/exe" in src)
check("P1 页面：send() 在旁观时直接返回（失焦发零也不发）", "if(estop||!armed||OFF||!OPR)return;" in src)
check("P1 页面：api() 自动带 pid；只读回复不弹窗", "u+=(u.includes('?')?'&':'?')+'pid='+CPID" in src and "if(j&&j.readonly){roFlash(); poll(); return j}" in src)
check("P1 页面：轮询带 pid（= 心跳），旁观时清掉按住的键", "fetch('/api/state?pid='+CPID)" in src and "if(!OPR&&held.size){held.clear();paint()}" in src)
check("P1 页面：热键、手柄在旁观时不发指令", "if(!OPR){" in src and "const ro=!OPR;" in src and "if(e3)climb(1,'手柄Y')" in src)
check("P1 页面：空洞率为 None 时显示「未知」", src.count("hs.hole==null?'未知'") == 2)
check("P2 页面和代码里不再出现「真实扫到 / 实扫 / 真实覆盖」", not any(k in src for k in ("真实扫到", "实扫", "真实覆盖", "under_cov\"")))

# ======================================================================= M：night4m 自动对正 / 侧倾冻结 / 转向上限
print("---- M：night4m ----")
C.SQUARE_AUTO = True   # night4n：M 段测闭环逻辑
HB["PAGEA"] = True; http("/api/op/claim?pid=PAGEA"); time.sleep(0.3)
good(); time.sleep(0.5); base(); YAW["v"] = 0.0; C.IMU_NOW["roll"] = 0.0
t = feed(mk(0.4, 0.7, 0.12))
check("M1 左 0.4 / 右 0.7 → yaw_off ≈ +21°（台沿法线在左，狗要左转）", t and t.get("yaw_off") is not None and 17 <= t["yaw_off"] <= 26, t)
check("M1b 左右差 0.3 > 0.10 → square False；TERR_SQUARE_M = 0.10", t and t["square"] is False and abs(C.TERR_SQUARE_M - 0.10) < 1e-9, (t, C.TERR_SQUARE_M))
t = feed(mk(0.45, 0.55, 0.12))
check("M2 左 0.45 / 右 0.55 → yaw_off ≈ +7°", t and t.get("yaw_off") is not None and 5 <= t["yaw_off"] <= 10, t)
time.sleep(1.6)   # night4n：等 2 s 中位窗口里只剩 +7° 的帧
r = try_enter("PAGEA"); ref = C.STATE["stair_yaw_ref"]
check("M3 切入时航向目标 = 当前偏航 0° + 台沿修正 ≈ +7°", r.get("ok") and C.STATE["climb"] == 2 and ref is not None and 4 <= math.degrees(ref) <= 10, (r, ref))
check("M3b 切入日志写了「按台沿修正」", any("按台沿修正" in l for l in logs()[:3]), logs()[:3])
t = feed(mk(0.6, 0.5, 0.12)); time.sleep(2.0)     # 台沿反向偏 ≈ −7° → 2 s 中位换过来后目标往 −7 方向改，每次 ≤5°
ref2 = C.STATE["stair_yaw_ref"]
check("M4 爬楼中复校：目标从 +7° 往 −7° 方向改、单次 ≤5°（现 ≈ +2°）", ref2 is not None and 0.5 <= math.degrees(ref2) <= 4.5 and any("楼梯复校航向" in l for l in logs()[:4]), (ref2, logs()[:3]))
C.IMU_NOW["roll"] = 0.30; time.sleep(1.8); ref3 = C.STATE["stair_yaw_ref"]
check("M5 侧倾 17° 时不复校（目标不变）", ref3 is not None and abs(ref3 - ref2) < 1e-9, (ref2, ref3))
with C.LK: w_roll = C.stair_wz(0.0)
C.IMU_NOW["roll"] = 0.0
with C.LK: w_ok = C.stair_wz(0.0)
check("M6 侧倾 17° 时航向纠偏输出 0；回平后恢复非 0", abs(w_roll) < 1e-9 and abs(w_ok) > 1e-6, (w_roll, w_ok))
with C.LK:
    C.STATE["stair_yaw_ref"] = 1.0; w_cap = C.stair_wz(0.0)
check("M7 转向上限 0.3（目标差 57° 也只给 0.3）", abs(abs(w_cap) - 0.3) < 1e-9 and abs(C.STAIR_WZ_MAX - 0.3) < 1e-9, (w_cap, C.STAIR_WZ_MAX))
release(); check("M8 松键照旧退出专家", C.STATE["climb"] == 0, C.STATE["climb"])
good(); time.sleep(0.5); base(); C.IMU_NOW["roll"] = 0.0
t = feed(mk()); r = try_enter("PAGEA")
check("M9 平地（没判到台阶）时切入：目标 = 当时朝向，日志写「台沿看不准」", r.get("ok") and C.STATE["stair_yaw_ref"] is not None and abs(math.degrees(C.STATE["stair_yaw_ref"])) < 0.5 and any("台沿看不准" in l for l in logs()[:3]), (r, C.STATE["stair_yaw_ref"], logs()[:2]))
release()
C.SQUARE_AUTO = False
good(); time.sleep(0.5); base(); C.IMU_NOW["roll"] = 0.0; YAW["v"] = 0.0
t = feed(mk(0.45, 0.55, 0.12)); r = try_enter("PAGEA")
check("M11 不闭环（默认）：台沿偏 +7° 只显示，目标 = 当时朝向 0°，日志写「只显示未闭环」", r.get("ok") and C.STATE["stair_yaw_ref"] is not None and abs(math.degrees(C.STATE["stair_yaw_ref"])) < 0.5 and any("只显示未闭环" in l for l in logs()[:3]), (r, C.STATE["stair_yaw_ref"], logs()[:2]))
time.sleep(0.8); tt2 = C.LIVE.get("terrain")
check("M11b 页面字段 yaw_med（2 s 中位）≈ +7°", tt2 and tt2.get("yaw_med") is not None and 5 <= tt2["yaw_med"] <= 10, tt2)
t = feed(mk(0.6, 0.5, 0.12)); time.sleep(2.0)
check("M11c 不闭环时爬楼中不复校", abs(math.degrees(C.STATE["stair_yaw_ref"])) < 0.5 and not any("楼梯复校航向" in l for l in logs()[:2]), (C.STATE["stair_yaw_ref"], logs()[:2]))
release()
srcm = open(PATH, encoding="utf-8").read()
check("M10 页面前方栏显示「未正对，需左/右转 N°」（优先 1 s 中位）", "'!! 未正对'" in srcm and "T.yaw_med!=null?T.yaw_med:T.yaw_off" in srcm)
http("/api/op/release?pid=PAGEA"); HB["PAGEA"] = False; time.sleep(0.3)
# ---- night4q：楼梯槽可选
check("Q1 默认楼梯槽仍是 9300", C.STAIR == "StairN-C_9300" and C.STAIR_ONNX.endswith("stairnc_9300.onnx") and C.STAIR_ONNX_MD5.startswith("c0a72923"), (C.STAIR, C.STAIR_ONNX))
check("Q2 表里有 E2 且策略列表有 E2 表项", "StairN-E2_10300" in C.STAIR_TABLE and "StairN-E4_10600" in C.STAIR_TABLE and "StairN-E4_11099" in C.STAIR_TABLE and "StairN-S3_13800" in C.STAIR_TABLE and "StairN-E2_10300" in C.POLICIES and C.POLICIES["StairN-E2_10300"].get("stair"), list(C.STAIR_TABLE))
# ---- night4r：简版首页 + 完整页 /full
r0 = {}; hh = C.H.__new__(C.H); hh.path = "/"; hh.client_address = ("127.0.0.9", 0); hh._s = lambda code, body, ct=None: r0.update(code=code, body=body, ct=ct); hh.do_GET()
r1 = {}; hh.path = "/full"; hh._s = lambda code, body, ct=None: r1.update(code=code, body=body, ct=ct); hh.do_GET()
check("R1 首页是简版（含一键接管），/full 是完整页", r0.get("code") == 200 and "一键接管" in r0.get("body", "") and r1.get("code") == 200 and "id=speedbar" in r1.get("body", "") and "一键接管" not in r1.get("body", ""), (r0.get("code"), r1.get("code")))
check("R3 简版页录制状态用 running 字段", "r.running" in r0.get("body","") and "r.on" not in r0.get("body",""))
check("R2 简版页只用已有接口", all(k in r0.get("body", "") for k in ("/api/op/claim", "/api/runner/start?yes=1", "/api/arm?on=1", "/api/mode?m=1", "/api/mode?m=6", "/api/setmode?m=0", "/api/op/release", "/api/runner/stop", "/api/segment?s=", "/api/lim?vx=", "/api/rec/start?name=", "/api/estop", "/api/clear", "/api/cmd?vx=")))
# ---- night4u：楼梯槽运行时可选 + 主策略候选
HB["PAGEA"] = True; http("/api/op/claim?pid=PAGEA"); time.sleep(0.3)
r = http("/api/stairpol?p=StairN-E5_11200&pid=PAGEA")
check("W1 /api/stairpol 选 E5 → STAIR/STAIR_ONNX/STAIR_ONNX_MD5 同步、state.stair_choice", r.get("ok") and C.STAIR == "StairN-E5_11200" and C.STAIR_ONNX.endswith("stairne5_11200.onnx") and C.STAIR_ONNX_MD5 == "315c942b" and C.STATE["stair_choice"] == "StairN-E5_11200", (r, C.STAIR, C.STAIR_ONNX))
r2 = http("/api/stairpol?p=NoSuch&pid=PAGEA")
check("W2 选表外名字被拒，不改", (not r2.get("ok")) and C.STAIR == "StairN-E5_11200", (r2, C.STAIR))
C.STATE["running_pit"] = "StairN-C_9300"; C.STATE["runner_pid"] = 4242
r3 = http("/api/stairpol?p=StairN-E2_10300&pid=PAGEA")
check("W3 runner 在跑且带的不是所选 → 回复提示要重起 runner", r3.get("ok") and "重起 runner" in r3.get("msg", ""), r3)
C.STATE["runner_pid"] = None; C.STATE["running_pit"] = C.STAIR
http("/api/stairpol?p=StairN-C_9300&pid=PAGEA"); check("W4 改回 9300", C.STAIR == "StairN-C_9300", C.STAIR)
r4 = http("/api/policy?p=V1HTrotH4_11500&pid=PAGEA"); check("W5 主策略可选 V1HTrotH4_11500", r4.get("ok") and C.STATE["policy"] == "V1HTrotH4_11500", (r4, C.STATE["policy"]))
http("/api/policy?p=V1H_9196&pid=PAGEA"); check("W5b 改回 V1H", C.STATE["policy"] == "V1H_9196", C.STATE["policy"])
srcw = open(PATH, encoding="utf-8").read()
check("W6 简版页有两个下拉（平地策略 / 楼梯策略）", "id=polsel" in srcw and "id=stairsel" in srcw and "/api/stairpol?p=" in srcw and "FLAT=['T10_15197','V1HTrotT3_13400','V1HTrotH7_11800','V1HTrotH4_11500','V1HTrotH6_12099'],CLIMBS=['ClimbR13','ClimbH_8797'],STAIRS=['StairS16_15200','StairN-S8_14800','StairN-S3_13800','StairN-E2_10300','StairN-E5_11200']" in srcw)
http("/api/op/release?pid=PAGEA"); HB["PAGEA"] = False; time.sleep(0.3)
# ---- night4z：切入后 1 s 内纠偏限 ±0.1
with C.LK:
    C.STATE["climb"] = 2; C.STATE["stair_yaw_ref"] = 1.0; C.STATE["stair_k"] = 2.0; C.STATE["climb_t0"] = time.time(); C.IMU_NOW["roll"] = 0.0
    w1 = C.stair_wz(0.0)
    C.STATE["climb_t0"] = time.time() - 2.0
    w2 = C.stair_wz(0.0)
    C.STATE["climb"] = 0
check("Z1 切入 1 s 内纠偏 |wz| ≤ 0.1，1 s 后回到 0.3 上限", abs(abs(w1) - 0.1) < 1e-9 and abs(abs(w2) - 0.3) < 1e-9, (w1, w2))
# ---- night5a：楼梯槽装 S3（能停）时楼梯赛段自动全程专家
HB["PAGEA"] = True; http("/api/op/claim?pid=PAGEA"); time.sleep(0.3)
good(); time.sleep(0.5); base(); C.STATE["segment"] = "flat"; C.SEG.update(t_exit=0.0, t_try=0.0, t_log=0.0, deny=None); YAW["v"] = 0.0; C.IMU_NOW["roll"] = 0.0
http("/api/stairpol?p=StairN-S3_13800&pid=PAGEA"); C.STATE["running_pit"] = "StairN-S3_13800"; C.STATE["pit_loaded"] = True
check("S1 楼梯槽 = S3，stair_can_stop 为真、seg_full_now 在楼梯赛段才为真", C.stair_can_stop() and not C.seg_full_now(), (C.STAIR, C.seg_full_now()))
m = mark(); r = http("/api/segment?s=stairs&pid=PAGEA"); time.sleep(1.2); ev = since(m); cm = [e for e in ev if e[0] == "cmd"][-3:]
check("S2 选楼梯 → 不按 W 也立刻切专家（climb=2），指令 0、档位默认 0.3", r.get("ok") and C.STATE["climb"] == 2 and ("climb_mode", 2) in ev and cm and all(abs(e[1]) < 1e-6 for e in cm) and abs(C.STATE["max_vx"] - 0.3) < 1e-9, (r, C.STATE["climb"], cm, C.STATE["max_vx"]))
check("S2b 回复文案说明专家能停", "能停" in r.get("msg", ""), r.get("msg"))
m = mark(); OPS.update(on=True, vx=0.5, wz=0.0); time.sleep(1.0); cm = [e for e in since(m) if e[0] == "cmd"][-3:]
check("S3 按 W → 0.3（档位）", C.STATE["climb"] == 2 and cm and all(abs(e[1] - 0.3) < 1e-6 for e in cm), cm)
m = mark(); OPS["on"] = False; time.sleep(1.2); ev = since(m); cm = [e for e in ev if e[0] == "cmd"][-3:]
check("S4 松 W → 仍在专家、指令 0、没发 /climb_mode 0", C.STATE["climb"] == 2 and ("climb_mode", 0) not in ev and cm and all(abs(e[1]) < 1e-6 for e in cm), (C.STATE["climb"], cm))
r = http("/api/segment?s=flat&pid=PAGEA"); time.sleep(0.6)
check("S5 点平地 → 退专家", r.get("ok") and C.STATE["climb"] == 0 and C.STATE["segment"] == "flat", (C.STATE["climb"], C.STATE["segment"]))
http("/api/stairpol?p=StairN-C_9300&pid=PAGEA"); C.STATE["running_pit"] = C.STAIR
check("S6 换回 9300 后楼梯赛段不再全程专家", not C.stair_can_stop(), C.STAIR)
srcs = open(PATH, encoding="utf-8").read()
check("S7 简版页楼梯下拉含 S8/S3，能停策略档位按训练范围过滤", "'StairN-S8_14800','StairN-S3_13800'" in srcs and "q.stop_ok&&r" in srcs)
HB["PAGEA"] = True; http("/api/op/claim?pid=PAGEA"); time.sleep(0.3)
good(); time.sleep(0.5); base(); C.STATE["segment"] = "flat"; C.SEG.update(t_exit=0.0, t_try=0.0, t_log=0.0, deny=None, flat_vx=None); C.STATE["max_vx"] = 0.3
http("/api/stairpol?p=StairN-S8_14800&pid=PAGEA"); C.STATE["running_pit"] = "StairN-S8_14800"; C.STATE["pit_loaded"] = True
r = http("/api/segment?s=stairs&pid=PAGEA"); time.sleep(1.2)
check("S8 楼梯槽 = S8：选楼梯默认档 0.25、立刻切专家", r.get("ok") and abs(C.STATE["max_vx"] - 0.25) < 1e-9 and C.STATE["climb"] == 2, (r, C.STATE["max_vx"], C.STATE["climb"]))
http("/api/segment?s=flat&pid=PAGEA"); time.sleep(0.5); http("/api/stairpol?p=StairN-C_9300&pid=PAGEA"); C.STATE["running_pit"] = C.STAIR
http("/api/op/release?pid=PAGEA"); HB["PAGEA"] = False; time.sleep(0.3)
# ---- night5g：三槽交付包 + 爬墙槽可选
HB["PAGEA"] = True; http("/api/op/claim?pid=PAGEA"); time.sleep(0.3)
check("N5g1 三个新策略都在表里、槽位属性正确",
      C.POLICIES["T10_15197"]["expert"] is False
      and C.POLICIES["StairS16_15200"].get("stair") and not C.POLICIES["StairS16_15200"].get("stop_ok")
      and C.POLICIES["ClimbR13"]["expert"] and not C.POLICIES["ClimbR13"].get("stair"),
      [C.POLICIES[k].get("expert") for k in ("T10_15197", "StairS16_15200", "ClimbR13")])
check("N5g2 三个 onnx 路径按交付文件名", C.POLICIES["T10_15197"]["onnx"] == "T10_15197/t10_15197.onnx"
      and C.POLICIES["StairS16_15200"]["onnx"] == "StairS16_15200/stairs16_15200.onnx"
      and C.POLICIES["ClimbR13"]["onnx"] == "ClimbR13/climb_r13.onnx")
r = http("/api/stairpol?p=StairS16_15200&pid=PAGEA")
check("N5g3 楼梯槽可选 S16，md5 前缀 faa37c9a", r.get("ok") and C.STAIR == "StairS16_15200" and C.STAIR_ONNX_MD5 == "faa37c9a", (r, C.STAIR, C.STAIR_ONNX_MD5))
check("N5g4 S16 不是能停策略 → 楼梯赛段仍按「按住 W 切、松开切回」", not C.stair_can_stop(), C.STAIR)
e0 = C.EXPERT
r = http("/api/climbpol?p=ClimbR13&pid=PAGEA")
check("N5g5 /api/climbpol 选 R13 → EXPERT 与 state.climb_choice 同步", r.get("ok") and C.EXPERT == "ClimbR13" and C.STATE["climb_choice"] == "ClimbR13", (r, C.EXPERT))
r2 = http("/api/climbpol?p=NoSuch&pid=PAGEA")
check("N5g6 爬墙槽选表外名字被拒、不改", (not r2.get("ok")) and C.EXPERT == "ClimbR13", (r2, C.EXPERT))
C.STATE["runner_pid"] = 4242; C.STATE["running_climb"] = "ClimbH_8797"
r3 = http("/api/climbpol?p=ClimbH_8797&pid=PAGEA")
check("N5g7 runner 在跑且带的不是所选 → 提示重起 runner", r3.get("ok"), r3)
C.STATE["runner_pid"] = None
http("/api/climbpol?p=%s&pid=PAGEA" % e0); http("/api/stairpol?p=StairN-C_9300&pid=PAGEA")
check("N5g8 改得回来", C.EXPERT == e0 and C.STAIR == "StairN-C_9300", (C.EXPERT, C.STAIR))
src5g = open(PATH, encoding="utf-8").read()
check("N5g9 页面有第三个下拉（石笼策略）", "id=climbsel" in src5g and "/api/climbpol?p=" in src5g and "CLIMBS=['ClimbR13','ClimbH_8797']" in src5g)
http("/api/op/release?pid=PAGEA"); HB["PAGEA"] = False; time.sleep(0.3)
print("失败 %d 项" % len(FAIL))
RUN["on"] = False
sys.exit(1 if FAIL else 0)
