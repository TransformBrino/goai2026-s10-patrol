#!/usr/bin/env python3
"""S10 本体监控协议（UDP/TCP）编解码与探针 —— 按《S10软件开发指南 V0.0.1 (2026-06)》§1 实现。

用途（D0 现场 10 分钟级）：
  python3 s10proto.py selftest                                    # 离线自检：帧头字节布局
  python3 s10proto.py probe  --host 10.21.33.103 --port 30000     # 明文口探针：两种编码各发一次心跳，收回包
  python3 s10proto.py listen --host 10.21.33.103 --port 30000 --secs 10   # 持续心跳 + 打印 Model/Version/MotionState/Location
  python3 s10proto.py send axis --x 0.2                           # 单发一条（会动！只在架起时用）

已知不确定项（都要 D0 用真机验证，不要在纸上争）：
  1. Type/Command 线上编码：指南写 0x0010 0064 这类 32 位十六进制，但 JSON 不能带 0x 字面量；
     M20 手册与两份指南的附录都用短十进制（心跳 Type 100 / Command 100）。本工具默认按 32 位十六进制转十进制发，
     --encoding m20 则按短十进制发；probe 两种都试，谁收到 §1.5 通用响应谁对。
  2. 端口：S10 指南正文说 30004(DTLS)/30003(TLS) 默认加密、关加密需技术支持；但其附录示例仍是 10.21.31.103:30000 明文。
     本工具只做明文 UDP；DTLS 不实现（要证书/PSK，问技术支持）。
  3. 导航模式/真实轴指令/NAV_CMD 疑为 Pro 版独占（§1.5 错误码 0xE00A 注释「导航功能只有Pro版本支持」）。
     listen 打印 Version 字段（STD/PRO）就能定。
"""
import argparse
import json
import socket
import struct
import time
from datetime import datetime

SYNC = bytes([0xEB, 0x91, 0xEB, 0x90])
FMT_XML, FMT_JSON = 0x00, 0x01
VERSION = 0x01

# —— 指南里的 Type/Command（32 位十六进制原文） ——
T = dict(
    heartbeat=(0x00100064, 0x00000005),
    usage_mode=(0x00100002, 0x00500002),     # Items.Mode 0 常规 / 1 导航 / 2 辅助
    motion_state=(0x00100001, 0x00200002),   # Items.MotionParam 1 站立 / 4 趴下 / 17 RL / 2 关节阻尼
    gait=(0x00100001, 0x00300002),           # Items.GaitParam 0x1001 基础 / 0x1003 楼梯(标准) / 0x3002 平地(导航) / 0x3003 楼梯(导航)
    axis=(0x00100001, 0x00100002),           # X/Y/Z/Roll/Pitch/Yaw ∈ [-1,1]，仅常规/辅助模式
    real_axis=(0x00100001, 0x00110002),      # 同上但 m/s、rad/s，原样透传，仅导航模式
    sdk_mode=(0x00100005, 0x00300002),       # SDKEnable bool, Frequency int
    loc_reset=(0x00100003, 0x00200002),      # PosX PosY PosZ Yaw
    map_pose=(0x00100003, 0x00200001),       # → Location(0正常/1丢失) PosX PosY PosZ Roll Pitch Yaw
    nav_status=(0x00100003, 0x00110001),     # → Location, ObsState
    nav_task=(0x00100003, 0x00100001),       # → Value Status ErrorCode
)
# 机器人主动上报（发给最近一次心跳的来源 IP:port）
PUSH = {(0x00100064, 0x00F00000): "基础状态(2Hz)", (0x00100001, 0x00F00000): "运控状态(10Hz)",
        (0x00100002, 0x00F00000): "设备状态(2Hz)", (0x0010007F, 0x00F00000): "异常状态(2Hz)"}
# M20 手册 / 附录的短十进制写法（只确认过心跳）
M20 = dict(heartbeat=(100, 100))

_msg_id = 0
_pkt_no = 0


def build_asdu(type_, cmd, items=None):
    return {"PatrolDevice": {"Type": int(type_), "Command": int(cmd),
                             "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             "Items": items or {}}}


def encode(asdu, fmt=FMT_JSON):
    """16 字节帧头 + ASDU。帧头：同步4 | 长度2 LE | 报文ID2 LE | 格式1 | 包编号1 | 版本1 | 预留5"""
    global _msg_id, _pkt_no
    body = json.dumps(asdu, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    hdr = SYNC + struct.pack("<HH", len(body), _msg_id) + bytes([fmt, _pkt_no & 0xFF, VERSION]) + b"\x00" * 5
    _msg_id = (_msg_id + 1) & 0xFFFF
    _pkt_no = (_pkt_no + 1) & 0xFF
    assert len(hdr) == 16
    return hdr + body


def decode(buf):
    if len(buf) < 16 or buf[:4] != SYNC:
        raise ValueError("非本协议帧（前 4 字节 %s）" % buf[:4].hex())
    length, msg_id = struct.unpack("<HH", buf[4:8])
    fmt, pkt, ver = buf[8], buf[9], buf[10]
    body = buf[16:16 + length]
    data = json.loads(body.decode("utf-8")) if fmt == FMT_JSON else {"_xml": body.decode("utf-8", "replace")}
    return dict(length=length, msg_id=msg_id, fmt=fmt, pkt=pkt, ver=ver), data


def describe(data):
    pd = data.get("PatrolDevice", {})
    key = (pd.get("Type"), pd.get("Command"))
    items = pd.get("Items", {})
    name = PUSH.get(key) or next((k for k, v in T.items() if v == key), None) or "?"
    out = ["[%s] Type=0x%08x Cmd=0x%08x" % (name, pd.get("Type", 0), pd.get("Command", 0))]
    if "ErrorCode" in items and "ErrorMessage" in items:
        out.append("  通用响应 ErrorCode=0x%04x %s" % (int(items["ErrorCode"]), items["ErrorMessage"]))
    bs = items.get("BasicStatus")
    if bs:
        out.append("  Model=%s Version=%s  <- STD/PRO 在这里  MotionState=%s Gait=0x%x UsageMode=%s HES=%s" % (
            bs.get("Model"), bs.get("Version"), bs.get("MotionState"), int(bs.get("Gait", 0)),
            bs.get("ControlUsageMode"), bs.get("HES")))
    ms = items.get("MotionStatus")
    if ms:
        b = ms.get("Body", {})
        out.append("  Body rpy=(%s,%s,%s) wz=%s v=(%s,%s) h=%s" % (
            b.get("Roll"), b.get("Pitch"), b.get("Yaw"), b.get("OmegaZ"),
            ms.get("LinearX"), ms.get("LinearY"), ms.get("Height")))
    if "Location" in items:
        out.append("  Location=%s(0正常/1丢失) pos=(%s,%s,%s) yaw=%s ObsState=%s" % (
            items["Location"], items.get("PosX"), items.get("PosY"), items.get("PosZ"),
            items.get("Yaw"), items.get("ObsState")))
    if "ErrorList" in items:
        for e in items["ErrorList"]:
            out.append("  故障 0x%04x %s sev=%s src=%s" % (int(e.get("Code", 0)), e.get("Name"),
                                                         e.get("Severities"), e.get("Source")))
    if "Status" in items and "Value" in items:
        out.append("  导航任务 目标点=%s Status=%s Err=0x%04x" % (items["Value"], items["Status"],
                                                             int(items.get("ErrorCode", 0))))
    return "\n".join(out)


def make(cmd, encoding="s10", **kw):
    tc = (M20 if encoding == "m20" else T).get(cmd) or T[cmd]
    items = {}
    if cmd == "usage_mode":
        items = {"Mode": int(kw.get("mode", 0))}
    elif cmd == "motion_state":
        items = {"MotionParam": int(kw.get("param", 1))}
    elif cmd == "gait":
        items = {"GaitParam": int(kw.get("gait", 0x1001)), "ActionParam": 12288}
    elif cmd in ("axis", "real_axis"):
        items = {k: float(kw.get(k.lower(), 0.0)) for k in ("X", "Y", "Z", "Roll", "Pitch", "Yaw")}
    elif cmd == "sdk_mode":
        items = {"SDKEnable": bool(kw.get("enable", False)), "Frequency": int(kw.get("freq", 100))}
    elif cmd == "loc_reset":
        items = {k: float(kw.get(k.lower(), 0.0)) for k in ("PosX", "PosY", "PosZ", "Yaw")}
    return build_asdu(tc[0], tc[1], items)


def selftest():
    global _msg_id, _pkt_no
    _msg_id, _pkt_no = 7, 3
    frame = encode(make("heartbeat"))
    h = frame[:16]
    assert h[:4] == SYNC, "同步字"
    assert struct.unpack("<H", h[4:6])[0] == len(frame) - 16, "长度 = ASDU 字节数，小端"
    assert struct.unpack("<H", h[6:8])[0] == 7, "报文 ID 小端"
    assert h[8] == FMT_JSON and h[9] == 3 and h[10] == VERSION and h[11:16] == b"\0" * 5
    hdr, data = decode(frame)
    assert data["PatrolDevice"]["Type"] == 0x00100064 and data["PatrolDevice"]["Command"] == 5
    # 附录示例固定字节：header[6]=ID 低字节, header[7]=ID 高字节, header[8]=0x01(JSON) —— 与本实现一致
    ax = make("axis", x=0.3, yaw=-0.1)["PatrolDevice"]["Items"]
    assert ax == {"X": 0.3, "Y": 0.0, "Z": 0.0, "Roll": 0.0, "Pitch": 0.0, "Yaw": -0.1}
    m20 = make("heartbeat", encoding="m20")["PatrolDevice"]
    assert (m20["Type"], m20["Command"]) == (100, 100)
    print("selftest OK  帧头:", h.hex(" "), " ASDU:", frame[16:].decode())


def _sock(bind_port=0, timeout=1.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", bind_port))
    s.settimeout(timeout)
    return s


def probe(host, port, secs):
    """两种编码各发一次心跳，收 secs 秒，打印所有回包。谁有通用响应/上报谁就是对的编码。"""
    s = _sock(timeout=0.5)
    for enc in ("s10", "m20"):
        s.sendto(encode(make("heartbeat", encoding=enc)), (host, port))
        print("-> 心跳(%s) 已发至 %s:%d" % (enc, host, port))
    t0 = time.time()
    n = 0
    while time.time() - t0 < secs:
        try:
            buf, src = s.recvfrom(65535)
        except socket.timeout:
            continue
        n += 1
        try:
            hdr, data = decode(buf)
            print("<- %s id=%d fmt=%d\n%s" % (src, hdr["msg_id"], hdr["fmt"], describe(data)))
        except Exception as e:
            print("<- %s 无法解码 (%s) 前 32 字节: %s" % (src, e, buf[:32].hex(" ")))
    tail = "" if n else "  -> 明文口无响应：端口/编码不对，或默认加密生效（问技术支持关加密或给 DTLS 凭据）"
    print("共收到 %d 个数据报%s" % (n, tail))


def listen(host, port, secs, encoding):
    s = _sock(timeout=0.2)
    t0 = time.time()
    last_hb = 0.0
    seen = {}
    while time.time() - t0 < secs:
        if time.time() - last_hb > 0.9:      # >=1Hz 心跳，机器人才会往这个 IP:port 上报
            s.sendto(encode(make("heartbeat", encoding=encoding)), (host, port))
            last_hb = time.time()
        try:
            buf, src = s.recvfrom(65535)
        except socket.timeout:
            continue
        try:
            hdr, data = decode(buf)
        except Exception:
            continue
        pd = data.get("PatrolDevice", {})
        key = (pd.get("Type"), pd.get("Command"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] in (1, 20, 100):        # 每类只打印几次，免刷屏
            print("[%5.1fs] %s" % (time.time() - t0, describe(data)))
    print("统计:", {PUSH.get(k, "0x%08x/0x%08x" % k): v for k, v in seen.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="op", required=True)
    sub.add_parser("selftest")
    for name in ("probe", "listen", "send"):
        p = sub.add_parser(name)
        p.add_argument("--host", default="10.21.33.103")
        p.add_argument("--port", type=int, default=30000)
        p.add_argument("--secs", type=float, default=5.0)
        p.add_argument("--encoding", choices=("s10", "m20"), default="s10")
        if name == "send":
            p.add_argument("cmd", choices=sorted(T))
            for k in ("x", "y", "z", "roll", "pitch", "yaw", "posx", "posy", "posz"):
                p.add_argument("--" + k, type=float, default=0.0)
            p.add_argument("--mode", type=int, default=0)
            p.add_argument("--param", type=int, default=1)
            p.add_argument("--gait", type=lambda v: int(v, 0), default=0x1001)
            p.add_argument("--enable", action="store_true")
            p.add_argument("--freq", type=int, default=100)
    a = ap.parse_args()
    if a.op == "selftest":
        return selftest()
    if a.op == "probe":
        return probe(a.host, a.port, a.secs)
    if a.op == "listen":
        return listen(a.host, a.port, a.secs, a.encoding)
    if a.op == "send":
        s = _sock(timeout=a.secs)
        frame = encode(make(a.cmd, a.encoding, **vars(a)))
        s.sendto(frame, (a.host, a.port))
        print("->", frame[16:].decode())
        try:
            buf, src = s.recvfrom(65535)
            hdr, data = decode(buf)
            print("<-", describe(data))
        except socket.timeout:
            print("<- 无响应")


if __name__ == "__main__":
    main()
