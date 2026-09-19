#!/usr/bin/env python3
"""点云 → 俯视高度图（PNG）+ 元数据（JSON），给 waypoint_picker.html 点航点用。

支持：.pcd（ASCII / binary，字段含 x y z）、.ply（ASCII / binary_little_endian，含 x y z）、
      .las/.laz（需 pip install laspy[lazrs]）、.npy（N×3）
用法：python3 cloud2map.py <点云文件> <输出前缀> [--res 0.10] [--zmin -1] [--zmax 3] [--origin auto|x,y]
输出：<前缀>.png（按高度上色，深=低、亮=高；白=无数据）、<前缀>.json（原点、分辨率、尺寸、z 范围）
坐标约定：地图系 x 向右、y 向上；图像行 0 在上，所以像素 (col,row) ↔ (x0 + col·res, y0 + (H−1−row)·res)。
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_pcd(p):
    data = Path(p).read_bytes()
    hdr_end = data.find(b"DATA ")
    hdr = data[:hdr_end + 40].decode("ascii", "replace").splitlines()
    fields = sizes = types = counts = None
    width = height = points = 0
    fmt = "ascii"
    for ln in hdr:
        t = ln.split()
        if not t:
            continue
        if t[0] == "FIELDS": fields = t[1:]
        elif t[0] == "SIZE": sizes = [int(v) for v in t[1:]]
        elif t[0] == "TYPE": types = t[1:]
        elif t[0] == "COUNT": counts = [int(v) for v in t[1:]]
        elif t[0] == "POINTS": points = int(t[1])
        elif t[0] == "DATA": fmt = t[1]; break
    counts = counts or [1] * len(fields)
    body_start = data.find(b"\n", hdr_end) + 1
    if fmt == "ascii":
        arr = np.loadtxt(data[body_start:].decode("ascii", "replace").splitlines(), dtype=float)
        ix = [fields.index(k) for k in ("x", "y", "z")]
        return arr[:, ix]
    if fmt != "binary":
        sys.exit("不支持 PCD DATA=%s（binary_compressed 请先用 pcl_convert_pcd_ascii_binary 转）" % fmt)
    dt = []
    for f, s, t, c in zip(fields, sizes, types, counts):
        np_t = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4"}[(t, s)]
        dt.append((f, np_t, (c,)) if c > 1 else (f, np_t))
    arr = np.frombuffer(data[body_start:body_start + points * np.dtype(dt).itemsize], dtype=np.dtype(dt))
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)


def read_ply(p):
    data = Path(p).read_bytes()
    end = data.find(b"end_header\n") + len(b"end_header\n")
    hdr = data[:end].decode("ascii", "replace").splitlines()
    fmt = "ascii"; n = 0; props = []
    for ln in hdr:
        t = ln.split()
        if t[:1] == ["format"]: fmt = t[1]
        elif t[:2] == ["element", "vertex"]: n = int(t[2]); props = []
        elif t[:1] == ["property"] and n and len(t) == 3: props.append((t[2], t[1]))
    if fmt == "ascii":
        arr = np.loadtxt(data[end:].decode("ascii", "replace").splitlines()[:n], dtype=float)
        ix = [[k for k, _ in props].index(k) for k in ("x", "y", "z")]
        return arr[:, ix]
    m = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2", "char": "i1"}
    dt = np.dtype([(k, m[t]) for k, t in props])
    arr = np.frombuffer(data[end:end + n * dt.itemsize], dtype=dt)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)


def read_las(p):
    try:
        import laspy
    except ImportError:
        sys.exit("读 LAS/LAZ 需要：pip install laspy[lazrs]")
    f = laspy.read(p)
    return np.stack([np.asarray(f.x), np.asarray(f.y), np.asarray(f.z)], axis=1).astype(float)


def load(p):
    ext = Path(p).suffix.lower()
    if ext == ".pcd": return read_pcd(p)
    if ext == ".ply": return read_ply(p)
    if ext in (".las", ".laz"): return read_las(p)
    if ext == ".npy": return np.load(p)[:, :3].astype(float)
    sys.exit("不认识的格式 " + ext)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud"); ap.add_argument("out")
    ap.add_argument("--res", type=float, default=0.10)
    ap.add_argument("--zmin", type=float, default=None); ap.add_argument("--zmax", type=float, default=None)
    ap.add_argument("--origin", default="auto", help="auto=点云最小角；或 'x,y' 指定图左下角的地图坐标")
    ap.add_argument("--stat", choices=("max", "p95", "median"), default="p95", help="每格高度取法（p95 抗离群）")
    ap.add_argument("--ref", default=None, help="'x,y'：航点/显示坐标的参考原点（如 LidarWorks 全局偏移 214765.68,3359032.95）；图像原点仍是点云最小角")
    a = ap.parse_args()
    pts = load(a.cloud)
    pts = pts[np.isfinite(pts).all(axis=1)]
    print("点数 %d  x[%.1f,%.1f] y[%.1f,%.1f] z[%.2f,%.2f]" % (len(pts), *pts[:, 0].min(axis=0, keepdims=True), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max(), pts[:, 2].min(), pts[:, 2].max()))
    x0, y0 = (pts[:, 0].min(), pts[:, 1].min()) if a.origin == "auto" else map(float, a.origin.split(","))
    W = int(np.ceil((pts[:, 0].max() - x0) / a.res)) + 1
    H = int(np.ceil((pts[:, 1].max() - y0) / a.res)) + 1
    col = ((pts[:, 0] - x0) / a.res).astype(int); row = (H - 1 - (pts[:, 1] - y0) / a.res).astype(int)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    col, row, z = col[ok], row[ok], pts[ok, 2]
    cell = row * W + col
    order = np.argsort(cell, kind="stable"); cell, z = cell[order], z[order]
    uniq, start = np.unique(cell, return_index=True)
    hmap = np.full(W * H, np.nan)
    if a.stat == "max":
        hmap[uniq] = np.maximum.reduceat(z, start)
    else:
        q = 0.95 if a.stat == "p95" else 0.5
        ends = np.append(start[1:], len(z))
        hmap[uniq] = [np.quantile(z[s:e], q) for s, e in zip(start, ends)]
    hmap = hmap.reshape(H, W)
    zmin = a.zmin if a.zmin is not None else float(np.nanpercentile(hmap, 1))
    zmax = a.zmax if a.zmax is not None else float(np.nanpercentile(hmap, 99))
    norm = np.clip((hmap - zmin) / max(zmax - zmin, 1e-6), 0, 1)
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    valid = ~np.isnan(hmap)
    # 深蓝(低)→绿→黄→红(高)，肉眼分得清路面/台阶/花坛
    t = norm[valid]
    r = np.clip(np.interp(t, [0, .35, .65, 1], [20, 30, 230, 200]), 0, 255)
    g = np.clip(np.interp(t, [0, .35, .65, 1], [40, 170, 210, 30]), 0, 255)
    b = np.clip(np.interp(t, [0, .35, .65, 1], [120, 60, 40, 30]), 0, 255)
    img[valid] = np.stack([r, g, b], axis=1).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(img).save(a.out + ".png")
    except ImportError:
        import zlib
        def png(arr):
            raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(arr.shape[0]))
            def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
            return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", arr.shape[1], arr.shape[0], 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
        Path(a.out + ".png").write_bytes(png(img))
    meta = dict(x0=float(x0), y0=float(y0), res=a.res, W=W, H=H, zmin=zmin, zmax=zmax, source=str(a.cloud),
                note="像素(col,row) ↔ 地图(x0+col*res, y0+(H-1-row)*res)；颜色 蓝低→红高，白=无点")
    if a.ref:
        rx, ry = map(float, a.ref.split(","))
        meta.update(ref_x=rx, ref_y=ry, note2="航点/显示坐标 = 地图坐标 − (ref_x, ref_y)")
    Path(a.out + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出 %s.png (%dx%d, %.2fm/px)  %s.json  z 显示范围 [%.2f, %.2f]" % (a.out, W, H, a.res, a.out, zmin, zmax))


if __name__ == "__main__":
    main()
