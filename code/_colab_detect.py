#!/usr/bin/env python3
"""Deteksi pohon sawit pada ortofoto baru DI DALAM VM Colab (GPU).

Pipeline sengaja dibuat identik dengan `predict.py` yang dipakai pada ORTO 47N:
  tile model 640 px, GSD acuan 0.155138 m/px, overlap 96 px, conf 0.25,
  NMS iou 0.7, filter jarak minimum 5 m.
Bedanya hanya pembaca raster: di sini `rasterio` (GDAL ikut di wheel) supaya
tidak perlu python3-gdal di Colab. Resampling nearest, sama seperti default
`GDALReadAsArray(buf_xsize=...)` di predict.py.

Pemakaian (dari laptop):
    colab exec -f _colab_detect.py --ortho /content/orto/ortoPMBN1_47N.tif \
        --weights /content/weights/roboflow_palm_best.pt

`colab exec -f` menambahkan argumen kernel ke sys.argv, jadi dipakai
parse_known_args().
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def proximity_filter(points: np.ndarray, min_dist: float) -> np.ndarray:
    """Buang deteksi berdekatan (< min_dist), simpan conf tertinggi.

    Salinan persis fungsi di predict.py supaya hasil 47N dan orto baru
    bisa dibandingkan apel-ke-apel.
    """
    from scipy.spatial import cKDTree

    order = np.argsort(-points[:, 2])
    kept = []
    tree = None
    for i in order:
        p = points[i]
        if kept:
            if tree is None:
                tree = cKDTree(np.array([[k[0], k[1]] for k in kept]))
            dist, _idx = tree.query([p[0], p[1]], k=1)
            if float(dist) < min_dist:
                continue
            kept.append(p)
            tree = cKDTree(np.array([[k[0], k[1]] for k in kept]))
        else:
            kept.append(p)
    return np.array(kept) if kept else np.empty((0, 3))


def windows(W, H, src, stride):
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            w, h = min(src, W - x0), min(src, H - y0)
            if w < src // 4 or h < src // 4:
                continue
            yield x0, y0, w, h


def footprint_ha(path: str) -> float:
    """Luas area valid (band alpha > 0) dalam hektar."""
    import rasterio

    with rasterio.open(path) as ds:
        px_area = abs(ds.transform.a * ds.transform.e)
        if ds.count < 4:
            return ds.width * ds.height * px_area / 1e4
        sc = 2000.0 / max(ds.width, ds.height)
        ow = max(64, int(ds.width * sc))
        oh = max(64, int(ds.height * sc))
        a = ds.read(4, out_shape=(oh, ow))
        return float((a > 0).sum()) * px_area * (ds.width / ow) * (ds.height / oh) / 1e4


def detect(path: str, model, args) -> dict:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    name = os.path.splitext(os.path.basename(path))[0]
    ds = rasterio.open(path)
    W, H = ds.width, ds.height
    aff = ds.transform
    native = abs(aff.a)
    gt = (aff.c, aff.a, aff.b, aff.f, aff.d, aff.e)  # GDAL GeoTransform
    alpha_idx = 4 if ds.count >= 4 else None

    src = max(64, int(round(args.tile * args.target_gsd / native)))
    stride = max(1, src - int(round(args.overlap * args.target_gsd / native)))
    epsg = ds.crs.to_epsg() if ds.crs else None
    print(f"\n=== {name}: {W}x{H} px @ {native:.4f} m/px, EPSG:{epsg}")
    print(f"  window {src} px ({src * native:.0f} m) -> {args.tile} px, "
          f"stride {stride} px", flush=True)

    t0 = time.time()
    dets, buf, meta, n_tile = [], [], [], 0

    def flush():
        nonlocal buf, meta, n_tile
        if not buf:
            return
        results = model.predict(buf, imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, device=args.device, verbose=False)
        for res, (x0, y0, w, h, ow, oh) in zip(results, meta):
            sx, sy = w / ow, h / oh
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].cpu()]
                c = float(b.conf[0].cpu())
                cx = x0 + (x1 + x2) / 2.0 * sx
                cy = y0 + (y1 + y2) / 2.0 * sy
                mx = gt[0] + cx * gt[1] + cy * gt[2]
                my = gt[3] + cx * gt[4] + cy * gt[5]
                dets.append((mx, my, c, (x2 - x1) * sx * native,
                             (y2 - y1) * sy * native))
        n_tile += len(buf)
        buf, meta = [], []
        if n_tile % 200 < args.batch:
            print(f"  {n_tile} tile, {len(dets)} deteksi, "
                  f"{time.time() - t0:.0f}s ...", flush=True)

    for x0, y0, w, h in windows(W, H, src, stride):
        if alpha_idx:
            a = ds.read(alpha_idx, window=Window(x0, y0, w, h))
            if a is None or a.max() == 0:
                continue
        ow = max(1, int(round(w * args.tile / src)))
        oh = max(1, int(round(h * args.tile / src)))
        rgb = ds.read([1, 2, 3], window=Window(x0, y0, w, h),
                      out_shape=(3, oh, ow), resampling=Resampling.nearest)
        buf.append(np.transpose(rgb, (1, 2, 0)).astype(np.uint8))
        meta.append((x0, y0, w, h, ow, oh))
        if len(buf) >= args.batch:
            flush()
        if args.max_tiles and n_tile >= args.max_tiles:
            break
    flush()
    ds.close()

    if not dets:
        print(f"  TIDAK ADA DETEKSI di {name}")
        return {"name": name, "n": 0}

    raw = np.array([(d[0], d[1], d[2]) for d in dets])
    kept = proximity_filter(raw, args.min_dist)
    keys = {(round(x, 3), round(y, 3)) for x, y, _ in kept}
    final = [d for d in dets if (round(d[0], 3), round(d[1], 3)) in keys]
    print(f"  {len(dets)} deteksi mentah -> {len(final)} titik setelah filter "
          f"{args.min_dist} m ({n_tile} tile, {time.time() - t0:.0f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{name}.csv")
    with open(csv_path, "w") as f:
        f.write("x,y,conf,box_w_m,box_h_m\n")
        for mx, my, c, bw, bh in final:
            f.write(f"{mx:.2f},{my:.2f},{c:.3f},{bw:.2f},{bh:.2f}\n")

    # titik sampel untuk pratinjau cepat di VM (tanpa perlu unduh CSV penuh)
    with open(os.path.join(args.out_dir, f"{name}_sample.csv"), "w") as f:
        f.write("x,y,conf\n")
        for mx, my, c, _, _ in final[::max(1, len(final) // 300)]:
            f.write(f"{mx:.2f},{my:.2f},{c:.3f}\n")

    ha = footprint_ha(path)
    stat = {
        "name": name, "n_raw": len(dets), "n": len(final),
        "n_tile": n_tile, "footprint_ha": round(ha, 2),
        "density_per_ha": round(len(final) / ha, 1) if ha else None,
        "conf_median": round(float(np.median([d[2] for d in final])), 3),
        "native_gsd_m": round(native, 6), "epsg": epsg,
        "window_px": src, "stride_px": stride,
        "seconds": round(time.time() - t0, 1), "csv": csv_path,
    }
    print("  STAT " + json.dumps(stat), flush=True)
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", action="append", default=[],
                    help="path tif di VM (boleh diulang)")
    ap.add_argument("--weights", default="/content/weights/roboflow_palm_best.pt")
    ap.add_argument("--out-dir", default="/content/new_ortho")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--target-gsd", type=float, default=0.155138)
    ap.add_argument("--overlap", type=int, default=96)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--min-dist", type=float, default=5.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tiles", type=int, default=0)
    args, _unknown = ap.parse_known_args()

    if not args.ortho:
        sys.exit("tidak ada --ortho")
    import torch
    from ultralytics import YOLO

    print("GPU:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)
    print("bobot:", args.weights, "| kelas:", model.names)

    stats = [detect(p, model, args) for p in args.ortho]
    total = sum(s.get("n", 0) for s in stats)
    ha = sum(s.get("footprint_ha", 0) or 0 for s in stats)
    print(f"\nTOTAL {total} pohon | {ha:.1f} ha | "
          f"{total / ha:.1f} pohon/ha" if ha else f"\nTOTAL {total} pohon")
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump({"orthos": stats, "total_trees": total,
                   "total_ha": round(ha, 2),
                   "density_per_ha": round(total / ha, 1) if ha else None},
                  f, indent=2)
    print("SELESAI | stats:", os.path.join(args.out_dir, "stats.json"))


if __name__ == "__main__":
    main()
