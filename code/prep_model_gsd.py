#!/usr/bin/env python3
"""Siapkan ortofoto baru agar bisa dideteksi model tanpa upload 2 GB.

Model dilatih pada GSD 0.155138 m/px. `predict.py` selalu menurunkan tile
sumber ke ukuran itu sebelum masuk YOLO, jadi menyalin ortofoto 0.03 / 0.099 /
0.136 m/px apa adanya ke VM hanya membuang bandwidth: 99% piksel dibuang saat
resampling di GPU.

Skrip ini melakukan resampling itu SEKALI di laptop:
  - RGB  : rata-rata (anti-alias, tidak ada aliasing saat turun 5x)
  - alpha: nearest, lalu di-threshold biner 0/255 (footprint tetap tegas,
           jadi tile di luar area terbang tetap dilewati seperti run 47N)
  - georeferensi & origin tidak diubah, hanya ukuran piksel

Hasil: GTiff deflate 4 band, native GSD = target GSD, sehingga di Colab
src = tile = 640 px dan pipeline-nya sama dengan `predict.py`.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
TARGET_GSD = 0.155138


def prep(src_path: str, dst_path: str, target: float) -> dict:
    src = gdal.Open(src_path)
    W, H, nband = src.RasterXSize, src.RasterYSize, src.RasterCount
    gt = src.GetGeoTransform()
    native = abs(gt[1])

    ow = int(round(W * native / target))
    oh = int(round(H * native / target))
    out_gt = (gt[0], target, 0.0, gt[3], 0.0, -target)

    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(dst_path, ow, oh, 4, gdal.GDT_Byte,
                     options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2",
                              "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"])
    dst.SetGeoTransform(out_gt)
    dst.SetProjection(src.GetProjection())

    for b in (1, 2, 3):
        arr = src.GetRasterBand(b).ReadAsArray(
            0, 0, W, H, buf_xsize=ow, buf_ysize=oh,
            resample_alg=gdal.GRIORA_Average)
        ob = dst.GetRasterBand(b)
        ob.WriteArray(arr)
        ob.SetColorInterpretation((gdal.GCI_RedBand, gdal.GCI_GreenBand,
                                   gdal.GCI_BlueBand)[b - 1])

    if nband >= 4:
        a = src.GetRasterBand(4).ReadAsArray(
            0, 0, W, H, buf_xsize=ow, buf_ysize=oh,
            resample_alg=gdal.GRIORA_NearestNeighbour)
        alpha = np.where(a > 127, 255, 0).astype(np.uint8)
    else:
        alpha = np.full((oh, ow), 255, np.uint8)
    ab = dst.GetRasterBand(4)
    ab.WriteArray(alpha)
    ab.SetColorInterpretation(gdal.GCI_AlphaBand)
    dst.SetMetadataItem("AREA_OR_POINT", "Area")
    dst = None

    crs = osr.SpatialReference(wkt=src.GetProjection())
    ha = float((alpha > 0).sum()) * target * target / 1e4
    stat = {
        "src": os.path.basename(src_path), "dst": os.path.basename(dst_path),
        "src_px": f"{W}x{H}", "dst_px": f"{ow}x{oh}",
        "native_gsd": round(native, 6), "target_gsd": target,
        "footprint_ha": round(ha, 2),
        "epsg": crs.GetAuthorityCode(None),
        "size_mb": round(os.path.getsize(dst_path) / 1e6, 1),
        "extent_ul": (round(gt[0], 1), round(gt[3], 1)),
    }
    print(f"{stat['src']}: {stat['src_px']} @ {native:.4f} -> "
          f"{stat['dst_px']} @ {target} | {stat['footprint_ha']} ha | "
          f"{stat['size_mb']} MB", flush=True)
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="../New Orto File")
    ap.add_argument("--out-dir", default="../intermediates/new_orto_155")
    ap.add_argument("--target-gsd", type=float, default=TARGET_GSD)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stats = []
    for f in sorted(os.listdir(args.src_dir)):
        if not f.lower().endswith(".tif"):
            continue
        dst = os.path.join(args.out_dir, f)
        if os.path.exists(dst):
            os.remove(dst)
        stats.append(prep(os.path.join(args.src_dir, f), dst,
                          args.target_gsd))
    print("\nselesai:", len(stats), "berkas ->", os.path.abspath(args.out_dir))
    total = sum(s["footprint_ha"] for s in stats)
    print(f"total footprint: {total:.1f} ha")


if __name__ == "__main__":
    main()
