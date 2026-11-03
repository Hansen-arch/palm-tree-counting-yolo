#!/usr/bin/env python3
"""Pecah mosaik (GSD model) jadi strip yang muat diunggah ke Colab.

Mosaik 47N selebar 40.681 px x 90.867 px; setelah diturunkan ke GSD model
0.155138 m hasilnya masih ~18.7k x 41.8k px. Unggah Colab praktis terbatas
~100 MB per berkas, jadi dipotong per pita baris.

Tiap pita dipangkas dulu ke kotak alpha-nya sendiri, karena footprint-nya
koridor sungai: sebagian besar kanvas transparan dan tidak perlu diunggah.
Kalau satu strip masih > --max-mb, strip itu dibelah dua (rekursif).

Hasil: <out-dir>/stripN.tif + manifest.json (baris, ukuran, luas valid).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def strip_bbox(ds, y0, y1):
    """Kotak kolom berisi data valid di pita baris y0..y1."""
    a = ds.GetRasterBand(4).ReadAsArray(0, y0, ds.RasterXSize, y1 - y0)
    cols = np.nonzero((a > 0).any(axis=0))[0]
    if len(cols) == 0:
        return None
    return int(cols[0]), int(cols[-1]) + 1


def write_strip(ds, path, x0, y0, w, h):
    opt = gdal.TranslateOptions(
        format="GTiff", srcWin=[x0, y0, w, h],
        creationOptions=["COMPRESS=DEFLATE", "ZLEVEL=9", "PREDICTOR=2",
                         "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512",
                         "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"])
    gdal.Translate(path, ds, options=opt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="intermediates/merge_155/mosaik_155.tif")
    ap.add_argument("--out-dir", default="intermediates/merge_155/strips")
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--max-mb", type=float, default=90.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for f in os.listdir(args.out_dir):
        if f.endswith(".tif"):
            os.remove(os.path.join(args.out_dir, f))

    ds = gdal.Open(args.src)
    W, H = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    gsd = abs(gt[1])
    px_area = gsd * gsd / 1e4
    print(f"sumber {args.src}: {W} x {H} px @ {gsd:.6f} m")

    manifest, n = [], 0
    for band_y0 in range(0, H, args.rows):
        band_y1 = min(H, band_y0 + args.rows)
        bb = strip_bbox(ds, band_y0, band_y1)
        if bb is None:
            print(f"  baris {band_y0}-{band_y1}: seluruhnya transparan, dilewati")
            continue
        todo = [(bb[0], band_y0, bb[1] - bb[0], band_y1 - band_y0)]
        while todo:
            x0, y0, w, h = todo.pop(0)
            n += 1
            out = os.path.join(args.out_dir, f"strip{n}.tif")
            # tulis ke nama sementara dulu, supaya penomoran tidak bolong kalau
            # strip ternyata kebesaran dan harus dibelah
            tmp = out + ".part"
            write_strip(ds, tmp, x0, y0, w, h)
            mb = os.path.getsize(tmp) / 1e6
            if mb > args.max_mb and h > 400:
                os.remove(tmp)
                n -= 1
                half = h // 2
                todo = [(x0, y0, w, half), (x0, y0 + half, w, h - half)] + todo
                continue
            os.rename(tmp, out)
            a = gdal.Open(out)
            al = a.GetRasterBand(4).ReadAsArray()
            ha = float((al > 0).sum()) * px_area
            a = None
            manifest.append({"file": os.path.basename(out), "x0": x0, "y0": y0,
                             "w": w, "h": h, "mb": round(mb, 1),
                             "valid_ha": round(ha, 2)})
            print(f"  {os.path.basename(out):12s} {w}x{h} @({x0},{y0}) "
                  f"{mb:6.1f} MB  {ha:6.2f} ha valid")

    tot_mb = sum(m["mb"] for m in manifest)
    tot_ha = sum(m["valid_ha"] for m in manifest)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"src": args.src, "gsd": gsd, "strips": manifest,
                   "total_mb": round(tot_mb, 1),
                   "total_valid_ha": round(tot_ha, 2)}, f, indent=2)
    print(f"\n{len(manifest)} strip | {tot_mb:.0f} MB | {tot_ha:.1f} ha valid")


if __name__ == "__main__":
    main()
