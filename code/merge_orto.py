#!/usr/bin/env python3
"""Gabung 4 ortofoto 47N jadi satu GeoTIFF lossless (DEFLATE).

Masalah yang diselesaikan: tiga ortofoto drone saling tumpang tindih, jadi
pohon di area irisan terdeteksi dua kali saat dianalisis per-berkas. Mosaik
memilih satu sumber per piksel, sehingga irisan hanya muncul sekali.

Aturan prioritas (gdalbuildvrt: berkas TERAKHIR menang):
    ORTO 47N (0.155 m)  <  PMBN3 (0.136 m)  <  PMBN1 (0.099 m)  <  PMBN2 (0.071 m)
Sumber paling halus menang, dan PMBN2 (drone terbaru) selalu utuh.

Resolusi keluaran = GSD asli sumber terhalus (PMBN2, 0.071346 m), jadi tidak
ada detail yang dibuang, dan grid-nya dipaksa sejajar piksel PMBN2 supaya
sumber terhalus itu tidak ikut di-resample sama sekali. Sumber yang lebih
kasar di-upsample cubic ke grid itu (tidak menambah informasi, hanya
menyamakan grid; cubic dipilih karena tepi tajuk lebih tegas daripada
bilinear). Kompresi DEFLATE
ZLEVEL 9 + PREDICTOR 2 (horizontal differencing) bersifat lossless, jadi
piksel hasil dekompresi identik dengan mosaik sebelum kompresi.

Pemakaian:
    python3 merge_orto.py --build      # bikin VRT + tif deflate
    python3 merge_orto.py --verify     # cek ukuran, cakupan, kesetiaan piksel
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from osgeo import gdal, osr
from PIL import Image

gdal.UseExceptions()

# urutan = prioritas naik (terakhir paling dipakai)
SOURCES = [
    ("ORTO 47N", "ORTO & DEM 47N/ORTO 47N.tif", 0.155076),
    ("PMBN3", "New Orto File/ortoPMBN3_47N.tif", 0.135692),
    ("PMBN1", "New Orto File/ortoPMBN1_47N.tif", 0.098649),
    ("PMBN2", "new pmbn 2 orto/reprojectedPmbn2.tif", 0.071346),
]
OUT_DIR = "merged rasters"
VRT = os.path.join(OUT_DIR, "orto_47n_merge.vrt")
TIF = os.path.join(OUT_DIR, "orto_47n_merge.tif")
GSD = 0.071346  # GSD asli PMBN2 -> tidak ada downsampling sama sekali


def build() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    lst = os.path.join(OUT_DIR, "_sources.txt")
    with open(lst, "w") as f:
        for label, path, g in SOURCES:
            if not os.path.exists(path):
                sys.exit(f"sumber hilang: {path}")
            f.write(os.path.abspath(path) + "\n")
            print(f"  prioritas {label:<9} GSD {g:.6f}  {path}")

    x0, y0, x1, y1, gx, gy = _aligned_grid()
    print("\n[1/3] gdalbuildvrt (cubic, grid sejajar PMBN2, alpha = validity)")
    print(f"  te=({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f})  tr=({gx:.9f},{gy:.9f})")
    subprocess.run(["gdalbuildvrt", "-q",
                    "-te", f"{x0:.6f}", f"{y0:.6f}", f"{x1:.6f}", f"{y1:.6f}",
                    "-tr", f"{gx:.9f}", f"{gy:.9f}",
                    "-r", "cubic", "-input_file_list", lst, VRT], check=True)
    v = gdal.Open(VRT)
    gt = v.GetGeoTransform()
    W, H, nb = v.RasterXSize, v.RasterYSize, v.RasterCount
    print(f"  VRT {W}x{H} px @ {abs(gt[1]):.6f} m  nb={nb}  "
          f"{W * abs(gt[1]) * H * abs(gt[5]) / 1e4:.0f} ha kanvas")
    print(f"  kanvas {W * H / 1e6:.0f} Mpx  (mentah {W * H * nb / 1e9:.1f} GB)")
    v = None

    print("\n[2/3] gdal_translate -> DEFLATE ZLEVEL 9, PREDICTOR 2, tile 512")
    subprocess.run([
        "gdal_translate", "-q", VRT, TIF,
        "-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=9", "-co", "PREDICTOR=2",
        "-co", "TILED=YES", "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512",
        "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS", "-co", "ALPHA=YES",
    ], check=True)
    print(f"  selesai: {TIF}  {os.path.getsize(TIF) / 1e6:.0f} MB")

    # WAJIB: tanpa overview, QGIS/penampil lain menurunkan 3.7 Gpx dengan
    # nearest sampling -> tajuk sawit hilang saat zoom-out dan citra tampak
    # rata. Overview -r average menyimpan versi terdownsample yang benar.
    print("\n[3/4] gdaladdo: overview 2..512x (average, deflate)")
    subprocess.run(["gdaladdo", "-q", "-r", "average",
                    "--config", "COMPRESS_OVERWRITE", "DEFLATE",
                    "--config", "BIGTIFF_OVERVIEW", "IF_SAFER",
                    "--config", "GDAL_NUM_THREADS", "ALL_CPUS",
                    TIF, "2", "4", "8", "16", "32", "64", "128", "256", "512"],
                   check=True)
    print(f"  dengan overview: {os.path.getsize(TIF) / 1e6:.0f} MB")

    print("\n[4/4] buang VRT antara (daftar sumber tetap disimpan)")
    for p in (VRT, VRT + ".ovr"):
        if os.path.exists(p):
            os.remove(p)
    print("  OK")


def _aligned_grid() -> tuple[float, float, float, float, float, float]:
    """Grid mosaik yang SEJAJAR PIKsel dengan sumber terhalus (PMBN2).

    `-resolution highest` menaruh origin di union upper-left, yang bisa jatuh
    di tengah piksel PMBN2. Akibatnya PMBN2 di-resample bilinear walau GSD-nya
    sama -> tepi tajuk melembut (terukur: ketajaman mosaik hanya 0.82 dari
    sumber). Dengan origin digeser kelipatan GSD PMBN2, piksel PMBN2 dipakai
    apa adanya.
    """
    finest = gdal.Open(SOURCES[-1][1])
    fgt = finest.GetGeoTransform()
    gx, gy = fgt[1], -fgt[5]
    xs, ys = [], []
    for _label, path, _g in SOURCES:
        ds = gdal.Open(path)
        gt = ds.GetGeoTransform()
        xs += [gt[0], gt[0] + ds.RasterXSize * gt[1]]
        ys += [gt[3] + ds.RasterYSize * gt[5], gt[3]]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    ox, oy = fgt[0], fgt[3]
    x0 = ox - int(np.ceil((ox - xmin) / gx)) * gx
    y1 = oy + int(np.ceil((ymax - oy) / gy)) * gy
    x1 = x0 + int(np.ceil((xmax - x0) / gx)) * gx
    y0 = y1 - int(np.ceil((y1 - ymin) / gy)) * gy
    return x0, y0, x1, y1, gx, gy


def verify() -> None:
    ds = gdal.Open(TIF)
    gt = ds.GetGeoTransform()
    W, H, nb = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
    sr = osr.SpatialReference(wkt=ds.GetProjection())
    size = os.path.getsize(TIF)
    comp = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE")
    res = {
        "file": TIF,
        "size_mb": round(size / 1e6, 1),
        "size_gb": round(size / 1e9, 3),
        "px": f"{W}x{H}",
        "mpix": round(W * H / 1e6, 1),
        "bands": nb,
        "dtype": gdal.GetDataTypeName(ds.GetRasterBand(1).DataType),
        "gsd_m": round(abs(gt[1]), 6),
        "epsg": int(sr.GetAuthorityCode(None)),
        "compression": comp,
        "raw_gb": round(W * H * nb / 1e9, 2),
        "ratio_pct": round(size / (W * H * nb) * 100, 2),
        "overviews": ds.GetRasterBand(1).GetOverviewCount(),
        "extent": [round(gt[0], 1), round(gt[3] + H * gt[5], 1),
                   round(gt[0] + W * gt[1], 1), round(gt[3], 1)],
    }
    print(json.dumps(res, indent=2))

    # cakupan valid (alpha) -> harus ~jumlah luas valid sumber, bukan 340 ha dua kali
    sc = 4000.0 / max(W, H)
    a = ds.GetRasterBand(4).ReadAsArray(buf_xsize=max(64, int(W * sc)),
                                        buf_ysize=max(64, int(H * sc)))
    ha = float((a > 0).sum()) * GSD * GSD * (W / a.shape[1]) * (H / a.shape[0]) / 1e4
    print(f"luas valid (alpha>0): {ha:.1f} ha")
    print(f">= 3 GB? {'YA (melanggar)' if size > 3e9 else 'tidak'}")
    res["valid_ha"] = round(ha, 1)

    # Kesetiaan piksel: jendela 60 m dari sumber yang SAMA LUAS TANAHNYA di
    # mosaik (bukan sama jumlah piksel - GSD-nya beda, itu pernah bikin angka
    # palsu). Mosaik dikembalikan ke grid sumber pakai nearest sebelum dibanding.
    print("\nuji kesetiaan piksel (jendela 60 m, luas tanah sama):")
    checks = []
    for label, path, g in SOURCES:
        src = gdal.Open(path)
        sgt = src.GetGeoTransform()
        pt = _densest_point(src)
        if pt is None:
            print(f"  {label}: lewati (tidak ada data valid)")
            continue
        cx, cy = pt
        w_px = int(round(60.0 / abs(sgt[1])))
        m_px = int(round(60.0 / GSD))
        sx = int((cx - sgt[0]) / sgt[1]) - w_px // 2
        sy = int((sgt[3] - cy) / -sgt[5]) - w_px // 2
        mx = int((cx - gt[0]) / gt[1]) - m_px // 2
        my = int((gt[3] - cy) / -gt[5]) - m_px // 2
        s_arr = src.GetRasterBand(1).ReadAsArray(sx, sy, w_px, w_px).astype(np.float32)
        s_al = src.GetRasterBand(4).ReadAsArray(sx, sy, w_px, w_px)
        m_arr = ds.GetRasterBand(1).ReadAsArray(mx, my, m_px, m_px).astype(np.float32)
        m_al = ds.GetRasterBand(4).ReadAsArray(mx, my, m_px, m_px)
        m2 = np.asarray(Image.fromarray(m_arr).resize((w_px, w_px), Image.NEAREST))
        a2 = np.asarray(Image.fromarray(m_al).resize((w_px, w_px), Image.NEAREST))
        mask = (s_al > 127) & (a2 > 127)
        if mask.sum() < 500:
            print(f"  {label}: lewati (valid bersama {int(mask.sum())} px)")
            continue
        diff = np.abs(s_arr[mask] - m2[mask])
        corr = float(np.corrcoef(s_arr[mask], m2[mask])[0, 1])
        # ketajaman: beda antar-piksel bertetangga, mosaik vs sumber (1.0 = sama)
        mm = mask[:, :-1] & mask[:, 1:]
        d_src = np.abs(np.diff(s_arr, axis=1))[mm].mean()
        d_mos = np.abs(np.diff(m2.astype(np.float32), axis=1))[mm].mean()
        sharp = float(d_mos / (d_src + 1e-6))
        print(f"  {label}: {int(mask.sum())} px | corr {corr:.4f} | "
              f"median |diff| {np.median(diff):.0f} | ketajaman {sharp:.3f} | "
              f"mean {s_arr[mask].mean():.0f} vs {m2[mask].mean():.0f}")
        checks.append({"src": label, "n_px": int(mask.sum()),
                       "corr": round(corr, 4),
                       "median_abs_diff": float(np.median(diff)),
                       "sharpness_ratio": round(sharp, 3)})
    res["pixel_check"] = checks
    with open(os.path.join(OUT_DIR, "ringkasan_merge.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nringkasan ->", os.path.join(OUT_DIR, "ringkasan_merge.json"))


def _densest_point(src) -> tuple[float, float] | None:
    """Titik tengah baris terpadat data valid, supaya uji tidak jatuh di
    area transparan (kesalahan yang membuat uji pertama melaporkan 0)."""
    a = src.GetRasterBand(4).ReadAsArray(buf_xsize=800, buf_ysize=800)
    rows = (a > 127).sum(axis=1)
    r = int(np.argmax(rows))
    cols = np.nonzero(a[r] > 127)[0]
    if len(cols) == 0:
        return None
    c = int(cols[len(cols) // 2])
    sgt = src.GetGeoTransform()
    return (sgt[0] + (c + 0.5) / 800 * src.RasterXSize * sgt[1],
            sgt[3] + (r + 0.5) / 800 * src.RasterYSize * sgt[5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if not (a.build or a.verify):
        ap.error("pilih --build atau --verify")
    if a.build:
        build()
    if a.verify:
        verify()
