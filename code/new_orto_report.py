#!/usr/bin/env python3
"""Gabungkan hasil deteksi Colab jadi satu set per ortofoto + QA visual.

Input : output/new_orto/ortoPMBN*.csv  (unduhan dari VM)
Output: output/new_orto/pohon_<orto>.{gpkg,geojson,csv}
        output/new_orto/qa_<orto>_overview.jpg, qa_<orto>_zoom.jpg
        output/new_orto/ringkasan.json

Dipisah dari `_colab_detect.py` supaya VM hanya melakukan inferensi; penulisan
GeoPackage dan render QA pakai GDAL lokal (butuh PROJ_DATA=/usr/share/proj).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from osgeo import gdal, ogr, osr
from PIL import Image, ImageDraw

gdal.UseExceptions()
ogr.UseExceptions()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUPS = {
    "ortoPMBN1": ["ortoPMBN1_s1", "ortoPMBN1_s2", "ortoPMBN1_s3", "ortoPMBN1_s4"],
    "ortoPMBN2": ["ortoPMBN2_47N"],
    "ortoPMBN3": ["ortoPMBN3_47N"],
}
RASTER = {
    "ortoPMBN1": "intermediates/new_orto_155/ortoPMBN1_47N.tif",
    "ortoPMBN2": "intermediates/new_orto_155/ortoPMBN2_47N.tif",
    "ortoPMBN3": "intermediates/new_orto_155/ortoPMBN3_47N.tif",
}
EPSG = 32647


def read_csv(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 5:
                rows.append((float(p[0]), float(p[1]), float(p[2]),
                             float(p[3]), float(p[4])))
    return rows


def dedupe_min_dist(pts, min_dist):
    """Buang titik yang lebih dekat dari min_dist (conf tertinggi menang).

    Filter jarak di VM berjalan per berkas strip, jadi di batas strip dua
    deteksi untuk tajuk yang sama bisa lolos (pusatnya bergeser beberapa
    meter). Fungsi ini menyapu ulang gabungan semua strip.
    """
    if min_dist <= 0 or len(pts) < 2:
        return pts
    from scipy.spatial import cKDTree

    coords = np.array([(p[0], p[1]) for p in pts])
    nbrs = cKDTree(coords).query_ball_point(coords, min_dist)
    keep = np.zeros(len(pts), dtype=bool)
    dropped = 0
    for i in np.argsort(-np.array([p[2] for p in pts])):
        if any(keep[j] for j in nbrs[i] if j != i):
            dropped += 1
            continue
        keep[i] = True
    print(f"  dedupe {min_dist} m global: {dropped} titik duplikat dibuang")
    return [p for p, k in zip(pts, keep) if k]


def footprint_ha(path):
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    sc = 2500.0 / max(W, H)
    ow, oh = max(64, int(W * sc)), max(64, int(H * sc))
    if ds.RasterCount >= 4:
        a = ds.GetRasterBand(4).ReadAsArray(buf_xsize=ow, buf_ysize=oh)
    else:
        a = np.full((oh, ow), 255, np.uint8)
    frac = float((a > 0).mean())
    return frac * W * H * abs(gt[1] * gt[5]) / 1e4


def write_vectors(base, pts, epsg=EPSG):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    gpkg = base + ".gpkg"
    if os.path.exists(gpkg):
        os.remove(gpkg)
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(gpkg)
    lyr = ds.CreateLayer("palm_detected", srs, ogr.wkbPoint)
    for fld in ("conf", "box_w_m", "box_h_m", "box_area_m2"):
        lyr.CreateField(ogr.FieldDefn(fld, ogr.OFTReal))
    defn = lyr.GetLayerDefn()
    lyr.StartTransaction()
    for x, y, c, bw, bh in pts:
        f = ogr.Feature(defn)
        f.SetField("conf", round(c, 3))
        f.SetField("box_w_m", round(bw, 2))
        f.SetField("box_h_m", round(bh, 2))
        f.SetField("box_area_m2", round(bw * bh, 2))
        g = ogr.Geometry(ogr.wkbPoint)
        g.AddPoint_2D(x, y)
        f.SetGeometry(g)
        lyr.CreateFeature(f)
    lyr.CommitTransaction()
    ds = None

    gj = base + ".geojson"
    if os.path.exists(gj):
        os.remove(gj)
    ds = ogr.GetDriverByName("GeoJSON").CreateDataSource(gj)
    lyr = ds.CreateLayer("palm_detected", srs, ogr.wkbPoint)
    lyr.CreateField(ogr.FieldDefn("conf", ogr.OFTReal))
    defn = lyr.GetLayerDefn()
    for x, y, c, _, _ in pts:
        f = ogr.Feature(defn)
        f.SetField("conf", round(c, 3))
        g = ogr.Geometry(ogr.wkbPoint)
        g.AddPoint_2D(x, y)
        f.SetGeometry(g)
        lyr.CreateFeature(f)
    ds = None

    with open(base + ".csv", "w") as f:
        f.write("x,y,conf,box_w_m,box_h_m\n")
        for x, y, c, bw, bh in pts:
            f.write(f"{x:.2f},{y:.2f},{c:.3f},{bw:.2f},{bh:.2f}\n")


def read_window(ds, cx, cy, size_m):
    gt = ds.GetGeoTransform()
    px = abs(gt[1])
    h = int(round(size_m / px))
    x0 = max(0, min(ds.RasterXSize - h, int((cx - gt[0]) / px - h / 2)))
    y0 = max(0, min(ds.RasterYSize - h, int((gt[3] - cy) / px - h / 2)))
    ow, oh = min(h, ds.RasterXSize - x0), min(h, ds.RasterYSize - y0)
    rgb = np.dstack([ds.GetRasterBand(b).ReadAsArray(x0, y0, ow, oh)
                      for b in (1, 2, 3)])
    a = ds.GetRasterBand(4).ReadAsArray(x0, y0, ow, oh)
    img = Image.fromarray(rgb.astype(np.uint8))
    mask = Image.fromarray((a > 0).astype(np.uint8) * 255)
    img.paste((20, 20, 20), mask=Image.eval(mask, lambda v: 255 - v))
    return img, gt, x0, y0, px


def draw_points(img, pts, gt, x0, y0, px, scale=1.0, r=5):
    d = ImageDraw.Draw(img)
    for x, y in pts:
        px_ = (x - gt[0]) / px - x0
        py_ = (y - gt[3]) / -px - y0
        cx, cy = px_ * scale, py_ * scale
        if -10 < cx < img.width + 10 and -10 < cy < img.height + 10:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 40, 40),
                      width=2)
            d.line([cx - r * 1.7, cy, cx + r * 1.7, cy], fill=(255, 255, 0), width=1)
            d.line([cx, cy - r * 1.7, cx, cy + r * 1.7], fill=(255, 255, 0), width=1)
    return img


def densest_window(pts, size_m, px_x, px_y):
    """Pusat kotak size_m dengan jumlah deteksi terbanyak."""
    a = np.array([(p[0], p[1]) for p in pts])
    best, centre = -1, (a[:, 0].mean(), a[:, 1].mean())
    for i in range(0, len(a), max(1, len(a) // 200)):
        x, y = a[i]
        n = ((np.abs(a[:, 0] - x) < size_m / 2) &
             (np.abs(a[:, 1] - y) < size_m / 2)).sum()
        if n > best:
            best, centre = n, (float(x), float(y))
    return centre, int(best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "output/new_orto"))
    ap.add_argument("--name", default=None,
                    help="nama orto tunggal, mis. ortoPMBN4 (default: 3 grup bawaan)")
    ap.add_argument("--parts", nargs="+", default=None,
                    help="nama CSV tanpa .csv untuk --name")
    ap.add_argument("--raster", default=None,
                    help="tif hasil resample (relatif root proyek) untuk --name")
    ap.add_argument("--epsg", type=int, default=EPSG,
                    help="CRS keluaran, default 32647 (pakai 32748 untuk UTM 48S)")
    ap.add_argument("--min-dist", type=float, default=5.0,
                    help="dedupe jarak minimum pada gabungan strip (0 = mati)")
    args = ap.parse_args()
    out = args.dir
    groups, raster, epsg = GROUPS, RASTER, EPSG
    if args.name:
        if not args.parts or not args.raster:
            raise SystemExit("--name butuh --parts dan --raster")
        groups = {args.name: args.parts}
        raster = {args.name: args.raster}
        epsg = args.epsg
    for k, v in raster.items():
        raster[k] = os.path.join(ROOT, v)
    os.makedirs(out, exist_ok=True)

    summary = {}
    for name, parts in groups.items():
        pts = []
        for p in parts:
            f = os.path.join(out, p + ".csv")
            if os.path.exists(f):
                pts += read_csv(f)
        if not pts:
            print(f"{name}: tidak ada CSV, lewati")
            continue
        pts.sort(key=lambda t: -t[2])
        pts = dedupe_min_dist(pts, args.min_dist)
        base = os.path.join(out, "pohon_" + name)
        write_vectors(base, pts, epsg)

        ha = footprint_ha(raster[name])
        ds = gdal.Open(raster[name])
        gt = ds.GetGeoTransform()
        px = abs(gt[1])

        ov = ds.GetRasterBand(4).ReadAsArray(buf_xsize=1000)
        scale = 1000.0 / ds.RasterXSize
        oh = int(ds.RasterYSize * scale)
        rgb = np.dstack([ds.GetRasterBand(b).ReadAsArray(buf_xsize=1000,
                                                          buf_ysize=oh)
                          for b in (1, 2, 3)])
        img = Image.fromarray(rgb.astype(np.uint8))
        draw_points(img, [(p[0], p[1]) for p in pts], gt, 0, 0, px,
                    scale=scale, r=2)
        img.save(os.path.join(out, f"qa_{name}_overview.jpg"), quality=88)

        centre, cnt = densest_window(pts, 200.0, px, px)
        zi, zgt, zx0, zy0, zpx = read_window(ds, centre[0], centre[1], 200.0)
        if zi.width > 1400:
            zi = zi.resize((1400, int(zi.height * 1400 / zi.width)))
        zscale = zi.width / 200.0 * zpx
        draw_points(zi, [(p[0], p[1]) for p in pts], zgt, zx0, zy0, zpx,
                    scale=zscale, r=6)
        zi.save(os.path.join(out, f"qa_{name}_zoom.jpg"), quality=90, subsampling=0)

        summary[name] = {
            "pohon": len(pts), "footprint_ha": round(ha, 2),
            "kepadatan_per_ha": round(len(pts) / ha, 1),
            "conf_median": round(float(np.median([p[2] for p in pts])), 3),
            "conf_p10": round(float(np.percentile([p[2] for p in pts], 10)), 3),
            "pohon_conf_ge_050": sum(1 for p in pts if p[2] >= 0.5),
            "box_area_median_m2": round(float(np.median(
                [p[3] * p[4] for p in pts])), 1),
            "raster": raster[name], "tile_px": ds.RasterXSize * ds.RasterYSize,
            "zoom_200m_pohon": cnt,
        }
        print(f"{name}: {len(pts)} pohon / {ha:.1f} ha = "
              f"{len(pts) / ha:.1f} pohon/ha, conf median "
              f"{summary[name]['conf_median']}")

    tot = sum(v["pohon"] for v in summary.values())
    tot_ha = sum(v["footprint_ha"] for v in summary.values())
    summary["_total"] = {"pohon": tot, "footprint_ha": round(tot_ha, 1),
                         "kepadatan_per_ha": round(tot / tot_ha, 1)}
    with open(os.path.join(out, "ringkasan.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"TOTAL {tot} pohon / {tot_ha:.1f} ha = {tot / tot_ha:.1f} pohon/ha")


if __name__ == "__main__":
    main()
