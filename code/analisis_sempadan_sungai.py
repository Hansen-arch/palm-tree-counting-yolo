#!/usr/bin/env python3
"""Analisis titik pohon vs sempadan sungai 500 m dan badan sungai 5 m.

Input:
  SHP_SEGMEN500_SEMPADAN_SUNGAI_UTM47N/SEMPADAN_SEGMEN_500M_UTM47N.shp  (polygon)
  SHP_SEGMEN500_SEMPADAN_SUNGAI_UTM47N/SUNGAI_LEBAR_5M_UTM47N.shp       (polygon)
  output/new_orto/pohon_ortoPMBN{1,2,3}.csv                            (deteksi)
  intermediates/new_orto_155/ortoPMBN*.tif                             (grid 0.155 m)
  New Orto File/ortoPMBN*_47N.tif                                      (sumber clip)

Keluaran:
  output/sempadan/pohon_dalam_sempadan_500m.gpkg
  output/sempadan/pohon_dalam_sungai_5m.gpkg
  output/sempadan/pohon_per_segmen_sungai.csv
  output/sempadan/ringkasan_sempadan.json
  output/sempadan/clip/<orto>_{sempadan500m,sungai5m}.tif   (dengan --clip)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from osgeo import gdal, ogr

gdal.UseExceptions()
ogr.UseExceptions()

SHP_DIR = "SHP_SEGMEN500_SEMPADAN_SUNGAI_UTM47N"
SEMPADAN = os.path.join(SHP_DIR, "SEMPADAN_SEGMEN_500M_UTM47N.shp")
SUNGAI = os.path.join(SHP_DIR, "SUNGAI_LEBAR_5M_UTM47N.shp")

ORTHOS = {
    "ortoPMBN1": {"pts": "output/new_orto/pohon_ortoPMBN1.csv",
                  "grid": "intermediates/new_orto_155/ortoPMBN1_47N.tif",
                  "native": "New Orto File/ortoPMBN1_47N.tif"},
    "ortoPMBN2": {"pts": "output/new_orto/pohon_ortoPMBN2.csv",
                  "grid": "intermediates/new_orto_155/ortoPMBN2_47N.tif",
                  "native": "New Orto File/ortoPMBN2_47N.tif"},
    "ortoPMBN3": {"pts": "output/new_orto/pohon_ortoPMBN3.csv",
                  "grid": "intermediates/new_orto_155/ortoPMBN3_47N.tif",
                  "native": "New Orto File/ortoPMBN3_47N.tif"},
}
OUT = "output/sempadan"


def load_union(path):
    """Gabungkan semua fitur satu shapefile jadi satu geometri."""
    ds = ogr.Open(path)
    lyr = ds.GetLayer(0)
    srs = lyr.GetSpatialRef()
    feats = []
    for f in lyr:
        g = f.GetGeometryRef().Clone()
        if not g.IsValid():
            try:
                g = g.MakeValid()
            except Exception:
                pass
        feats.append((f.GetField("SEGMEN"), g))
    union = feats[0][1]
    for _, g in feats[1:]:
        union = union.Union(g)
    return union, srs, feats


def point(x, y):
    g = ogr.Geometry(ogr.wkbPoint)
    g.AddPoint_2D(x, y)
    return g


def read_pts(path):
    pts = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 3:
                pts.append((float(p[0]), float(p[1]), float(p[2])))
    return pts


def clip_polygon(shp, raster, dst_gpkg):
    """Iris polygon shp dengan footprint raster, simpan satu geometri valid.

    Cutline langsung dari shapefile gagal (TopologyException) dan GDAL lalu
    menulis SELURUH raster tanpa terpotong - karena itu irisannya dihitung
    dengan GEOS dulu. Kembalikan None kalau tidak ada irisan.
    """
    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    x0, y0 = gt[0], gt[3]
    x1 = x0 + ds.RasterXSize * gt[1]
    y1 = y0 + ds.RasterYSize * gt[5]
    ds = None
    foot = ogr.CreateGeometryFromWkt(
        f"POLYGON (({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))")

    vds = ogr.Open(shp)
    lyr = vds.GetLayer(0)
    srs = lyr.GetSpatialRef()
    union = None
    for f in lyr:
        g = f.GetGeometryRef().Clone()
        if not g.IsValid():
            g = g.MakeValid()
        union = g if union is None else union.Union(g)
    inter = union.Intersection(foot)
    if inter is None or inter.IsEmpty():
        return None, 0.0

    if os.path.exists(dst_gpkg):
        os.remove(dst_gpkg)
    ods = ogr.GetDriverByName("GPKG").CreateDataSource(dst_gpkg)
    olyr = ods.CreateLayer("cutline", srs, ogr.wkbMultiPolygon)
    of = ogr.Feature(olyr.GetLayerDefn())
    of.SetGeometry(inter)
    olyr.CreateFeature(of)
    ods = None
    return dst_gpkg, inter.GetArea() / 1e4


def clip_all(summary):
    """Potong ketiga ortofoto dengan sempadan dan sungai, resolusi asli."""
    os.makedirs(os.path.join(OUT, "clip"), exist_ok=True)
    tmp = os.path.join(OUT, "clip", "_cutline.gpkg")
    for name, cfg in ORTHOS.items():
        for tag, shp in (("sempadan500m", SEMPADAN), ("sungai5m", SUNGAI)):
            cut, luas = clip_polygon(shp, cfg["native"], tmp)
            if cut is None:
                print(f"{name} {tag}: tidak ada irisan, dilewati", flush=True)
                continue
            dst = os.path.join(OUT, "clip", f"{name}_{tag}.tif")
            gdal.Warp(dst, cfg["native"], options=gdal.WarpOptions(
                format="GTiff", cutlineDSName=cut, cropToCutline=True,
                dstAlpha=True, multithread=True,
                creationOptions=["TILED=YES", "COMPRESS=DEFLATE",
                                 "PREDICTOR=2", "BIGTIFF=IF_SAFER",
                                 "NUM_THREADS=ALL_CPUS"]))
            mb = os.path.getsize(dst) / 1e6
            print(f"clip {dst} ({mb:.1f} MB, luas {luas:.2f} ha)", flush=True)
            summary.setdefault("clip", {}).setdefault(name, {})[tag] = {
                "berkas": dst, "mb": round(mb, 1), "luas_ha": round(luas, 2),
                "gsd_m": clip_gsd(dst),
            }
    if os.path.exists(tmp):
        os.remove(tmp)
    with open(os.path.join(OUT, "ringkasan_sempadan.json"), "w") as f:
        json.dump(summary, f, indent=2)


def clip_gsd(raster):
    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    ds = None
    return round(abs(gt[1]), 4)


def valid_area_ha(raster):
    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    px_ha = abs(gt[1] * gt[5]) / 1e4
    if ds.RasterCount < 4:
        return ds.RasterXSize * ds.RasterYSize * px_ha
    a = ds.GetRasterBand(4).ReadAsArray() > 0
    ds = None
    return float(a.sum()) * px_ha


def masked_area_ha(raster, shp):
    """Luas piksel valid (alpha>0) yang masuk ke shp, dalam hektar."""
    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    m = gdal.GetDriverByName("GTiff").Create("/tmp/_mask.tif", W, H, 1,
                                             gdal.GDT_Byte)
    m.SetGeoTransform(gt)
    m.SetProjection(ds.GetProjection())
    gdal.Rasterize(m, shp, burnValues=[1])
    inside = m.GetRasterBand(1).ReadAsArray() == 1
    alpha = (ds.GetRasterBand(4).ReadAsArray() > 0) if ds.RasterCount >= 4 \
        else np.ones((H, W), bool)
    px_ha = abs(gt[1] * gt[5]) / 1e4
    m = None
    ds = None
    return float((inside & alpha).sum()) * px_ha


def write_gpkg(path, rows, srs):
    if os.path.exists(path):
        os.remove(path)
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(path)
    lyr = ds.CreateLayer("pohon", srs, ogr.wkbPoint)
    for name, typ in (("orto", ogr.OFTString), ("conf", ogr.OFTReal),
                      ("segmen", ogr.OFTInteger)):
        lyr.CreateField(ogr.FieldDefn(name, typ))
    defn = lyr.GetLayerDefn()
    lyr.StartTransaction()
    for (orto, conf, segmen), x, y in rows:
        f = ogr.Feature(defn)
        f.SetField("orto", orto)
        f.SetField("conf", conf)
        if segmen is not None:
            f.SetField("segmen", int(segmen))
        f.SetGeometry(point(x, y))
        lyr.CreateFeature(f)
    lyr.CommitTransaction()
    ds = None


def segment_of(feats, g):
    """Segmen yang benar-benar memuat titik; kalau hanya bersinggungan, ambil
    yang pertama."""
    for seg, geom in feats:
        if geom.Contains(g):
            return seg
    for seg, geom in feats:
        if geom.Intersects(g):
            return seg
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="store_true", help="ikut clip ortofoto")
    ap.add_argument("--resample", type=float, default=0.0,
                    help="GSD keluaran clip; 0 = resolusi asli")
    ap.add_argument("--clip-only", action="store_true",
                    help="hanya clip (hitungan sudah ada)")
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "clip"), exist_ok=True)
    if args.clip_only:
        clip_all({"clip": {}})
        return

    semp, srs, semp_feats = load_union(SEMPADAN)
    sung, _, sung_feats = load_union(SUNGAI)
    print(f"sempadan: {len(semp_feats)} segmen, {semp.GetArea()/1e4:.1f} ha")
    print(f"sungai  : {len(sung_feats)} segmen, {sung.GetArea()/1e4:.1f} ha")

    semp_rows, sung_rows, per_seg = [], [], {}
    summary = {"orto": {}, "sempadan_segmen": len(semp_feats),
               "sungai_segmen": len(sung_feats)}

    for name, cfg in ORTHOS.items():
        pts = read_pts(cfg["pts"])
        n_semp = n_sung = 0
        for x, y, c in pts:
            g = point(x, y)
            if sung.Intersects(g):
                n_sung += 1
                seg = segment_of(sung_feats, g)
                sung_rows.append(([name, round(c, 3), seg], x, y))
                per_seg[seg] = per_seg.get(seg, 0) + 1
            if semp.Intersects(g):
                n_semp += 1
                semp_rows.append(([name, round(c, 3), segment_of(semp_feats, g)], x, y))

        a_semp = masked_area_ha(cfg["grid"], SEMPADAN)
        a_sung = masked_area_ha(cfg["grid"], SUNGAI)
        a_tot = valid_area_ha(cfg["grid"])
        n_out = len(pts) - n_semp
        summary["orto"][name] = {
            "pohon_total": len(pts),
            "pohon_dalam_sempadan_500m": n_semp,
            "pohon_dalam_sungai_5m": n_sung,
            "pohon_di_luar_sempadan": n_out,
            "persen_dalam_sempadan": round(n_semp / len(pts) * 100, 1),
            "luas_valid_total_ha": round(a_tot, 2),
            "luas_valid_dalam_sempadan_ha": round(a_semp, 2),
            "luas_valid_dalam_sungai_ha": round(a_sung, 2),
            "kepadatan_dalam_sempadan_per_ha": round(n_semp / a_semp, 1) if a_semp else None,
            "kepadatan_luar_sempadan_per_ha":
                round(n_out / (a_tot - a_semp), 1) if a_tot - a_semp > 0 else None,
        }
        print(f"{name}: {len(pts)} pohon | dalam sempadan {n_semp} "
              f"({n_semp/len(pts)*100:.1f}%) | dalam sungai {n_sung} | "
              f"luas sempadan {a_semp:.2f} ha -> {n_semp/a_semp:.1f} pohon/ha | "
              f"luar sempadan {n_out}")

    write_gpkg(os.path.join(OUT, "pohon_dalam_sempadan_500m.gpkg"), semp_rows, srs)
    write_gpkg(os.path.join(OUT, "pohon_dalam_sungai_5m.gpkg"), sung_rows, srs)
    with open(os.path.join(OUT, "pohon_per_segmen_sungai.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segmen", "jumlah_pohon"])
        for seg in sorted(per_seg, key=lambda s: (s is None, s)):
            w.writerow([seg, per_seg[seg]])

    summary["total"] = {
        "pohon_semua_orto": sum(v["pohon_total"] for v in summary["orto"].values()),
        "pohon_dalam_sempadan_500m": sum(v["pohon_dalam_sempadan_500m"]
                                        for v in summary["orto"].values()),
        "pohon_dalam_sungai_5m": sum(v["pohon_dalam_sungai_5m"]
                                     for v in summary["orto"].values()),
    }
    with open(os.path.join(OUT, "ringkasan_sempadan.json"), "w") as f:
        json.dump(summary, f, indent=2)
    t = summary["total"]
    print(f"\nTOTAL: {t['pohon_semua_orto']} pohon | dalam sempadan 500 m "
          f"{t['pohon_dalam_sempadan_500m']} | dalam sungai 5 m "
          f"{t['pohon_dalam_sungai_5m']}")

    if args.clip:
        clip_all(summary)


if __name__ == "__main__":
    main()
