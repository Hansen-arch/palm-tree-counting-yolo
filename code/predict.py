#!/usr/bin/env python3
"""
Inferensi tile-by-tile -> titik pohon (GeoJSON/GPKG/CSV).

Mengikuti notebook 03 dari repo oilpalm-tree-detection-yolov26: tile beroverlap,
centroid bbox -> koordinat peta, lalu filter jarak (default 5 m) untuk membuang
duplikat di batas tile.

Dua hal penting untuk kasus tiga ortofoto ini:

  * GSD berbeda-beda (0.155 / 0.099 / 0.071 m). Model dilatih pada tile 640 px
    dengan GSD 0.155 m, jadi tiap tile dibaca pada luas tanah yang sama
    (640 x 0.155 m) lalu di-resample ke 640 px -- `--target-gsd`. Tanpa ini
    pohon di PMBN_ORTO2 tampak 2x lebih besar daripada saat training.
  * PMBN_ORTO2.tif bertanda UTM 48S (EPSG:32748) padahal koordinatnya 48N.
    Pakai `--assume-srs EPSG:32648` supaya titik keluar di belahan yang benar.
"""
import argparse
import os
import sys

import numpy as np
from osgeo import gdal, ogr, osr
from scipy.spatial import cKDTree

gdal.UseExceptions()
ogr.UseExceptions()


def make_srs(wkt_or_epsg):
    srs = osr.SpatialReference()
    if wkt_or_epsg.upper().startswith("EPSG:"):
        srs.ImportFromEPSG(int(wkt_or_epsg.split(":")[1]))
    else:
        srs.ImportFromWkt(wkt_or_epsg)
    try:
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    return srs


def proximity_filter(points, min_dist):
    """Buang deteksi yang berdekatan (< min_dist), simpan conf tertinggi."""
    order = np.argsort(-points[:, 2])
    kept = []
    tree = None
    for i in order:
        p = points[i]
        if kept:
            if tree is None:
                tree = cKDTree(np.array([[k[0], k[1]] for k in kept]))
            if tree.query([[p[0], p[1]]], k=1)[0][0] < min_dist:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", default="../rasters/ORTO271026.tif")
    ap.add_argument("--weights", default="runs/oilpalm/weights/best.pt")
    ap.add_argument("--out", default="hasil/pohon", help="basename output")
    ap.add_argument("--tile", type=int, default=640, help="ukuran tile model (px)")
    ap.add_argument("--target-gsd", type=float, default=0.155138,
                    help="GSD training model (m/px); tile lain di-resample ke ini")
    ap.add_argument("--overlap", type=int, default=96,
                    help="overlap antar tile, dalam px tile model")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--min-dist", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=4, help="tile per prediksi")
    ap.add_argument("--assume-srs", default=None,
                    help="paksa CRS, mis. EPSG:32648 untuk PMBN_ORTO2 yang salah tag")
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = semua")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics belum terpasang: pip install -r requirements.txt")
    if not os.path.exists(args.weights):
        sys.exit(f"bobot tidak ditemukan: {args.weights}\n"
                 "Repo GitHub-nya tidak menyertakan .pt; taruh bobot sawit di sini "
                 "atau latih dulu dengan train.py")

    ds = gdal.Open(args.ortho)
    W, H = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    native = abs(gt[1])
    wkt = ds.GetProjection()
    alpha = 4 if ds.RasterCount >= 4 else None
    srs = make_srs(args.assume_srs or wkt)

    src = max(64, int(round(args.tile * args.target_gsd / native)))
    stride = max(1, src - int(round(args.overlap * args.target_gsd / native)))
    print(f"{os.path.basename(args.ortho)}: {W}x{H} px @ {native:.4f} m/px")
    print(f"  window {src} px ({src*native:.0f} m) -> {args.tile} px, "
          f"stride {stride} px, CRS {srs.GetAuthorityCode(None) or 'custom'}")

    model = YOLO(args.weights)
    dets, n_tile = [], 0
    buf, meta = [], []

    def flush():
        nonlocal buf, meta, n_tile
        if not buf:
            return
        results = model.predict(buf, imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, device=args.device, verbose=False)
        for res, (x0, y0, w, h, ow, oh) in zip(results, meta):
            sx, sy = w / ow, h / oh
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                c = float(b.conf[0])
                cx = x0 + (x1 + x2) / 2.0 * sx
                cy = y0 + (y1 + y2) / 2.0 * sy
                mx = gt[0] + cx * gt[1] + cy * gt[2]
                my = gt[3] + cx * gt[4] + cy * gt[5]
                dets.append((mx, my, c, (x2 - x1) * sx * native,
                             (y2 - y1) * sy * native))
        n_tile += len(buf)
        buf, meta = [], []
        if n_tile % 200 < args.batch:
            print(f"  {n_tile} tile, {len(dets)} deteksi ...", flush=True)

    for x0, y0, w, h in windows(W, H, src, stride):
        if alpha:
            a = ds.GetRasterBand(alpha).ReadAsArray(x0, y0, w, h)
            if a is None or a.max() == 0:
                continue
        ow = max(1, int(round(w * args.tile / src)))
        oh = max(1, int(round(h * args.tile / src)))
        rgb = np.dstack([ds.GetRasterBand(b).ReadAsArray(
            x0, y0, w, h, buf_xsize=ow, buf_ysize=oh) for b in (1, 2, 3)])
        buf.append(rgb.astype(np.uint8))
        meta.append((x0, y0, w, h, ow, oh))
        if len(buf) >= args.batch:
            flush()
        if args.max_tiles and n_tile >= args.max_tiles:
            break
    flush()

    if not dets:
        sys.exit("tidak ada deteksi - turunkan --conf atau periksa bobotnya")

    raw = np.array([(d[0], d[1], d[2]) for d in dets])
    kept = proximity_filter(raw, args.min_dist)
    keys = {(round(x, 3), round(y, 3)) for x, y, _ in kept}
    final = [d for d in dets if (round(d[0], 3), round(d[1], 3)) in keys]
    print(f"{len(dets)} deteksi mentah -> {len(final)} titik setelah filter "
          f"{args.min_dist} m")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    gpkg_path = args.out + ".gpkg"
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)
    gds = ogr.GetDriverByName("GPKG").CreateDataSource(gpkg_path)
    lyr = gds.CreateLayer("palm_detected", srs, ogr.wkbPoint)
    for fld, typ in (("conf", ogr.OFTReal), ("box_w_m", ogr.OFTReal),
                     ("box_h_m", ogr.OFTReal), ("box_area_m2", ogr.OFTReal)):
        lyr.CreateField(ogr.FieldDefn(fld, typ))
    defn = lyr.GetLayerDefn()
    lyr.StartTransaction()
    for mx, my, c, bw, bh in final:
        f = ogr.Feature(defn)
        f.SetField("conf", round(c, 3))
        f.SetField("box_w_m", round(bw, 2))
        f.SetField("box_h_m", round(bh, 2))
        f.SetField("box_area_m2", round(bw * bh, 2))
        g = ogr.Geometry(ogr.wkbPoint)
        g.AddPoint_2D(mx, my)
        f.SetGeometry(g)
        lyr.CreateFeature(f)
    lyr.CommitTransaction()
    gds = None

    gj_path = args.out + ".geojson"
    if os.path.exists(gj_path):
        os.remove(gj_path)
    gj = ogr.GetDriverByName("GeoJSON").CreateDataSource(gj_path)
    glyr = gj.CreateLayer("palm_detected", srs, ogr.wkbPoint)
    glyr.CreateField(ogr.FieldDefn("conf", ogr.OFTReal))
    gdefn = glyr.GetLayerDefn()
    for mx, my, c, _, _ in final:
        f = ogr.Feature(gdefn)
        f.SetField("conf", round(c, 3))
        g = ogr.Geometry(ogr.wkbPoint)
        g.AddPoint_2D(mx, my)
        f.SetGeometry(g)
        glyr.CreateFeature(f)
    gj = None

    with open(args.out + ".csv", "w") as f:
        f.write("x,y,conf,box_w_m,box_h_m\n")
        for mx, my, c, bw, bh in final:
            f.write(f"{mx:.2f},{my:.2f},{c:.3f},{bw:.2f},{bh:.2f}\n")

    print(f"tulis {gpkg_path} / .geojson / .csv")
    print(f"kerapatan kasar: {len(final)} pohon, {n_tile} tile diproses")


if __name__ == "__main__":
    main()
