#!/usr/bin/env python3
"""
Buat dataset YOLO (tiles + label) dari ortofoto + digitasi tajuk yang sudah ada.

Sumber label = hasil digitasi manual/otomatis kita sendiri
(`output/palm_crowns.gpkg`), jadi tidak perlu anotasi Roboflow dulu. Setiap
poligon tajuk jadi satu bounding box kelas 0 (oil palm).

    python3 make_tiles.py --ortho ../rasters/ORTO271026.tif \
        --crowns ../output/palm_crowns.gpkg --out dataset

Hasil:
    dataset/images/{train,val}/*.png
    dataset/labels/{train,val}/*.txt      (format YOLO: cls cx cy w h, 0..1)
    dataset/tiles.csv                     (offset tiap tile di raster penuh)
    dataset/data.yaml                     (dipakai train.py)
"""
import argparse, csv, os, random

import numpy as np
from osgeo import gdal, ogr
from PIL import Image

gdal.UseExceptions()
ogr.UseExceptions()


def tile_boxes(layer, x0, y0, w, h, gt):
    """Bounding box tiap tajuk di dalam tile, dalam koordinat tile (piksel)."""
    ulx, uly = gt[0] + x0 * gt[1], gt[3] + y0 * gt[5]
    lrx, lry = gt[0] + (x0 + w) * gt[1], gt[3] + (y0 + h) * gt[5]
    layer.SetSpatialFilterRect(min(ulx, lrx), min(uly, lry),
                               max(ulx, lrx), max(uly, lry))
    out = []
    for f in layer:
        g = f.GetGeometryRef()
        if g is None:
            continue
        env = g.GetEnvelope()          # minx, maxx, miny, maxy
        cx0 = (env[0] - gt[0]) / gt[1] - x0
        cx1 = (env[1] - gt[0]) / gt[1] - x0
        cy0 = (env[3] - gt[3]) / gt[5] - y0
        cy1 = (env[2] - gt[3]) / gt[5] - y0
        cx0, cx1 = max(0.0, min(cx0, cx1)), min(w, max(cx0, cx1))
        cy0, cy1 = max(0.0, min(cy0, cy1)), min(h, max(cy0, cy1))
        if cx1 - cx0 < 3 or cy1 - cy0 < 3:      # terlalu kecil / di tepi tile
            continue
        out.append(((cx0 + cx1) / 2 / w, (cy0 + cy1) / 2 / h,
                    (cx1 - cx0) / w, (cy1 - cy0) / h))
    layer.SetSpatialFilter(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", default="../rasters/ORTO271026.tif")
    ap.add_argument("--crowns", default="../output/palm_crowns.gpkg",
                    help="layer poligon tajuk (sumber label)")
    ap.add_argument("--layer", default=None, help="nama layer di dalam gpkg")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--tile", type=int, default=640, help="ukuran tile (px)")
    ap.add_argument("--stride", type=int, default=None,
                    help="langkah tile, default = --tile (tanpa overlap)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--empty-ratio", type=float, default=0.10,
                    help="0..1: peluang menyertakan tile tanpa pohon")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    stride = args.stride or args.tile
    ds = gdal.Open(args.ortho)
    W, H = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    alpha_band = 4 if ds.RasterCount >= 4 else None

    cds = ogr.Open(args.crowns)
    layer = (cds.GetLayerByName(args.layer) if args.layer
             else cds.GetLayer(0))
    srs_wkt = layer.GetSpatialRef().ExportToWkt()

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        for f in os.listdir(os.path.join(args.out, sub)):
            os.remove(os.path.join(args.out, sub, f))

    rows, n_tiles, n_boxes, n_empty = [], 0, 0, 0
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            w, h = min(args.tile, W - x0), min(args.tile, H - y0)
            if w < args.tile // 2 or h < args.tile // 2:
                continue                       # sisa tepi raster, lewati
            if alpha_band:
                a = ds.GetRasterBand(alpha_band).ReadAsArray(x0, y0, w, h)
                if a is None or a.max() == 0:
                    continue                   # di luar footprint terbang
            boxes = tile_boxes(layer, x0, y0, w, h, gt)
            if not boxes and random.random() > args.empty_ratio:
                continue
            split = "val" if random.random() < args.val_frac else "train"
            name = f"t{x0}_{y0}"
            rgb = np.dstack([ds.GetRasterBand(b).ReadAsArray(x0, y0, w, h)
                             for b in (1, 2, 3)]).astype(np.uint8)
            Image.fromarray(rgb).save(
                os.path.join(args.out, "images", split, name + ".png"),
                optimize=False)
            with open(os.path.join(args.out, "labels", split, name + ".txt"),
                      "w") as f:
                for cx, cy, bw, bh in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            rows.append(dict(tile=name, split=split, x0=x0, y0=y0, w=w, h=h,
                             boxes=len(boxes)))
            n_tiles += 1
            n_boxes += len(boxes)
            n_empty += (len(boxes) == 0)
            if n_tiles % 200 == 0:
                print(f"  {n_tiles} tiles, {n_boxes} boxes ...", flush=True)

    with open(os.path.join(args.out, "tiles.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["tile", "split", "x0", "y0", "w", "h", "boxes"])
        wr.writeheader()
        wr.writerows(rows)

    meta = dict(ortho=os.path.abspath(args.ortho), geotransform=list(gt),
                projection=srs_wkt, tile=args.tile, stride=stride,
                tiles=n_tiles, boxes=n_boxes)
    with open(os.path.join(args.out, "tiles_meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)

    yaml = (f"path: {os.path.abspath(args.out)}\n"
            "train: images/train\nval: images/val\n"
            "names:\n  0: oil_palm\n")
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(yaml)

    print(f"\n{n_tiles} tiles ({n_empty} tanpa pohon), {n_boxes} bounding box")
    print(f"train/val split + data.yaml -> {args.out}")


if __name__ == "__main__":
    main()
