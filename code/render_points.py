#!/usr/bin/env python3
"""
QA cepat: gambar titik hasil deteksi di atas ortofoto (tanpa torch).

    python3 render_points.py --ortho ../rasters/ORTO271026.tif \
        --points hasil/ORTO271026_pohon.csv --crop 17447 10806 300 300 \
        --out hasil/qa_ORTO271026.jpg

--crop: x0 y0 lebar tinggi dalam piksel resolusi penuh (opsional; tanpa ini
seluruh raster dikecilkan ke --width).
"""
import argparse

import numpy as np
from osgeo import gdal
from PIL import Image

gdal.UseExceptions()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", default="../rasters/ORTO271026.tif")
    ap.add_argument("--points", required=True, help="csv hasil predict.py")
    ap.add_argument("--out", default="qa.jpg")
    ap.add_argument("--crop", nargs=4, type=int, default=None,
                    help="x0 y0 lebar tinggi (piksel penuh)")
    ap.add_argument("--width", type=int, default=900, help="lebar gambar keluaran")
    ap.add_argument("--circle-m", type=float, default=2.6,
                    help="radius lingkaran QA (m), 0 = tanpa lingkaran")
    args = ap.parse_args()

    ds = gdal.Open(args.ortho)
    gt = ds.GetGeoTransform()
    px = abs(gt[1])
    if args.crop:
        x0, y0, w, h = args.crop
    else:
        x0, y0, w, h = 0, 0, ds.RasterXSize, ds.RasterYSize
    w, h = min(w, ds.RasterXSize - x0), min(h, ds.RasterYSize - y0)
    arr = np.dstack([ds.GetRasterBand(b).ReadAsArray(x0, y0, w, h)
                     for b in (1, 2, 3)]).astype(np.uint8)
    ds = None

    import csv
    n = 0
    rad = int(round(args.circle_m / px))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1] if rad > 0 else (None, None)
    ring = np.abs(np.hypot(yy, xx) - rad) < 0.9 if rad > 0 else None
    with open(args.points) as f:
        for row in csv.DictReader(f):
            cx = (float(row["x"]) - gt[0]) / gt[1] - x0
            cy = (float(row["y"]) - gt[3]) / gt[5] - y0
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            n += 1
            cx, cy = int(round(cx)), int(round(cy))
            if ring is not None:
                ry, rx = cy + yy[ring], cx + xx[ring]
                ok = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
                arr[ry[ok], rx[ok]] = [0, 220, 255]
            arr[max(0, cy - 1):cy + 2, max(0, cx - 5):cx + 6] = [255, 255, 0]
            arr[max(0, cy - 5):cy + 6, max(0, cx - 1):cx + 2] = [255, 255, 0]

    scale = args.width / float(w)
    im = Image.fromarray(arr).resize((args.width, max(1, int(h * scale))),
                                     Image.LANCZOS)
    im.save(args.out, quality=88)
    print(f"{n} titik dalam jendela ini -> {args.out} ({im.size[0]}x{im.size[1]})")


if __name__ == "__main__":
    main()
