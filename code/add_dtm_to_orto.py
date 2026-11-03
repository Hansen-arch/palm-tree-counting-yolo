#!/usr/bin/env python3
"""
Attach elevation (DTM / DSM) to an orthophoto as an extra band.

An ortho GeoTIFF carries only the 4 visible bands (R,G,B,alpha); elevation is a
separate product.  This tool puts them in one file: the DTM is warped onto the
ortho's exact grid and added as band 5 (band name "DTM_elevation_m").

Modes
    # 1. empty template on the ortho grid: hand this to QGIS/your drone
    #    software as the target grid for the DTM export
    python3 add_dtm_to_orto.py --ortho rasters/ORTO271026.tif --template

    # 2. stack a DTM you already have onto the ortho grid
    python3 add_dtm_to_orto.py --ortho rasters/ORTO271026.tif --dtm my_dtm.tif \
            --out ORTO271026_with_DTM.vrt
    python3 add_dtm_to_orto.py --ortho rasters/ORTO271026.tif --dtm my_dtm.tif \
            --out ORTO271026_with_DTM.tif --cog

    # 3. check what is available in the project
    python3 add_dtm_to_orto.py --ortho rasters/ORTO271026.tif --check
"""
import argparse, os, subprocess, sys
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

DTM_BAND_NAME = "DTM_elevation_m"


def ortho_grid(path):
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    wkt = ds.GetProjection()
    ds = None
    te = (gt[0], gt[3] + h * gt[5], gt[0] + w * gt[1], gt[3])
    return w, h, wkt, (gt[1], gt[5]), te


def make_template(ortho, out):
    w, h, wkt, (px, py), te = ortho_grid(ortho)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(out, w, h, 1, gdal.GDT_Float32,
                    options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3",
                             "BIGTIFF=IF_SAFER"])
    ds.SetGeoTransform((te[0], px, 0.0, te[1], 0.0, py))
    ds.SetProjection(wkt)
    b = ds.GetRasterBand(1)
    b.SetDescription(DTM_BAND_NAME)
    b.SetNoDataValue(-9999.0)
    b.Fill(-9999.0)
    ds = None
    print(f"wrote empty DTM template {out} ({w}x{h}, {px:.6f} m, same grid as {os.path.basename(ortho)})")


def stack(ortho, dtm, out, as_cog=False):
    w, h, wkt, (px, py), te = ortho_grid(ortho)
    workdir = os.path.dirname(os.path.abspath(out)) or "."
    aligned = os.path.join(workdir, "_dtm_aligned.tif")
    gdal.Warp(aligned, dtm, options=gdal.WarpOptions(
        dstSRS=wkt, outputBounds=te, xRes=px, yRes=abs(py),
        width=w, height=h, resampleAlg="bilinear", dstNodata=-9999.0,
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3"]))
    print("DTM aligned to the ortho grid ->", aligned)

    if as_cog:
        rgb = os.path.join(workdir, "_orto_f32.tif")
        gdal.Translate(rgb, ortho, options=gdal.TranslateOptions(
            bandList=[1, 2, 3, 4], outputType=gdal.GDT_Float32,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3"]))
        vrt = os.path.join(workdir, "_stack.vrt")
        gdal.BuildVRT(vrt, [rgb, aligned], separate=True)
        gdal.Translate(out, vrt, options=gdal.TranslateOptions(
            format="COG", creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=3",
                                           "BIGTIFF=IF_SAFER"]))
        for p in (rgb, vrt):
            os.remove(p)
        print(f"wrote {out} (4 float RGB+alpha bands + band 5 = {DTM_BAND_NAME})")
    else:
        gdal.BuildVRT(out, [ortho, aligned], separate=True)
        ds = gdal.Open(out, gdal.GA_Update)
        ds.GetRasterBand(5).SetDescription(DTM_BAND_NAME)
        ds = None
        print(f"wrote {out} (bands 1-4 from the ortho, band 5 = {DTM_BAND_NAME})")

    ds = gdal.Open(aligned, gdal.GA_Update)
    z = ds.GetRasterBand(1).ReadAsArray()
    if z.size:
        good = z != -9999.0
        if good.any():
            print(f"elevation range: {np.nanmin(z[good]):.1f} .. {np.nanmax(z[good]):.1f} m "
                  f"(mean {np.nanmean(z[good]):.1f} m, {100.0*good.mean():.1f} % of pixels valid)")
    ds = None
    if aligned != out:
        os.remove(aligned)


def check(ortho):
    ds = gdal.Open(ortho)
    print(f"{os.path.basename(ortho)}: {ds.RasterXSize}x{ds.RasterYSize}, "
          f"{ds.RasterCount} bands")
    for i in range(1, ds.RasterCount + 1):
        b = ds.GetRasterBand(i)
        print(f"  band {i}: {gdal.GetDataTypeName(b.DataType)}, "
              f"colour={gdal.GetColorInterpretationName(b.GetColorInterpretation())}, "
              f"desc={b.GetDescription()!r}")
    md = ds.GetMetadata()
    el = {k: v for k, v in md.items() if any(s in k.lower() for s in ("elev", "height", "z", "dtm", "dsm"))}
    print("  elevation-looking metadata:", el if el else "none")
    ds = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", default="rasters/ORTO271026.tif")
    ap.add_argument("--dtm")
    ap.add_argument("--out", default="ORTO271026_with_DTM.vrt")
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--cog", action="store_true", help="write a Cloud-Optimised GeoTIFF instead of a VRT")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        check(args.ortho)
    elif args.template:
        out = args.out if args.out.endswith(".tif") else "ORTO271026_DTM_template.tif"
        make_template(args.ortho, out)
    elif args.dtm:
        if not os.path.exists(args.dtm):
            sys.exit(f"DTM not found: {args.dtm}")
        stack(args.ortho, args.dtm, args.out, args.cog)
    else:
        ap.error("give --check, --template or --dtm <file>")


if __name__ == "__main__":
    main()
